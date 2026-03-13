"""
Direction-aware backtest engine for downtrend_breakout v1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    FEE_PCT,
    HOLD_TIMEOUT_BARS,
    INITIAL_CAPITAL,
    MAX_POSITIONS,
    POSITION_PCT,
    SLIPPAGE_PCT,
    STOP_LOSS_ATR_MULT,
    TAKE_PROFIT_R_MULT,
    TRAIL_ACTIVATION_R,
    TRAIL_ATR_MULT,
)


def compute_exit_levels(
    entry_price: float,
    direction: int,
    atr: float,
    stop_loss_atr_mult: float = STOP_LOSS_ATR_MULT,
    take_profit_r: float = TAKE_PROFIT_R_MULT,
) -> dict:
    """Compute directional stop-loss and take-profit from ATR risk distance."""
    risk_distance = max(float(atr) * float(stop_loss_atr_mult), 1e-9)

    if direction == 1:
        stop_loss = entry_price - risk_distance
        take_profit = entry_price + risk_distance * take_profit_r
    else:
        stop_loss = entry_price + risk_distance
        take_profit = entry_price - risk_distance * take_profit_r

    return {
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "risk_distance": risk_distance,
    }


def _apply_slippage(price: float, direction: int, is_entry: bool, slippage_pct: float) -> float:
    """Apply adverse slippage for entries and exits."""
    if direction == 1:
        # LONG: buy on entry, sell on exit.
        return price * (1 + slippage_pct) if is_entry else price * (1 - slippage_pct)
    # SHORT: sell on entry, buy on exit.
    return price * (1 - slippage_pct) if is_entry else price * (1 + slippage_pct)


def compute_metrics(trades_df: pd.DataFrame, equity_curve: np.ndarray, initial_capital: float) -> dict:
    """Compute portfolio-level performance metrics."""
    if equity_curve.size == 0:
        equity_curve = np.array([initial_capital], dtype=float)

    if len(trades_df) == 0:
        return {
            "total_return": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "total_trades": 0,
            "expectancy": 0.0,
            "final_capital": float(equity_curve[-1]),
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "long_trades": 0,
            "short_trades": 0,
        }

    total_return = (equity_curve[-1] - initial_capital) / initial_capital

    returns = np.diff(equity_curve) / np.where(equity_curve[:-1] == 0, np.nan, equity_curve[:-1])
    returns = returns[np.isfinite(returns)]
    sharpe = 0.0
    if returns.size > 1 and np.std(returns) > 0:
        sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(8760))

    peak = np.maximum.accumulate(equity_curve)
    dd = (equity_curve - peak) / np.where(peak == 0, np.nan, peak)
    dd = dd[np.isfinite(dd)]
    max_dd = float(dd.min()) if dd.size else 0.0

    pnl_col = "pnl_after_fees"
    wins = trades_df[trades_df[pnl_col] > 0]
    losses = trades_df[trades_df[pnl_col] <= 0]

    win_rate = len(wins) / len(trades_df)

    gross_profit = wins[pnl_col].sum()
    gross_loss = abs(losses[pnl_col].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

    avg_win = wins[pnl_col].mean() if len(wins) else 0.0
    avg_loss = abs(losses[pnl_col].mean()) if len(losses) else 0.0
    expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss

    long_trades = int((trades_df["direction"] == 1).sum()) if "direction" in trades_df.columns else 0
    short_trades = int((trades_df["direction"] == -1).sum()) if "direction" in trades_df.columns else 0

    return {
        "total_return": round(float(total_return), 4),
        "sharpe": round(float(sharpe), 2),
        "max_drawdown": round(float(max_dd), 4),
        "win_rate": round(float(win_rate), 4),
        "profit_factor": round(float(profit_factor), 2),
        "total_trades": int(len(trades_df)),
        "expectancy": round(float(expectancy), 6),
        "final_capital": round(float(equity_curve[-1]), 2),
        "avg_win": round(float(avg_win), 4),
        "avg_loss": round(float(-avg_loss), 4),
        "long_trades": long_trades,
        "short_trades": short_trades,
    }


def run_backtest(
    df: pd.DataFrame,
    initial_capital: float = INITIAL_CAPITAL,
    stop_loss_atr_mult: float = STOP_LOSS_ATR_MULT,
    take_profit_r: float = TAKE_PROFIT_R_MULT,
    hold_timeout: int = HOLD_TIMEOUT_BARS,
    trail_activation_r: float = TRAIL_ACTIVATION_R,
    trail_atr_mult: float = TRAIL_ATR_MULT,
    position_pct: float = POSITION_PCT,
    max_positions: int = MAX_POSITIONS,
    fee_pct: float = FEE_PCT,
    slippage_pct: float = SLIPPAGE_PCT,
) -> dict:
    """Run a long+short backtest on precomputed signals."""
    capital = float(initial_capital)
    positions: list[dict] = []
    trades: list[dict] = []
    equity_curve = []

    for i in range(len(df)):
        row = df.iloc[i]
        current_close = float(row["close"])
        current_high = float(row["high"])
        current_low = float(row["low"])
        ts = row["timestamp"]
        current_atr = row.get("atr14", np.nan)

        # ── 1) manage exits ───────────────────────────────────
        to_close = []
        for j, pos in enumerate(positions):
            pos["bars_held"] += 1
            direction = int(pos["direction"])

            atr_for_trail = float(current_atr) if pd.notna(current_atr) and current_atr > 0 else float(pos["atr_entry"])

            if direction == 1:
                if not pos["trail_active"] and current_high >= pos["entry_price"] + trail_activation_r * pos["risk_distance"]:
                    pos["trail_active"] = True
                if pos["trail_active"]:
                    pos["best_price"] = max(pos["best_price"], current_high)
                    trail_stop = pos["best_price"] - trail_atr_mult * atr_for_trail
                    pos["stop_loss"] = max(pos["stop_loss"], trail_stop)

                if current_low <= pos["stop_loss"]:
                    exit_price = pos["stop_loss"]
                    exit_reason = "stop_loss"
                elif current_high >= pos["take_profit"]:
                    exit_price = pos["take_profit"]
                    exit_reason = "take_profit"
                elif pos["bars_held"] >= hold_timeout:
                    exit_price = current_close
                    exit_reason = "timeout"
                else:
                    continue
            else:
                if not pos["trail_active"] and current_low <= pos["entry_price"] - trail_activation_r * pos["risk_distance"]:
                    pos["trail_active"] = True
                if pos["trail_active"]:
                    pos["best_price"] = min(pos["best_price"], current_low)
                    trail_stop = pos["best_price"] + trail_atr_mult * atr_for_trail
                    pos["stop_loss"] = min(pos["stop_loss"], trail_stop)

                if current_high >= pos["stop_loss"]:
                    exit_price = pos["stop_loss"]
                    exit_reason = "stop_loss"
                elif current_low <= pos["take_profit"]:
                    exit_price = pos["take_profit"]
                    exit_reason = "take_profit"
                elif pos["bars_held"] >= hold_timeout:
                    exit_price = current_close
                    exit_reason = "timeout"
                else:
                    continue

            exit_exec = _apply_slippage(exit_price, direction=direction, is_entry=False, slippage_pct=slippage_pct)
            if direction == 1:
                pnl_pct = (exit_exec - pos["entry_price"]) / pos["entry_price"]
            else:
                pnl_pct = (pos["entry_price"] - exit_exec) / pos["entry_price"]

            pnl_after_fees = pnl_pct - fee_pct
            capital_change = pos["allocated"] * pnl_after_fees
            capital += pos["allocated"] + capital_change

            trades.append(
                {
                    "entry_time": pos["entry_time"],
                    "exit_time": ts,
                    "entry_price": round(pos["entry_price"], 6),
                    "exit_price": round(exit_exec, 6),
                    "direction": direction,
                    "pnl_pct": round(float(pnl_pct), 6),
                    "pnl_after_fees": round(float(pnl_after_fees), 6),
                    "exit_reason": exit_reason,
                    "duration_bars": int(pos["bars_held"]),
                    "entry_reason": pos.get("entry_reason", ""),
                }
            )

            to_close.append(j)

        for j in sorted(to_close, reverse=True):
            positions.pop(j)

        # ── 2) open new entries ───────────────────────────────
        open_slots = max_positions - len(positions)
        signal = row.get("signal", "NO TRADE")
        direction = int(row.get("signal_direction", 0))

        if open_slots > 0 and capital > 0 and signal in ("LONG", "SHORT") and direction in (1, -1):
            atr = row.get("atr14", np.nan)
            if pd.notna(atr) and atr > 0:
                allocated = capital * position_pct
                if allocated > 1:
                    entry_exec = _apply_slippage(current_close, direction=direction, is_entry=True, slippage_pct=slippage_pct)
                    levels = compute_exit_levels(
                        entry_price=entry_exec,
                        direction=direction,
                        atr=float(atr),
                        stop_loss_atr_mult=stop_loss_atr_mult,
                        take_profit_r=take_profit_r,
                    )

                    entry_fee = allocated * fee_pct
                    allocated_after_fee = allocated - entry_fee
                    capital -= allocated

                    positions.append(
                        {
                            "entry_price": entry_exec,
                            "entry_time": ts,
                            "allocated": allocated_after_fee,
                            "bars_held": 0,
                            "direction": direction,
                            "stop_loss": levels["stop_loss"],
                            "take_profit": levels["take_profit"],
                            "risk_distance": levels["risk_distance"],
                            "atr_entry": float(atr),
                            "trail_active": False,
                            "best_price": entry_exec,
                            "entry_reason": row.get("entry_reason", ""),
                        }
                    )

        # ── 3) mark-to-market equity ─────────────────────────
        total_equity = capital
        for pos in positions:
            if pos["direction"] == 1:
                pos_value = pos["allocated"] * (current_close / pos["entry_price"])
            else:
                unrealized = (pos["entry_price"] - current_close) / pos["entry_price"]
                pos_value = pos["allocated"] * (1 + unrealized)
            total_equity += max(0.0, pos_value)

        equity_curve.append(total_equity)

    # ── 4) close remaining at end-of-data ─────────────────────
    if len(df) > 0:
        final_ts = df["timestamp"].iloc[-1]
        final_close = float(df["close"].iloc[-1])

        for pos in positions:
            direction = int(pos["direction"])
            exit_exec = _apply_slippage(final_close, direction=direction, is_entry=False, slippage_pct=slippage_pct)

            if direction == 1:
                pnl_pct = (exit_exec - pos["entry_price"]) / pos["entry_price"]
            else:
                pnl_pct = (pos["entry_price"] - exit_exec) / pos["entry_price"]

            pnl_after_fees = pnl_pct - fee_pct
            capital_change = pos["allocated"] * pnl_after_fees
            capital += pos["allocated"] + capital_change

            trades.append(
                {
                    "entry_time": pos["entry_time"],
                    "exit_time": final_ts,
                    "entry_price": round(pos["entry_price"], 6),
                    "exit_price": round(exit_exec, 6),
                    "direction": direction,
                    "pnl_pct": round(float(pnl_pct), 6),
                    "pnl_after_fees": round(float(pnl_after_fees), 6),
                    "exit_reason": "end_of_data",
                    "duration_bars": int(pos["bars_held"]),
                    "entry_reason": pos.get("entry_reason", ""),
                }
            )

    equity = np.array(equity_curve, dtype=float)
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    metrics = compute_metrics(trades_df, equity, initial_capital)

    return {
        "trades": trades_df,
        "equity_curve": equity,
        "equity_timestamps": df["timestamp"].values if len(df) > 0 else np.array([]),
        "metrics": metrics,
    }
