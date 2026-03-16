"""
Simple backtest engine for Sell-the-Rip (short-only, no ATR complexity).
"""

import numpy as np
import pandas as pd
import sys

sys.path.insert(0, "..")

from config import (
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
    HOLD_TIMEOUT_HOURS,
    POSITION_PCT,
    MAX_POSITIONS,
    FEE_PCT,
    INITIAL_CAPITAL,
)
from strategy import compute_exit_levels


def compute_metrics(trades_df: pd.DataFrame, equity_curve: np.ndarray, initial_capital: float) -> dict:
    if len(trades_df) == 0:
        return {
            "total_return": 0,
            "sharpe": 0,
            "max_drawdown": 0,
            "win_rate": 0,
            "profit_factor": 0,
            "total_trades": 0,
            "expectancy": 0,
            "final_capital": initial_capital,
        }

    total_return = (equity_curve[-1] - initial_capital) / initial_capital

    returns = np.diff(equity_curve) / np.where(equity_curve[:-1] == 0, np.nan, equity_curve[:-1])
    returns = returns[np.isfinite(returns)]
    sharpe = 0.0
    if len(returns) > 1 and np.std(returns) > 0:
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(8760)

    peak = np.maximum.accumulate(equity_curve)
    dd = (equity_curve - peak) / np.where(peak == 0, np.nan, peak)
    dd = dd[np.isfinite(dd)]
    max_dd = dd.min() if len(dd) else 0.0

    pnl_col = "pnl_after_fees"
    wins = trades_df[trades_df[pnl_col] > 0]
    losses = trades_df[trades_df[pnl_col] <= 0]
    win_rate = len(wins) / len(trades_df)

    gross_profit = wins[pnl_col].sum()
    gross_loss = abs(losses[pnl_col].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

    avg_win = wins[pnl_col].mean() if len(wins) > 0 else 0.0
    avg_loss = abs(losses[pnl_col].mean()) if len(losses) > 0 else 0.0
    expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss

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
    }


def run_backtest(
    df: pd.DataFrame,
    initial_capital: float = INITIAL_CAPITAL,
    stop_loss_pct: float = STOP_LOSS_PCT,
    take_profit_pct: float = TAKE_PROFIT_PCT,
    hold_timeout: int = HOLD_TIMEOUT_HOURS,
    position_pct: float = POSITION_PCT,
    max_positions: int = MAX_POSITIONS,
    fee_pct: float = FEE_PCT,
) -> dict:
    capital = float(initial_capital)
    positions = []
    trades = []
    equity_curve = []

    for i in range(len(df)):
        row = df.iloc[i]
        current_close = float(row["close"])
        current_high = float(row["high"])
        current_low = float(row["low"])
        ts = row["timestamp"]

        # Exits
        to_close = []
        for j, pos in enumerate(positions):
            pos["bars_held"] += 1

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

            pnl_pct = (pos["entry_price"] - exit_price) / pos["entry_price"]
            pnl_after_fees = pnl_pct - fee_pct
            capital_change = pos["allocated"] * pnl_after_fees
            capital += pos["allocated"] + capital_change

            trades.append(
                {
                    "entry_time": pos["entry_time"],
                    "exit_time": ts,
                    "entry_price": pos["entry_price"],
                    "exit_price": round(float(exit_price), 6),
                    "pnl_pct": round(float(pnl_pct), 6),
                    "pnl_after_fees": round(float(pnl_after_fees), 6),
                    "exit_reason": exit_reason,
                    "duration_bars": int(pos["bars_held"]),
                    "rip_size": pos.get("rip_size", 0.0),
                }
            )
            to_close.append(j)

        for j in sorted(to_close, reverse=True):
            positions.pop(j)

        # Entries
        open_slots = max_positions - len(positions)
        if open_slots > 0 and capital > 0 and row.get("signal", "NO TRADE") == "SHORT":
            allocated = capital * position_pct
            if allocated > 1:
                entry_fee = allocated * fee_pct
                allocated_after_fee = allocated - entry_fee
                capital -= allocated

                levels = compute_exit_levels(
                    current_close,
                    take_profit_pct=take_profit_pct,
                    stop_loss_pct=stop_loss_pct,
                )
                positions.append(
                    {
                        "entry_price": current_close,
                        "entry_time": ts,
                        "allocated": allocated_after_fee,
                        "bars_held": 0,
                        "stop_loss": levels["stop_loss"],
                        "take_profit": levels["take_profit"],
                        "rip_size": row.get("rip_size", 0.0),
                    }
                )

        # Mark-to-market equity for shorts
        total_equity = capital
        for pos in positions:
            unrealized_pct = (pos["entry_price"] - current_close) / pos["entry_price"]
            total_equity += pos["allocated"] * (1 + unrealized_pct)
        equity_curve.append(total_equity)

    # Force close remaining positions
    for pos in positions:
        exit_price = float(df["close"].iloc[-1])
        pnl_pct = (pos["entry_price"] - exit_price) / pos["entry_price"]
        pnl_after_fees = pnl_pct - fee_pct
        capital_change = pos["allocated"] * pnl_after_fees
        capital += pos["allocated"] + capital_change

        trades.append(
            {
                "entry_time": pos["entry_time"],
                "exit_time": df["timestamp"].iloc[-1],
                "entry_price": pos["entry_price"],
                "exit_price": round(float(exit_price), 6),
                "pnl_pct": round(float(pnl_pct), 6),
                "pnl_after_fees": round(float(pnl_after_fees), 6),
                "exit_reason": "end_of_data",
                "duration_bars": int(pos["bars_held"]),
                "rip_size": pos.get("rip_size", 0.0),
            }
        )

    equity = np.array(equity_curve)
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    metrics = compute_metrics(trades_df, equity, initial_capital)

    return {
        "trades": trades_df,
        "equity_curve": equity,
        "equity_timestamps": df["timestamp"].values,
        "metrics": metrics,
    }
