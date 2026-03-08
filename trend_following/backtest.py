"""
backtest.py
Realistic backtest engine for trend following strategy.
Features: ATR-based stops, trailing stops, transaction fees,
position sizing, max simultaneous positions, equity curve.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    STOP_LOSS_ATR_MULT,
    TAKE_PROFIT_RR,
    HOLD_TIMEOUT,
    POSITION_PCT,
    MAX_POSITIONS,
    FEE_PCT,
    INITIAL_CAPITAL,
    USE_TRAILING_STOP,
    TRAILING_STOP_ATR_MULT,
)
from strategy import compute_exit_levels, check_exit
from metrics import compute_all_metrics, print_performance_report


def run_backtest(
    df: pd.DataFrame,
    initial_capital: float = INITIAL_CAPITAL,
    stop_loss_atr_mult: float = STOP_LOSS_ATR_MULT,
    take_profit_rr: float = TAKE_PROFIT_RR,
    hold_timeout: int = HOLD_TIMEOUT,
    position_pct: float = POSITION_PCT,
    max_positions: int = MAX_POSITIONS,
    fee_pct: float = FEE_PCT,
    use_trailing_stop: bool = USE_TRAILING_STOP,
    trailing_atr_mult: float = TRAILING_STOP_ATR_MULT,
) -> dict:
    """
    Run backtest on a single symbol using pre-computed signals.

    Args:
        df: DataFrame with signals already computed (signal, signal_direction columns).
        initial_capital: Starting capital.
        stop_loss_atr_mult: ATR multiplier for stop loss.
        take_profit_rr: Risk:Reward ratio for take profit.
        hold_timeout: Max candles to hold a position.
        position_pct: Fraction of capital per trade.
        max_positions: Max simultaneous positions.
        fee_pct: Fee per side (0.001 = 0.1%).
        use_trailing_stop: Enable trailing stops.
        trailing_atr_mult: ATR multiplier for trailing stop.

    Returns:
        dict with trades_df, equity_curve, metrics.
    """
    capital = initial_capital
    positions = []  # list of active position dicts
    trades = []
    equity_curve = []

    for i in range(len(df)):
        row = df.iloc[i]
        current_close = row["close"]
        current_high = row["high"]
        current_low = row["low"]
        current_atr = row.get("atr_14", 0.0) * current_close  # Denormalize ATR
        ts = row["timestamp"]

        # ── 1. Check exits on open positions ──
        positions_to_close = []
        for j, pos in enumerate(positions):
            pos["bars_held"] += 1

            should_exit, exit_price, exit_reason, updated_pos = check_exit(
                pos, current_high, current_low, current_close,
                current_atr, pos["bars_held"], hold_timeout,
                use_trailing_stop, trailing_atr_mult,
            )

            # Update position with new trailing stop
            positions[j] = updated_pos

            if should_exit:
                # Compute PnL
                if pos["direction"] == 1:  # LONG
                    pnl_pct = (exit_price - pos["entry_price"]) / pos["entry_price"]
                else:  # SHORT
                    pnl_pct = (pos["entry_price"] - exit_price) / pos["entry_price"]

                # Apply exit fee
                pnl_after_fees = pnl_pct - fee_pct  # entry fee already deducted

                # Return capital
                capital_change = pos["allocated"] * pnl_after_fees
                capital += pos["allocated"] + capital_change

                trades.append({
                    "entry_time": pos["entry_time"],
                    "exit_time": ts,
                    "direction": pos["direction"],
                    "entry_price": pos["entry_price"],
                    "exit_price": round(exit_price, 6),
                    "pnl_pct": round(pnl_pct, 6),
                    "pnl_after_fees": round(pnl_after_fees, 6),
                    "exit_reason": exit_reason,
                    "duration_bars": pos["bars_held"],
                })

                positions_to_close.append(j)

        # Remove closed positions (in reverse order to preserve indices)
        for j in sorted(positions_to_close, reverse=True):
            positions.pop(j)

        # ── 2. Check for new entries ──
        open_slots = max_positions - len(positions)

        if (
            open_slots > 0
            and capital > 0
            and row.get("signal", "NO TRADE") in ("LONG", "SHORT")
            and current_atr > 0
        ):
            direction = int(row["signal_direction"])
            allocated = capital * position_pct

            if allocated > 1:
                # Deduct entry fee from allocated capital
                entry_fee = allocated * fee_pct
                allocated_after_fee = allocated - entry_fee
                capital -= allocated

                levels = compute_exit_levels(
                    current_close, direction, current_atr,
                    stop_loss_atr_mult, take_profit_rr
                )

                positions.append({
                    "entry_price": current_close,
                    "entry_time": ts,
                    "direction": direction,
                    "allocated": allocated_after_fee,
                    "bars_held": 0,
                    "stop_loss": levels["stop_loss"],
                    "take_profit": levels["take_profit"],
                })

        # ── 3. Compute equity ──
        total_equity = capital
        for pos in positions:
            if pos["direction"] == 1:
                pos_value = pos["allocated"] * (current_close / pos["entry_price"])
            else:
                pnl = (pos["entry_price"] - current_close) / pos["entry_price"]
                pos_value = pos["allocated"] * (1 + pnl)
            total_equity += pos_value

        equity_curve.append(total_equity)

    # ── Close remaining positions at end ──
    for pos in positions:
        exit_price = df["close"].iloc[-1]
        if pos["direction"] == 1:
            pnl_pct = (exit_price - pos["entry_price"]) / pos["entry_price"]
        else:
            pnl_pct = (pos["entry_price"] - exit_price) / pos["entry_price"]

        pnl_after_fees = pnl_pct - fee_pct
        capital_change = pos["allocated"] * pnl_after_fees
        capital += pos["allocated"] + capital_change

        trades.append({
            "entry_time": pos["entry_time"],
            "exit_time": df["timestamp"].iloc[-1],
            "direction": pos["direction"],
            "entry_price": pos["entry_price"],
            "exit_price": exit_price,
            "pnl_pct": round(pnl_pct, 6),
            "pnl_after_fees": round(pnl_after_fees, 6),
            "exit_reason": "end_of_data",
            "duration_bars": pos["bars_held"],
        })

    # ── Build results ──
    equity = np.array(equity_curve)
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

    metrics = compute_all_metrics(
        trades_df, equity, initial_capital,
        periods_per_year=8760,
        pnl_col="pnl_after_fees",
    )

    return {
        "trades": trades_df,
        "equity_curve": equity,
        "equity_timestamps": df["timestamp"].values,
        "metrics": metrics,
    }


def run_multi_backtest(
    trained: dict,
    signals_by_symbol: dict,
    initial_capital: float = INITIAL_CAPITAL,
    stop_loss_atr_mult: float = STOP_LOSS_ATR_MULT,
    take_profit_rr: float = TAKE_PROFIT_RR,
    hold_timeout: int = HOLD_TIMEOUT,
    position_pct: float = POSITION_PCT,
    max_positions: int = MAX_POSITIONS,
    fee_pct: float = FEE_PCT,
    use_trailing_stop: bool = USE_TRAILING_STOP,
    trailing_atr_mult: float = TRAILING_STOP_ATR_MULT,
) -> dict:
    """
    Multi-crypto backtest: all symbols share one capital pool.

    Args:
        trained: dict with symbol → {test_df, model, ...}
        signals_by_symbol: dict with symbol → signals DataFrame
    """
    # Collect all symbol data with signals
    symbol_data = {}
    all_timestamps = set()

    for symbol, sig_df in signals_by_symbol.items():
        symbol_data[symbol] = sig_df.set_index("timestamp")
        all_timestamps.update(sig_df["timestamp"].tolist())

    timestamps = sorted(all_timestamps)

    capital = initial_capital
    positions = {}  # symbol → position dict
    trades = []
    equity_curve = []
    equity_timestamps = []

    for ts in timestamps:
        # ── 1. Check exits ──
        symbols_to_close = []
        for symbol, pos in positions.items():
            if ts not in symbol_data[symbol].index:
                continue

            row = symbol_data[symbol].loc[ts]
            pos["bars_held"] += 1

            current_atr = row.get("atr_14", 0.0) * row["close"]

            should_exit, exit_price, exit_reason, updated_pos = check_exit(
                pos, row["high"], row["low"], row["close"],
                current_atr, pos["bars_held"], hold_timeout,
                use_trailing_stop, trailing_atr_mult,
            )

            positions[symbol] = updated_pos

            if should_exit:
                if pos["direction"] == 1:
                    pnl_pct = (exit_price - pos["entry_price"]) / pos["entry_price"]
                else:
                    pnl_pct = (pos["entry_price"] - exit_price) / pos["entry_price"]

                pnl_after_fees = pnl_pct - fee_pct
                capital_change = pos["allocated"] * pnl_after_fees
                capital += pos["allocated"] + capital_change

                trades.append({
                    "symbol": symbol,
                    "entry_time": pos["entry_time"],
                    "exit_time": ts,
                    "direction": pos["direction"],
                    "entry_price": pos["entry_price"],
                    "exit_price": round(exit_price, 6),
                    "pnl_pct": round(pnl_pct, 6),
                    "pnl_after_fees": round(pnl_after_fees, 6),
                    "exit_reason": exit_reason,
                    "duration_bars": pos["bars_held"],
                })

                symbols_to_close.append(symbol)

        for symbol in symbols_to_close:
            del positions[symbol]

        # ── 2. Check for new entries ──
        open_slots = max_positions - len(positions)

        if open_slots > 0 and capital > 0:
            # Collect all signals at this timestamp
            candidates = []
            for symbol, data in symbol_data.items():
                if symbol in positions:
                    continue
                if ts not in data.index:
                    continue

                row = data.loc[ts]
                if row.get("signal", "NO TRADE") in ("LONG", "SHORT"):
                    prob = row.get("probability", 0.0)
                    candidates.append((symbol, row, prob))

            # Sort by probability (highest first)
            candidates.sort(key=lambda x: x[2], reverse=True)

            # Take best candidates up to open_slots
            for symbol, row, prob in candidates[:open_slots]:
                direction = int(row["signal_direction"])
                allocated = capital * position_pct
                current_atr = row.get("atr_14", 0.0) * row["close"]

                if allocated > 1 and current_atr > 0:
                    entry_fee = allocated * fee_pct
                    allocated_after_fee = allocated - entry_fee
                    capital -= allocated

                    levels = compute_exit_levels(
                        row["close"], direction, current_atr,
                        stop_loss_atr_mult, take_profit_rr
                    )

                    positions[symbol] = {
                        "entry_price": row["close"],
                        "entry_time": ts,
                        "direction": direction,
                        "allocated": allocated_after_fee,
                        "bars_held": 0,
                        "stop_loss": levels["stop_loss"],
                        "take_profit": levels["take_profit"],
                    }

                    open_slots -= 1
                    if open_slots <= 0:
                        break

        # ── 3. Compute equity ──
        total_equity = capital
        for symbol, pos in positions.items():
            if ts not in symbol_data[symbol].index:
                continue
            current_close = symbol_data[symbol].loc[ts, "close"]

            if pos["direction"] == 1:
                pos_value = pos["allocated"] * (current_close / pos["entry_price"])
            else:
                pnl = (pos["entry_price"] - current_close) / pos["entry_price"]
                pos_value = pos["allocated"] * (1 + pnl)
            total_equity += pos_value

        equity_curve.append(total_equity)
        equity_timestamps.append(ts)

    # ── Close remaining positions ──
    for symbol, pos in positions.items():
        last_ts = symbol_data[symbol].index[-1]
        exit_price = symbol_data[symbol].loc[last_ts, "close"]

        if pos["direction"] == 1:
            pnl_pct = (exit_price - pos["entry_price"]) / pos["entry_price"]
        else:
            pnl_pct = (pos["entry_price"] - exit_price) / pos["entry_price"]

        pnl_after_fees = pnl_pct - fee_pct
        capital_change = pos["allocated"] * pnl_after_fees
        capital += pos["allocated"] + capital_change

        trades.append({
            "symbol": symbol,
            "entry_time": pos["entry_time"],
            "exit_time": last_ts,
            "direction": pos["direction"],
            "entry_price": pos["entry_price"],
            "exit_price": exit_price,
            "pnl_pct": round(pnl_pct, 6),
            "pnl_after_fees": round(pnl_after_fees, 6),
            "exit_reason": "end_of_data",
            "duration_bars": pos["bars_held"],
        })

    # ── Build results ──
    equity = np.array(equity_curve)
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

    metrics = compute_all_metrics(
        trades_df, equity, initial_capital,
        periods_per_year=8760,
        pnl_col="pnl_after_fees",
    )

    return {
        "trades": trades_df,
        "equity_curve": equity,
        "equity_timestamps": np.array(equity_timestamps),
        "metrics": metrics,
    }


def plot_equity_curve(
    equity_curve: np.ndarray,
    equity_timestamps: np.ndarray,
    initial_capital: float,
    filename: str = "equity_curve.png",
):
    """Plot and save equity curve."""
    plt.figure(figsize=(14, 7))

    # Convert timestamps to datetime if they aren't already
    if len(equity_timestamps) > 0:
        if isinstance(equity_timestamps[0], pd.Timestamp):
            dates = equity_timestamps
        else:
            dates = pd.to_datetime(equity_timestamps)
    else:
        dates = range(len(equity_curve))

    plt.plot(dates, equity_curve, label="Portfolio Equity", linewidth=1.5)
    plt.axhline(y=initial_capital, color="gray", linestyle="--", label="Initial Capital")

    plt.xlabel("Date")
    plt.ylabel("Equity ($)")
    plt.title("Trend Following Strategy — Equity Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

    print(f"  Chart saved: {filename}")