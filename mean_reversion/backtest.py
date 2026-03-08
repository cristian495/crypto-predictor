"""
backtest.py
Realistic backtest engine for mean reversion strategy.
Includes: transaction fees, stop-loss, take-profit, long+short,
position sizing, max simultaneous positions, equity curve.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
    HOLD_TIMEOUT,
    POSITION_PCT,
    MAX_POSITIONS,
    FEE_PCT,
    INITIAL_CAPITAL,
)
from strategy import compute_exit_levels, check_exit
from metrics import compute_all_metrics, print_performance_report


def run_backtest(
    df: pd.DataFrame,
    zscore_col: str = "zscore_50",
    initial_capital: float = INITIAL_CAPITAL,
    stop_loss_pct: float = STOP_LOSS_PCT,
    take_profit_pct: float = TAKE_PROFIT_PCT,
    hold_timeout: int = HOLD_TIMEOUT,
    position_pct: float = POSITION_PCT,
    max_positions: int = MAX_POSITIONS,
    fee_pct: float = FEE_PCT,
) -> dict:
    """
    Run backtest on a single symbol using pre-computed signals.

    Args:
        df: DataFrame with signals already computed (signal, signal_direction columns).
        zscore_col: Z-score column for mean reversion exit detection.
        initial_capital: Starting capital.
        stop_loss_pct: Stop loss percentage.
        take_profit_pct: Take profit percentage.
        hold_timeout: Max candles to hold a position.
        position_pct: Fraction of capital per trade.
        max_positions: Max simultaneous positions.
        fee_pct: Fee per side (0.001 = 0.1%).

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
        current_zscore = row.get(zscore_col, 0.0)
        ts = row["timestamp"]

        # ── 1. Check exits on open positions ──
        positions_to_close = []
        for j, pos in enumerate(positions):
            pos["bars_held"] += 1

            should_exit, exit_price, exit_reason = check_exit(
                pos, current_high, current_low, current_close,
                current_zscore, pos["bars_held"], hold_timeout,
            )

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
                    "entry_zscore": pos["entry_zscore"],
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
        ):
            direction = int(row["signal_direction"])
            allocated = capital * position_pct

            if allocated > 1:
                # Deduct entry fee from allocated capital
                entry_fee = allocated * fee_pct
                allocated_after_fee = allocated - entry_fee
                capital -= allocated

                levels = compute_exit_levels(
                    current_close, direction, stop_loss_pct, take_profit_pct
                )

                positions.append({
                    "entry_price": current_close,
                    "entry_time": ts,
                    "direction": direction,
                    "allocated": allocated_after_fee,
                    "bars_held": 0,
                    "stop_loss": levels["stop_loss"],
                    "take_profit": levels["take_profit"],
                    "entry_zscore": current_zscore,
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
            "entry_zscore": pos.get("entry_zscore", 0),
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
    stop_loss_pct: float = STOP_LOSS_PCT,
    take_profit_pct: float = TAKE_PROFIT_PCT,
    hold_timeout: int = HOLD_TIMEOUT,
    position_pct: float = POSITION_PCT,
    max_positions: int = MAX_POSITIONS,
    fee_pct: float = FEE_PCT,
    zscore_col: str = "zscore_50",
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

            current_zscore = row.get(zscore_col, 0.0)

            should_exit, exit_price, exit_reason = check_exit(
                pos, row["high"], row["low"], row["close"],
                current_zscore, pos["bars_held"], hold_timeout,
            )

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

        for s in symbols_to_close:
            del positions[s]

        # ── 2. Check new entries ──
        open_slots = max_positions - len(positions)
        if open_slots > 0 and capital > 0:
            candidates = []
            for symbol, sdf in symbol_data.items():
                if symbol in positions:
                    continue
                if ts not in sdf.index:
                    continue
                row = sdf.loc[ts]
                if row.get("signal", "NO TRADE") in ("LONG", "SHORT"):
                    candidates.append((
                        symbol,
                        row["probability"],
                        row["close"],
                        int(row["signal_direction"]),
                        row.get(zscore_col, 0.0),
                    ))

            # Rank by probability (highest first)
            candidates.sort(key=lambda x: x[1], reverse=True)

            for symbol, prob, close_price, direction, zscore in candidates[:open_slots]:
                allocated = capital * position_pct
                if allocated < 1:
                    break

                entry_fee = allocated * fee_pct
                allocated_after_fee = allocated - entry_fee
                capital -= allocated

                levels = compute_exit_levels(
                    close_price, direction, stop_loss_pct, take_profit_pct
                )

                positions[symbol] = {
                    "entry_price": close_price,
                    "entry_time": ts,
                    "direction": direction,
                    "allocated": allocated_after_fee,
                    "bars_held": 0,
                    "stop_loss": levels["stop_loss"],
                    "take_profit": levels["take_profit"],
                    "entry_zscore": zscore,
                }

        # ── 3. Equity ──
        total_equity = capital
        for symbol, pos in positions.items():
            if ts in symbol_data[symbol].index:
                current = symbol_data[symbol].loc[ts]["close"]
                if pos["direction"] == 1:
                    pos_value = pos["allocated"] * (current / pos["entry_price"])
                else:
                    pnl = (pos["entry_price"] - current) / pos["entry_price"]
                    pos_value = pos["allocated"] * (1 + pnl)
                total_equity += pos_value
            else:
                total_equity += pos["allocated"]

        equity_curve.append(total_equity)
        equity_timestamps.append(ts)

    # ── Close remaining positions ──
    for symbol, pos in positions.items():
        last_ts = timestamps[-1]
        if last_ts in symbol_data[symbol].index:
            exit_price = symbol_data[symbol].loc[last_ts]["close"]
        else:
            exit_price = pos["entry_price"]

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
        "equity_timestamps": equity_timestamps,
        "metrics": metrics,
    }


def plot_equity(equity_curve: np.ndarray,
                timestamps,
                initial_capital: float = INITIAL_CAPITAL,
                title: str = "Equity Curve",
                filename: str = "equity_curve.png"):
    """Save equity curve plot."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(timestamps[:len(equity_curve)], equity_curve, linewidth=1)
    ax.axhline(y=initial_capital, color="gray", linestyle="--", alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Capital (USD)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  Chart saved: {filename}")


def print_trades_summary(trades_df: pd.DataFrame):
    """Print per-symbol breakdown of trades."""
    if len(trades_df) == 0:
        print("  No trades.")
        return

    # Exit reason breakdown
    print(f"\n  Exit reasons:")
    for reason, count in trades_df["exit_reason"].value_counts().items():
        print(f"    {reason:<15} {count}")

    # Per-symbol breakdown
    if "symbol" in trades_df.columns:
        print(f"\n  Per symbol:")
        by_sym = trades_df.groupby("symbol").agg(
            trades=("pnl_after_fees", "count"),
            win_rate=("pnl_after_fees", lambda x: (x > 0).mean()),
            avg_pnl=("pnl_after_fees", "mean"),
            total_pnl=("pnl_after_fees", "sum"),
        ).sort_values("avg_pnl", ascending=False)

        for sym, row in by_sym.iterrows():
            print(f"    {sym:<12} {int(row['trades']):>3} trades  "
                  f"WR: {row['win_rate']:.0%}  "
                  f"Avg: {row['avg_pnl']:.2%}  "
                  f"Total: {row['total_pnl']:.2%}")

    # Direction breakdown
    if "direction" in trades_df.columns:
        print(f"\n  Per direction:")
        for d, label in [(1, "LONG"), (-1, "SHORT")]:
            subset = trades_df[trades_df["direction"] == d]
            if len(subset) > 0:
                wr = (subset["pnl_after_fees"] > 0).mean()
                avg = subset["pnl_after_fees"].mean()
                print(f"    {label:<8} {len(subset):>3} trades  "
                      f"WR: {wr:.0%}  Avg: {avg:.2%}")
