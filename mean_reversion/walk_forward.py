"""
walk_forward.py
Walk-Forward Validation for mean reversion strategy.

Tests model robustness across different market regimes by:
- Training on rolling windows
- Testing on out-of-sample periods
- Analyzing performance consistency

This is CRITICAL before live trading to detect overfitting.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data_loader import fetch_ohlcv, fetch_funding_rates, fetch_open_interest
from features import add_features
from model import EnsembleModel
from target import compute_reversion_target, prepare_dataset
from backtest import run_backtest
from strategy import generate_signals
from features import FEATURE_COLS
from config import (
    DEFAULT_SYMBOLS,
    TIMEFRAME,
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
    INITIAL_CAPITAL,
    FEE_PCT,
)


def walk_forward_validation(
    symbol: str,
    train_months: int = 6,
    test_months: int = 1,
    step_months: int = 1,
    total_days: int = 1095,
):
    """
    Perform walk-forward validation on a single symbol.

    Args:
        symbol: Trading pair (e.g., "ETH/USDT")
        train_months: Training window size (default 6 months)
        test_months: Test window size (default 1 month)
        step_months: How much to move window forward (default 1 month)
        total_days: Total historical data to fetch

    Returns:
        List of test results for each window
    """
    print(f"\n{'='*70}")
    print(f"  WALK-FORWARD VALIDATION: {symbol}")
    print(f"{'='*70}")
    print(f"  Train window:  {train_months} months")
    print(f"  Test window:   {test_months} month(s)")
    print(f"  Step size:     {step_months} month(s)")
    print(f"{'='*70}\n")

    # 1. Download data
    print(f"Downloading {symbol} data...")
    df = fetch_ohlcv(symbol, timeframe=TIMEFRAME, days=total_days)

    # Add funding/OI if available
    try:
        funding_symbol = symbol.replace("/", "/") + ":USDT"
        df_funding = fetch_funding_rates(funding_symbol, days=total_days)
        if not df_funding.empty:
            df = df.merge(df_funding, on="timestamp", how="left")
            df["funding_rate"] = df["funding_rate"].fillna(0)
    except Exception as e:
        print(f"  Warning: Could not fetch funding rates: {e}")
        df["funding_rate"] = 0.0

    df["open_interest"] = 0.0  # OI not available for most pairs

    # 2. Add features and compute targets (includes tradeable column)
    df = add_features(df)
    df = compute_reversion_target(df)

    # Create tradeable column (rows with valid targets)
    df["tradeable"] = df["target"].notna().astype(int)

    print(f"  Total rows: {len(df)}")
    print(f"  Tradeable rows: {df['tradeable'].sum()}")
    print(f"  Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")

    # 5. Define walk-forward windows
    results = []
    train_size = int(train_months * 30 * 24)  # approximate hours
    test_size = int(test_months * 30 * 24)
    step_size = int(step_months * 30 * 24)

    start_idx = 0
    window_num = 1

    while True:
        train_end_idx = start_idx + train_size
        test_end_idx = train_end_idx + test_size

        # Check if we have enough data
        if test_end_idx > len(df):
            break

        # Split data
        train_data = df.iloc[start_idx:train_end_idx].copy()
        test_data = df.iloc[train_end_idx:test_end_idx].copy()

        # Ensure we have enough samples
        if len(train_data) < 500 or len(test_data) < 50:
            print(f"  [Window {window_num}] Skipping - insufficient data")
            start_idx += step_size
            window_num += 1
            continue

        train_start = train_data["timestamp"].min()
        train_end = train_data["timestamp"].max()
        test_start = test_data["timestamp"].min()
        test_end = test_data["timestamp"].max()

        print(f"\n  [Window {window_num}] ────────────────────────────────")
        print(f"    Train: {train_start.date()} → {train_end.date()} ({len(train_data)} rows)")
        print(f"    Test:  {test_start.date()} → {test_end.date()} ({len(test_data)} rows)")

        try:
            # 6. Prepare training data (only rows with valid targets)
            train_tradeable = train_data[train_data["tradeable"] == 1].copy()
            train_tradeable = train_tradeable.dropna(subset=["target"] + FEATURE_COLS)

            if len(train_tradeable) < 100:
                print(f"    Skipping - only {len(train_tradeable)} tradeable rows")
                start_idx += step_size
                window_num += 1
                continue

            # 7. Train model
            model = EnsembleModel()
            model.fit(train_tradeable, feature_cols=FEATURE_COLS)

            # 8. Generate signals on test data
            test_data_with_signals = generate_signals(
                test_data,
                model,
                feature_cols=FEATURE_COLS,
            )

            # 9. Backtest on test period
            backtest_results = run_backtest(
                test_data_with_signals,
                initial_capital=INITIAL_CAPITAL,
                stop_loss_pct=STOP_LOSS_PCT,
                take_profit_pct=TAKE_PROFIT_PCT,
                fee_pct=FEE_PCT,
            )

            # 10. Store results
            metrics = backtest_results["metrics"]
            result = {
                "window": window_num,
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "train_size": len(train_data),
                "test_size": len(test_data),
                "total_return": metrics.get("total_return", 0) * 100,  # Convert to %
                "sharpe": metrics.get("sharpe", 0),
                "max_dd": metrics.get("max_drawdown", 0) * 100,  # Convert to %
                "win_rate": metrics.get("win_rate", 0) * 100,  # Convert to %
                "profit_factor": metrics.get("profit_factor", 0),
                "num_trades": metrics.get("total_trades", 0),
                "expectancy": metrics.get("expectancy", 0) * 100,  # Convert to %
            }

            results.append(result)

            # Print summary
            print(f"    Return:  {result['total_return']:>7.2f}%")
            print(f"    Sharpe:  {result['sharpe']:>7.2f}")
            print(f"    Win rate:{result['win_rate']:>7.1f}%")
            print(f"    Trades:  {result['num_trades']:>7}")

        except Exception as e:
            print(f"    Error: {e}")

        # Move window forward
        start_idx += step_size
        window_num += 1

    return results


def analyze_walk_forward_results(results: list, symbol: str):
    """
    Analyze and summarize walk-forward results.

    Args:
        results: List of dictionaries with test results
        symbol: Trading pair name
    """
    if not results:
        print("\n⚠️  No walk-forward results to analyze")
        return

    df_results = pd.DataFrame(results)

    print(f"\n{'='*70}")
    print(f"  WALK-FORWARD ANALYSIS: {symbol}")
    print(f"{'='*70}")

    print(f"\n  Summary Statistics ({len(results)} windows):")
    print(f"  ─────────────────────────────────────────")

    metrics = {
        "Return (%)": df_results["total_return"],
        "Sharpe": df_results["sharpe"],
        "Win Rate (%)": df_results["win_rate"],
        "Max DD (%)": df_results["max_dd"],
        "Profit Factor": df_results["profit_factor"],
        "# Trades": df_results["num_trades"],
        "Expectancy (%)": df_results["expectancy"],
    }

    for metric_name, values in metrics.items():
        mean_val = values.mean()
        median_val = values.median()
        std_val = values.std()
        min_val = values.min()
        max_val = values.max()

        print(f"\n  {metric_name}:")
        print(f"    Mean:   {mean_val:>8.2f}")
        print(f"    Median: {median_val:>8.2f}")
        print(f"    Std:    {std_val:>8.2f}")
        print(f"    Min:    {min_val:>8.2f}")
        print(f"    Max:    {max_val:>8.2f}")

    # Consistency analysis
    print(f"\n  Consistency Metrics:")
    print(f"  ─────────────────────────────────────────")

    positive_returns = (df_results["total_return"] > 0).sum()
    positive_pct = positive_returns / len(results) * 100
    print(f"    Positive returns:  {positive_returns}/{len(results)} ({positive_pct:.1f}%)")

    sharpe_above_1 = (df_results["sharpe"] > 1.0).sum()
    sharpe_pct = sharpe_above_1 / len(results) * 100
    print(f"    Sharpe > 1.0:      {sharpe_above_1}/{len(results)} ({sharpe_pct:.1f}%)")

    wr_above_50 = (df_results["win_rate"] > 50).sum()
    wr_pct = wr_above_50 / len(results) * 100
    print(f"    Win rate > 50%:    {wr_above_50}/{len(results)} ({wr_pct:.1f}%)")

    # Stability score (0-100)
    stability_score = (positive_pct * 0.4 + sharpe_pct * 0.3 + wr_pct * 0.3)

    print(f"\n  ⭐ STABILITY SCORE: {stability_score:.1f}/100")

    if stability_score >= 75:
        print(f"     ✅ EXCELLENT - Very stable across regimes")
    elif stability_score >= 60:
        print(f"     🟢 GOOD - Reasonably stable")
    elif stability_score >= 45:
        print(f"     🟡 MODERATE - Some instability")
    else:
        print(f"     ❌ POOR - Highly unstable, likely overfitting")

    # Detailed results table
    print(f"\n  Detailed Results by Window:")
    print(f"  ─────────────────────────────────────────")
    print(f"  {'Win':<4} {'Test Period':<22} {'Return':>8} {'Sharpe':>7} {'WR':>6} {'Trades':>7}")
    print(f"  {'-'*65}")

    for _, row in df_results.iterrows():
        test_period = f"{row['test_start'].strftime('%Y-%m-%d')} → {row['test_end'].strftime('%m-%d')}"
        print(f"  {int(row['window']):<4} {test_period:<22} {row['total_return']:>7.2f}% "
              f"{row['sharpe']:>7.2f} {row['win_rate']:>5.1f}% {int(row['num_trades']):>7}")

    print(f"{'='*70}\n")

    return df_results, stability_score


def run_multi_symbol_walk_forward(
    symbols: list = None,
    train_months: int = 6,
    test_months: int = 1,
):
    """
    Run walk-forward validation on multiple symbols.

    Args:
        symbols: List of trading pairs (default: DEFAULT_SYMBOLS)
        train_months: Training window size
        test_months: Test window size
    """
    if symbols is None:
        symbols = DEFAULT_SYMBOLS

    print(f"\n{'='*70}")
    print(f"  MULTI-SYMBOL WALK-FORWARD VALIDATION")
    print(f"{'='*70}")
    print(f"  Symbols: {len(symbols)}")
    print(f"  Train: {train_months} months | Test: {test_months} month(s)")
    print(f"{'='*70}\n")

    all_results = {}
    all_stability_scores = {}

    for symbol in symbols:
        try:
            results = walk_forward_validation(
                symbol,
                train_months=train_months,
                test_months=test_months,
                step_months=1,
                total_days=1095,
            )

            df_results, stability_score = analyze_walk_forward_results(results, symbol)

            all_results[symbol] = df_results
            all_stability_scores[symbol] = stability_score

        except Exception as e:
            print(f"\n❌ Error with {symbol}: {e}")
            continue

    # Final summary across all symbols
    print(f"\n{'='*70}")
    print(f"  FINAL SUMMARY - ALL SYMBOLS")
    print(f"{'='*70}\n")

    for symbol in all_stability_scores:
        score = all_stability_scores[symbol]
        status = "✅" if score >= 60 else "⚠️" if score >= 45 else "❌"
        print(f"  {status} {symbol:<12} Stability: {score:>5.1f}/100")

    avg_stability = np.mean(list(all_stability_scores.values()))
    print(f"\n  Average Stability: {avg_stability:.1f}/100")

    if avg_stability >= 60:
        print(f"\n  ✅ Strategy is STABLE across symbols and time periods")
        print(f"     → Safe to proceed with paper trading")
    elif avg_stability >= 45:
        print(f"\n  🟡 Strategy has MODERATE stability")
        print(f"     → Consider more optimization before live trading")
    else:
        print(f"\n  ❌ Strategy is UNSTABLE")
        print(f"     → HIGH RISK of overfitting, DO NOT trade live")

    print(f"\n{'='*70}\n")

    return all_results, all_stability_scores


if __name__ == "__main__":
    # Run walk-forward validation on all symbols
    results, scores = run_multi_symbol_walk_forward(
        symbols=DEFAULT_SYMBOLS[:2],  # Start with 2 symbols for faster testing
        train_months=6,
        test_months=1,
    )