"""
walk_forward.py
Walk-Forward Validation for Buy-the-Dip strategy.

Tests strategy robustness by:
- Testing on rolling out-of-sample periods
- Measuring consistency across time
- No training needed (rule-based strategy)
"""

import sys
sys.path.insert(0, "..")

import pandas as pd
import numpy as np
from breakout_momentum.data_loader import fetch_ohlcv
from strategy import add_dip_features, generate_signals
from backtest import run_backtest, compute_metrics
from config import (
    DEFAULT_SYMBOLS,
    TIMEFRAME,
    DAYS,
    DIP_THRESHOLD_PCT,
    DIP_THRESHOLD_STRONG,
    USE_STRONG_THRESHOLD,
    INITIAL_CAPITAL,
    FEE_PCT,
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
    HOLD_TIMEOUT_HOURS,
    WF_TRAIN_MONTHS,
    WF_TEST_MONTHS,
    WF_STEP_MONTHS,
)


def walk_forward_validation(
    symbol: str,
    train_months: int = WF_TRAIN_MONTHS,
    test_months: int = WF_TEST_MONTHS,
    step_months: int = WF_STEP_MONTHS,
    total_days: int = DAYS,
    dip_threshold: float = None,
):
    """
    Walk-forward validation for Buy-the-Dip strategy.

    Since this is rule-based (no ML), we just test on rolling windows
    to see if the edge is consistent across time.

    Args:
        symbol: Trading pair
        train_months: Warmup period (for indicators)
        test_months: Test window size
        step_months: Step size
        total_days: Total historical data
        dip_threshold: Dip threshold to use

    Returns:
        List of test results for each window
    """
    print(f"\n{'='*70}")
    print(f"  WALK-FORWARD VALIDATION: {symbol}")
    print(f"{'='*70}")
    print(f"  Warmup window: {train_months} months")
    print(f"  Test window:   {test_months} month(s)")
    print(f"  Step size:     {step_months} month(s)")
    print(f"{'='*70}\n")

    # Determine threshold
    threshold = dip_threshold if dip_threshold else (
        DIP_THRESHOLD_STRONG if USE_STRONG_THRESHOLD else DIP_THRESHOLD_PCT
    )
    print(f"  Dip threshold: {threshold*100:.1f}%")

    # 1. Download data
    print(f"\n  Downloading {symbol} data...")
    df = fetch_ohlcv(symbol, timeframe=TIMEFRAME, days=total_days)

    # 2. Add features
    df = add_dip_features(df)

    print(f"  Total rows: {len(df)}")
    print(f"  Date range: {df['timestamp'].min()} -> {df['timestamp'].max()}")

    # 3. Walk through windows
    results = []
    warmup_size = int(train_months * 30 * 24)  # hours
    test_size = int(test_months * 30 * 24)
    step_size = int(step_months * 30 * 24)

    start_idx = warmup_size  # Start after warmup
    window_num = 1

    while True:
        test_end_idx = start_idx + test_size

        if test_end_idx > len(df):
            break

        test_data = df.iloc[start_idx:test_end_idx].copy()

        if len(test_data) < 100:
            start_idx += step_size
            window_num += 1
            continue

        test_start = test_data["timestamp"].min()
        test_end = test_data["timestamp"].max()

        # print(f"\n  [Window {window_num}] ────────────────────────────────")
        # print(f"    Test: {test_start.date()} -> {test_end.date()} ({len(test_data)} rows)")

        try:
            # Generate signals
            test_with_signals = generate_signals(test_data, dip_threshold=threshold)

            # Count signals
            n_signals = (test_with_signals["signal"] == "LONG").sum()
            # print(f"    Signals: {n_signals}")

            if n_signals == 0:
                # print(f"    Skipping - no signals")
                start_idx += step_size
                window_num += 1
                continue

            # Backtest
            backtest_results = run_backtest(
                test_with_signals,
                initial_capital=INITIAL_CAPITAL,
                stop_loss_pct=STOP_LOSS_PCT,
                take_profit_pct=TAKE_PROFIT_PCT,
                hold_timeout=HOLD_TIMEOUT_HOURS,
                fee_pct=FEE_PCT,
            )

            metrics = backtest_results["metrics"]

            result = {
                "window": window_num,
                "test_start": test_start,
                "test_end": test_end,
                "total_return": metrics.get("total_return", 0) * 100,
                "sharpe": metrics.get("sharpe", 0),
                "max_dd": metrics.get("max_drawdown", 0) * 100,
                "win_rate": metrics.get("win_rate", 0) * 100,
                "profit_factor": metrics.get("profit_factor", 0),
                "num_trades": metrics.get("total_trades", 0),
                "expectancy": metrics.get("expectancy", 0) * 100,
            }

            results.append(result)

            # print(f"    Return:   {result['total_return']:>7.2f}%")
            # print(f"    Sharpe:   {result['sharpe']:>7.2f}")
            # print(f"    Win rate: {result['win_rate']:>7.1f}%")
            # print(f"    Trades:   {result['num_trades']:>7}")

        except Exception as e:
            print(f"    Error: {e}")

        start_idx += step_size
        window_num += 1

    return results


def analyze_results(results: list, symbol: str):
    """Analyze walk-forward results."""
    if not results:
        print("\n  No results to analyze")
        return None, 0

    df_results = pd.DataFrame(results)

    print(f"\n{'='*70}")
    print(f"  WALK-FORWARD ANALYSIS: {symbol}")
    print(f"{'='*70}")

    print(f"\n  Summary ({len(results)} windows):")
    print(f"  ─────────────────────────────────────────")

    for metric in ["total_return", "sharpe", "win_rate", "profit_factor"]:
        values = df_results[metric]
        print(f"\n  {metric}:")
        print(f"    Mean:   {values.mean():>8.2f}")
        print(f"    Std:    {values.std():>8.2f}")
        print(f"    Min:    {values.min():>8.2f}")
        print(f"    Max:    {values.max():>8.2f}")

    # Consistency
    positive_returns = (df_results["total_return"] > 0).sum()
    positive_pct = positive_returns / len(results) * 100

    sharpe_positive = (df_results["sharpe"] > 0).sum()
    sharpe_pct = sharpe_positive / len(results) * 100

    pf_above_1 = (df_results["profit_factor"] > 1).sum()
    pf_pct = pf_above_1 / len(results) * 100

    print(f"\n  Consistency:")
    print(f"  ─────────────────────────────────────────")
    print(f"    Positive returns: {positive_returns}/{len(results)} ({positive_pct:.1f}%)")
    print(f"    Sharpe > 0:       {sharpe_positive}/{len(results)} ({sharpe_pct:.1f}%)")
    print(f"    PF > 1:           {pf_above_1}/{len(results)} ({pf_pct:.1f}%)")

    # Stability score
    stability = (positive_pct * 0.4 + sharpe_pct * 0.3 + pf_pct * 0.3)

    print(f"\n  STABILITY SCORE: {stability:.1f}/100")

    if stability >= 70:
        print(f"     EXCELLENT - Strategy is robust")
    elif stability >= 55:
        print(f"     GOOD - Reasonably stable")
    elif stability >= 40:
        print(f"     MODERATE - Some instability")
    else:
        print(f"     POOR - Likely not robust")

    print(f"\n{'='*70}\n")

    return df_results, stability


def run_multi_symbol(
    symbols: list = None,
    test_months: int = WF_TEST_MONTHS,
    dip_threshold: float = None,
):
    """Run walk-forward on multiple symbols."""
    if symbols is None:
        symbols = DEFAULT_SYMBOLS

    print(f"\n{'='*70}")
    print(f"  MULTI-SYMBOL WALK-FORWARD: BUY-THE-DIP")
    print(f"{'='*70}")

    all_scores = {}

    for symbol in symbols:
        try:
            results = walk_forward_validation(
                symbol,
                test_months=test_months,
                dip_threshold=dip_threshold,
            )
            _, stability = analyze_results(results, symbol)
            all_scores[symbol] = stability
        except Exception as e:
            print(f"\n  Error with {symbol}: {e}")

    # Summary
    print(f"\n{'='*70}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*70}\n")

    for symbol, score in all_scores.items():
        status = "GOOD" if score >= 55 else "MODERATE" if score >= 40 else "POOR"
        print(f"  {status:<10} {symbol:<12} Stability: {score:>5.1f}/100")

    if all_scores:
        avg = np.mean(list(all_scores.values()))
        print(f"\n  Average: {avg:.1f}/100")

    print(f"\n{'='*70}\n")

    return all_scores


if __name__ == "__main__":
    # Run walk-forward on symbols with best edge
    scores = run_multi_symbol(
        symbols=DEFAULT_SYMBOLS,
        test_months=2,
        dip_threshold=-0.07,  # Stronger threshold
    )
