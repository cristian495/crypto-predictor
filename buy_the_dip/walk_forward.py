"""
walk_forward.py
Walk-Forward Validation for Buy-the-Dip strategy.

Tests strategy robustness by:
- Testing on rolling out-of-sample periods
- Measuring consistency across time
- No training needed (rule-based strategy)
"""

import argparse
import sys
sys.path.insert(0, "..")

import pandas as pd
import numpy as np
from breakout_momentum.data_loader import fetch_ohlcv
from strategy import add_dip_features, generate_signals
from backtest import run_backtest
from config import (
    DEFAULT_SYMBOLS,
    TIMEFRAME,
    DAYS,
    INITIAL_CAPITAL,
    FEE_PCT,
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
    HOLD_TIMEOUT_HOURS,
    WF_TRAIN_MONTHS,
    WF_TEST_MONTHS,
    WF_STEP_MONTHS,
    get_symbol_signal_params,
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

    signal_params = get_symbol_signal_params(
        symbol, dip_threshold_override=dip_threshold
    )
    threshold = signal_params["dip_threshold"]
    print(f"  Dip threshold: {threshold*100:.1f}%")
    print(
        "  Filters: "
        f"RSI={'on' if signal_params['use_rsi_filter'] else 'off'} "
        f"VOL={'on' if signal_params['use_volume_filter'] else 'off'}"
    )

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
            test_with_signals = generate_signals(
                test_data,
                dip_threshold=threshold,
                use_volume_filter=signal_params["use_volume_filter"],
                use_rsi_filter=signal_params["use_rsi_filter"],
            )

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


def analyze_results(
    results: list,
    symbol: str,
    min_trades_for_pf: int = 5,
):
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

    for metric in ["total_return", "sharpe", "win_rate", "profit_factor", "max_dd", "num_trades"]:
        values = df_results[metric]
        print(f"\n  {metric}:")
        print(f"    Mean:   {values.mean():>8.2f}")
        print(f"    Median: {values.median():>8.2f}")
        print(f"    Std:    {values.std():>8.2f}")
        print(f"    Min:    {values.min():>8.2f}")
        print(f"    Max:    {values.max():>8.2f}")

    # Consistency
    positive_returns = (df_results["total_return"] > 0).sum()
    positive_pct = positive_returns / len(results) * 100

    sharpe_positive = (df_results["sharpe"] > 0).sum()
    sharpe_pct = sharpe_positive / len(results) * 100

    enough_trades = df_results["num_trades"] >= min_trades_for_pf
    low_trade_windows = (~enough_trades).sum()

    pf_above_1 = ((df_results["profit_factor"] > 1) & enough_trades).sum()
    pf_pct = pf_above_1 / len(results) * 100

    print(f"\n  Consistency:")
    print(f"  ─────────────────────────────────────────")
    print(f"    Positive returns: {positive_returns}/{len(results)} ({positive_pct:.1f}%)")
    print(f"    Sharpe > 0:       {sharpe_positive}/{len(results)} ({sharpe_pct:.1f}%)")
    print(
        f"    PF > 1 (trades>={min_trades_for_pf}): "
        f"{pf_above_1}/{len(results)} ({pf_pct:.1f}%)"
    )
    print(
        f"    Low-trade windows (<{min_trades_for_pf}): "
        f"{low_trade_windows}/{len(results)}"
    )

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
    train_months: int = WF_TRAIN_MONTHS,
    test_months: int = WF_TEST_MONTHS,
    step_months: int = WF_STEP_MONTHS,
    total_days: int = DAYS,
    dip_threshold: float = None,
    min_trades_for_pf: int = 5,
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
                train_months=train_months,
                test_months=test_months,
                step_months=step_months,
                total_days=total_days,
                dip_threshold=dip_threshold,
            )
            _, stability = analyze_results(
                results,
                symbol,
                min_trades_for_pf=min_trades_for_pf,
            )
            all_scores[symbol] = stability
        except Exception as e:
            print(f"\n  Error with {symbol}: {e}")

    # Summary
    print(f"\n{'='*70}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*70}\n")

    for symbol, score in all_scores.items():
        status = (
            "EXCELLENT" if score >= 70
            else "GOOD" if score >= 55
            else "MODERATE" if score >= 40
            else "POOR"
        )
        print(f"  {status:<10} {symbol:<12} Stability: {score:>5.1f}/100")

    if all_scores:
        avg = np.mean(list(all_scores.values()))
        print(f"\n  Average: {avg:.1f}/100")

    print(f"\n{'='*70}\n")

    return all_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Walk-forward validation for Buy-the-Dip strategy"
    )
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="Symbols to evaluate (default: config list)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Global dip threshold override (e.g., -0.07)")
    parser.add_argument("--train-months", type=int, default=WF_TRAIN_MONTHS,
                        help=f"Warmup months (default: {WF_TRAIN_MONTHS})")
    parser.add_argument("--test-months", type=int, default=WF_TEST_MONTHS,
                        help=f"Test window months (default: {WF_TEST_MONTHS})")
    parser.add_argument("--step-months", type=int, default=WF_STEP_MONTHS,
                        help=f"Step months (default: {WF_STEP_MONTHS})")
    parser.add_argument("--days", type=int, default=DAYS,
                        help=f"Historical days to download (default: {DAYS})")
    parser.add_argument("--min-trades", type=int, default=5,
                        help="Min trades per window for PF consistency checks")

    args = parser.parse_args()

    run_multi_symbol(
        symbols=args.symbols or DEFAULT_SYMBOLS,
        train_months=args.train_months,
        test_months=args.test_months,
        step_months=args.step_months,
        total_days=args.days,
        dip_threshold=args.threshold,
        min_trades_for_pf=args.min_trades,
    )
