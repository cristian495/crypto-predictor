"""
Walk-forward validation for downtrend_breakout v1.
"""

from __future__ import annotations

import argparse
import sys

sys.path.insert(0, "..")

import numpy as np
import pandas as pd

from breakout_momentum.data_loader import fetch_ohlcv
from backtest import run_backtest
from config import (
    DAYS,
    DEFAULT_SYMBOLS,
    MODE_DEFAULT,
    STOP_LOSS_ATR_MULT,
    TAKE_PROFIT_R_MULT,
    HOLD_TIMEOUT_BARS,
    TRAIL_ACTIVATION_R,
    TRAIL_ATR_MULT,
    TIMEFRAME,
    VALID_MODES,
    WF_MIN_TRADES_FOR_PF,
    WF_STEP_MONTHS,
    WF_TEST_MONTHS,
    WF_TRAIN_MONTHS,
    get_symbol_params,
)
from strategy import add_features, generate_signals


def walk_forward_validation(
    symbol: str,
    train_months: int = WF_TRAIN_MONTHS,
    test_months: int = WF_TEST_MONTHS,
    step_months: int = WF_STEP_MONTHS,
    total_days: int = DAYS,
    symbol_params: dict | None = None,
    mode: str = MODE_DEFAULT,
    preloaded_df: pd.DataFrame | None = None,
    preloaded_btc_df: pd.DataFrame | None = None,
    verbose: bool = True,
) -> list:
    """Run rolling out-of-sample validation for one symbol."""
    if mode not in VALID_MODES:
        raise ValueError(f"Unsupported mode: {mode}")

    params = get_symbol_params(symbol)
    if symbol_params:
        params.update(symbol_params)

    if verbose:
        print(f"\n{'='*70}")
        print(f"  WALK-FORWARD VALIDATION: {symbol}")
        print(f"{'='*70}")
        print(f"  Warmup window: {train_months} months")
        print(f"  Test window:   {test_months} month(s)")
        print(f"  Step size:     {step_months} month(s)")
        print(f"{'='*70}\n")
        print(
            "  Config: "
            f"mode={mode} "
            f"ret24_short<={params['short_ret24_threshold']*100:.1f}% "
            f"donchian={params['donchian_window']} "
            f"adx>={params['adx_min']} "
            f"btc_filter={'on' if params.get('use_btc_bear_filter', False) else 'off'}"
        )

    if preloaded_df is None:
        if verbose:
            print(f"\n  Downloading {symbol} data...")
        raw_df = fetch_ohlcv(symbol, timeframe=TIMEFRAME, days=total_days)
    else:
        raw_df = preloaded_df.copy()

    btc_context_df = (
        preloaded_btc_df
        if bool(params.get("use_btc_bear_filter", False)) or bool(params.get("use_btc_regime_switch", False))
        else None
    )
    df = add_features(
        raw_df,
        donchian_window=int(params["donchian_window"]),
        btc_df=btc_context_df,
    )

    if verbose:
        print(f"  Total rows: {len(df)}")
        print(f"  Date range: {df['timestamp'].min()} -> {df['timestamp'].max()}")

    results = []
    warmup_size = int(train_months * 30 * 24)
    test_size = int(test_months * 30 * 24)
    step_size = int(step_months * 30 * 24)

    start_idx = warmup_size
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

        try:
            test_with_signals = generate_signals(test_data, symbol_params=params, mode=mode)
            n_signals = (test_with_signals["signal"] != "NO TRADE").sum()
            if n_signals == 0:
                start_idx += step_size
                window_num += 1
                continue

            backtest_results = run_backtest(
                test_with_signals,
                stop_loss_atr_mult=float(params.get("stop_loss_atr_mult", STOP_LOSS_ATR_MULT)),
                take_profit_r=float(params.get("take_profit_r_mult", TAKE_PROFIT_R_MULT)),
                hold_timeout=int(params.get("hold_timeout_bars", HOLD_TIMEOUT_BARS)),
                trail_activation_r=float(params.get("trail_activation_r", TRAIL_ACTIVATION_R)),
                trail_atr_mult=float(params.get("trail_atr_mult", TRAIL_ATR_MULT)),
            )
            metrics = backtest_results["metrics"]

            results.append(
                {
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
            )
        except Exception as exc:
            if verbose:
                print(f"    Error in window {window_num}: {exc}")

        start_idx += step_size
        window_num += 1

    return results


def analyze_results(
    results: list,
    symbol: str,
    min_trades_for_pf: int = WF_MIN_TRADES_FOR_PF,
    verbose: bool = True,
):
    """Analyze walk-forward results and compute stability score."""
    if not results:
        if verbose:
            print("\n  No results to analyze")
        return None, 0.0

    df_results = pd.DataFrame(results)

    positive_returns = (df_results["total_return"] > 0).sum()
    positive_pct = positive_returns / len(results) * 100

    sharpe_positive = (df_results["sharpe"] > 0).sum()
    sharpe_pct = sharpe_positive / len(results) * 100

    enough_trades = df_results["num_trades"] >= min_trades_for_pf
    low_trade_windows = (~enough_trades).sum()

    pf_above_1 = ((df_results["profit_factor"] > 1) & enough_trades).sum()
    pf_pct = pf_above_1 / len(results) * 100

    stability = positive_pct * 0.4 + sharpe_pct * 0.3 + pf_pct * 0.3

    if verbose:
        print(f"\n{'='*70}")
        print(f"  WALK-FORWARD ANALYSIS: {symbol}")
        print(f"{'='*70}")

        print(f"\n  Summary ({len(results)} windows):")
        print("  ─────────────────────────────────────────")
        for metric in ["total_return", "sharpe", "win_rate", "profit_factor", "max_dd", "num_trades"]:
            values = df_results[metric]
            print(f"\n  {metric}:")
            print(f"    Mean:   {values.mean():>8.2f}")
            print(f"    Median: {values.median():>8.2f}")
            print(f"    Std:    {values.std():>8.2f}")
            print(f"    Min:    {values.min():>8.2f}")
            print(f"    Max:    {values.max():>8.2f}")

        print("\n  Consistency:")
        print("  ─────────────────────────────────────────")
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

        print(f"\n  STABILITY SCORE: {stability:.1f}/100")
        if stability >= 70:
            print("     EXCELLENT - Strategy is robust")
        elif stability >= 55:
            print("     GOOD - Reasonably stable")
        elif stability >= 40:
            print("     MODERATE - Some instability")
        else:
            print("     POOR - Likely not robust")

        print(f"\n{'='*70}\n")

    return df_results, float(stability)


def run_multi_symbol(
    symbols: list | None = None,
    train_months: int = WF_TRAIN_MONTHS,
    test_months: int = WF_TEST_MONTHS,
    step_months: int = WF_STEP_MONTHS,
    total_days: int = DAYS,
    min_trades_for_pf: int = WF_MIN_TRADES_FOR_PF,
    mode: str = MODE_DEFAULT,
    global_param_overrides: dict | None = None,
    title_suffix: str | None = None,
) -> dict:
    """Run walk-forward across multiple symbols."""
    if mode not in VALID_MODES:
        raise ValueError(f"Unsupported mode: {mode}")

    if symbols is None:
        symbols = DEFAULT_SYMBOLS

    title = "  MULTI-SYMBOL WALK-FORWARD: DOWNTREND BREAKOUT"
    if title_suffix:
        title = f"{title} ({title_suffix})"

    print(f"\n{'='*70}")
    print(title)
    print(f"{'='*70}")
    print(f"  Mode: {mode}")

    all_scores = {}
    btc_df = None

    def _resolved_params(sym: str) -> dict:
        params = get_symbol_params(sym)
        if global_param_overrides:
            params.update(global_param_overrides)
        return params

    needs_btc = any(
        bool(_resolved_params(sym).get("use_btc_bear_filter", False))
        or bool(_resolved_params(sym).get("use_btc_regime_switch", False))
        for sym in symbols
    )
    if needs_btc:
        print("\n  Loading BTC context for short filter...")
        btc_df = fetch_ohlcv("BTC/USDT", timeframe=TIMEFRAME, days=total_days)

    for symbol in symbols:
        try:
            params = _resolved_params(symbol)
            results = walk_forward_validation(
                symbol=symbol,
                train_months=train_months,
                test_months=test_months,
                step_months=step_months,
                total_days=total_days,
                symbol_params=params,
                mode=mode,
                preloaded_btc_df=btc_df,
                verbose=True,
            )
            _, stability = analyze_results(
                results,
                symbol=symbol,
                min_trades_for_pf=min_trades_for_pf,
                verbose=True,
            )
            all_scores[symbol] = stability
        except Exception as exc:
            print(f"\n  Error with {symbol}: {exc}")

    print(f"\n{'='*70}")
    print("  FINAL SUMMARY")
    print(f"{'='*70}\n")

    for symbol, score in all_scores.items():
        status = (
            "EXCELLENT" if score >= 70 else "GOOD" if score >= 55 else "MODERATE" if score >= 40 else "POOR"
        )
        print(f"  {status:<10} {symbol:<12} Stability: {score:>5.1f}/100")

    if all_scores:
        avg = np.mean(list(all_scores.values()))
        print(f"\n  Average: {avg:.1f}/100")

    print(f"\n{'='*70}\n")

    return all_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Walk-forward validation for downtrend_breakout")
    parser.add_argument("--symbols", nargs="+", default=None, help="Symbols to evaluate")
    parser.add_argument("--days", type=int, default=DAYS, help=f"Historical days (default: {DAYS})")
    parser.add_argument("--train-months", type=int, default=WF_TRAIN_MONTHS)
    parser.add_argument("--test-months", type=int, default=WF_TEST_MONTHS)
    parser.add_argument("--step-months", type=int, default=WF_STEP_MONTHS)
    parser.add_argument("--min-trades", type=int, default=WF_MIN_TRADES_FOR_PF)
    parser.add_argument("--mode", choices=sorted(VALID_MODES), default=MODE_DEFAULT)

    args = parser.parse_args()

    run_multi_symbol(
        symbols=args.symbols or DEFAULT_SYMBOLS,
        train_months=args.train_months,
        test_months=args.test_months,
        step_months=args.step_months,
        total_days=args.days,
        min_trades_for_pf=args.min_trades,
        mode=args.mode,
    )
