"""
Grid search for downtrend_breakout v1.

Tunes per-symbol parameters and writes selected overrides to matrix_overrides.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "..")

import pandas as pd

from breakout_momentum.data_loader import fetch_ohlcv
from config import (
    DAYS,
    DEFAULT_SYMBOLS,
    MATRIX_ADX_MIN_GRID,
    MATRIX_DONCHIAN_GRID,
    MATRIX_OVERRIDES_PATH,
    MATRIX_RET24_GRID,
    MATRIX_RUNS_DIR,
    MODE_DEFAULT,
    TIMEFRAME,
    VALID_MODES,
    WF_MIN_TRADES_FOR_PF,
    WF_STEP_MONTHS,
    WF_TEST_MONTHS,
    WF_TRAIN_MONTHS,
    get_symbol_params,
)
from walk_forward import analyze_results, walk_forward_validation


def _safe_symbol(symbol: str) -> str:
    return symbol.replace("/", "_")


def _evaluate_symbol_grid(
    symbol: str,
    raw_df: pd.DataFrame,
    btc_df: pd.DataFrame | None,
    train_months: int,
    test_months: int,
    step_months: int,
    min_trades_for_pf: int,
    mode: str,
) -> tuple[pd.DataFrame, dict]:
    rows = []
    base = get_symbol_params(symbol)

    for ret24_threshold in MATRIX_RET24_GRID:
        for donchian_window in MATRIX_DONCHIAN_GRID:
            for adx_min in MATRIX_ADX_MIN_GRID:
                params = dict(base)
                params["short_ret24_threshold"] = float(ret24_threshold)
                params["donchian_window"] = int(donchian_window)
                params["adx_min"] = float(adx_min)

                results = walk_forward_validation(
                    symbol=symbol,
                    train_months=train_months,
                    test_months=test_months,
                    step_months=step_months,
                    total_days=0,  # ignored when preloaded_df is used
                    symbol_params=params,
                    mode=mode,
                    preloaded_df=raw_df,
                    preloaded_btc_df=btc_df,
                    verbose=False,
                )

                df_results, stability = analyze_results(
                    results,
                    symbol,
                    min_trades_for_pf=min_trades_for_pf,
                    verbose=False,
                )

                if df_results is None or df_results.empty:
                    windows = 0
                    median_return = -999.0
                    median_max_dd = -999.0
                    median_trades = 0.0
                else:
                    windows = int(len(df_results))
                    median_return = float(df_results["total_return"].median())
                    median_max_dd = float(df_results["max_dd"].median())
                    median_trades = float(df_results["num_trades"].median())

                rows.append(
                    {
                        "symbol": symbol,
                        "short_ret24_threshold": ret24_threshold,
                        "donchian_window": donchian_window,
                        "adx_min": adx_min,
                        "stability": float(stability),
                        "median_return": median_return,
                        "median_max_dd": median_max_dd,
                        "abs_median_max_dd": abs(median_max_dd),
                        "median_trades": median_trades,
                        "windows": windows,
                    }
                )

    df = pd.DataFrame(rows)
    df_sorted = df.sort_values(
        by=["stability", "median_return", "abs_median_max_dd"],
        ascending=[False, False, True],
    )

    best = df_sorted.iloc[0].to_dict()
    return df_sorted.reset_index(drop=True), best


def run_matrix_search(
    symbols: list | None = None,
    days: int = DAYS,
    train_months: int = WF_TRAIN_MONTHS,
    test_months: int = WF_TEST_MONTHS,
    step_months: int = WF_STEP_MONTHS,
    min_trades_for_pf: int = WF_MIN_TRADES_FOR_PF,
    mode: str = MODE_DEFAULT,
) -> dict:
    """Run parameter grid search and persist selected per-symbol overrides."""
    if mode not in VALID_MODES:
        raise ValueError(f"Unsupported mode: {mode}")

    if symbols is None:
        symbols = DEFAULT_SYMBOLS

    MATRIX_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    selected: dict[str, dict] = {}
    btc_df = None
    needs_btc = any(
        bool(get_symbol_params(sym).get("use_btc_bear_filter", False))
        or bool(get_symbol_params(sym).get("use_btc_regime_switch", False))
        for sym in symbols
    )

    print(f"\n{'='*70}")
    print("  MATRIX SEARCH: DOWNTREND BREAKOUT")
    print(f"{'='*70}")
    print(
        "  Grid: "
        f"ret24={MATRIX_RET24_GRID} "
        f"donchian={MATRIX_DONCHIAN_GRID} "
        f"adx_min={MATRIX_ADX_MIN_GRID}"
    )
    print(f"  Mode: {mode}")

    if needs_btc:
        print("\n  Loading BTC context for short filter...")
        btc_df = fetch_ohlcv("BTC/USDT", timeframe=TIMEFRAME, days=days)

    for symbol in symbols:
        print(f"\n{'-'*70}")
        print(f"  {symbol}")
        print(f"{'-'*70}")

        raw_df = fetch_ohlcv(symbol, timeframe=TIMEFRAME, days=days)
        grid_df, best = _evaluate_symbol_grid(
            symbol=symbol,
            raw_df=raw_df,
            btc_df=btc_df,
            train_months=train_months,
            test_months=test_months,
            step_months=step_months,
            min_trades_for_pf=min_trades_for_pf,
            mode=mode,
        )

        csv_path = MATRIX_RUNS_DIR / f"{_safe_symbol(symbol)}_matrix.csv"
        grid_df.to_csv(csv_path, index=False)

        selected[symbol] = {
            "short_ret24_threshold": float(best["short_ret24_threshold"]),
            "donchian_window": int(best["donchian_window"]),
            "adx_min": float(best["adx_min"]),
        }

        print(
            "  Best -> "
            f"stability={best['stability']:.1f} "
            f"ret24={best['short_ret24_threshold']:.2f} "
            f"donchian={int(best['donchian_window'])} "
            f"adx={best['adx_min']:.0f}"
        )
        print(f"  Saved grid: {csv_path}")

    with open(MATRIX_OVERRIDES_PATH, "w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2, ensure_ascii=True, sort_keys=True)

    print(f"\nSaved overrides: {MATRIX_OVERRIDES_PATH}")
    print(f"{'='*70}\n")

    return selected


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Matrix tuning for downtrend_breakout")
    parser.add_argument("--symbols", nargs="+", default=None, help="Symbols to tune")
    parser.add_argument("--days", type=int, default=DAYS)
    parser.add_argument("--train-months", type=int, default=WF_TRAIN_MONTHS)
    parser.add_argument("--test-months", type=int, default=WF_TEST_MONTHS)
    parser.add_argument("--step-months", type=int, default=WF_STEP_MONTHS)
    parser.add_argument("--min-trades", type=int, default=WF_MIN_TRADES_FOR_PF)
    parser.add_argument("--mode", choices=sorted(VALID_MODES), default=MODE_DEFAULT)

    args = parser.parse_args()

    run_matrix_search(
        symbols=args.symbols or DEFAULT_SYMBOLS,
        days=args.days,
        train_months=args.train_months,
        test_months=args.test_months,
        step_months=args.step_months,
        min_trades_for_pf=args.min_trades,
        mode=args.mode,
    )
