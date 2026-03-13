"""
CLI orchestrator for downtrend_breakout v1.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, "..")

import numpy as np
import pandas as pd

from breakout_momentum.data_loader import fetch_ohlcv
from backtest import compute_exit_levels, run_backtest
from config import (
    BTC_RET24_MAX,
    DAYS,
    DEFAULT_SYMBOLS,
    ENABLE_SELL_THE_RIP,
    HOLD_TIMEOUT_BARS,
    MODE_DEFAULT,
    STOP_LOSS_ATR_MULT,
    TAKE_PROFIT_R_MULT,
    TIMEFRAME,
    TRAIL_ACTIVATION_R,
    TRAIL_ATR_MULT,
    USE_BTC_BEAR_FILTER,
    USE_BTC_REGIME_SWITCH,
    VALID_MODES,
    WF_MIN_TRADES_FOR_PF,
    WF_STEP_MONTHS,
    WF_TEST_MONTHS,
    WF_TRAIN_MONTHS,
    get_symbol_params,
)
from matrix_test import run_matrix_search
from strategy import add_features, generate_signals
from walk_forward import run_multi_symbol

STRATEGY_NAME = Path(__file__).resolve().parent.name


def _resolve_symbol_params(symbol: str, global_param_overrides: dict | None = None) -> dict:
    params = get_symbol_params(symbol)
    if global_param_overrides:
        params.update(global_param_overrides)
    return params


def _write_signals_json(path: str, signals: list, strategy: str):
    payload = {
        "strategy": strategy,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "signals": signals,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True)


def _build_signal_payload(df: pd.DataFrame, symbol: str) -> dict | None:
    last = df.iloc[-1]
    signal = last.get("signal", "NO TRADE")

    if signal not in ["LONG", "SHORT"]:
        return None

    prob = float(last.get("probability", 0.5))
    ret24 = float(last.get("ret_24h", 0)) * 100
    adx = float(last.get("adx14", 0))
    atr_rank = float(last.get("atr_pct_rank_90d", 0))

    extra = f"ret24={ret24:+.2f}% ADX={adx:.1f} ATRrank={atr_rank:.1f}"

    return {
        "symbol": symbol,
        "signal": signal,
        "prob": prob,
        "extra": extra,
    }


def print_current_signal(df: pd.DataFrame, symbol: str, symbol_params: dict | None = None):
    """Print current signal status for a symbol."""
    params = symbol_params or {}
    last = df.iloc[-1]
    signal = last.get("signal", "NO TRADE")
    ret24 = last.get("ret_24h", 0) * 100

    if signal in ["LONG", "SHORT"]:
        direction = 1 if signal == "LONG" else -1
        atr = float(last.get("atr14", np.nan))
        entry = float(last.get("close", np.nan))
        if np.isfinite(atr) and np.isfinite(entry) and atr > 0:
            levels = compute_exit_levels(
                entry_price=entry,
                direction=direction,
                atr=atr,
                stop_loss_atr_mult=float(params.get("stop_loss_atr_mult", STOP_LOSS_ATR_MULT)),
                take_profit_r=float(params.get("take_profit_r_mult", TAKE_PROFIT_R_MULT)),
            )
            print(f"  {symbol:<12} {signal} SIGNAL")
            print(f"               24h move: {ret24:+.2f}%")
            print(f"               Entry:    ${entry:,.4f}")
            print(f"               TP:       ${levels['take_profit']:,.4f}")
            print(f"               SL:       ${levels['stop_loss']:,.4f}")
        else:
            print(f"  {symbol:<12} {signal} SIGNAL  (24h move: {ret24:+.2f}%)")
    else:
        print(f"  {symbol:<12} NO SIGNAL  (24h move: {ret24:+.2f}%)")


def main_scan(
    symbols: list | None = None,
    timeframe: str = TIMEFRAME,
    days: int = DAYS,
    mode: str = MODE_DEFAULT,
    signals_json: str | None = None,
    global_param_overrides: dict | None = None,
    title_suffix: str | None = None,
):
    """Multi-symbol scan + backtest summary."""
    if mode not in VALID_MODES:
        raise ValueError(f"Unsupported mode: {mode}")

    if symbols is None:
        symbols = DEFAULT_SYMBOLS

    title = "  DOWNTREND BREAKOUT — MULTI-CRYPTO SCANNER"
    if title_suffix:
        title = f"{title} ({title_suffix})"

    example_params = _resolve_symbol_params(symbols[0], global_param_overrides) if symbols else {}

    print("=" * 70)
    print(title)
    print("=" * 70)
    print("  Mode:          Scanner")
    print(f"  Symbols:       {len(symbols)}")
    print(f"  Timeframe:     {timeframe}")
    print(f"  History:       {days} days")
    print(f"  Trade mode:    {mode}")
    print(f"  Stop loss:     {float(example_params.get('stop_loss_atr_mult', STOP_LOSS_ATR_MULT)):.2f}x ATR14")
    print(f"  Take profit:   {float(example_params.get('take_profit_r_mult', TAKE_PROFIT_R_MULT)):.2f}R")
    print(
        "  Trailing:      "
        f"on @ +{float(example_params.get('trail_activation_r', TRAIL_ACTIVATION_R)):.1f}R, "
        f"dist {float(example_params.get('trail_atr_mult', TRAIL_ATR_MULT)):.1f}x ATR"
    )
    print(f"  Hold timeout:  {int(example_params.get('hold_timeout_bars', HOLD_TIMEOUT_BARS))} bars")
    print(
        f"  BTC short filter:  {'on' if bool(example_params.get('use_btc_bear_filter', USE_BTC_BEAR_FILTER)) else 'off'} "
        f"(BTC24h<={float(example_params.get('btc_ret24_max', BTC_RET24_MAX))*100:.1f}%)"
    )
    print(f"  BTC regime switch: {'on' if bool(example_params.get('use_btc_regime_switch', USE_BTC_REGIME_SWITCH)) else 'off'}")
    print(f"  Sell-the-rip:      {'on' if bool(example_params.get('enable_sell_the_rip', ENABLE_SELL_THE_RIP)) else 'off'}")
    print("=" * 70)

    print("\n[1/3] Processing symbols...")
    results = {}
    btc_df = None

    needs_btc = any(
        bool(_resolve_symbol_params(sym, global_param_overrides).get("use_btc_bear_filter", False))
        or bool(_resolve_symbol_params(sym, global_param_overrides).get("use_btc_regime_switch", False))
        for sym in symbols
    )
    if needs_btc:
        print("\n[Pre] Loading BTC context for short filter...")
        btc_df = fetch_ohlcv("BTC/USDT", timeframe=timeframe, days=days)

    for i, symbol in enumerate(symbols, 1):
        print(f"\n{'─'*60}")
        print(f"[{i}/{len(symbols)}] {symbol}")
        print("─" * 60)

        try:
            params = _resolve_symbol_params(symbol, global_param_overrides)

            df = fetch_ohlcv(symbol, timeframe=timeframe, days=days)
            df = add_features(
                df,
                donchian_window=int(params["donchian_window"]),
                btc_df=btc_df,
            )
            df = generate_signals(df, symbol_params=params, mode=mode)

            n_signals = (df["signal"] != "NO TRADE").sum()
            n_long = (df["signal"] == "LONG").sum()
            n_short = (df["signal"] == "SHORT").sum()
            print(f"  Signals: {n_signals} ({n_long} LONG / {n_short} SHORT)")
            print(
                "  Config: "
                f"donchian={params['donchian_window']} "
                f"adx>={params['adx_min']} "
                f"short24<={params['short_ret24_threshold']*100:.1f}%"
            )

            bt = run_backtest(
                df,
                stop_loss_atr_mult=float(params.get("stop_loss_atr_mult", STOP_LOSS_ATR_MULT)),
                take_profit_r=float(params.get("take_profit_r_mult", TAKE_PROFIT_R_MULT)),
                hold_timeout=int(params.get("hold_timeout_bars", HOLD_TIMEOUT_BARS)),
                trail_activation_r=float(params.get("trail_activation_r", TRAIL_ACTIVATION_R)),
                trail_atr_mult=float(params.get("trail_atr_mult", TRAIL_ATR_MULT)),
            )
            m = bt["metrics"]

            print(
                f"  Return: {m['total_return']:.2%}  Sharpe: {m['sharpe']:.2f}  "
                f"WR: {m['win_rate']:.1%}  PF: {m['profit_factor']:.2f}  "
                f"Trades: {m['total_trades']}"
            )

            results[symbol] = {
                "df": df,
                "metrics": m,
            }

        except Exception as exc:
            print(f"  ERROR: {exc}")

    if not results:
        print("No symbols processed successfully.")
        return None

    print("\n[2/3] Current signals...")
    print(f"\n{'=' * 70}")
    print("  CURRENT SIGNALS — Downtrend Breakout")
    print(f"{'=' * 70}")

    payload = []
    for symbol, data in results.items():
        symbol_params = _resolve_symbol_params(symbol, global_param_overrides)
        print_current_signal(data["df"], symbol, symbol_params=symbol_params)
        sig = _build_signal_payload(data["df"], symbol)
        if sig:
            payload.append(sig)

    print(f"{'=' * 70}")

    print("\n[3/3] Summary...")
    print(f"\n{'=' * 70}")
    print("  BACKTEST SUMMARY — ALL SYMBOLS")
    print(f"{'=' * 70}")
    print(f"\n  {'Symbol':<12} {'Return':>10} {'Sharpe':>8} {'WR':>8} {'PF':>8} {'Trades':>8}")
    print(f"  {'-'*56}")

    total_return = 0.0
    total_sharpe = 0.0
    total_wr = 0.0
    count = 0

    for symbol, data in results.items():
        m = data["metrics"]
        ret = m["total_return"] * 100
        sharpe = m["sharpe"]
        wr = m["win_rate"] * 100
        pf = m["profit_factor"]
        trades = m["total_trades"]

        print(f"  {symbol:<12} {ret:>9.2f}% {sharpe:>8.2f} {wr:>7.1f}% {pf:>8.2f} {trades:>8}")

        total_return += ret
        total_sharpe += sharpe
        total_wr += wr
        count += 1

    if count > 0:
        print(f"  {'-'*56}")
        print(
            f"  {'AVERAGE':<12} {total_return/count:>9.2f}% "
            f"{total_sharpe/count:>8.2f} {total_wr/count:>7.1f}%"
        )

    print(f"\n{'=' * 70}")

    if signals_json:
        _write_signals_json(signals_json, payload, STRATEGY_NAME)

    return results


def main_walk_forward(
    symbols: list | None = None,
    days: int = DAYS,
    train_months: int = WF_TRAIN_MONTHS,
    test_months: int = WF_TEST_MONTHS,
    step_months: int = WF_STEP_MONTHS,
    min_trades: int = WF_MIN_TRADES_FOR_PF,
    mode: str = MODE_DEFAULT,
    global_param_overrides: dict | None = None,
    title_suffix: str | None = None,
):
    return run_multi_symbol(
        symbols=symbols or DEFAULT_SYMBOLS,
        train_months=train_months,
        test_months=test_months,
        step_months=step_months,
        total_days=days,
        min_trades_for_pf=min_trades,
        mode=mode,
        global_param_overrides=global_param_overrides,
        title_suffix=title_suffix,
    )


def main_matrix(
    symbols: list | None = None,
    days: int = DAYS,
    train_months: int = WF_TRAIN_MONTHS,
    test_months: int = WF_TEST_MONTHS,
    step_months: int = WF_STEP_MONTHS,
    min_trades: int = WF_MIN_TRADES_FOR_PF,
    mode: str = MODE_DEFAULT,
):
    return run_matrix_search(
        symbols=symbols or DEFAULT_SYMBOLS,
        days=days,
        train_months=train_months,
        test_months=test_months,
        step_months=step_months,
        min_trades_for_pf=min_trades,
        mode=mode,
    )


def main_ab(
    symbols: list | None = None,
    days: int = DAYS,
    train_months: int = WF_TRAIN_MONTHS,
    test_months: int = WF_TEST_MONTHS,
    step_months: int = WF_STEP_MONTHS,
    min_trades: int = WF_MIN_TRADES_FOR_PF,
    mode: str = MODE_DEFAULT,
):
    symbols = symbols or DEFAULT_SYMBOLS

    baseline_overrides = {
        "use_btc_regime_switch": False,
        "enable_sell_the_rip": False,
    }
    variant_overrides = {
        "use_btc_regime_switch": True,
        "enable_sell_the_rip": True,
    }

    print(f"\n{'=' * 70}")
    print("  WALK-FORWARD A/B COMPARISON")
    print(f"{'=' * 70}")
    print("  Baseline: BTC regime switch=off, sell-the-rip=off")
    print("  Variant:  BTC regime switch=on,  sell-the-rip=on")
    print(f"{'=' * 70}")

    baseline_scores = main_walk_forward(
        symbols=symbols,
        days=days,
        train_months=train_months,
        test_months=test_months,
        step_months=step_months,
        min_trades=min_trades,
        mode=mode,
        global_param_overrides=baseline_overrides,
        title_suffix="Baseline",
    )

    variant_scores = main_walk_forward(
        symbols=symbols,
        days=days,
        train_months=train_months,
        test_months=test_months,
        step_months=step_months,
        min_trades=min_trades,
        mode=mode,
        global_param_overrides=variant_overrides,
        title_suffix="Variant",
    )

    print(f"\n{'=' * 70}")
    print("  A/B SUMMARY")
    print(f"{'=' * 70}\n")
    print("  Symbol           Before      After      Delta")
    print("  ----------------------------------------------")

    before_vals = []
    after_vals = []
    for symbol in symbols:
        before = float(baseline_scores.get(symbol, 0.0))
        after = float(variant_scores.get(symbol, 0.0))
        delta = after - before
        before_vals.append(before)
        after_vals.append(after)
        print(f"  {symbol:<14} {before:>7.1f}    {after:>7.1f}    {delta:>+7.1f}")

    if before_vals and after_vals:
        b_avg = float(np.mean(before_vals))
        a_avg = float(np.mean(after_vals))
        d_avg = a_avg - b_avg
        print("  ----------------------------------------------")
        print(f"  {'AVERAGE':<14} {b_avg:>7.1f}    {a_avg:>7.1f}    {d_avg:>+7.1f}")

    print(f"\n{'=' * 70}\n")
    return {"baseline": baseline_scores, "variant": variant_scores}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="downtrend_breakout v1")
    parser.add_argument("--timeframe", default=TIMEFRAME, help=f"Candle timeframe (default: {TIMEFRAME})")
    parser.add_argument("--days", type=int, default=DAYS, help=f"Days of history (default: {DAYS})")
    parser.add_argument("--scan", action="store_true", help="Run scanner mode")
    parser.add_argument("--walk-forward", action="store_true", help="Run walk-forward validation")
    parser.add_argument("--matrix", action="store_true", help="Run matrix parameter tuning")
    parser.add_argument("--ab", action="store_true", help="Run walk-forward A/B comparison (regime switch + sell-the-rip)")
    parser.add_argument("--symbols", nargs="+", default=None, help="Symbols to process")
    parser.add_argument("--signals-json", default=None, help="Write current signals to JSON")
    parser.add_argument("--mode", choices=sorted(VALID_MODES), default=MODE_DEFAULT, help="Signal mode")
    parser.add_argument("--train-months", type=int, default=WF_TRAIN_MONTHS)
    parser.add_argument("--test-months", type=int, default=WF_TEST_MONTHS)
    parser.add_argument("--step-months", type=int, default=WF_STEP_MONTHS)
    parser.add_argument("--min-trades", type=int, default=WF_MIN_TRADES_FOR_PF)

    args = parser.parse_args()

    if args.scan:
        main_scan(
            symbols=args.symbols or DEFAULT_SYMBOLS,
            timeframe=args.timeframe,
            days=args.days,
            mode=args.mode,
            signals_json=args.signals_json,
        )
    elif args.walk_forward:
        main_walk_forward(
            symbols=args.symbols or DEFAULT_SYMBOLS,
            days=args.days,
            train_months=args.train_months,
            test_months=args.test_months,
            step_months=args.step_months,
            min_trades=args.min_trades,
            mode=args.mode,
        )
    elif args.matrix:
        main_matrix(
            symbols=args.symbols or DEFAULT_SYMBOLS,
            days=args.days,
            train_months=args.train_months,
            test_months=args.test_months,
            step_months=args.step_months,
            min_trades=args.min_trades,
            mode=args.mode,
        )
    elif args.ab:
        main_ab(
            symbols=args.symbols or DEFAULT_SYMBOLS,
            days=args.days,
            train_months=args.train_months,
            test_months=args.test_months,
            step_months=args.step_months,
            min_trades=args.min_trades,
            mode=args.mode,
        )
    else:
        parser.error("Use one mode: --scan, --walk-forward, --matrix, or --ab")
