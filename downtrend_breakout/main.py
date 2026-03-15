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
from strategy import (
    _sell_rip_confidence,
    _signal_confidence,
    add_features,
    generate_signals,
)
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


def _evaluate_downtrend_conditions(
    df: pd.DataFrame, symbol_params: dict, mode: str
) -> tuple[list[tuple[str, bool, str]], str]:
    """Evaluate current-bar rules and return condition lines + blocking reason."""
    if df.empty:
        return [("data", False, "empty dataframe")], "data (empty dataframe)"

    row = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else None

    checks: list[tuple[str, bool, str]] = []

    use_btc_bear_filter = bool(symbol_params.get("use_btc_bear_filter", False))
    use_btc_regime_switch = bool(symbol_params.get("use_btc_regime_switch", False))
    enable_sell_the_rip = bool(symbol_params.get("enable_sell_the_rip", False))

    short_ret24_threshold = float(symbol_params["short_ret24_threshold"])
    short_ret6_threshold = float(symbol_params["short_ret6_threshold"])
    adx_min = float(symbol_params["adx_min"])
    atr_rank_max = float(symbol_params["atr_pct_rank_max"])
    rel_vol_min = float(symbol_params["relative_volume_min"])
    min_prob = float(symbol_params.get("min_probability", 0.50))

    ret_24h = row.get("ret_24h", np.nan)
    ret_6h = row.get("ret_6h", np.nan)
    close = row.get("close", np.nan)
    donchian_low = row.get("donchian_low", np.nan)
    donchian_high = row.get("donchian_high", np.nan)
    adx14 = row.get("adx14", np.nan)
    atr_rank = row.get("atr_pct_rank_90d", np.nan)
    rel_vol = row.get("relative_volume_20", np.nan)
    bearish = bool(row.get("regime_bearish", False))
    bullish = bool(row.get("regime_bullish", False))

    # Short breakdown conditions
    checks.append(("regime_bearish", bearish, f"regime_bearish={bearish}"))

    c_break_low = pd.notna(close) and pd.notna(donchian_low) and float(close) < float(donchian_low)
    checks.append(
        (
            "break_donchian_low",
            bool(c_break_low),
            (
                f"close={float(close):.6f} < donchian_low={float(donchian_low):.6f}"
                if pd.notna(close) and pd.notna(donchian_low)
                else "close/donchian_low NaN"
            ),
        )
    )

    c_ret24 = pd.notna(ret_24h) and float(ret_24h) <= short_ret24_threshold
    checks.append(
        (
            "ret_24h_short",
            bool(c_ret24),
            (
                f"ret_24h={float(ret_24h)*100:+.2f}% <= {short_ret24_threshold*100:+.2f}%"
                if pd.notna(ret_24h)
                else "ret_24h=NaN"
            ),
        )
    )

    c_ret6 = pd.notna(ret_6h) and float(ret_6h) <= short_ret6_threshold
    checks.append(
        (
            "ret_6h_short",
            bool(c_ret6),
            (
                f"ret_6h={float(ret_6h)*100:+.2f}% <= {short_ret6_threshold*100:+.2f}%"
                if pd.notna(ret_6h)
                else "ret_6h=NaN"
            ),
        )
    )

    c_adx = pd.notna(adx14) and float(adx14) >= adx_min
    checks.append(
        (
            "adx_min",
            bool(c_adx),
            f"adx14={float(adx14):.2f} >= {adx_min:.2f}" if pd.notna(adx14) else "adx14=NaN",
        )
    )

    c_atr_rank = pd.notna(atr_rank) and float(atr_rank) <= atr_rank_max
    checks.append(
        (
            "atr_rank_max",
            bool(c_atr_rank),
            (
                f"atr_rank={float(atr_rank):.2f} <= {atr_rank_max:.2f}"
                if pd.notna(atr_rank)
                else "atr_rank=NaN"
            ),
        )
    )

    c_rel_vol = pd.notna(rel_vol) and float(rel_vol) >= rel_vol_min
    checks.append(
        (
            "relative_volume",
            bool(c_rel_vol),
            (
                f"rel_vol={float(rel_vol):.2f} >= {rel_vol_min:.2f}"
                if pd.notna(rel_vol)
                else "rel_vol=NaN"
            ),
        )
    )

    short_breakdown_ok = all([bearish, c_break_low, c_ret24, c_ret6, c_adx, c_atr_rank, c_rel_vol])

    # BTC macro filters
    macro_short_ok = True
    if use_btc_regime_switch:
        btc_ret24 = row.get("btc_ret_24h", np.nan)
        btc_ret7d = row.get("btc_ret_7d", np.nan)
        btc_regime_bearish = bool(row.get("btc_regime_bearish", False))
        btc_ret24_max = float(symbol_params.get("btc_ret24_max", 0.0))
        btc_ret7d_max = float(symbol_params.get("btc_ret7d_max", 0.0))

        macro_short_ok = (
            pd.notna(btc_ret24)
            and pd.notna(btc_ret7d)
            and btc_regime_bearish
            and float(btc_ret24) <= btc_ret24_max
            and float(btc_ret7d) <= btc_ret7d_max
        )
        detail = (
            f"btc_regime_bearish={btc_regime_bearish}, "
            f"btc24={float(btc_ret24)*100:+.2f}%<={btc_ret24_max*100:+.2f}%, "
            f"btc7d={float(btc_ret7d)*100:+.2f}%<={btc_ret7d_max*100:+.2f}%"
            if pd.notna(btc_ret24) and pd.notna(btc_ret7d)
            else "btc_ret_24h/btc_ret_7d NaN"
        )
        checks.append(("btc_macro_short", bool(macro_short_ok), detail))
    elif use_btc_bear_filter:
        btc_ret24 = row.get("btc_ret_24h", np.nan)
        btc_regime_bearish = bool(row.get("btc_regime_bearish", False))
        btc_ret24_max = float(symbol_params.get("btc_ret24_max", 0.0))
        macro_short_ok = (
            pd.notna(btc_ret24)
            and btc_regime_bearish
            and float(btc_ret24) <= btc_ret24_max
        )
        detail = (
            f"btc_regime_bearish={btc_regime_bearish}, btc24={float(btc_ret24)*100:+.2f}%<={btc_ret24_max*100:+.2f}%"
            if pd.notna(btc_ret24)
            else "btc_ret_24h=NaN"
        )
        checks.append(("btc_macro_short", bool(macro_short_ok), detail))
    else:
        checks.append(("btc_macro_short", True, "disabled"))

    # Sell-the-rip conditions
    sell_rip_ok = False
    if enable_sell_the_rip and prev is not None and bearish:
        prev_close = prev.get("close", np.nan)
        prev_ema50 = prev.get("ema50", np.nan)
        ema50 = row.get("ema50", np.nan)
        prev_ret6h = prev.get("ret_6h", np.nan)
        sell_rip_ret6h_min = float(symbol_params.get("sell_rip_ret6h_min", 0.02))
        sell_rip_ret24h_max = float(symbol_params.get("sell_rip_ret24h_max", 0.03))

        prev_above_ema50 = pd.notna(prev_close) and pd.notna(prev_ema50) and float(prev_close) > float(prev_ema50)
        cross_back_below_ema50 = prev_above_ema50 and pd.notna(close) and pd.notna(ema50) and float(close) <= float(ema50)
        ret6h_rip = max(
            float(ret_6h) if pd.notna(ret_6h) else float("-inf"),
            float(prev_ret6h) if pd.notna(prev_ret6h) else float("-inf"),
        )
        c_rip6 = ret6h_rip >= sell_rip_ret6h_min
        c_rip24 = pd.notna(ret_24h) and float(ret_24h) <= sell_rip_ret24h_max

        sell_rip_ok = all([cross_back_below_ema50, c_rip6, c_rip24, c_adx, c_atr_rank, c_rel_vol])
        checks.append(
            (
                "sell_the_rip",
                bool(sell_rip_ok),
                (
                    f"cross_ema50={cross_back_below_ema50}, ret6h_rip={ret6h_rip*100:+.2f}%>={sell_rip_ret6h_min*100:.2f}%, "
                    f"ret24={float(ret_24h)*100:+.2f}%<={sell_rip_ret24h_max*100:.2f}%"
                    if pd.notna(ret_24h)
                    else "ret_24h=NaN or missing prev"
                ),
            )
        )
    elif enable_sell_the_rip:
        checks.append(("sell_the_rip", False, "not enough data or bearish regime=False"))
    else:
        checks.append(("sell_the_rip", True, "disabled"))

    short_ok = macro_short_ok and (short_breakdown_ok or sell_rip_ok)
    checks.append(("short_setup", bool(short_ok), f"macro_short_ok={macro_short_ok}, breakdown={short_breakdown_ok}, sell_rip={sell_rip_ok}"))

    if short_ok:
        try:
            confidence = (
                _sell_rip_confidence(row, symbol_params)
                if sell_rip_ok and not short_breakdown_ok
                else _signal_confidence(row, symbol_params, direction=-1)
            )
            prob_ok = confidence >= min_prob
            checks.append(("min_probability", bool(prob_ok), f"prob={confidence:.3f} >= min={min_prob:.3f}"))
        except Exception as exc:
            checks.append(("min_probability", False, f"error computing confidence: {exc}"))
            prob_ok = False
        short_ok = short_ok and prob_ok

    if mode == "long_short":
        long_ret24_min = float(symbol_params["long_ret24_min"])
        long_ret6_min = float(symbol_params["long_ret6_min"])
        c_long_break = pd.notna(close) and pd.notna(donchian_high) and float(close) > float(donchian_high)
        c_long_ret24 = pd.notna(ret_24h) and float(ret_24h) >= long_ret24_min
        c_long_ret6 = pd.notna(ret_6h) and float(ret_6h) >= long_ret6_min
        long_ok = all([bullish, c_long_break, c_long_ret24, c_long_ret6, c_adx, c_atr_rank, c_rel_vol])
        checks.append(("regime_bullish", bool(bullish), f"regime_bullish={bullish}"))
        checks.append(("long_setup", bool(long_ok), f"close>donchian_high={c_long_break}, ret24_ok={c_long_ret24}, ret6_ok={c_long_ret6}"))

    blocking = "none"
    for name, ok, detail in checks:
        if not ok:
            blocking = f"{name} ({detail})"
            break

    return checks, blocking


def print_current_signal(df: pd.DataFrame, symbol: str, symbol_params: dict | None = None, mode: str = MODE_DEFAULT):
    """Print current signal status for a symbol."""
    params = symbol_params or {}
    last = df.iloc[-1]
    signal = last.get("signal", "NO TRADE")
    ret24 = last.get("ret_24h", 0) * 100
    checks, blocking = _evaluate_downtrend_conditions(df, params, mode=mode)

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
    print("               Conditions:")
    for name, ok, detail in checks:
        state = "PASS" if ok else "FAIL"
        print(f"                 - {name:<16} {state:<4} | {detail}")
    if signal not in ["LONG", "SHORT"]:
        print(f"               Blocked by: {blocking}")


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
        print_current_signal(data["df"], symbol, symbol_params=symbol_params, mode=mode)
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
