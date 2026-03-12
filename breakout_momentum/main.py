"""
main.py
CLI orchestrator for the breakout momentum trading system.

Usage:
    python main.py                                  # Single: BNB/USDT
    python main.py --symbol SOL/USDT                # Single: specific pair
    python main.py --symbol BNB/USDT --optimize     # Single + Optuna
    python main.py --scan                           # Multi-crypto scan
    python main.py --scan --optimize                # Multi + Optuna
    python main.py --scan --symbols BNB/USDT SOL/USDT AVAX/USDT
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

from config import (
    TIMEFRAME,
    DAYS,
    DEFAULT_SYMBOLS,
    BREAKOUT_PERIOD,
    BREAKOUT_THRESHOLD,
    TARGET_TP_PCT,
    TARGET_SL_ATR_MULT,
    TARGET_HORIZON,
    STOP_LOSS_ATR_MULT,
    TAKE_PROFIT_PCT,
    TAKE_PROFIT_RR,
    USE_PERCENTAGE_TP,
    HOLD_TIMEOUT,
    POSITION_PCT,
    MAX_POSITIONS,
    FEE_PCT,
    INITIAL_CAPITAL,
    BUY_THRESHOLD,
    MIN_AGREE,
    VOL_PROBABILITY_THRESHOLD,
    SURGE_PROBABILITY_THRESHOLD,
    COMP_PROBABILITY_THRESHOLD,
)
from data_loader import load_all_data, load_btc_for_context
from features import add_features, FEATURE_COLS
from target import prepare_dataset, get_training_data
from model import EnsembleModel, temporal_split_3way, optimize_ensemble
from strategy import generate_signals, print_current_signals
from backtest import (
    run_backtest,
    run_multi_backtest,
    plot_equity_curve,
)
from metrics import print_performance_report

STRATEGY_NAME = Path(__file__).resolve().parent.name


# ── Single symbol mode ───────────────────────────────────────


def main_single(
    symbol: str = "BNB/USDT",
    timeframe: str = TIMEFRAME,
    days: int = DAYS,
    optimize: bool = False,
    buy_threshold: float = BUY_THRESHOLD,
):
    """Full pipeline for a single symbol."""

    print("=" * 70)
    print("  BREAKOUT MOMENTUM TRADING SYSTEM")
    print("=" * 70)
    print(f"  Mode:          Single")
    print(f"  Symbol:        {symbol}")
    print(f"  Timeframe:     {timeframe}")
    print(f"  History:       {days} days")
    print(f"  Breakout:      {BREAKOUT_PERIOD}-period Donchian (threshold {BREAKOUT_THRESHOLD})")
    print(f"  Target:        {TARGET_TP_PCT:.0%} TP in {TARGET_HORIZON} candles")
    print(f"  TP / SL:       {'%' if USE_PERCENTAGE_TP else 'RR'}-based: "
          f"{TAKE_PROFIT_PCT:.1%} / {STOP_LOSS_ATR_MULT}xATR")
    print(f"  Fees:          {FEE_PCT:.2%} per side")
    print(f"  Strategy 1:    Volatility Breakout (prob > {VOL_PROBABILITY_THRESHOLD})")
    print(f"  Strategy 2:    Volume Surge (prob > {SURGE_PROBABILITY_THRESHOLD})")
    print(f"  Strategy 3:    Range Compression (prob > {COMP_PROBABILITY_THRESHOLD})")
    print(f"  Ensemble:      {MIN_AGREE}/3 models must agree")
    print(f"  Features:      {len(FEATURE_COLS)}")
    print(f"  Optimize:      {'Yes' if optimize else 'No'}")
    print("=" * 70)

    # 1. Download data
    print(f"\n[1/8] Downloading data...")
    raw_df = load_all_data(symbol, timeframe, days)

    # Load BTC for market context (if not BTC itself)
    btc_df = None
    if symbol not in ["BTC/USDT", "BTCUSDT"]:
        print(f"  Loading BTC for market context...")
        btc_df = load_btc_for_context(timeframe, days)

    # 2. Feature engineering + target
    print(f"\n[2/8] Computing features and breakout target...")
    full_df = prepare_dataset(raw_df, symbol=symbol, btc_df=btc_df)
    train_data = get_training_data(full_df)
    n_tradeable = len(train_data)
    n_total = len(full_df)
    print(f"  Total rows: {n_total}")
    print(f"  Tradeable rows (breakouts): {n_tradeable} ({n_tradeable/n_total:.1%})")
    if n_tradeable > 0:
        success_pct = train_data["target"].mean()
        print(f"  Breakout success rate: {success_pct:.1%}")
        n_long = (train_data["direction"] == 1).sum()
        n_short = (train_data["direction"] == -1).sum()
        print(f"  Long / Short breakouts: {n_long} / {n_short}")

    if n_tradeable < 100:
        print(f"\n  WARNING: Only {n_tradeable} tradeable rows. Need at least 100.")
        print("  Try increasing --days or adjusting BREAKOUT_PERIOD in config.py")
        return None

    # 3. Temporal split
    print(f"\n[3/8] Temporal split (70/15/15)...")
    train, val, test = temporal_split_3way(train_data)

    # 4. Optimize (optional)
    params = None
    if optimize:
        print(f"\n[4/8] Optimizing hyperparameters with Optuna...")
        params = optimize_ensemble(train)
    else:
        print(f"\n[4/8] Skipping optimization (use --optimize to enable)")

    # 5. Train ensemble
    print(f"\n[5/8] Training ensemble model...")
    model = EnsembleModel()
    model.fit(train, params=params)

    # Evaluate on validation
    X_val = val[FEATURE_COLS].values
    y_val = val["target"].values.astype(int)
    val_proba = model.predict_proba(X_val)
    from sklearn.metrics import roc_auc_score
    val_auc = roc_auc_score(y_val, val_proba)
    print(f"  Validation AUC: {val_auc:.4f}")

    # 6. Generate signals on full test period and backtest
    print(f"\n[6/8] Backtesting on test period...")

    # We need the FULL dataframe (not just tradeable rows) for backtest
    # Re-split the full_df by timestamp ranges
    test_start = test["timestamp"].iloc[0]
    test_end = test["timestamp"].iloc[-1]
    full_test = full_df[
        (full_df["timestamp"] >= test_start) &
        (full_df["timestamp"] <= test_end)
    ].copy()

    # Drop rows with NaN in features
    full_test = full_test.dropna(subset=FEATURE_COLS).reset_index(drop=True)

    # Generate signals
    full_test_signals = generate_signals(
        full_test, model,
        buy_threshold=buy_threshold,
        min_agree=MIN_AGREE,
    )

    n_signals = (full_test_signals["signal"] != "NO TRADE").sum()
    n_long_sig = (full_test_signals["signal"] == "LONG").sum()
    n_short_sig = (full_test_signals["signal"] == "SHORT").sum()
    print(f"  Signals in test period: {n_signals} "
          f"({n_long_sig} LONG, {n_short_sig} SHORT)")

    # Count signals per strategy
    if "strategy_name" in full_test_signals.columns:
        strategy_counts = full_test_signals[full_test_signals["signal"] != "NO TRADE"]["strategy_name"].value_counts()
        print(f"  Strategy breakdown:")
        for strat, count in strategy_counts.items():
            print(f"    {strat}: {count}")

    # Run backtest
    print(f"\n[7/8] Running backtest...")
    bt = run_backtest(full_test_signals)
    print_performance_report(bt["metrics"], title=f"BACKTEST RESULTS — {symbol}")

    # Print trades summary
    if len(bt["trades"]) > 0:
        trades = bt["trades"]
        print(f"\n  Trades by exit reason:")
        for reason, count in trades["exit_reason"].value_counts().items():
            print(f"    {reason}: {count}")

    # Plot equity
    plot_equity_curve(
        bt["equity_curve"],
        bt["equity_timestamps"],
        INITIAL_CAPITAL,
        filename=f"equity_{symbol.replace('/', '_')}.png",
    )

    # 8. Current signal
    print(f"\n[8/8] Current signal...")
    latest_df = add_features(raw_df, symbol=symbol, btc_df=btc_df)
    latest_df = latest_df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    latest_signals = generate_signals(
        latest_df, model,
        buy_threshold=buy_threshold,
        min_agree=MIN_AGREE,
    )
    print_current_signals(latest_signals, symbol)

    current_payload = _build_signal_payload(latest_signals, symbol)

    # Summary
    m = bt["metrics"]
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY — {symbol}")
    print(f"{'=' * 70}")
    print(f"  Val AUC:       {val_auc:.4f}")
    print(f"  Sharpe:        {m['sharpe_ratio']:.2f}")
    print(f"  Return:        {m['total_return']:.2%}")
    print(f"  Win rate:      {m['win_rate']:.2%}")
    print(f"  Profit factor: {m['profit_factor']:.2f}")
    print(f"  Max DD:        {m['max_drawdown']:.2%}")
    print(f"  Trades:        {m['total_trades']} ({m['long_trades']}L / {m['short_trades']}S)")
    print(f"  Expectancy:    {m['expectancy']:.4%}")
    print(f"{'=' * 70}")

    return {
        "metrics": bt["metrics"],
        "trades": bt["trades"],
        "val_auc": val_auc,
        "current_signal": current_payload,
    }


def _build_signal_payload(df: pd.DataFrame, symbol: str) -> dict | None:
    """Build a structured signal dict for machine consumption."""
    last = df.iloc[-1]
    signal = last.get("signal", "NO TRADE")

    if signal not in ["LONG", "SHORT"]:
        return None

    prob = float(last.get("probability", 0.0))
    n_agree = int(last.get("n_agree", 0))
    strategy_name = last.get("strategy_name", "UNKNOWN")
    rsi = float(last.get("rsi_14", 0))
    adx = float(last.get("adx", 0))
    rel_vol = float(last.get("relative_volume", 0))

    extra = (
        f"{strategy_name} agree={n_agree}/3 "
        f"RSI={rsi:.1f} ADX={adx:.1f} Vol={rel_vol:.2f}x"
    )

    return {
        "symbol": symbol,
        "signal": signal,
        "prob": prob,
        "extra": extra,
    }


def _write_signals_json(path: str, signals: list, strategy: str):
    """Write signals in a standard JSON format."""
    payload = {
        "strategy": strategy,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "signals": signals,
    }
    with open(path, "w") as f:
        json.dump(payload, f, ensure_ascii=True)


# ── Multi-crypto scan mode ────────────────────────────────────


def main_scan(
    symbols: list = None,
    timeframe: str = TIMEFRAME,
    days: int = DAYS,
    optimize: bool = False,
    buy_threshold: float = BUY_THRESHOLD,
    signals_json: str | None = None,
):
    """Multi-crypto pipeline: train, scan, backtest."""

    if symbols is None:
        symbols = DEFAULT_SYMBOLS

    print("=" * 70)
    print("  BREAKOUT MOMENTUM — MULTI-CRYPTO SCANNER")
    print("=" * 70)
    print(f"  Mode:          Scanner")
    print(f"  Symbols:       {len(symbols)}")
    print(f"  Timeframe:     {timeframe}")
    print(f"  History:       {days} days")
    print(f"  Breakout:      {BREAKOUT_PERIOD}-period Donchian")
    print(f"  TP / SL:       {TAKE_PROFIT_PCT:.1%} / {STOP_LOSS_ATR_MULT}xATR")
    print(f"  Fees:          {FEE_PCT:.2%} per side")
    print(f"  Strategy 1:    Volatility Breakout (prob > {VOL_PROBABILITY_THRESHOLD})")
    print(f"  Strategy 2:    Volume Surge (prob > {SURGE_PROBABILITY_THRESHOLD})")
    print(f"  Strategy 3:    Range Compression (prob > {COMP_PROBABILITY_THRESHOLD})")
    print(f"  Optimize:      {'Yes' if optimize else 'No'}")
    print("=" * 70)

    # Load BTC once for all symbols
    print(f"\n[Pre] Loading BTC for market context...")
    btc_df = load_btc_for_context(timeframe, days)

    # 1. Train models for all symbols
    print(f"\n[1/3] Training models per symbol...")
    trained = {}
    signals_by_symbol = {}

    for i, symbol in enumerate(symbols, 1):
        print(f"\n{'─' * 60}")
        print(f"[{i}/{len(symbols)}] {symbol}")
        print('─' * 60)

        try:
            raw_df = load_all_data(symbol, timeframe, days)

            # Use BTC context for non-BTC symbols
            symbol_btc = btc_df if symbol not in ["BTC/USDT", "BTCUSDT"] else None

            full_df = prepare_dataset(raw_df, symbol=symbol, btc_df=symbol_btc)
            train_data = get_training_data(full_df)

            n_tradeable = len(train_data)
            print(f"  Tradeable rows (breakouts): {n_tradeable}")

            if n_tradeable < 100:
                print(f"  Insufficient data ({n_tradeable} rows), skipping.")
                continue

            train, val, test = temporal_split_3way(train_data)

            # Optimize or use defaults
            params = None
            if optimize:
                print(f"  Optimizing ensemble for {symbol}...")
                params = optimize_ensemble(train, n_trials=20)

            # Train
            model = EnsembleModel()
            model.fit(train, params=params)

            # Validation AUC
            X_val = val[FEATURE_COLS].values
            y_val = val["target"].values.astype(int)
            val_proba = model.predict_proba(X_val)
            from sklearn.metrics import roc_auc_score
            val_auc = roc_auc_score(y_val, val_proba)
            print(f"  Ensemble val AUC: {val_auc:.4f}")

            # Full test dataframe for backtest
            test_start = test["timestamp"].iloc[0]
            test_end = test["timestamp"].iloc[-1]
            full_test = full_df[
                (full_df["timestamp"] >= test_start) &
                (full_df["timestamp"] <= test_end)
            ].copy()
            full_test = full_test.dropna(subset=FEATURE_COLS).reset_index(drop=True)

            # Generate signals on test set
            full_test_signals = generate_signals(
                full_test, model,
                buy_threshold=buy_threshold,
                min_agree=MIN_AGREE,
            )

            n_sig = (full_test_signals["signal"] != "NO TRADE").sum()
            print(f"  Signals in test: {n_sig}")

            trained[symbol] = {
                "model": model,
                "val_auc": val_auc,
                "raw_df": raw_df,
                "full_df": full_df,
                "test_df": full_test_signals,
            }
            signals_by_symbol[symbol] = full_test_signals

        except Exception as e:
            print(f"  ERROR in {symbol}: {e}")
            import traceback
            traceback.print_exc()

    if not trained:
        print("No models trained successfully.")
        return None

    print(f"\n{'=' * 60}")
    print(f"Models trained: {len(trained)}/{len(symbols)}")
    avg_auc = np.mean([d["val_auc"] for d in trained.values()])
    print(f"Average val AUC: {avg_auc:.4f}")
    print('=' * 60)

    # 2. Current signals
    print(f"\n[2/3] Scanning current signals...")
    print(f"\n{'=' * 80}")
    print(f"  CURRENT SIGNALS — Breakout Momentum")
    print(f"{'=' * 80}")

    signals_payload = []

    for symbol, data in trained.items():
        symbol_btc = btc_df if symbol not in ["BTC/USDT", "BTCUSDT"] else None
        latest_df = add_features(data["raw_df"], symbol=symbol, btc_df=symbol_btc)
        latest_df = latest_df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
        latest_signals = generate_signals(
            latest_df, data["model"],
            buy_threshold=buy_threshold,
            min_agree=MIN_AGREE,
        )
        print_current_signals(latest_signals, symbol)
        sig = _build_signal_payload(latest_signals, symbol)
        if sig:
            signals_payload.append(sig)

    print(f"{'=' * 80}")

    # 3. Multi-crypto backtest
    print(f"\n[3/3] Multi-crypto backtest...")
    bt = run_multi_backtest(trained, signals_by_symbol)

    print_performance_report(bt["metrics"], title="MULTI-CRYPTO BACKTEST")

    # Trades summary
    if len(bt["trades"]) > 0:
        trades = bt["trades"]
        print(f"\n  Trades by symbol:")
        for sym, count in trades["symbol"].value_counts().items():
            print(f"    {sym}: {count}")

        print(f"\n  Trades by exit reason:")
        for reason, count in trades["exit_reason"].value_counts().items():
            print(f"    {reason}: {count}")

    plot_equity_curve(
        bt["equity_curve"],
        bt["equity_timestamps"],
        INITIAL_CAPITAL,
        filename="equity_multi.png",
    )

    # Summary
    m = bt["metrics"]
    print(f"\n{'=' * 70}")
    print(f"  FINAL SUMMARY — SCANNER")
    print(f"{'=' * 70}")
    print(f"  Symbols trained: {len(trained)}")
    print(f"  Avg val AUC:     {avg_auc:.4f}")
    print(f"  Sharpe:          {m['sharpe_ratio']:.2f}")
    print(f"  Return:          {m['total_return']:.2%}")
    print(f"  Win rate:        {m['win_rate']:.2%}")
    print(f"  Profit factor:   {m['profit_factor']:.2f}")
    print(f"  Max DD:          {m['max_drawdown']:.2%}")
    print(f"  Trades:          {m['total_trades']} ({m['long_trades']}L / {m['short_trades']}S)")
    print(f"  Expectancy:      {m['expectancy']:.4%}")
    print(f"{'=' * 70}")

    if signals_json:
        _write_signals_json(signals_json, signals_payload, STRATEGY_NAME)

    return {"trained": trained, "backtest": bt}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Breakout Momentum Trading System"
    )
    parser.add_argument("--symbol", default="BNB/USDT",
                        help="Trading pair (default: BNB/USDT)")
    parser.add_argument("--timeframe", default=TIMEFRAME,
                        help=f"Candle timeframe (default: {TIMEFRAME})")
    parser.add_argument("--days", type=int, default=DAYS,
                        help=f"Days of history (default: {DAYS})")
    parser.add_argument("--optimize", action="store_true",
                        help="Enable Optuna hyperparameter optimization")
    parser.add_argument("--scan", action="store_true",
                        help="Multi-crypto scanner mode")
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="Symbols for scan mode")
    parser.add_argument("--buy-threshold", type=float, default=BUY_THRESHOLD,
                        help=f"Min probability for signal (default: {BUY_THRESHOLD})")
    parser.add_argument("--signals-json", default=None,
                        help="Write current signals to JSON file")

    args = parser.parse_args()

    if args.scan:
        main_scan(
            symbols=args.symbols or DEFAULT_SYMBOLS,
            timeframe=args.timeframe,
            days=args.days,
            optimize=args.optimize,
            buy_threshold=args.buy_threshold,
            signals_json=args.signals_json,
        )
    else:
        result = main_single(
            symbol=args.symbol,
            timeframe=args.timeframe,
            days=args.days,
            optimize=args.optimize,
            buy_threshold=args.buy_threshold,
        )

        if args.signals_json and result:
            payload = []
            current = result.get("current_signal")
            if current:
                payload.append(current)
            _write_signals_json(args.signals_json, payload, STRATEGY_NAME)
