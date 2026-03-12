"""
main.py
CLI orchestrator for the mean reversion trading system.

Usage:
    python main.py                                  # Single: BTC/USDT
    python main.py --symbol ETH/USDT                # Single: specific pair
    python main.py --symbol BTC/USDT --optimize     # Single + Optuna
    python main.py --scan                           # Multi-crypto scan
    python main.py --scan --optimize                # Multi + Optuna
    python main.py --scan --symbols BTC/USDT PAXG/USDT
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
    ZSCORE_ENTRY_THRESHOLD,
    REVERSION_TARGET_PCT,
    REVERSION_HORIZON,
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
    HOLD_TIMEOUT,
    POSITION_PCT,
    MAX_POSITIONS,
    FEE_PCT,
    INITIAL_CAPITAL,
    BUY_THRESHOLD,
    MIN_AGREE,
)
from data_loader import load_all_data
from features import add_features, FEATURE_COLS
from target import prepare_dataset, get_training_data
from model import EnsembleModel, temporal_split_3way, optimize_ensemble
from strategy import generate_signals, print_current_signals
from backtest import (
    run_backtest,
    run_multi_backtest,
    plot_equity,
    print_trades_summary,
)
from metrics import print_performance_report

STRATEGY_NAME = Path(__file__).resolve().parent.name


# ── Single symbol mode ───────────────────────────────────────


def main_single(
    symbol: str = "BTC/USDT",
    timeframe: str = TIMEFRAME,
    days: int = DAYS,
    optimize: bool = False,
    buy_threshold: float = BUY_THRESHOLD,
):
    """Full pipeline for a single symbol."""

    print("=" * 60)
    print("  MEAN REVERSION TRADING SYSTEM")
    print("=" * 60)
    print(f"  Mode:          Single")
    print(f"  Symbol:        {symbol}")
    print(f"  Timeframe:     {timeframe}")
    print(f"  History:       {days} days")
    print(f"  Z-score entry: |Z| > {ZSCORE_ENTRY_THRESHOLD}")
    print(f"  Target:        {REVERSION_TARGET_PCT:.0%} reversion in {REVERSION_HORIZON} candles")
    print(f"  TP / SL:       +{TAKE_PROFIT_PCT:.1%} / -{STOP_LOSS_PCT:.1%} (RR 1:{TAKE_PROFIT_PCT/STOP_LOSS_PCT:.0f})")
    print(f"  Fees:          {FEE_PCT:.2%} per side")
    print(f"  Signal SHORT:  prob > {buy_threshold - 0.05} + {MIN_AGREE}/3 votes")
    print(f"  Signal LONG:   prob > {buy_threshold + 0.05} + {MIN_AGREE}/3 votes")
    print(f"  Features:      {len(FEATURE_COLS)}")
    print(f"  Optimize:      {'Yes' if optimize else 'No'}")
    print("=" * 60)

    # 1. Download data
    print(f"\n[1/7] Downloading data...")
    raw_df = load_all_data(symbol, timeframe, days)

    # 2. Feature engineering + target
    print(f"\n[2/7] Computing features and mean reversion target...")
    full_df = prepare_dataset(raw_df)
    train_data = get_training_data(full_df)
    n_tradeable = len(train_data)
    n_total = len(full_df)
    print(f"  Total rows: {n_total}")
    print(f"  Tradeable rows (|Z|>{ZSCORE_ENTRY_THRESHOLD}): {n_tradeable} ({n_tradeable/n_total:.1%})")
    if n_tradeable > 0:
        revert_pct = train_data["target"].mean()
        print(f"  Reversion rate: {revert_pct:.1%}")
        n_long = (train_data["direction"] == 1).sum()
        n_short = (train_data["direction"] == -1).sum()
        print(f"  Long / Short opportunities: {n_long} / {n_short}")

    if n_tradeable < 100:
        print(f"\n  WARNING: Only {n_tradeable} tradeable rows. Need at least 100.")
        print("  Try increasing --days or lowering Z-score threshold in config.py")
        return None

    # 3. Temporal split
    print(f"\n[3/7] Temporal split (70/15/15)...")
    train, val, test = temporal_split_3way(train_data)

    # 4. Optimize (optional)
    params = None
    if optimize:
        print(f"\n[4/7] Optimizing hyperparameters with Optuna...")
        params = optimize_ensemble(train)
    else:
        print(f"\n[4/7] Skipping optimization (use --optimize to enable)")

    # 5. Train ensemble
    print(f"\n[5/7] Training ensemble model...")
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
    print(f"\n[6/7] Backtesting on test period...")

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

    # Run backtest
    bt = run_backtest(full_test_signals)
    print_performance_report(bt["metrics"], title="BACKTEST RESULTS")
    print_trades_summary(bt["trades"])

    # Plot equity
    plot_equity(
        bt["equity_curve"],
        bt["equity_timestamps"],
        title=f"Equity Curve — {symbol} Mean Reversion",
        filename="equity_curve.png",
    )

    # 7. Current signal
    print(f"\n[7/7] Current signal...")
    latest_df = add_features(raw_df)
    latest_df = latest_df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    latest_signals = generate_signals(
        latest_df, model,
        buy_threshold=buy_threshold,
        min_agree=MIN_AGREE,
    )
    print_current_signals(latest_signals, symbol, verbose=True)

    current_payload = _build_signal_payload(latest_signals, symbol)

    # Summary
    m = bt["metrics"]
    print(f"\n{'=' * 60}")
    print(f"  SUMMARY — {symbol}")
    print(f"{'=' * 60}")
    print(f"  Val AUC:       {val_auc:.4f}")
    print(f"  Sharpe:        {m['sharpe_ratio']:.2f}")
    print(f"  Return:        {m['total_return']:.2%}")
    print(f"  Win rate:      {m['win_rate']:.2%}")
    print(f"  Profit factor: {m['profit_factor']:.2f}")
    print(f"  Max DD:        {m['max_drawdown']:.2%}")
    print(f"  Trades:        {m['total_trades']}")
    print(f"  Expectancy:    {m['expectancy']:.4%}")
    print(f"{'=' * 60}")

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
    zscore = float(last.get("zscore_50", 0))

    extra = f"agree={n_agree}/3 Z={zscore:.2f}"

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

    print("=" * 60)
    print("  MEAN REVERSION — MULTI-CRYPTO SCANNER")
    print("=" * 60)
    print(f"  Mode:          Scanner")
    print(f"  Symbols:       {len(symbols)}")
    print(f"  Timeframe:     {timeframe}")
    print(f"  History:       {days} days")
    print(f"  Z-score entry: |Z| > {ZSCORE_ENTRY_THRESHOLD}")
    print(f"  TP / SL:       +{TAKE_PROFIT_PCT:.1%} / -{STOP_LOSS_PCT:.1%}")
    print(f"  Fees:          {FEE_PCT:.2%} per side")
    print(f"  Signal SHORT:  prob > {buy_threshold - 0.05} + {MIN_AGREE}/3 votes")
    print(f"  Signal LONG:   prob > {buy_threshold + 0.05} + {MIN_AGREE}/3 votes")
    print(f"  Optimize:      {'Yes' if optimize else 'No'}")
    print("=" * 60)

    # 1. Train models for all symbols
    print(f"\n[1/3] Training models per symbol...")
    trained = {}
    signals_by_symbol = {}

    for i, symbol in enumerate(symbols, 1):
        print(f"\n{'─' * 50}")
        print(f"[{i}/{len(symbols)}] {symbol}")
        print('─' * 50)

        try:
            raw_df = load_all_data(symbol, timeframe, days)
            full_df = prepare_dataset(raw_df)
            train_data = get_training_data(full_df)

            n_tradeable = len(train_data)
            print(f"  Tradeable rows: {n_tradeable}")

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

    print(f"\n{'=' * 50}")
    print(f"Models trained: {len(trained)}/{len(symbols)}")
    avg_auc = np.mean([d["val_auc"] for d in trained.values()])
    print(f"Average val AUC: {avg_auc:.4f}")
    print('=' * 50)

    # 2. Current signals
    print(f"\n[2/3] Scanning current signals...")
    print(f"\n{'=' * 70}")
    print(f"  CURRENT SIGNALS — Mean Reversion")
    print(f"{'=' * 70}")

    signals_payload = []

    for symbol, data in trained.items():
        latest_df = add_features(data["raw_df"])
        latest_df = latest_df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
        latest_signals = generate_signals(
            latest_df, data["model"],
            buy_threshold=buy_threshold,
            min_agree=MIN_AGREE,
        )
        print_current_signals(latest_signals, symbol, verbose=True)
        sig = _build_signal_payload(latest_signals, symbol)
        if sig:
            signals_payload.append(sig)

    print(f"{'=' * 70}")

    # 3. Multi-crypto backtest
    print(f"\n[3/3] Multi-crypto backtest...")
    bt = run_multi_backtest(trained, signals_by_symbol)

    print_performance_report(bt["metrics"], title="MULTI-CRYPTO BACKTEST")
    print_trades_summary(bt["trades"])

    plot_equity(
        bt["equity_curve"],
        bt["equity_timestamps"],
        title="Equity Curve — Multi-Crypto Mean Reversion",
        filename="equity_multi.png",
    )

    # Summary
    m = bt["metrics"]
    print(f"\n{'=' * 60}")
    print(f"  FINAL SUMMARY — SCANNER")
    print(f"{'=' * 60}")
    print(f"  Symbols trained: {len(trained)}")
    print(f"  Avg val AUC:     {avg_auc:.4f}")
    print(f"  Sharpe:          {m['sharpe_ratio']:.2f}")
    print(f"  Return:          {m['total_return']:.2%}")
    print(f"  Win rate:        {m['win_rate']:.2%}")
    print(f"  Profit factor:   {m['profit_factor']:.2f}")
    print(f"  Max DD:          {m['max_drawdown']:.2%}")
    print(f"  Trades:          {m['total_trades']} ({m['long_trades']}L / {m['short_trades']}S)")
    print(f"  Expectancy:      {m['expectancy']:.4%}")
    print(f"{'=' * 60}")

    if signals_json:
        _write_signals_json(signals_json, signals_payload, STRATEGY_NAME)

    return {"trained": trained, "backtest": bt}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mean Reversion Trading System"
    )
    parser.add_argument("--symbol", default="BTC/USDT",
                        help="Trading pair (default: BTC/USDT)")
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
