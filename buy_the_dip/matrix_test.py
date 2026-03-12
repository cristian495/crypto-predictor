"""
matrix_test.py
Run a parameter matrix for Buy-the-Dip on one symbol and rank configurations.

Matrix:
- threshold in [-0.05, -0.06, -0.07, -0.08]
- RSI filter on/off
- Volume filter on/off
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "..")

from breakout_momentum.data_loader import fetch_ohlcv
from strategy import add_dip_features, generate_signals
from backtest import run_backtest
from config import (
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
)


def _parse_thresholds(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _walk_forward_for_config(
    df_features: pd.DataFrame,
    dip_threshold: float,
    use_volume_filter: bool,
    use_rsi_filter: bool,
    train_months: int,
    test_months: int,
    step_months: int,
) -> dict:
    warmup_size = int(train_months * 30 * 24)
    test_size = int(test_months * 30 * 24)
    step_size = int(step_months * 30 * 24)

    start_idx = warmup_size
    windows_total = 0
    windows_with_trades = 0
    positive_return_windows = 0
    sharpe_positive_windows = 0
    pf_above_1_windows = 0

    returns = []
    sharpes = []
    pfs = []
    mdds = []
    trades_total = 0

    while True:
        test_end_idx = start_idx + test_size
        if test_end_idx > len(df_features):
            break

        windows_total += 1
        test_data = df_features.iloc[start_idx:test_end_idx].copy()
        if len(test_data) < 100:
            start_idx += step_size
            continue

        test_with_signals = generate_signals(
            test_data,
            dip_threshold=dip_threshold,
            use_volume_filter=use_volume_filter,
            use_rsi_filter=use_rsi_filter,
        )

        bt = run_backtest(
            test_with_signals,
            initial_capital=INITIAL_CAPITAL,
            stop_loss_pct=STOP_LOSS_PCT,
            take_profit_pct=TAKE_PROFIT_PCT,
            hold_timeout=HOLD_TIMEOUT_HOURS,
            fee_pct=FEE_PCT,
        )
        m = bt["metrics"]
        trades = int(m.get("total_trades", 0))

        if trades <= 0:
            start_idx += step_size
            continue

        windows_with_trades += 1
        trades_total += trades

        total_return = float(m.get("total_return", 0.0))
        sharpe = float(m.get("sharpe", 0.0))
        pf = float(m.get("profit_factor", 0.0))
        max_dd = float(m.get("max_drawdown", 0.0))

        returns.append(total_return)
        sharpes.append(sharpe)
        pfs.append(pf)
        mdds.append(max_dd)

        if total_return > 0:
            positive_return_windows += 1
        if sharpe > 0:
            sharpe_positive_windows += 1
        if pf > 1:
            pf_above_1_windows += 1

        start_idx += step_size

    if windows_with_trades == 0:
        return {
            "wf_windows_total": windows_total,
            "wf_windows_with_trades": 0,
            "wf_positive_pct": 0.0,
            "wf_sharpe_positive_pct": 0.0,
            "wf_pf_above_1_pct": 0.0,
            "wf_avg_return_pct": 0.0,
            "wf_avg_sharpe": 0.0,
            "wf_avg_pf": 0.0,
            "wf_avg_max_dd_pct": 0.0,
            "wf_total_trades": 0,
        }

    denom = windows_with_trades
    return {
        "wf_windows_total": windows_total,
        "wf_windows_with_trades": windows_with_trades,
        "wf_positive_pct": round(positive_return_windows / denom * 100, 2),
        "wf_sharpe_positive_pct": round(sharpe_positive_windows / denom * 100, 2),
        "wf_pf_above_1_pct": round(pf_above_1_windows / denom * 100, 2),
        "wf_avg_return_pct": round(sum(returns) / denom * 100, 3),
        "wf_avg_sharpe": round(sum(sharpes) / denom, 3),
        "wf_avg_pf": round(sum(pfs) / denom, 3),
        "wf_avg_max_dd_pct": round(sum(mdds) / denom * 100, 3),
        "wf_total_trades": trades_total,
    }


def _score_row(row: dict) -> float:
    """Composite score for ranking. Higher is better."""
    if row["total_trades"] < 50 or row["wf_windows_with_trades"] < 3:
        return -1.0

    wf_pos = row["wf_positive_pct"] / 100.0
    wf_pf = min(max((row["wf_avg_pf"] - 1.0) / 1.0, 0.0), 1.0)
    sharpe = min(max(row["sharpe"] / 2.0, 0.0), 1.0)
    drawdown = max(0.0, 1.0 - abs(row["max_drawdown"]) / 0.50)
    trades = min(row["total_trades"] / 600.0, 1.0)

    score = (
        wf_pos * 0.40
        + wf_pf * 0.20
        + sharpe * 0.20
        + drawdown * 0.10
        + trades * 0.10
    )
    return round(score * 100, 2)


def run_matrix(
    symbol: str,
    days: int,
    thresholds: list[float],
    train_months: int,
    test_months: int,
    step_months: int,
) -> pd.DataFrame:
    print(f"\n{'='*72}")
    print("  BUY-THE-DIP MATRIX TEST")
    print(f"{'='*72}")
    print(f"  Symbol:      {symbol}")
    print(f"  Timeframe:   {TIMEFRAME}")
    print(f"  History:     {days} days")
    print(f"  Thresholds:  {thresholds}")
    print(f"  Matrix size: {len(thresholds) * 2 * 2} configs")
    print(f"{'='*72}\n")

    df = fetch_ohlcv(symbol, timeframe=TIMEFRAME, days=days)
    df_features = add_dip_features(df)

    rows = []
    total_configs = len(thresholds) * 2 * 2
    idx = 0

    for threshold in thresholds:
        for use_rsi in (False, True):
            for use_vol in (False, True):
                idx += 1
                print(
                    f"[{idx:>2}/{total_configs}] "
                    f"th={threshold:.2f} rsi={'on' if use_rsi else 'off'} "
                    f"vol={'on' if use_vol else 'off'}"
                )

                with_signals = generate_signals(
                    df_features,
                    dip_threshold=threshold,
                    use_volume_filter=use_vol,
                    use_rsi_filter=use_rsi,
                )

                bt = run_backtest(
                    with_signals,
                    initial_capital=INITIAL_CAPITAL,
                    stop_loss_pct=STOP_LOSS_PCT,
                    take_profit_pct=TAKE_PROFIT_PCT,
                    hold_timeout=HOLD_TIMEOUT_HOURS,
                    fee_pct=FEE_PCT,
                )
                m = bt["metrics"]

                wf = _walk_forward_for_config(
                    df_features=df_features,
                    dip_threshold=threshold,
                    use_volume_filter=use_vol,
                    use_rsi_filter=use_rsi,
                    train_months=train_months,
                    test_months=test_months,
                    step_months=step_months,
                )

                row = {
                    "symbol": symbol,
                    "threshold": threshold,
                    "use_rsi_filter": use_rsi,
                    "use_volume_filter": use_vol,
                    "total_return_pct": round(float(m["total_return"]) * 100, 3),
                    "sharpe": float(m["sharpe"]),
                    "max_drawdown": float(m["max_drawdown"]),
                    "max_drawdown_pct": round(float(m["max_drawdown"]) * 100, 3),
                    "win_rate_pct": round(float(m["win_rate"]) * 100, 3),
                    "profit_factor": float(m["profit_factor"]),
                    "total_trades": int(m["total_trades"]),
                    "expectancy_pct": round(float(m["expectancy"]) * 100, 4),
                }
                row.update(wf)
                row["score"] = _score_row({
                    "total_trades": row["total_trades"],
                    "wf_windows_with_trades": row["wf_windows_with_trades"],
                    "wf_positive_pct": row["wf_positive_pct"],
                    "wf_avg_pf": row["wf_avg_pf"],
                    "sharpe": row["sharpe"],
                    "max_drawdown": row["max_drawdown"],
                })

                rows.append(row)

    results = pd.DataFrame(rows)
    results = results.sort_values(
        by=["score", "wf_positive_pct", "sharpe", "profit_factor", "total_trades"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)
    return results


def main():
    parser = argparse.ArgumentParser(description="Run Buy-the-Dip matrix test")
    parser.add_argument("--symbol", default="SUI/USDT", help="Trading pair")
    parser.add_argument("--days", type=int, default=DAYS, help="History days")
    parser.add_argument(
        "--thresholds",
        default="-0.05,-0.06,-0.07,-0.08",
        help="Comma-separated thresholds",
    )
    parser.add_argument("--wf-train-months", type=int, default=WF_TRAIN_MONTHS)
    parser.add_argument("--wf-test-months", type=int, default=WF_TEST_MONTHS)
    parser.add_argument("--wf-step-months", type=int, default=WF_STEP_MONTHS)
    parser.add_argument(
        "--out-csv",
        default="buy_the_dip/matrix_results.csv",
        help="Path to write CSV results",
    )
    parser.add_argument(
        "--out-json",
        default="buy_the_dip/matrix_results.json",
        help="Path to write JSON results",
    )
    args = parser.parse_args()

    thresholds = _parse_thresholds(args.thresholds)
    if not thresholds:
        raise ValueError("No thresholds provided")

    results = run_matrix(
        symbol=args.symbol,
        days=args.days,
        thresholds=thresholds,
        train_months=args.wf_train_months,
        test_months=args.wf_test_months,
        step_months=args.wf_step_months,
    )

    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    results.to_csv(out_csv, index=False)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results.to_dict(orient="records"), f, ensure_ascii=True, indent=2)

    print(f"\n{'='*72}")
    print("  TOP 5 CONFIGS")
    print(f"{'='*72}")
    top = results.head(5)
    cols = [
        "threshold",
        "use_rsi_filter",
        "use_volume_filter",
        "score",
        "total_return_pct",
        "sharpe",
        "profit_factor",
        "total_trades",
        "wf_positive_pct",
        "wf_windows_with_trades",
    ]
    print(top[cols].to_string(index=False))

    best = results.iloc[0].to_dict()
    print(f"\nRecommended config:")
    print(
        f"  threshold={best['threshold']:.2f}, "
        f"rsi={'on' if best['use_rsi_filter'] else 'off'}, "
        f"volume={'on' if best['use_volume_filter'] else 'off'}"
    )
    print(
        f"  score={best['score']}, return={best['total_return_pct']:.2f}%, "
        f"sharpe={best['sharpe']:.2f}, pf={best['profit_factor']:.2f}, "
        f"trades={best['total_trades']}, wf_positive={best['wf_positive_pct']:.1f}%"
    )
    print(f"\nSaved:")
    print(f"  - {out_csv}")
    print(f"  - {out_json}")


if __name__ == "__main__":
    main()
