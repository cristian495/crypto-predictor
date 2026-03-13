"""
Signal and feature logic for downtrend_breakout v1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    ADX_PERIOD,
    ATR_PERCENTILE_WINDOW,
    ATR_PERIOD,
    BASE_SYMBOL_PARAMS,
    DONCHIAN_WINDOW,
    EMA_FAST,
    EMA_SLOW,
    EMA_SLOPE_BARS,
    MODE_DEFAULT,
    RELATIVE_VOLUME_WINDOW,
    RET_6H_BARS,
    RET_24H_BARS,
)

RET_7D_BARS = 24 * 7


def _compute_atr_adx(df: pd.DataFrame, atr_period: int = ATR_PERIOD, adx_period: int = ADX_PERIOD) -> tuple[pd.Series, pd.Series]:
    """Compute ATR(14) and ADX(14) with Wilder smoothing."""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.ewm(alpha=1 / atr_period, adjust=False).mean()

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)

    plus_di = 100 * (plus_dm.ewm(alpha=1 / adx_period, adjust=False).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.ewm(alpha=1 / adx_period, adjust=False).mean() / atr.replace(0, np.nan))

    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0.0)
    adx = dx.ewm(alpha=1 / adx_period, adjust=False).mean()

    return atr, adx


def _rolling_percentile_rank(series: pd.Series, window: int) -> pd.Series:
    """Rolling percentile rank of the latest value in each window (0-100)."""
    if window <= 1:
        return pd.Series(50.0, index=series.index)

    return series.rolling(window, min_periods=window).apply(
        lambda arr: float((arr <= arr[-1]).mean() * 100.0),
        raw=True,
    )


def add_features(
    df: pd.DataFrame,
    donchian_window: int = DONCHIAN_WINDOW,
    btc_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add deterministic features needed by the strategy."""
    out = df.copy()

    out["ret_6h"] = out["close"].pct_change(RET_6H_BARS)
    out["ret_24h"] = out["close"].pct_change(RET_24H_BARS)

    out["ema50"] = out["close"].ewm(span=EMA_FAST, adjust=False).mean()
    out["ema200"] = out["close"].ewm(span=EMA_SLOW, adjust=False).mean()
    out["ema200_slope_24h"] = out["ema200"].pct_change(EMA_SLOPE_BARS)

    # Shift(1) to avoid lookahead bias on breakout checks.
    out["donchian_high"] = out["high"].rolling(donchian_window).max().shift(1)
    out["donchian_low"] = out["low"].rolling(donchian_window).min().shift(1)

    atr, adx = _compute_atr_adx(out)
    out["atr14"] = atr
    out["adx14"] = adx

    out["atr_pct_rank_90d"] = _rolling_percentile_rank(out["atr14"], ATR_PERCENTILE_WINDOW)

    vol_ma = out["volume"].rolling(RELATIVE_VOLUME_WINDOW).mean()
    out["relative_volume_20"] = out["volume"] / vol_ma.replace(0, np.nan)

    out["regime_bullish"] = (out["close"] > out["ema200"]) & (out["ema200_slope_24h"] > 0)
    out["regime_bearish"] = (out["close"] < out["ema200"]) & (out["ema200_slope_24h"] < 0)

    # BTC market context can be injected to avoid shorting altcoins against
    # broad risk-on conditions.
    if btc_df is not None and not btc_df.empty:
        btc_ctx = btc_df[["timestamp", "close"]].copy()
        btc_ctx = btc_ctx.sort_values("timestamp").rename(columns={"close": "btc_close"})

        out = out.sort_values("timestamp")
        out = pd.merge_asof(out, btc_ctx, on="timestamp", direction="backward")

        out["btc_ret_24h"] = out["btc_close"].pct_change(RET_24H_BARS)
        out["btc_ret_7d"] = out["btc_close"].pct_change(RET_7D_BARS)
        out["btc_ema200"] = out["btc_close"].ewm(span=EMA_SLOW, adjust=False).mean()
        out["btc_ema200_slope_24h"] = out["btc_ema200"].pct_change(EMA_SLOPE_BARS)
        out["btc_regime_bearish"] = (out["btc_close"] < out["btc_ema200"]) & (out["btc_ema200_slope_24h"] < 0)
        out["btc_regime_bullish"] = (out["btc_close"] > out["btc_ema200"]) & (out["btc_ema200_slope_24h"] > 0)
    else:
        out["btc_close"] = np.nan
        out["btc_ret_24h"] = np.nan
        out["btc_ret_7d"] = np.nan
        out["btc_ema200"] = np.nan
        out["btc_ema200_slope_24h"] = np.nan
        out["btc_regime_bearish"] = False
        out["btc_regime_bullish"] = False

    return out


def _score_to_confidence(score: float, base: float = 0.50) -> float:
    """Map normalized score into [base, 0.95]."""
    return float(round(base + max(0.0, min(0.95 - base, (score - 1.0) * 0.25)), 3))


def _signal_confidence(row: pd.Series, params: dict, direction: int) -> float:
    """Build a deterministic confidence score in [0.50, 0.95]."""
    adx_min = max(float(params["adx_min"]), 1e-9)

    if direction == -1:
        s1 = abs(float(row["ret_24h"])) / max(abs(float(params["short_ret24_threshold"])), 1e-9)
        s2 = abs(float(row["ret_6h"])) / max(abs(float(params["short_ret6_threshold"])), 1e-9)
    else:
        s1 = abs(float(row["ret_24h"])) / max(abs(float(params["long_ret24_min"])), 1e-9)
        s2 = abs(float(row["ret_6h"])) / max(abs(float(params["long_ret6_min"])), 1e-9)

    s3 = float(row["adx14"]) / adx_min

    score = (min(s1, 2.0) + min(s2, 2.0) + min(s3, 2.0)) / 3.0
    return _score_to_confidence(score, base=0.50)


def _sell_rip_confidence(row: pd.Series, params: dict) -> float:
    """Confidence model for sell-the-rip entries."""
    adx_min = max(float(params["adx_min"]), 1e-9)
    rip_ret6_min = max(float(params["sell_rip_ret6h_min"]), 1e-9)
    rip_ret24_cap = max(abs(float(params["sell_rip_ret24h_max"])), 1e-9)

    s1 = max(float(row["ret_6h"]), 0.0) / rip_ret6_min
    s2 = float(row["adx14"]) / adx_min
    # Lower 24h return (after a short-term rip) tends to be safer for fading.
    s3 = 1.0 + max(0.0, float(params["sell_rip_ret24h_max"]) - float(row["ret_24h"])) / rip_ret24_cap

    score = (min(s1, 2.0) + min(s2, 2.0) + min(s3, 2.0)) / 3.0
    return _score_to_confidence(score, base=0.52)


def generate_signals(df: pd.DataFrame, symbol_params: dict | None = None, mode: str = MODE_DEFAULT) -> pd.DataFrame:
    """Generate LONG/SHORT signals using rule-based momentum breakdown logic."""
    params = dict(BASE_SYMBOL_PARAMS)
    if symbol_params:
        params.update(symbol_params)

    out = df.copy()
    out["signal"] = "NO TRADE"
    out["signal_direction"] = 0
    out["probability"] = 0.0
    out["entry_reason"] = ""

    cooldown_bars = int(params["cooldown_bars"])
    min_probability = float(params.get("min_probability", 0.50))
    use_btc_bear_filter = bool(params.get("use_btc_bear_filter", False))
    use_btc_regime_switch = bool(params.get("use_btc_regime_switch", False))
    btc_ret24_max = float(params.get("btc_ret24_max", 0.0))
    btc_ret7d_max = float(params.get("btc_ret7d_max", 0.0))
    btc_ret7d_long_min = float(params.get("btc_ret7d_long_min", 0.0))
    enable_sell_the_rip = bool(params.get("enable_sell_the_rip", False))
    sell_rip_ret6h_min = float(params.get("sell_rip_ret6h_min", 0.02))
    sell_rip_ret24h_max = float(params.get("sell_rip_ret24h_max", 0.03))
    next_entry_i = 0

    for i, idx in enumerate(out.index):
        if i < next_entry_i:
            continue

        row = out.loc[idx]

        required = [
            "ret_6h",
            "ret_24h",
            "ema200",
            "ema200_slope_24h",
            "donchian_high",
            "donchian_low",
            "adx14",
            "atr14",
            "atr_pct_rank_90d",
            "relative_volume_20",
            "close",
        ]
        if any(pd.isna(row.get(col, np.nan)) for col in required):
            continue

        bearish = bool(row["regime_bearish"])
        bullish = bool(row["regime_bullish"])

        macro_short_ok = True
        macro_long_ok = True

        if use_btc_regime_switch:
            btc_ret24 = row.get("btc_ret_24h", np.nan)
            btc_ret7d = row.get("btc_ret_7d", np.nan)
            macro_short_ok = (
                pd.notna(btc_ret24)
                and pd.notna(btc_ret7d)
                and bool(row.get("btc_regime_bearish", False))
                and float(btc_ret24) <= btc_ret24_max
                and float(btc_ret7d) <= btc_ret7d_max
            )
            macro_long_ok = (
                pd.notna(btc_ret7d)
                and bool(row.get("btc_regime_bullish", False))
                and float(btc_ret7d) >= btc_ret7d_long_min
            )
        elif use_btc_bear_filter:
            btc_ret24 = row.get("btc_ret_24h", np.nan)
            macro_short_ok = (
                pd.notna(btc_ret24)
                and float(btc_ret24) <= btc_ret24_max
                and bool(row.get("btc_regime_bearish", False))
            )

        short_breakdown_ok = (
            bearish
            and row["close"] < row["donchian_low"]
            and row["ret_24h"] <= float(params["short_ret24_threshold"])
            and row["ret_6h"] <= float(params["short_ret6_threshold"])
            and row["adx14"] >= float(params["adx_min"])
            and row["atr_pct_rank_90d"] <= float(params["atr_pct_rank_max"])
            and row["relative_volume_20"] >= float(params["relative_volume_min"])
        )

        sell_rip_ok = False
        if enable_sell_the_rip and i > 0 and bearish:
            prev = out.iloc[i - 1]
            prev_above_ema50 = pd.notna(prev.get("close", np.nan)) and pd.notna(prev.get("ema50", np.nan)) and float(prev["close"]) > float(prev["ema50"])
            cross_back_below_ema50 = prev_above_ema50 and float(row["close"]) <= float(row["ema50"])
            prev_ret6h = prev.get("ret_6h", np.nan)
            ret6h_rip = max(
                float(row["ret_6h"]),
                float(prev_ret6h) if pd.notna(prev_ret6h) else float(row["ret_6h"]),
            )

            sell_rip_ok = (
                cross_back_below_ema50
                and ret6h_rip >= sell_rip_ret6h_min
                and float(row["ret_24h"]) <= sell_rip_ret24h_max
                and float(row["adx14"]) >= float(params["adx_min"])
                and float(row["atr_pct_rank_90d"]) <= float(params["atr_pct_rank_max"])
                and float(row["relative_volume_20"]) >= float(params["relative_volume_min"])
            )

        short_ok = macro_short_ok and (short_breakdown_ok or sell_rip_ok)

        long_ok = (
            bullish
            and row["close"] > row["donchian_high"]
            and row["ret_24h"] >= float(params["long_ret24_min"])
            and row["ret_6h"] >= float(params["long_ret6_min"])
            and row["adx14"] >= float(params["adx_min"])
            and row["atr_pct_rank_90d"] <= float(params["atr_pct_rank_max"])
            and row["relative_volume_20"] >= float(params["relative_volume_min"])
        )
        long_ok = long_ok and macro_long_ok

        if short_ok:
            if sell_rip_ok and not short_breakdown_ok:
                confidence = _sell_rip_confidence(row, params)
                entry_reason = "sell_the_rip"
            elif sell_rip_ok and short_breakdown_ok:
                confidence = max(_signal_confidence(row, params, direction=-1), _sell_rip_confidence(row, params))
                entry_reason = "breakdown_and_sell_the_rip"
            else:
                confidence = _signal_confidence(row, params, direction=-1)
                entry_reason = "momentum_breakdown"

            if confidence < min_probability:
                continue
            out.loc[idx, "signal"] = "SHORT"
            out.loc[idx, "signal_direction"] = -1
            out.loc[idx, "probability"] = confidence
            out.loc[idx, "entry_reason"] = entry_reason
            next_entry_i = i + cooldown_bars + 1
            continue

        if mode == "long_short" and long_ok:
            confidence = _signal_confidence(row, params, direction=1)
            if confidence < min_probability:
                continue
            out.loc[idx, "signal"] = "LONG"
            out.loc[idx, "signal_direction"] = 1
            out.loc[idx, "probability"] = confidence
            out.loc[idx, "entry_reason"] = "momentum_breakout"
            next_entry_i = i + cooldown_bars + 1

    return out
