"""
Sell-the-Rip strategy logic (simple, deterministic, short-only).
"""

import numpy as np
import pandas as pd

from config import (
    RIP_LOOKBACK_HOURS,
    RIP_THRESHOLD_PCT,
    RIP_THRESHOLD_STRONG,
    USE_STRONG_THRESHOLD,
    MIN_VOLUME_MULT,
    USE_VOLUME_FILTER,
    RSI_MIN,
    RSI_MAX,
    USE_RSI_FILTER,
    USE_TREND_FILTER,
    EMA_TREND_LEN,
    EMA_SLOPE_HOURS,
)


def add_rip_features(df: pd.DataFrame, lookback: int = RIP_LOOKBACK_HOURS) -> pd.DataFrame:
    df = df.copy()

    # 24h momentum used for rip detection
    df["past_return"] = df["close"].pct_change(lookback)

    # Relative volume
    vol_ma = df["volume"].rolling(20).mean()
    df["relative_volume"] = df["volume"] / vol_ma.replace(0, np.nan)

    # RSI(14)
    if "rsi_14" not in df.columns:
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df["rsi_14"] = 100 - (100 / (1 + rs))

    # Simple trend filter: bearish regime
    df["ema200"] = df["close"].ewm(span=EMA_TREND_LEN, adjust=False).mean()
    df["ema200_slope_24h"] = df["ema200"].pct_change(EMA_SLOPE_HOURS)
    df["trend_bearish"] = (df["close"] < df["ema200"]) & (df["ema200_slope_24h"] <= 0)

    return df


def generate_signals(
    df: pd.DataFrame,
    rip_threshold: float = None,
    use_volume_filter: bool = USE_VOLUME_FILTER,
    use_rsi_filter: bool = USE_RSI_FILTER,
    use_trend_filter: bool = USE_TREND_FILTER,
    min_volume_mult: float = MIN_VOLUME_MULT,
    rsi_min: float = RSI_MIN,
    rsi_max: float = RSI_MAX,
) -> pd.DataFrame:
    """Generate SHORT signals after strong 24h pumps."""
    out = df.copy()

    if rip_threshold is None:
        rip_threshold = RIP_THRESHOLD_STRONG if USE_STRONG_THRESHOLD else RIP_THRESHOLD_PCT

    if "past_return" not in out.columns:
        out = add_rip_features(out)

    out["signal"] = "NO TRADE"
    out["signal_direction"] = 0
    out["rip_size"] = 0.0

    for idx in out.index:
        row = out.loc[idx]
        past_ret = row.get("past_return", np.nan)
        rel_vol = row.get("relative_volume", np.nan)
        rsi = row.get("rsi_14", np.nan)
        trend_bearish = bool(row.get("trend_bearish", False))

        if pd.isna(past_ret):
            continue

        # Core rule: strong positive move in last 24h
        if float(past_ret) > float(rip_threshold):
            if use_volume_filter and (pd.isna(rel_vol) or float(rel_vol) < float(min_volume_mult)):
                continue

            if use_rsi_filter and (pd.isna(rsi) or float(rsi) < float(rsi_min) or float(rsi) > float(rsi_max)):
                continue

            if use_trend_filter and (not trend_bearish):
                continue

            out.loc[idx, "signal"] = "SHORT"
            out.loc[idx, "signal_direction"] = -1
            out.loc[idx, "rip_size"] = float(past_ret)

    return out


def compute_exit_levels(
    entry_price: float,
    take_profit_pct: float,
    stop_loss_pct: float,
) -> dict:
    """Compute short exit levels."""
    return {
        "stop_loss": entry_price * (1 + stop_loss_pct),
        "take_profit": entry_price * (1 - take_profit_pct),
    }
