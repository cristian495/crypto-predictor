"""
features.py
Technical indicators + mean reversion specific features.
No feature uses future information.
"""

import numpy as np
import pandas as pd


# ── Core indicator functions (from parent project) ──────────


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range."""
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.ewm(alpha=1 / period, min_periods=period).mean()


def compute_stochastic(df: pd.DataFrame, period: int = 14,
                       smooth_k: int = 3) -> tuple:
    """Stochastic Oscillator %K and %D."""
    lowest_low = df["low"].rolling(period).min()
    highest_high = df["high"].rolling(period).max()

    fast_k = 100 * (df["close"] - lowest_low) / (highest_high - lowest_low)
    slow_k = fast_k.rolling(smooth_k).mean()
    slow_d = slow_k.rolling(smooth_k).mean()

    return slow_k, slow_d


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average Directional Index (ADX).
    Measures trend strength (0-100).
    ADX < 20: weak/no trend (good for mean reversion)
    ADX 20-25: developing trend
    ADX > 25: strong trend (avoid mean reversion)
    ADX > 50: very strong trend
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    # Calculate +DM and -DM
    high_diff = high.diff()
    low_diff = -low.diff()

    plus_dm = pd.Series(0.0, index=df.index)
    minus_dm = pd.Series(0.0, index=df.index)

    plus_dm[(high_diff > low_diff) & (high_diff > 0)] = high_diff
    minus_dm[(low_diff > high_diff) & (low_diff > 0)] = low_diff

    # Calculate ATR
    atr = compute_atr(df, period)

    # Smooth DM
    plus_dm_smooth = plus_dm.ewm(alpha=1/period, min_periods=period).mean()
    minus_dm_smooth = minus_dm.ewm(alpha=1/period, min_periods=period).mean()

    # Calculate DI+ and DI-
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr

    # Calculate DX and ADX
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.ewm(alpha=1/period, min_periods=period).mean()

    return adx


def compute_hurst(series: pd.Series, window: int = 100) -> pd.Series:
    """
    Rolling Hurst exponent using rescaled range (R/S) method.
    H < 0.5 → mean reverting
    H = 0.5 → random walk
    H > 0.5 → trending
    """
    def _hurst_single(x):
        if len(x) < 20:
            return np.nan

        n = len(x)
        max_k = min(n // 2, 50)
        if max_k < 4:
            return np.nan

        rs_list = []
        ns_list = []

        for k in [max_k // 4, max_k // 2, max_k]:
            if k < 4:
                continue
            # Split into sub-series of length k
            n_subseries = n // k
            if n_subseries < 1:
                continue

            rs_vals = []
            for i in range(n_subseries):
                sub = x[i * k:(i + 1) * k]
                mean_sub = np.mean(sub)
                deviations = np.cumsum(sub - mean_sub)
                r = np.max(deviations) - np.min(deviations)
                s = np.std(sub, ddof=1)
                if s > 0:
                    rs_vals.append(r / s)

            if rs_vals:
                rs_list.append(np.log(np.mean(rs_vals)))
                ns_list.append(np.log(k))

        if len(rs_list) < 2:
            return np.nan

        # Linear regression: log(R/S) = H * log(n) + c
        coeffs = np.polyfit(ns_list, rs_list, 1)
        return np.clip(coeffs[0], 0, 1)

    returns = series.pct_change().dropna().values

    result = pd.Series(np.nan, index=series.index)
    for i in range(window, len(series)):
        result.iloc[i] = _hurst_single(returns[i - window:i])

    return result


# ── Feature engineering ─────────────────────────────────────


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all technical indicators and mean reversion features.
    Only uses past and present data (no data leakage).
    """
    df = df.copy()

    # ── Classic momentum / trend ─────────────────────────────

    # RSI
    df["rsi_14"] = compute_rsi(df["close"], 14)

    # EMAs
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema_diff"] = (df["ema_20"] - df["ema_50"]) / df["close"]

    # Log returns
    df["log_return_1"] = np.log(df["close"] / df["close"].shift(1))
    df["log_return_3"] = np.log(df["close"] / df["close"].shift(3))
    df["log_return_6"] = np.log(df["close"] / df["close"].shift(6))
    df["log_return_12"] = np.log(df["close"] / df["close"].shift(12))

    # Volatility
    df["volatility_10"] = df["log_return_1"].rolling(10).std()
    df["volatility_20"] = df["log_return_1"].rolling(20).std()

    # Relative volume
    df["relative_volume"] = df["volume"] / df["volume"].rolling(20).mean()

    # MACD (12, 26, 9) — normalized by price
    ema_12 = df["close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd_line"] = (ema_12 - ema_26) / df["close"]
    df["macd_signal"] = df["macd_line"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd_line"] - df["macd_signal"]

    # Bollinger Bands (20, 2)
    bb_mid = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    df["bb_percent_b"] = (df["close"] - bb_lower) / (bb_upper - bb_lower)
    df["bb_width"] = (bb_upper - bb_lower) / bb_mid

    # ATR (normalized)
    df["atr_14"] = compute_atr(df, 14) / df["close"]

    # Stochastic
    df["stoch_k"], df["stoch_d"] = compute_stochastic(df, 14, 3)

    # Candle patterns
    df["candle_range"] = (df["high"] - df["low"]) / df["close"]
    df["candle_body"] = (df["close"] - df["open"]) / df["close"]

    # Temporal encoding (cyclic)
    ts = pd.to_datetime(df["timestamp"])
    df["hour_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * ts.dt.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * ts.dt.dayofweek / 7)

    # ── Mean reversion specific ──────────────────────────────

    # Z-scores at multiple windows
    for w in [20, 50, 100]:
        sma = df["close"].rolling(w).mean()
        std = df["close"].rolling(w).std()
        df[f"zscore_{w}"] = (df["close"] - sma) / std

    # Normalized distance from moving averages
    sma_20 = df["close"].rolling(20).mean()
    sma_50 = df["close"].rolling(50).mean()
    df["mean_distance_20"] = (df["close"] - sma_20) / df["close"]
    df["mean_distance_50"] = (df["close"] - sma_50) / df["close"]

    # ATR-normalized deviation from SMA20
    atr = compute_atr(df, 14)
    df["atr_deviation"] = (df["close"] - sma_20) / atr

    # Bollinger squeeze: percentile of bb_width over 100 bars
    df["bb_squeeze"] = df["bb_width"].rolling(100).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )

    # RSI extremity (0 = neutral, 1 = extreme)
    df["rsi_extreme"] = (df["rsi_14"] - 50).abs() / 50

    # Volume Z-score
    vol_sma = df["volume"].rolling(20).mean()
    vol_std = df["volume"].rolling(20).std()
    df["zscore_volume"] = (df["volume"] - vol_sma) / vol_std

    # Hurst exponent (rolling)
    df["hurst"] = compute_hurst(df["close"], window=100)

    # ADX (trend strength) — CRITICAL for mean reversion
    df["adx_14"] = compute_adx(df, period=14)

    # Market regime classifier (0 = ranging, 1 = trending)
    # Mean reversion works best in ranging markets (ADX < 25)
    df["market_regime"] = (df["adx_14"] > 25).astype(int)

    # VWAP and distance
    # VWAP = volume-weighted average price (fair price)
    cumulative_vwap = (df["volume"] * (df["high"] + df["low"] + df["close"]) / 3).cumsum()
    cumulative_volume = df["volume"].cumsum()
    df["vwap"] = cumulative_vwap / cumulative_volume

    # Distance from VWAP (normalized)
    df["vwap_distance"] = (df["close"] - df["vwap"]) / df["vwap"]

    # Bollinger band distances (in addition to percent_b)
    bb_mid = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    df["bb_upper_dist"] = (bb_upper - df["close"]) / df["close"]
    df["bb_lower_dist"] = (df["close"] - bb_lower) / df["close"]

    # ── Derivatives data (if available) ──────────────────────

    if "funding_rate" in df.columns:
        df["funding_rate_feat"] = df["funding_rate"]
    else:
        df["funding_rate_feat"] = 0.0

    if "open_interest" in df.columns and (df["open_interest"] != 0).any():
        df["oi_change"] = df["open_interest"].pct_change()
        df["oi_change"] = df["oi_change"].clip(-1, 1)  # cap extreme values
    else:
        df["oi_change"] = 0.0

    return df


# Feature columns for the model
FEATURE_COLS = [
    # Classic
    "rsi_14",
    "ema_diff",
    "log_return_1",
    "log_return_3",
    "log_return_6",
    "log_return_12",
    "volatility_10",
    "volatility_20",
    "relative_volume",
    "macd_line",
    "macd_signal",
    "macd_hist",
    "bb_percent_b",
    "bb_width",
    "atr_14",
    "stoch_k",
    "stoch_d",
    "candle_range",
    "candle_body",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    # Mean reversion
    "zscore_20",
    "zscore_50",
    "zscore_100",
    "mean_distance_20",
    "mean_distance_50",
    "atr_deviation",
    "bb_squeeze",
    "rsi_extreme",
    "zscore_volume",
    "hurst",
    # NEW: Regime and VWAP features
    "adx_14",
    "market_regime",
    "vwap_distance",
    "bb_upper_dist",
    "bb_lower_dist",
    # Derivatives
    "funding_rate_feat",
    "oi_change",
]


if __name__ == "__main__":
    np.random.seed(42)
    n = 500
    fake = pd.DataFrame({
        "timestamp": pd.date_range("2023-01-01", periods=n, freq="1h"),
        "open": 100 + np.cumsum(np.random.randn(n) * 0.3),
        "high": 101 + np.cumsum(np.random.randn(n) * 0.3),
        "low": 99 + np.cumsum(np.random.randn(n) * 0.3),
        "close": 100 + np.cumsum(np.random.randn(n) * 0.3),
        "volume": np.random.uniform(100, 1000, n),
    })
    fake["high"] = fake[["open", "high", "close"]].max(axis=1) + 0.5
    fake["low"] = fake[["open", "low", "close"]].min(axis=1) - 0.5

    result = add_features(fake)
    print(f"Features: {len(FEATURE_COLS)}")
    print(f"Shape: {result.shape}")
    available = [c for c in FEATURE_COLS if c in result.columns]
    print(f"Available: {len(available)}/{len(FEATURE_COLS)}")
    print(result[available].describe())
