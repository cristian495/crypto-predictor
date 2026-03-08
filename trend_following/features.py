"""
features.py
Technical indicators optimized for trend following.
Focuses on momentum, trend strength, and breakout detection.

Updated with market context features for better altcoin prediction.
"""

import numpy as np
import pandas as pd
from typing import Optional


# ── Core indicator functions ────────────────────────────────


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


def compute_adx(df: pd.DataFrame, period: int = 14) -> tuple:
    """
    Average Directional Index (ADX) + DI+ and DI-.
    ADX measures trend strength (0-100).
    ADX > 25 = strong trend, ADX < 20 = weak/ranging market.
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    # Directional Movement
    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    # True Range
    tr = compute_atr(df, period) * period  # Undo EMA, get raw TR sum

    # Directional Indicators
    atr = compute_atr(df, period)
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, min_periods=period).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, min_periods=period).mean() / atr)

    # ADX
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.ewm(alpha=1/period, min_periods=period).mean()

    return adx, plus_di, minus_di


def compute_stochastic(df: pd.DataFrame, period: int = 14, smooth_k: int = 3) -> tuple:
    """Stochastic Oscillator %K and %D."""
    lowest_low = df["low"].rolling(period).min()
    highest_high = df["high"].rolling(period).max()

    fast_k = 100 * (df["close"] - lowest_low) / (highest_high - lowest_low)
    slow_k = fast_k.rolling(smooth_k).mean()
    slow_d = slow_k.rolling(smooth_k).mean()

    return slow_k, slow_d


def compute_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.Series:
    """
    Supertrend indicator for trend detection.
    Returns 1 for uptrend, -1 for downtrend.
    """
    atr = compute_atr(df, period)
    hl_avg = (df["high"] + df["low"]) / 2

    upper_band = hl_avg + (multiplier * atr)
    lower_band = hl_avg - (multiplier * atr)

    supertrend = pd.Series(0, index=df.index)
    direction = pd.Series(1, index=df.index)

    for i in range(1, len(df)):
        # Update bands
        if df["close"].iloc[i] > upper_band.iloc[i-1]:
            direction.iloc[i] = 1
        elif df["close"].iloc[i] < lower_band.iloc[i-1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i-1]

            # Adjust bands based on trend
            if direction.iloc[i] == 1:
                lower_band.iloc[i] = max(lower_band.iloc[i], lower_band.iloc[i-1])
            else:
                upper_band.iloc[i] = min(upper_band.iloc[i], upper_band.iloc[i-1])

    return direction


# ── Market Context (for altcoins) ───────────────────────────


def _load_btc_reference(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Load BTC data for market context features.
    Only loads if current symbol is NOT BTC.
    Uses cached data to avoid re-downloading.
    """
    # Extract symbol from df metadata if available
    # Otherwise, skip BTC loading (will be handled in add_features)
    return None  # Placeholder - will be loaded in add_features with symbol param


# ── Feature engineering ─────────────────────────────────────


def add_features(df: pd.DataFrame, symbol: str = "BTC/USDT",
                 btc_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Add trend following indicators and features.
    Focuses on momentum, trend strength, and volatility.

    Args:
        df: OHLCV dataframe for the target symbol
        symbol: Trading pair (e.g., "ETH/USDT")
        btc_df: Optional BTC OHLCV dataframe for market context features
    """
    df = df.copy()

    # Flag to determine if we need market context features
    is_altcoin = symbol not in ["BTC/USDT", "BTCUSDT"]

    # ── Trend Detection ──────────────────────────────────────

    # EMAs for trend
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema_200"] = df["close"].ewm(span=200, adjust=False).mean()

    # EMA relationships
    df["ema_20_50_diff"] = (df["ema_20"] - df["ema_50"]) / df["close"]
    df["ema_50_200_diff"] = (df["ema_50"] - df["ema_200"]) / df["close"]
    df["price_ema_20_dist"] = (df["close"] - df["ema_20"]) / df["close"]
    df["price_ema_200_dist"] = (df["close"] - df["ema_200"]) / df["close"]

    # Trend alignment (all EMAs ordered)
    df["trend_aligned_bull"] = (
        (df["ema_20"] > df["ema_50"]) &
        (df["ema_50"] > df["ema_200"])
    ).astype(int)
    df["trend_aligned_bear"] = (
        (df["ema_20"] < df["ema_50"]) &
        (df["ema_50"] < df["ema_200"])
    ).astype(int)

    # EMA slope (rate of change)
    df["ema_20_slope"] = df["ema_20"].pct_change(5)
    df["ema_50_slope"] = df["ema_50"].pct_change(10)

    # ── Momentum Indicators ──────────────────────────────────

    # RSI
    df["rsi_14"] = compute_rsi(df["close"], 14)

    # ROC (Rate of Change)
    df["roc_10"] = df["close"].pct_change(10)
    df["roc_20"] = df["close"].pct_change(20)

    # MACD
    ema_12 = df["close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd_line"] = (ema_12 - ema_26) / df["close"]
    df["macd_signal"] = df["macd_line"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd_line"] - df["macd_signal"]

    # Stochastic
    df["stoch_k"], df["stoch_d"] = compute_stochastic(df, 14, 3)

    # ADX and Directional Indicators
    df["adx"], df["di_plus"], df["di_minus"] = compute_adx(df, 14)
    df["di_diff"] = df["di_plus"] - df["di_minus"]

    # Supertrend
    df["supertrend"] = compute_supertrend(df, 10, 3.0)

    # ── Volatility ───────────────────────────────────────────

    # ATR (normalized)
    df["atr_14"] = compute_atr(df, 14) / df["close"]
    df["atr_20"] = compute_atr(df, 20) / df["close"]

    # ATR percentile (for volatility regime detection)
    df["atr_percentile"] = df["atr_14"].rolling(100).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )

    # ATR Expansion (ChatGPT suggestion for ETH)
    df["atr_ma_20"] = df["atr_14"].rolling(20).mean()
    df["atr_expanding"] = (df["atr_14"] > df["atr_ma_20"]).astype(int)
    df["atr_expansion_rate"] = df["atr_14"] / (df["atr_ma_20"] + 1e-8)

    # Bollinger Bands
    bb_mid = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    df["bb_percent_b"] = (df["close"] - bb_lower) / (bb_upper - bb_lower)
    df["bb_width"] = (bb_upper - bb_lower) / bb_mid

    # Log returns
    df["log_return_1"] = np.log(df["close"] / df["close"].shift(1))
    df["log_return_5"] = np.log(df["close"] / df["close"].shift(5))
    df["log_return_10"] = np.log(df["close"] / df["close"].shift(10))
    df["log_return_20"] = np.log(df["close"] / df["close"].shift(20))

    # Realized volatility
    df["volatility_10"] = df["log_return_1"].rolling(10).std()
    df["volatility_20"] = df["log_return_1"].rolling(20).std()

    # ── Breakout Detection ───────────────────────────────────

    # Highest/lowest over periods
    df["high_20"] = df["high"].rolling(20).max()
    df["low_20"] = df["low"].rolling(20).min()
    df["high_50"] = df["high"].rolling(50).max()
    df["low_50"] = df["low"].rolling(50).min()

    # Distance to highs/lows
    df["dist_to_high_20"] = (df["high_20"] - df["close"]) / df["close"]
    df["dist_to_low_20"] = (df["close"] - df["low_20"]) / df["close"]

    # Breakout signals (1 = at/near high, -1 = at/near low)
    df["near_high_20"] = (df["close"] >= df["high_20"] * 0.99).astype(int)
    df["near_low_20"] = (df["close"] <= df["low_20"] * 1.01).astype(int)

    # Range compression (precursor to breakouts)
    df["range_20"] = df["high_20"] - df["low_20"]
    df["range_50"] = df["high_50"] - df["low_50"]
    df["range_compression"] = df["range_20"] / (df["range_50"] + 1e-8)

    # Donchian Channel position
    donchian_mid = (df["high_20"] + df["low_20"]) / 2
    df["donchian_position"] = (df["close"] - donchian_mid) / (df["range_20"] + 1e-8)

    # ── Volume ───────────────────────────────────────────────

    # Relative volume
    df["volume_sma_20"] = df["volume"].rolling(20).mean()
    df["volume_std_20"] = df["volume"].rolling(20).std()
    df["relative_volume"] = df["volume"] / df["volume_sma_20"]

    # Volume Z-score (how unusual is current volume)
    df["volume_zscore"] = (
        (df["volume"] - df["volume_sma_20"]) / (df["volume_std_20"] + 1e-8)
    ).clip(-5, 5)

    # Volume spike detection
    df["volume_spike"] = (df["relative_volume"] > 2.0).astype(int)

    # VWAP (Volume-Weighted Average Price)
    df["vwap"] = (
        (df["close"] * df["volume"]).rolling(20).sum() /
        df["volume"].rolling(20).sum()
    )
    df["vwap_distance"] = (df["close"] - df["vwap"]) / df["close"]

    # Volume trend
    df["volume_ema_10"] = df["volume"].ewm(span=10, adjust=False).mean()
    df["volume_trend"] = df["volume_ema_10"] / df["volume_sma_20"]

    # On-Balance Volume (OBV)
    obv = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()
    df["obv"] = obv
    df["obv_ema"] = obv.ewm(span=20, adjust=False).mean()
    df["obv_divergence"] = (df["obv"] - df["obv_ema"]) / (df["obv_ema"].abs() + 1e-8)

    # ── Price Action ─────────────────────────────────────────

    # Candle patterns
    df["candle_range"] = (df["high"] - df["low"]) / df["close"]
    df["candle_body"] = (df["close"] - df["open"]) / df["close"]
    df["upper_wick"] = (df["high"] - df[["open", "close"]].max(axis=1)) / df["close"]
    df["lower_wick"] = (df[["open", "close"]].min(axis=1) - df["low"]) / df["close"]

    # Consecutive closes (streak detection)
    df["close_direction"] = np.sign(df["close"].diff())
    df["streak"] = df["close_direction"].groupby(
        (df["close_direction"] != df["close_direction"].shift()).cumsum()
    ).cumsum()

    # ── Acceleration & Momentum Derivatives ──────────────────

    # Multi-timeframe momentum
    df["momentum_5"] = df["close"].pct_change(5)
    df["momentum_10"] = df["close"].pct_change(10)
    df["momentum_20"] = df["close"].pct_change(20)

    # Momentum acceleration (rate of change of momentum)
    df["momentum_acceleration"] = df["momentum_10"].diff(5)

    # Return skewness (asymmetry in returns - bullish vs bearish)
    df["return_skew_20"] = df["log_return_1"].rolling(20).skew()

    # Momentum consistency (how often price closes in same direction)
    df["bullish_streak_pct"] = (
        df["close_direction"].clip(0, 1).rolling(20).mean()
    )

    # ── Temporal Features ────────────────────────────────────

    ts = pd.to_datetime(df["timestamp"])
    df["hour_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * ts.dt.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * ts.dt.dayofweek / 7)

    # Weekend bias (ChatGPT suggestion - ETH weekend pump)
    # Friday (4), Saturday (5), Sunday (6)
    df["is_weekend"] = (ts.dt.dayofweek >= 4).astype(int)

    # ── Market Context (for Altcoins) ────────────────────────

    if is_altcoin and btc_df is not None and len(btc_df) > 0:
        # Align BTC data with current symbol's timestamps
        btc_aligned = btc_df.set_index("timestamp").reindex(
            df["timestamp"], method="ffill"
        ).reset_index()

        # BTC returns for correlation
        btc_aligned["btc_return_1"] = btc_aligned["close"].pct_change(1)
        btc_aligned["btc_return_5"] = btc_aligned["close"].pct_change(5)
        btc_aligned["btc_return_20"] = btc_aligned["close"].pct_change(20)

        # BTC volatility
        btc_aligned["btc_volatility"] = (
            np.log(btc_aligned["close"] / btc_aligned["close"].shift(1))
            .rolling(20).std()
        )

        # Current symbol returns (for correlation calculation)
        symbol_return_20 = df["close"].pct_change(20)

        # Rolling correlation with BTC (30-day)
        df["btc_correlation_30"] = (
            df["log_return_1"].rolling(30 * 24)  # 30 days @ 1h
            .corr(btc_aligned["btc_return_1"])
        ).fillna(0)

        # Outperformance vs BTC (risk-adjusted)
        outperformance = (
            (symbol_return_20 - btc_aligned["btc_return_20"]) /
            (btc_aligned["btc_volatility"] + 1e-8)
        )
        df["btc_outperformance"] = outperformance.clip(-5, 5).fillna(0)

        # Relative strength (symbol vs BTC trend alignment)
        symbol_ema_20 = df["ema_20"]
        symbol_ema_50 = df["ema_50"]
        btc_ema_20 = btc_aligned["close"].ewm(span=20, adjust=False).mean()
        btc_ema_50 = btc_aligned["close"].ewm(span=50, adjust=False).mean()

        symbol_trend = (symbol_ema_20 > symbol_ema_50).astype(int)
        btc_trend = (btc_ema_20 > btc_ema_50).astype(int)
        df["trend_agrees_with_btc"] = (symbol_trend == btc_trend).astype(int)

        # Market regime based on BTC trend strength
        btc_adx, _, _ = compute_adx(btc_aligned[["high", "low", "close"]], 14)
        df["btc_trending"] = (btc_adx > 25).astype(int)

    else:
        # If BTC or no BTC data provided, fill with neutral values
        df["btc_correlation_30"] = 0.0
        df["btc_outperformance"] = 0.0
        df["trend_agrees_with_btc"] = 1  # Neutral
        df["btc_trending"] = 0

    # ── Derivatives Data (if available) ──────────────────────

    if "funding_rate" in df.columns:
        df["funding_rate_feat"] = df["funding_rate"]
        df["funding_rate_ema"] = df["funding_rate"].ewm(span=8, adjust=False).mean()
    else:
        df["funding_rate_feat"] = 0.0
        df["funding_rate_ema"] = 0.0

    if "open_interest" in df.columns and (df["open_interest"] != 0).any():
        df["oi_change"] = df["open_interest"].pct_change().clip(-1, 1)
        df["oi_trend"] = (
            df["open_interest"].ewm(span=20, adjust=False).mean() /
            df["open_interest"].rolling(50).mean()
        ) - 1
    else:
        df["oi_change"] = 0.0
        df["oi_trend"] = 0.0

    return df


# Feature columns for the model
FEATURE_COLS = [
    # Trend
    "ema_20_50_diff",
    "ema_50_200_diff",
    "price_ema_20_dist",
    "price_ema_200_dist",
    "trend_aligned_bull",
    "trend_aligned_bear",
    "ema_20_slope",
    "ema_50_slope",

    # Momentum
    "rsi_14",
    "roc_10",
    "roc_20",
    "macd_line",
    "macd_signal",
    "macd_hist",
    "stoch_k",
    "stoch_d",
    "adx",
    "di_plus",
    "di_minus",
    "di_diff",
    "supertrend",

    # Volatility
    "atr_14",
    "atr_20",
    "atr_percentile",
    "atr_expanding",
    "atr_expansion_rate",
    "bb_percent_b",
    "bb_width",
    "log_return_1",
    "log_return_5",
    "log_return_10",
    "log_return_20",
    "volatility_10",
    "volatility_20",

    # Breakout
    "dist_to_high_20",
    "dist_to_low_20",
    "near_high_20",
    "near_low_20",
    "range_compression",
    "donchian_position",

    # Acceleration & Momentum Derivatives
    "momentum_5",
    "momentum_10",
    "momentum_20",
    "momentum_acceleration",
    "return_skew_20",
    "bullish_streak_pct",

    # Volume
    "relative_volume",
    "volume_zscore",
    "volume_spike",
    "vwap_distance",
    "volume_trend",
    "obv_divergence",

    # Price action
    "candle_range",
    "candle_body",
    "upper_wick",
    "lower_wick",
    "streak",

    # Temporal
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",

    # Market Context (for altcoins vs BTC)
    "btc_correlation_30",
    "btc_outperformance",
    "trend_agrees_with_btc",
    "btc_trending",

    # Derivatives
    "funding_rate_feat",
    "funding_rate_ema",
    "oi_change",
    "oi_trend",
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
