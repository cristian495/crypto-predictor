"""
target.py
Trend following target definition.

The target answers: "Given current trend conditions and momentum,
will price continue in the trend direction and hit TP before SL within N candles?"

Only rows with clear trend signals get a label.
"""

import numpy as np
import pandas as pd

from config import (
    STOP_LOSS_ATR_MULT,
    TAKE_PROFIT_RR,
    HOLD_TIMEOUT,
    TARGET_SL_ATR_MULT,
    EMA_FAST,
    EMA_SLOW,
    ADX_MIN,
)


def compute_trend_target(
    df: pd.DataFrame,
    horizon: int = HOLD_TIMEOUT,
    sl_atr_mult: float = TARGET_SL_ATR_MULT,
    tp_rr: float = TAKE_PROFIT_RR,
    adx_min: float = ADX_MIN,
) -> pd.DataFrame:
    """
    Create binary target for trend following.

    Simulates each potential trade forward:
    - target = 1 if TP is hit before SL (within horizon)
    - target = 0 if SL is hit first, or neither within horizon

    Only labels rows where trend conditions are met:
    - ADX >= adx_min (trending market)
    - EMA(fast) crossed EMA(slow) in direction of trade

    Args:
        df: DataFrame with features computed (must have ema_20, ema_50, adx, atr_14, etc).
        horizon: Max candles to simulate forward.
        sl_atr_mult: ATR multiplier for stop loss (labeling).
        tp_rr: Risk:Reward ratio for take profit.
        adx_min: Minimum ADX for trend strength.

    Returns:
        DataFrame with 'target' and 'direction' columns added.
        direction: 1 for long (uptrend), -1 for short (downtrend), 0 for no trend.
    """
    df = df.copy()

    # Ensure required columns exist
    required = ["ema_20", "ema_50", "adx", "atr_14", "high", "low", "close"]
    if not all(col in df.columns for col in required):
        print(f"Warning: Missing required columns for target labeling")
        df["direction"] = 0
        df["target"] = np.nan
        return df

    # Calculate absolute ATR from normalized atr_14
    df["atr_abs"] = df["atr_14"] * df["close"]

    # Determine trend direction
    df["direction"] = 0

    # LONG: EMA20 > EMA50 and ADX >= threshold
    long_mask = (df["ema_20"] > df["ema_50"]) & (df["adx"] >= adx_min)
    df.loc[long_mask, "direction"] = 1

    # SHORT: EMA20 < EMA50 and ADX >= threshold
    short_mask = (df["ema_20"] < df["ema_50"]) & (df["adx"] >= adx_min)
    df.loc[short_mask, "direction"] = -1

    # Initialize target as NaN (rows without clear trend excluded)
    df["target"] = np.nan
    target_col = df.columns.get_loc("target")

    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    directions = df["direction"].values
    atr_abs = df["atr_abs"].values

    for i in range(len(df)):
        direction = directions[i]
        if direction == 0:
            continue
        if i + horizon >= len(df):
            continue

        entry_price = closes[i]
        atr = atr_abs[i]

        if np.isnan(atr) or atr <= 0:
            continue

        # Compute SL and TP levels
        sl_distance = atr * sl_atr_mult
        tp_distance = sl_distance * tp_rr

        if direction == 1:  # LONG
            sl_price = entry_price - sl_distance
            tp_price = entry_price + tp_distance

            outcome = 0  # default: no TP hit
            for j in range(1, horizon + 1):
                k = i + j
                # Check SL first (candle low touches SL)
                if lows[k] <= sl_price:
                    outcome = 0
                    break
                # Check TP (candle high touches TP)
                if highs[k] >= tp_price:
                    outcome = 1
                    break

        else:  # SHORT
            sl_price = entry_price + sl_distance
            tp_price = entry_price - tp_distance

            outcome = 0
            for j in range(1, horizon + 1):
                k = i + j
                # Check SL first (candle high touches SL)
                if highs[k] >= sl_price:
                    outcome = 0
                    break
                # Check TP (candle low touches TP)
                if lows[k] <= tp_price:
                    outcome = 1
                    break

        df.iloc[i, target_col] = float(outcome)

    return df


def prepare_dataset(
    df: pd.DataFrame,
    horizon: int = HOLD_TIMEOUT,
    adx_min: float = ADX_MIN,
    symbol: str = "BTC/USDT",
    btc_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Full pipeline: add features + target, clean NaNs.
    Returns dataframe with target labels for rows in trending conditions.

    Args:
        df: OHLCV dataframe
        horizon: Max candles for target simulation
        adx_min: Min ADX for trend strength
        symbol: Trading pair (for market context)
        btc_df: Optional BTC data for market context features
    """
    from features import add_features

    df = add_features(df, symbol=symbol, btc_df=btc_df)
    df = compute_trend_target(df, horizon=horizon, adx_min=adx_min)

    # Mark which rows are tradeable (have valid target)
    df["tradeable"] = df["target"].notna()

    # Return full dataframe (backtest needs all rows for context)
    return df


def get_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to only rows with valid targets (clear trend conditions).
    Drop any remaining NaN in feature columns.
    Used for model training.
    """
    from features import FEATURE_COLS

    train_df = df[df["tradeable"]].copy()
    cols_needed = FEATURE_COLS + ["target", "direction", "timestamp"]
    cols_available = [c for c in cols_needed if c in train_df.columns]
    train_df = train_df[cols_available].dropna().reset_index(drop=True)

    return train_df


if __name__ == "__main__":
    from features import add_features

    np.random.seed(42)
    n = 1000

    # Generate trending fake data
    trend = np.cumsum(np.random.randn(n) * 0.5)
    noise = np.random.randn(n) * 0.2

    fake = pd.DataFrame({
        "timestamp": pd.date_range("2023-01-01", periods=n, freq="1h"),
        "open": 100 + trend + noise,
        "high": 0,
        "low": 0,
        "close": 100 + trend + noise,
        "volume": np.random.uniform(100, 1000, n),
    })
    fake["high"] = fake[["open", "close"]].max(axis=1) + abs(np.random.randn(n) * 0.3)
    fake["low"] = fake[["open", "close"]].min(axis=1) - abs(np.random.randn(n) * 0.3)

    result = prepare_dataset(fake)

    tradeable = result[result["tradeable"]]
    print(f"Total rows: {len(result)}")
    print(f"Tradeable rows: {len(tradeable)} ({len(tradeable)/len(result):.1%})")

    if len(tradeable) > 0:
        print(f"\nTarget distribution:")
        print(f"  TP hit (1): {int(tradeable['target'].sum())} "
              f"({tradeable['target'].mean():.1%})")
        print(f"  SL/Timeout (0): {int((1 - tradeable['target']).sum())} "
              f"({1 - tradeable['target'].mean():.1%})")

        print(f"\nDirection distribution:")
        print(f"  Long:  {(tradeable['direction'] == 1).sum()}")
        print(f"  Short: {(tradeable['direction'] == -1).sum()}")