"""
target.py
Breakout momentum target definition.

The target answers: "Given a breakout signal (Donchian channel break),
will price reach TP before SL within N candles?"

Only rows with valid breakout signals (donchian_breakout_up/down) get a label.
"""

import numpy as np
import pandas as pd

from config import (
    TARGET_TP_PCT,
    TARGET_SL_ATR_MULT,
    TARGET_HORIZON,
    BREAKOUT_PERIOD,
    BREAKOUT_THRESHOLD,
)


def compute_breakout_target(
    df: pd.DataFrame,
    horizon: int = TARGET_HORIZON,
    tp_pct: float = TARGET_TP_PCT,
    sl_atr_mult: float = TARGET_SL_ATR_MULT,
) -> pd.DataFrame:
    """
    Create binary target for breakout momentum.

    Simulates each breakout forward:
    - target = 1 if TP is hit before SL (within horizon)
    - target = 0 if SL is hit first, or neither within horizon

    Only labels rows where breakout occurs:
    - donchian_breakout_up == 1 (price breaks above high_N)
    - donchian_breakout_down == 1 (price breaks below low_N)

    Args:
        df: DataFrame with features computed (must have donchian_breakout_up/down, atr_14, etc).
        horizon: Max candles to simulate forward.
        tp_pct: Take profit percentage target (e.g., 0.04 = 4%).
        sl_atr_mult: ATR multiplier for stop loss.

    Returns:
        DataFrame with 'target' and 'direction' columns added.
        direction: 1 for long (upward breakout), -1 for short (downward breakout), 0 for no breakout.
    """
    df = df.copy()

    # Ensure required columns exist
    required = ["donchian_breakout_up", "donchian_breakout_down", "atr_14", "high", "low", "close"]
    if not all(col in df.columns for col in required):
        print(f"Warning: Missing required columns for target labeling")
        df["direction"] = 0
        df["target"] = np.nan
        return df

    # Calculate absolute ATR from normalized atr_14
    df["atr_abs"] = df["atr_14"] * df["close"]

    # Determine breakout direction
    df["direction"] = 0

    # LONG: upward breakout
    long_mask = df["donchian_breakout_up"] == 1
    df.loc[long_mask, "direction"] = 1

    # SHORT: downward breakout
    short_mask = df["donchian_breakout_down"] == 1
    df.loc[short_mask, "direction"] = -1

    # Initialize target as NaN (rows without breakout excluded)
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
        tp_distance = entry_price * tp_pct

        if direction == 1:  # LONG (upward breakout)
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

        else:  # SHORT (downward breakout)
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
    symbol: str = "BTC/USDT",
    btc_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Full pipeline: add features + target, clean NaNs.
    Returns dataframe with target labels for rows with breakout signals.

    Args:
        df: OHLCV dataframe
        symbol: Trading pair (for market context)
        btc_df: Optional BTC data for market context features
    """
    from features import add_features

    df = add_features(df, symbol=symbol, btc_df=btc_df)
    df = compute_breakout_target(df)

    # Mark which rows are tradeable (have valid target)
    df["tradeable"] = df["target"].notna()

    # Return full dataframe (backtest needs all rows for context)
    return df


def get_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to only rows with valid targets (breakout conditions met).
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

    # Generate breakout-friendly fake data (trending with volatility)
    trend = np.cumsum(np.random.randn(n) * 0.8)
    noise = np.random.randn(n) * 0.5

    fake = pd.DataFrame({
        "timestamp": pd.date_range("2023-01-01", periods=n, freq="1h"),
        "open": 100 + trend + noise,
        "high": 0,
        "low": 0,
        "close": 100 + trend + noise,
        "volume": np.random.uniform(100, 1000, n),
    })
    fake["high"] = fake[["open", "close"]].max(axis=1) + abs(np.random.randn(n) * 0.5)
    fake["low"] = fake[["open", "close"]].min(axis=1) - abs(np.random.randn(n) * 0.5)

    result = prepare_dataset(fake)

    tradeable = result[result["tradeable"]]
    print(f"Total rows: {len(result)}")
    print(f"Tradeable rows (breakouts): {len(tradeable)} ({len(tradeable)/len(result):.1%})")

    if len(tradeable) > 0:
        print(f"\nTarget distribution:")
        print(f"  TP hit (1): {int(tradeable['target'].sum())} "
              f"({tradeable['target'].mean():.1%})")
        print(f"  SL/Timeout (0): {int((1 - tradeable['target']).sum())} "
              f"({1 - tradeable['target'].mean():.1%})")

        print(f"\nDirection distribution:")
        print(f"  Long:  {(tradeable['direction'] == 1).sum()}")
        print(f"  Short: {(tradeable['direction'] == -1).sum()}")