"""
target.py
Mean reversion target definition.

The target answers: "Given that price is extended (|Z| > threshold),
will it revert toward the mean within N candles?"

Only rows where price IS extended get a label.
Rows near the mean (|Z| < threshold) are excluded from training.
"""

import numpy as np
import pandas as pd

from config import (
    ZSCORE_ENTRY_THRESHOLD,
    REVERSION_TARGET_PCT,
    REVERSION_HORIZON,
    STOP_LOSS_PCT,
    TARGET_SL_PCT,
)


def compute_reversion_target(
    df: pd.DataFrame,
    zscore_col: str = "zscore_50",
    zscore_threshold: float = ZSCORE_ENTRY_THRESHOLD,
    reversion_pct: float = REVERSION_TARGET_PCT,
    horizon: int = REVERSION_HORIZON,
    sl_pct: float = TARGET_SL_PCT,
) -> pd.DataFrame:
    """
    Create binary target aligned with actual trade outcome.

    Simulates each potential trade forward, mirroring the backtest logic:
    - Checks stop loss hit (using candle high/low) before checking reversion.
    - target = 1 if Z-score reverts to 0 before SL is hit (within horizon).
    - target = 0 if SL is hit first, or no reversion within horizon.

    NOTE: sl_pct defaults to TARGET_SL_PCT (wider than trading STOP_LOSS_PCT).
    A wider labeling SL produces more positive examples, making classification
    easier and reducing overfitting. The actual trading SL remains tighter.

    Args:
        df: DataFrame with features computed (must have zscore_col, high, low, close).
        zscore_col: Z-score column to use for entry detection.
        zscore_threshold: Minimum |Z| to consider price extended.
        reversion_pct: Kept for API compatibility, not used in simulation.
        horizon: Max candles to simulate forward.
        sl_pct: SL fraction for labeling only (defaults to TARGET_SL_PCT).

    Returns:
        DataFrame with 'target' and 'direction' columns added.
        direction: 1 for long (oversold), -1 for short (overbought), 0 for neutral.
    """
    df = df.copy()

    # Determine direction based on Z-score
    df["direction"] = 0
    df.loc[df[zscore_col] < -zscore_threshold, "direction"] = 1    # oversold → long
    df.loc[df[zscore_col] > zscore_threshold, "direction"] = -1    # overbought → short

    # Initialize target as NaN (rows near mean excluded from training)
    df["target"] = np.nan
    target_col = df.columns.get_loc("target")

    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    zscores = df[zscore_col].values
    directions = df["direction"].values

    for i in range(len(df)):
        direction = directions[i]
        if direction == 0:
            continue
        if i + horizon >= len(df):
            continue

        entry_price = closes[i]

        if direction == 1:  # LONG
            sl_price = entry_price * (1 - sl_pct)

            outcome = 0  # default: no reversion or stopped out
            for j in range(1, horizon + 1):
                k = i + j
                # Stop loss check first (candle low touches SL)
                if lows[k] <= sl_price:
                    outcome = 0
                    break
                # Mean reversion: Z-score crosses back to 0
                if zscores[k] >= 0:
                    outcome = 1
                    break

        else:  # SHORT
            sl_price = entry_price * (1 + sl_pct)

            outcome = 0
            for j in range(1, horizon + 1):
                k = i + j
                # Stop loss check first (candle high touches SL)
                if highs[k] >= sl_price:
                    outcome = 0
                    break
                # Mean reversion: Z-score crosses back to 0
                if zscores[k] <= 0:
                    outcome = 1
                    break

        df.iloc[i, target_col] = float(outcome)

    return df


def prepare_dataset(df: pd.DataFrame,
                    zscore_col: str = "zscore_50",
                    zscore_threshold: float = ZSCORE_ENTRY_THRESHOLD,
                    reversion_pct: float = REVERSION_TARGET_PCT,
                    horizon: int = REVERSION_HORIZON) -> pd.DataFrame:
    """
    Full pipeline: add target + clean NaNs.
    Returns only rows where price is extended (has a valid target).
    """
    from features import add_features

    df = add_features(df)
    df = compute_reversion_target(
        df,
        zscore_col=zscore_col,
        zscore_threshold=zscore_threshold,
        reversion_pct=reversion_pct,
        horizon=horizon,
    )

    # Keep all rows for backtest context, but mark which are tradeable
    df["tradeable"] = df["target"].notna()

    # For model training, we'll filter to tradeable rows
    # But return the full dataframe for backtest use
    return df


def get_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to only rows with valid targets (|Z| > threshold).
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
    fake = pd.DataFrame({
        "timestamp": pd.date_range("2023-01-01", periods=n, freq="1h"),
        "open": 100 + np.cumsum(np.random.randn(n) * 0.5),
        "high": 0,
        "low": 0,
        "close": 100 + np.cumsum(np.random.randn(n) * 0.5),
        "volume": np.random.uniform(100, 1000, n),
    })
    fake["high"] = fake[["open", "close"]].max(axis=1) + abs(np.random.randn(n) * 0.3)
    fake["low"] = fake[["open", "close"]].min(axis=1) - abs(np.random.randn(n) * 0.3)

    result = prepare_dataset(fake)

    tradeable = result[result["tradeable"]]
    print(f"Total rows: {len(result)}")
    print(f"Tradeable rows: {len(tradeable)} ({len(tradeable)/len(result):.1%})")
    if len(tradeable) > 0:
        print(f"Target distribution:")
        print(f"  Revert (1): {int(tradeable['target'].sum())} "
              f"({tradeable['target'].mean():.1%})")
        print(f"  No revert (0): {int((1 - tradeable['target']).sum())} "
              f"({1 - tradeable['target'].mean():.1%})")
        print(f"Direction distribution:")
        print(f"  Long:  {(tradeable['direction'] == 1).sum()}")
        print(f"  Short: {(tradeable['direction'] == -1).sum()}")
