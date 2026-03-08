"""
strategy.py
Entry/exit logic for mean reversion trading.
Separates ML signal from trading rules.
"""

import numpy as np
import pandas as pd

from features import FEATURE_COLS
from config import (
    ZSCORE_ENTRY_THRESHOLD,
    BUY_THRESHOLD,
)

try:
    from config import ZSCORE_LONG_THRESHOLD, ZSCORE_SHORT_THRESHOLD
except ImportError:
    ZSCORE_LONG_THRESHOLD = ZSCORE_ENTRY_THRESHOLD
    ZSCORE_SHORT_THRESHOLD = ZSCORE_ENTRY_THRESHOLD

from config import (
    MIN_AGREE,
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
    HOLD_TIMEOUT,
    HURST_FILTER,
    RSI_LONG_MAX,
    RSI_SHORT_MIN,
    LONG_ONLY,
)


def generate_signals(
    df: pd.DataFrame,
    model,
    zscore_col: str = "zscore_50",
    z_long_threshold: float = ZSCORE_LONG_THRESHOLD,
    z_short_threshold: float = ZSCORE_SHORT_THRESHOLD,
    buy_threshold: float = BUY_THRESHOLD,
    min_agree: int = MIN_AGREE,
    feature_cols: list = FEATURE_COLS,
    short_bias: float = 0.10,
    hurst_filter: bool = HURST_FILTER,
    rsi_long_max: float = RSI_LONG_MAX,
    rsi_short_min: float = RSI_SHORT_MIN,
    long_only: bool = LONG_ONLY,
) -> pd.DataFrame:
    """
    Generate trading signals for every bar in the dataframe.

    Logic:
    1. Check if |Z| > threshold (price is extended)
    2. Apply regime filters:
       - Hurst filter: skip if Hurst >= 0.5 (trending market, not mean-reverting)
       - RSI filter:   LONG only if RSI < rsi_long_max (confirmed oversold)
                       SHORT only if RSI > rsi_short_min (confirmed overbought)
    3. Ask ensemble: probability of reversion?
    4. Directional thresholds (SHORT-friendly):
       - SHORT: prob > (buy_threshold - short_bias) AND >= min_agree agree
       - LONG:  prob > (buy_threshold + short_bias) AND >= min_agree agree
    5. Otherwise → NO TRADE

    Args:
        df: DataFrame with features computed (must include zscore_col).
        model: EnsembleModel instance (trained).
        zscore_col: Z-score column to use for entry detection.
        zscore_threshold: Minimum |Z| for entry consideration.
        buy_threshold: Base probability threshold for entry.
        min_agree: Minimum number of models that must agree.
        feature_cols: Feature columns for model input.
        short_bias: Reduces threshold for SHORT, increases for LONG.
            0.10 means SHORT needs 0.50 prob, LONG needs 0.70 prob.
        hurst_filter: If True, skip entries when Hurst >= 0.5 (trending regime).
        rsi_long_max: LONG signals only when RSI is below this value.
        rsi_short_min: SHORT signals only when RSI is above this value.

    Returns:
        DataFrame with added columns: signal, probability, n_agree, direction.
    """
    df = df.copy()

    long_threshold = buy_threshold + short_bias
    short_threshold = buy_threshold - short_bias

    # Initialize signal columns
    df["signal"] = "NO TRADE"
    df["probability"] = 0.0
    df["n_agree"] = 0
    df["signal_direction"] = 0  # 1=long, -1=short

    # Find rows where price is extended (asymmetric thresholds)
    extended_mask = (df[zscore_col] < -z_long_threshold) | (df[zscore_col] > z_short_threshold)
    extended_idx = df[extended_mask].index

    if len(extended_idx) == 0:
        return df

    # Get model predictions for extended rows
    X_extended = df.loc[extended_idx, feature_cols].values

    # Handle NaN in features (skip those rows)
    valid_mask = ~np.isnan(X_extended).any(axis=1)
    valid_idx = extended_idx[valid_mask]

    if len(valid_idx) == 0:
        return df

    X_valid = df.loc[valid_idx, feature_cols].values

    # Get individual model probabilities for directional voting
    avg_proba, _, _ = model.predict_with_voting(
        X_valid, threshold=buy_threshold, min_agree=min_agree
    )
    individual = model.predict_proba_individual(X_valid)

    # Assign probabilities
    df.loc[valid_idx, "probability"] = avg_proba

    # Generate directional signals with asymmetric thresholds and regime filters
    for i, idx in enumerate(valid_idx):
        z = df.loc[idx, zscore_col]
        prob = avg_proba[i]

        # Regime filter: skip trending markets (Hurst >= 0.5 = trending)
        if hurst_filter and "hurst" in df.columns:
            hurst_val = df.loc[idx, "hurst"]
            if not np.isnan(hurst_val) and hurst_val >= 0.5:
                continue

        rsi_val = df.loc[idx, "rsi_14"] if "rsi_14" in df.columns else np.nan

        if z < -z_long_threshold:
            # RSI filter: only enter LONG when price is confirmed oversold
            if not np.isnan(rsi_val) and rsi_val > rsi_long_max:
                continue

            # LONG candidate — stricter threshold
            n_agree_long = sum(
                1 for name in individual
                if individual[name][i] > long_threshold
            )
            df.loc[idx, "n_agree"] = n_agree_long
            if prob > long_threshold and n_agree_long >= min_agree:
                df.loc[idx, "signal"] = "LONG"
                df.loc[idx, "signal_direction"] = 1

        elif z > z_short_threshold and not long_only:
            # RSI filter: only enter SHORT when price is confirmed overbought
            if not np.isnan(rsi_val) and rsi_val < rsi_short_min:
                continue

            # SHORT candidate — relaxed threshold
            n_agree_short = sum(
                1 for name in individual
                if individual[name][i] > short_threshold
            )
            df.loc[idx, "n_agree"] = n_agree_short
            if prob > short_threshold and n_agree_short >= min_agree:
                df.loc[idx, "signal"] = "SHORT"
                df.loc[idx, "signal_direction"] = -1

    return df


def compute_exit_levels(
    entry_price: float,
    direction: int,
    stop_loss_pct: float = STOP_LOSS_PCT,
    take_profit_pct: float = TAKE_PROFIT_PCT,
) -> dict:
    """
    Compute TP and SL price levels for a position.

    Args:
        entry_price: Entry price.
        direction: 1 for long, -1 for short.
        stop_loss_pct: Stop loss as fraction (0.015 = 1.5%).
        take_profit_pct: Take profit as fraction (0.03 = 3%).

    Returns:
        dict with 'stop_loss' and 'take_profit' price levels.
    """
    if direction == 1:  # LONG
        return {
            "stop_loss": entry_price * (1 - stop_loss_pct),
            "take_profit": entry_price * (1 + take_profit_pct),
        }
    else:  # SHORT
        return {
            "stop_loss": entry_price * (1 + stop_loss_pct),
            "take_profit": entry_price * (1 - take_profit_pct),
        }


def check_exit(
    position: dict,
    current_high: float,
    current_low: float,
    current_close: float,
    current_zscore: float,
    bars_held: int,
    hold_timeout: int = HOLD_TIMEOUT,
    exit_z_threshold: float = -0.5,
) -> tuple:
    """
    Check if a position should be exited.

    Exit conditions (checked in order):
    1. Stop loss hit (using high/low of current candle)
    2. Take profit hit (using high/low of current candle)
    3. Mean reversion partial (Z-score crosses exit_z_threshold)
       LONG:  exits when Z >= -0.5 (before fully reaching mean)
       SHORT: exits when Z <= +0.5 (before fully reaching mean)
    4. Hold timeout exceeded

    exit_z_threshold: how far Z must revert before exiting.
       -0.5 = exit when Z is halfway back to mean (faster exits, higher WR)
        0.0 = exit when Z fully reaches mean (original behavior)

    Returns:
        (should_exit: bool, exit_price: float, exit_reason: str)
    """
    direction = position["direction"]
    sl = position["stop_loss"]
    tp = position["take_profit"]

    if direction == 1:  # LONG
        # Stop loss: price drops below SL
        if current_low <= sl:
            return True, sl, "stop_loss"

        # Take profit: price rises above TP
        if current_high >= tp:
            return True, tp, "take_profit"

        # Mean reversion: Z-score crosses back above exit threshold
        if current_zscore >= exit_z_threshold:
            return True, current_close, "mean_revert"

    else:  # SHORT
        # Stop loss: price rises above SL
        if current_high >= sl:
            return True, sl, "stop_loss"

        # Take profit: price drops below TP
        if current_low <= tp:
            return True, tp, "take_profit"

        # Mean reversion: Z-score crosses back below exit threshold
        if current_zscore <= -exit_z_threshold:
            return True, current_close, "mean_revert"

    # Timeout
    if bars_held >= hold_timeout:
        return True, current_close, "timeout"

    return False, 0.0, ""


def print_current_signals(df: pd.DataFrame, symbol: str = "", verbose: bool = True):
    """
    Print the latest signal for a symbol with detailed filter diagnostics.

    Args:
        df: DataFrame with signals generated
        symbol: Trading pair name
        verbose: If True, show detailed filter breakdown
    """
    last = df.iloc[-1]
    prefix = f"  {symbol:<12}" if symbol else "  "

    # Basic signal output (always shown)
    if last["signal"] != "NO TRADE":
        print(f"{prefix}{last['signal']:<8} "
              f"prob={last['probability']:.4f} "
              f"agree={int(last['n_agree'])}/3 "
              f"Z={last.get('zscore_50', 0):.2f}")
    else:
        # Show basic NO TRADE with Z-score
        z_val = last.get('zscore_50', 0)
        rsi_val = last.get('rsi_14', np.nan)
        prob_val = last.get('probability', 0)

        if verbose:
            # Detailed filter breakdown
            filters = []

            # 1. Z-score filter
            if z_val < -ZSCORE_LONG_THRESHOLD:
                filters.append(f"Z={z_val:.2f}✅(<-{ZSCORE_LONG_THRESHOLD})")
                direction = "LONG"
            elif z_val > ZSCORE_SHORT_THRESHOLD:
                filters.append(f"Z={z_val:.2f}✅(>{ZSCORE_SHORT_THRESHOLD})")
                direction = "SHORT"
            else:
                # Show which threshold is closer to help user understand what's needed
                if z_val < 0:
                    # Negative Z-score but not extreme enough for LONG
                    filters.append(f"Z={z_val:.2f}❌(need Z<-{ZSCORE_LONG_THRESHOLD} for LONG)")
                elif z_val > 0:
                    # Positive Z-score but not extreme enough for SHORT
                    filters.append(f"Z={z_val:.2f}❌(need Z>{ZSCORE_SHORT_THRESHOLD} for SHORT)")
                else:
                    # Z-score near zero (no opportunity)
                    filters.append(f"Z={z_val:.2f}❌(too neutral, need |Z|>{ZSCORE_SHORT_THRESHOLD})")
                print(f"{prefix}NO TRADE  {' '.join(filters)}")
                return

            # 2. RSI filter (only checked if Z-score passed)
            if direction == "LONG":
                if not np.isnan(rsi_val):
                    if rsi_val < RSI_LONG_MAX:
                        filters.append(f"RSI={rsi_val:.1f}✅(<{RSI_LONG_MAX})")
                    else:
                        filters.append(f"RSI={rsi_val:.1f}❌(need <{RSI_LONG_MAX})")
                        print(f"{prefix}NO TRADE  {' '.join(filters)}")
                        return
                else:
                    filters.append("RSI=N/A")
            else:  # SHORT
                if not np.isnan(rsi_val):
                    if rsi_val > RSI_SHORT_MIN:
                        filters.append(f"RSI={rsi_val:.1f}✅(>{RSI_SHORT_MIN})")
                    else:
                        filters.append(f"RSI={rsi_val:.1f}❌(need >{RSI_SHORT_MIN})")
                        print(f"{prefix}NO TRADE  {' '.join(filters)}")
                        return
                else:
                    filters.append("RSI=N/A")

            # 3. Hurst filter (if enabled)
            if HURST_FILTER and "hurst" in df.columns:
                hurst_val = last.get("hurst", np.nan)
                if not np.isnan(hurst_val):
                    if hurst_val < 0.5:
                        filters.append(f"Hurst={hurst_val:.2f}✅(<0.5)")
                    else:
                        filters.append(f"Hurst={hurst_val:.2f}❌(need <0.5)")
                        print(f"{prefix}NO TRADE  {' '.join(filters)}")
                        return

            # 4. Probability filter
            threshold = BUY_THRESHOLD + 0.10 if direction == "LONG" else BUY_THRESHOLD - 0.10
            if prob_val > 0:
                if prob_val > threshold:
                    filters.append(f"Prob={prob_val:.3f}✅(>{threshold:.2f})")
                else:
                    filters.append(f"Prob={prob_val:.3f}❌(need >{threshold:.2f})")
                    print(f"{prefix}NO TRADE  {' '.join(filters)}")
                    return
            else:
                # Probability not evaluated (previous filter failed or no extended price)
                filters.append("Prob=N/A⏭️")

            # 5. Ensemble voting
            n_agree = int(last.get('n_agree', 0))
            if n_agree > 0:
                if n_agree >= MIN_AGREE:
                    filters.append(f"Vote={n_agree}/3✅(>={MIN_AGREE})")
                else:
                    filters.append(f"Vote={n_agree}/3❌(need >={MIN_AGREE})")
                    print(f"{prefix}NO TRADE  {' '.join(filters)}")
                    return
            else:
                filters.append("Vote=N/A⏭️")

            print(f"{prefix}NO TRADE  {' '.join(filters)}")
        else:
            # Simple output (non-verbose)
            print(f"{prefix}NO TRADE  Z={z_val:.2f}")
