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

try:
    from config import ADX_MAX_THRESHOLD
except ImportError:
    ADX_MAX_THRESHOLD = 25.0

try:
    from config import USE_SCORING_SYSTEM, MIN_SCORE_THRESHOLD
except ImportError:
    USE_SCORING_SYSTEM = False
    MIN_SCORE_THRESHOLD = 70.0

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


def calculate_signal_score(
    row: pd.Series,
    direction: str,
    model_probs: list,
    z_threshold: float,
) -> float:
    """
    Calculate a continuous signal score (0-100) instead of binary pass/fail.

    Higher score = stronger signal.
    Entry threshold: score > 70 (configurable via MIN_SCORE_THRESHOLD).

    Score components:
    - Z-score strength (30 points): How extended is the price?
    - RSI confirmation (15 points): Is momentum confirming extremity?
    - Model confidence (30 points): What does ML ensemble predict?
    - Regime bonus (10 points): Is market ranging (ideal for mean reversion)?
    - Volume confirmation (10 points): Is volume elevated?
    - VWAP distance (5 points): How far from fair value?

    Args:
        row: DataFrame row with all features
        direction: "LONG" or "SHORT"
        model_probs: List of probabilities from individual models [p1, p2, p3]
        z_threshold: Z-score threshold for this direction

    Returns:
        Score from 0-100
    """
    score = 0.0

    # 1. Z-score strength (0-25 points)
    # More extreme Z-score = higher score
    z = abs(row.get('zscore_20', 0))
    if direction == "SHORT":
        # For SHORT: Z > 2.0 gets points
        # Scale: Z=2.0 → 0pts, Z=3.0 → 15pts, Z=4.0 → 25pts
        if z > z_threshold:
            score += min(25, (z - z_threshold) * 15)
    else:  # LONG
        # For LONG: Z < -2.5 gets points (more strict)
        # Scale: Z=-2.5 → 0pts, Z=-3.5 → 12pts, Z=-4.5 → 24pts
        if z > z_threshold:
            score += min(25, (z - z_threshold) * 12)

    # 2. RSI confirmation (0-15 points)
    # RSI extreme + aligned with Z-score = good
    rsi = row.get('rsi_14', 50)
    if not np.isnan(rsi):
        if direction == "SHORT" and rsi > RSI_SHORT_MIN:
            # More overbought = better SHORT signal
            # Scale: RSI=55 → 0pts, RSI=70 → 7.5pts, RSI=80 → 12.5pts
            score += min(15, (rsi - RSI_SHORT_MIN) * 0.5)
        elif direction == "LONG" and rsi < RSI_LONG_MAX:
            # More oversold = better LONG signal
            # Scale: RSI=45 → 0pts, RSI=30 → 7.5pts, RSI=20 → 12.5pts
            score += min(15, (RSI_LONG_MAX - rsi) * 0.5)

    # 3. Model confidence (0-40 points) — INCREASED WEIGHT
    # Average probability from ensemble
    if len(model_probs) > 0:
        avg_prob = np.mean(model_probs)
        # Map prob 0.5-1.0 to 0-40 points
        # Scale: prob=0.50 → 0pts, prob=0.65 → 12pts, prob=0.80 → 24pts
        score += max(0, (avg_prob - 0.5) * 80)

    # 4. Regime bonus (0-10 points)
    # Ranging market (ADX < 30) = ideal for mean reversion
    adx = row.get('adx_14', 50)
    if not np.isnan(adx):
        if adx < 30:
            score += 10
        elif adx < 40:
            # Partial credit for moderate trend
            score += 5

    # 5. Volume confirmation (0-5 points) — REDUCED WEIGHT
    # High volume = more conviction
    rel_volume = row.get('relative_volume', 1.0)
    if not np.isnan(rel_volume) and rel_volume > 1.2:
        score += min(5, (rel_volume - 1.0) * 5)

    # 6. VWAP distance bonus (0-5 points)
    # Far from VWAP = more reversion potential
    vwap_dist = abs(row.get('vwap_distance', 0))
    if not np.isnan(vwap_dist) and vwap_dist > 0.02:  # >2% from VWAP
        score += min(5, vwap_dist * 100)

    return min(100, score)


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
    use_scoring: bool = USE_SCORING_SYSTEM,
    min_score: float = MIN_SCORE_THRESHOLD,
    adx_max: float = ADX_MAX_THRESHOLD,
) -> pd.DataFrame:
    """
    Generate trading signals for every bar in the dataframe.

    Logic (when use_scoring=True):
    1. Check if |Z| > threshold (price is extended)
    2. Apply hard filters:
       - ADX filter: skip if ADX > adx_max (trending market)
       - Hurst filter: skip if Hurst >= 0.5 (if enabled)
       - RSI filter: LONG only if RSI < rsi_long_max, SHORT only if RSI > rsi_short_min
    3. Calculate signal score (0-100) based on:
       - Z-score strength
       - RSI confirmation
       - Model confidence
       - Market regime
       - Volume
       - VWAP distance
    4. Entry if score > min_score (default 70)

    Logic (when use_scoring=False, legacy mode):
    1. Check if |Z| > threshold
    2. Apply regime filters
    3. Ask ensemble: probability of reversion?
    4. Directional thresholds (SHORT-friendly):
       - SHORT: prob > (buy_threshold - short_bias) AND >= min_agree agree
       - LONG:  prob > (buy_threshold + short_bias) AND >= min_agree agree

    Args:
        df: DataFrame with features computed (must include zscore_col).
        model: EnsembleModel instance (trained).
        zscore_col: Z-score column to use for entry detection.
        z_long_threshold: Z-score threshold for LONG entries.
        z_short_threshold: Z-score threshold for SHORT entries.
        buy_threshold: Base probability threshold for entry (legacy mode).
        min_agree: Minimum number of models that must agree (legacy mode).
        feature_cols: Feature columns for model input.
        short_bias: Reduces threshold for SHORT, increases for LONG (legacy mode).
        hurst_filter: If True, skip entries when Hurst >= 0.5.
        rsi_long_max: LONG signals only when RSI is below this value.
        rsi_short_min: SHORT signals only when RSI is above this value.
        long_only: If True, only generate LONG signals (spot trading).
        use_scoring: If True, use 0-100 scoring system. If False, use legacy voting.
        min_score: Minimum score (0-100) required for entry (when use_scoring=True).
        adx_max: Maximum ADX value for entry (higher ADX = trending, bad for mean reversion).

    Returns:
        DataFrame with added columns: signal, probability, n_agree, direction, score (if use_scoring=True).
    """
    df = df.copy()

    long_threshold = buy_threshold + short_bias
    short_threshold = buy_threshold - short_bias

    # Initialize signal columns
    df["signal"] = "NO TRADE"
    df["probability"] = 0.0
    df["n_agree"] = 0
    df["signal_direction"] = 0  # 1=long, -1=short
    if use_scoring:
        df["score"] = 0.0

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
        row = df.loc[idx]

        # Hard filter 1: ADX (trending market filter)
        # Mean reversion doesn't work in strong trends
        if "adx_14" in df.columns:
            adx_val = row.get("adx_14", 0)
            if not np.isnan(adx_val) and adx_val > adx_max:
                continue

        # Hard filter 2: Hurst (if enabled)
        if hurst_filter and "hurst" in df.columns:
            hurst_val = row.get("hurst", 0)
            if not np.isnan(hurst_val) and hurst_val >= 0.5:
                continue

        rsi_val = row.get("rsi_14", np.nan)

        if z < -z_long_threshold:
            # LONG candidate
            # Hard filter 3: RSI (only enter LONG when oversold)
            if not np.isnan(rsi_val) and rsi_val > rsi_long_max:
                continue

            if use_scoring:
                # NEW: Scoring system
                model_probs = [individual[name][i] for name in individual]
                score = calculate_signal_score(row, "LONG", model_probs, z_long_threshold)
                df.loc[idx, "score"] = score

                if score >= min_score:
                    df.loc[idx, "signal"] = "LONG"
                    df.loc[idx, "signal_direction"] = 1
            else:
                # LEGACY: Voting system
                n_agree_long = sum(
                    1 for name in individual
                    if individual[name][i] > long_threshold
                )
                df.loc[idx, "n_agree"] = n_agree_long
                if prob > long_threshold and n_agree_long >= min_agree:
                    df.loc[idx, "signal"] = "LONG"
                    df.loc[idx, "signal_direction"] = 1

        elif z > z_short_threshold and not long_only:
            # SHORT candidate
            # Hard filter 3: RSI (only enter SHORT when overbought)
            if not np.isnan(rsi_val) and rsi_val < rsi_short_min:
                continue

            if use_scoring:
                # NEW: Scoring system
                model_probs = [individual[name][i] for name in individual]
                score = calculate_signal_score(row, "SHORT", model_probs, z_short_threshold)
                df.loc[idx, "score"] = score

                if score >= min_score:
                    df.loc[idx, "signal"] = "SHORT"
                    df.loc[idx, "signal_direction"] = -1
            else:
                # LEGACY: Voting system
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
        signal_str = f"{prefix}{last['signal']:<8} "

        # Show score if available (new scoring system)
        if "score" in last and last.get("score", 0) > 0:
            signal_str += f"score={last['score']:.1f}/100 "
        else:
            # Legacy: show probability and votes
            signal_str += f"prob={last['probability']:.4f} "
            signal_str += f"agree={int(last['n_agree'])}/3 "

        signal_str += f"Z={last.get('zscore_50', 0):.2f}"
        print(signal_str)
    else:
        # Show basic NO TRADE with Z-score
        z_val = last.get('zscore_50', 0)
        rsi_val = last.get('rsi_14', np.nan)
        prob_val = last.get('probability', 0)
        score_val = last.get('score', 0)
        use_scoring = score_val > 0 or USE_SCORING_SYSTEM

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

            # 4. ADX filter (trend strength)
            if "adx_14" in df.columns:
                adx_val = last.get("adx_14", 0)
                if not np.isnan(adx_val):
                    if adx_val < ADX_MAX_THRESHOLD:
                        filters.append(f"ADX={adx_val:.1f}✅(<{ADX_MAX_THRESHOLD})")
                    else:
                        filters.append(f"ADX={adx_val:.1f}❌(need <{ADX_MAX_THRESHOLD}, trending market)")
                        print(f"{prefix}NO TRADE  {' '.join(filters)}")
                        return

            # 5. Scoring or probability/voting
            if use_scoring and score_val > 0:
                # NEW: Show score
                if score_val >= MIN_SCORE_THRESHOLD:
                    filters.append(f"Score={score_val:.1f}/100✅(>={MIN_SCORE_THRESHOLD})")
                else:
                    filters.append(f"Score={score_val:.1f}/100❌(need >={MIN_SCORE_THRESHOLD})")
                    print(f"{prefix}NO TRADE  {' '.join(filters)}")
                    return
            else:
                # LEGACY: Probability + voting
                threshold = BUY_THRESHOLD + 0.10 if direction == "LONG" else BUY_THRESHOLD - 0.10
                if prob_val > 0:
                    if prob_val > threshold:
                        filters.append(f"Prob={prob_val:.3f}✅(>{threshold:.2f})")
                    else:
                        filters.append(f"Prob={prob_val:.3f}❌(need >{threshold:.2f})")
                        print(f"{prefix}NO TRADE  {' '.join(filters)}")
                        return
                else:
                    filters.append("Prob=N/A⏭️")

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
