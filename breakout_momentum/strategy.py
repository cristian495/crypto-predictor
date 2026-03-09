"""
strategy.py
Entry/exit logic for breakout momentum trading.
Combines 3 complementary breakout strategies with ML signals.

Strategy 1: Volatility Breakout (Donchian + ATR expansion + volume)
Strategy 2: Volume Surge Breakout (price breakout + 2x volume + RSI momentum)
Strategy 3: Range Compression → Expansion (tight range + BB squeeze + ADX low → breakout)
"""

import numpy as np
import pandas as pd

from features import FEATURE_COLS
from config import (
    BUY_THRESHOLD,
    MIN_AGREE,
    BREAKOUT_PERIOD,
    BREAKOUT_THRESHOLD,
    # Strategy 1: Volatility Breakout
    VOLATILITY_ENABLED,
    VOL_ATR_EXPANSION_MIN,
    VOL_ATR_MAX_PERCENTILE,
    VOL_VOLUME_MIN,
    VOL_PROBABILITY_THRESHOLD,
    # Strategy 2: Volume Surge
    VOLUME_SURGE_ENABLED,
    SURGE_VOLUME_MIN,
    SURGE_VOLUME_ZSCORE_MIN,
    SURGE_RSI_MIN,
    SURGE_RSI_MAX,
    SURGE_PROBABILITY_THRESHOLD,
    # Strategy 3: Range Compression
    COMPRESSION_ENABLED,
    COMP_RANGE_THRESHOLD,
    COMP_BB_WIDTH_PERCENTILE,
    COMP_ADX_MAX,
    COMP_ATR_EXPANSION_MIN,
    COMP_PROBABILITY_THRESHOLD,
    # Risk management
    STOP_LOSS_ATR_MULT,
    TAKE_PROFIT_PCT,
    TAKE_PROFIT_RR,
    USE_PERCENTAGE_TP,
    HOLD_TIMEOUT,
    USE_TRAILING_STOP,
    TRAILING_STOP_ATR_MULT,
    # Entry filters
    REQUIRE_VOLUME_CONFIRMATION,
    REQUIRE_BTC_ALIGNMENT,
    LONG_ONLY,
)


def generate_signals(
    df: pd.DataFrame,
    model,
    buy_threshold: float = BUY_THRESHOLD,
    min_agree: int = MIN_AGREE,
    feature_cols: list = FEATURE_COLS,
    long_only: bool = LONG_ONLY,
) -> pd.DataFrame:
    """
    Generate breakout momentum trading signals using 3 complementary strategies.

    Strategy logic:
    1. VOLATILITY BREAKOUT: Donchian breakout + ATR expansion + volume
    2. VOLUME SURGE: High breakout + 2x volume spike + RSI momentum
    3. RANGE COMPRESSION: Tight range → expansion + BB squeeze + low ADX

    Each strategy has its own ML probability threshold and filters.
    Final signal requires ML ensemble agreement (min_agree models).

    Args:
        df: DataFrame with features computed.
        model: Trained ensemble model.
        buy_threshold: Base ML probability threshold (adjusted per strategy).
        min_agree: Min number of models that must agree.
        feature_cols: Feature columns for model input.
        long_only: If True, only LONG trades.

    Returns:
        DataFrame with signal, probability, n_agree, signal_direction, strategy_name columns.
    """
    df = df.copy()

    # Initialize signal columns
    df["signal"] = "NO TRADE"
    df["probability"] = 0.0
    df["n_agree"] = 0
    df["signal_direction"] = 0  # 1=long, -1=short
    df["strategy_name"] = ""

    # Get required columns
    required = [
        "donchian_breakout_up", "donchian_breakout_down",
        "atr_expansion_rate", "atr_percentile",
        "relative_volume", "volume_zscore",
        "rsi_14", "adx",
        "range_compression", "bb_width_percentile",
        "close", "high", "low"
    ]
    if not all(col in df.columns for col in required):
        print("Warning: Missing required columns for strategy")
        return df

    # Find candidate rows (have all features, no NaN)
    candidates_mask = df[feature_cols].notna().all(axis=1)
    candidate_idx = df[candidates_mask].index
    if len(candidate_idx) == 0:
        return df

    # Get model predictions
    X_candidates = df.loc[candidate_idx, feature_cols].values
    avg_proba, _, _ = model.predict_with_voting(
        X_candidates, threshold=buy_threshold, min_agree=min_agree
    )
    individual = model.predict_proba_individual(X_candidates)

    df.loc[candidate_idx, "probability"] = avg_proba

    # Generate directional signals using each strategy
    for i, idx in enumerate(candidate_idx):
        prob = avg_proba[i]

        # Get current values
        breakout_up = df.loc[idx, "donchian_breakout_up"]
        breakout_down = df.loc[idx, "donchian_breakout_down"]
        atr_expansion = df.loc[idx, "atr_expansion_rate"]
        atr_percentile = df.loc[idx, "atr_percentile"]
        rel_volume = df.loc[idx, "relative_volume"]
        volume_zscore = df.loc[idx, "volume_zscore"]
        rsi = df.loc[idx, "rsi_14"]
        adx = df.loc[idx, "adx"]
        range_comp = df.loc[idx, "range_compression"]
        bb_width_pct = df.loc[idx, "bb_width_percentile"]

        # Count model agreement
        n_agree_long = sum(
            1 for name in individual if individual[name][i] > buy_threshold
        )
        df.loc[idx, "n_agree"] = n_agree_long

        # ═══════════════════════════════════════════════════════════
        # STRATEGY 1: VOLATILITY BREAKOUT
        # ═══════════════════════════════════════════════════════════
        if VOLATILITY_ENABLED:
            # LONG: Donchian upward breakout + ATR expansion + volume
            if (
                breakout_up == 1 and
                atr_expansion >= VOL_ATR_EXPANSION_MIN and
                atr_percentile <= VOL_ATR_MAX_PERCENTILE and
                rel_volume >= VOL_VOLUME_MIN and
                prob >= VOL_PROBABILITY_THRESHOLD and
                n_agree_long >= min_agree
            ):
                df.loc[idx, "signal"] = "LONG"
                df.loc[idx, "signal_direction"] = 1
                df.loc[idx, "strategy_name"] = "VOLATILITY_BREAKOUT"
                continue  # Skip other strategies once signal found

            # SHORT: Donchian downward breakout + ATR expansion + volume
            if (
                not long_only and
                breakout_down == 1 and
                atr_expansion >= VOL_ATR_EXPANSION_MIN and
                atr_percentile <= VOL_ATR_MAX_PERCENTILE and
                rel_volume >= VOL_VOLUME_MIN and
                prob >= VOL_PROBABILITY_THRESHOLD and
                n_agree_long >= min_agree
            ):
                df.loc[idx, "signal"] = "SHORT"
                df.loc[idx, "signal_direction"] = -1
                df.loc[idx, "strategy_name"] = "VOLATILITY_BREAKOUT"
                continue

        # ═══════════════════════════════════════════════════════════
        # STRATEGY 2: VOLUME SURGE BREAKOUT
        # ═══════════════════════════════════════════════════════════
        if VOLUME_SURGE_ENABLED:
            # LONG: Price at high + huge volume spike + RSI momentum
            if (
                breakout_up == 1 and
                rel_volume >= SURGE_VOLUME_MIN and
                volume_zscore >= SURGE_VOLUME_ZSCORE_MIN and
                rsi >= SURGE_RSI_MIN and
                rsi <= SURGE_RSI_MAX and
                prob >= SURGE_PROBABILITY_THRESHOLD and
                n_agree_long >= min_agree
            ):
                df.loc[idx, "signal"] = "LONG"
                df.loc[idx, "signal_direction"] = 1
                df.loc[idx, "strategy_name"] = "VOLUME_SURGE"
                continue

            # SHORT: Price at low + huge volume spike + RSI momentum
            if (
                not long_only and
                breakout_down == 1 and
                rel_volume >= SURGE_VOLUME_MIN and
                volume_zscore >= SURGE_VOLUME_ZSCORE_MIN and
                rsi >= (100 - SURGE_RSI_MAX) and  # Inverse RSI for shorts
                rsi <= (100 - SURGE_RSI_MIN) and
                prob >= SURGE_PROBABILITY_THRESHOLD and
                n_agree_long >= min_agree
            ):
                df.loc[idx, "signal"] = "SHORT"
                df.loc[idx, "signal_direction"] = -1
                df.loc[idx, "strategy_name"] = "VOLUME_SURGE"
                continue

        # ═══════════════════════════════════════════════════════════
        # STRATEGY 3: RANGE COMPRESSION → EXPANSION
        # ═══════════════════════════════════════════════════════════
        if COMPRESSION_ENABLED:
            # LONG: Tight range + BB squeeze + low ADX + upward breakout
            if (
                breakout_up == 1 and
                range_comp <= COMP_RANGE_THRESHOLD and
                bb_width_pct <= COMP_BB_WIDTH_PERCENTILE and
                adx <= COMP_ADX_MAX and
                atr_expansion >= COMP_ATR_EXPANSION_MIN and
                prob >= COMP_PROBABILITY_THRESHOLD and
                n_agree_long >= min_agree
            ):
                df.loc[idx, "signal"] = "LONG"
                df.loc[idx, "signal_direction"] = 1
                df.loc[idx, "strategy_name"] = "RANGE_COMPRESSION"
                continue

            # SHORT: Tight range + BB squeeze + low ADX + downward breakout
            if (
                not long_only and
                breakout_down == 1 and
                range_comp <= COMP_RANGE_THRESHOLD and
                bb_width_pct <= COMP_BB_WIDTH_PERCENTILE and
                adx <= COMP_ADX_MAX and
                atr_expansion >= COMP_ATR_EXPANSION_MIN and
                prob >= COMP_PROBABILITY_THRESHOLD and
                n_agree_long >= min_agree
            ):
                df.loc[idx, "signal"] = "SHORT"
                df.loc[idx, "signal_direction"] = -1
                df.loc[idx, "strategy_name"] = "RANGE_COMPRESSION"
                continue

    return df


def compute_exit_levels(
    entry_price: float,
    direction: int,
    atr: float,
    stop_loss_atr_mult: float = STOP_LOSS_ATR_MULT,
    take_profit_rr: float = TAKE_PROFIT_RR,
    take_profit_pct: float = TAKE_PROFIT_PCT,
    use_percentage_tp: bool = USE_PERCENTAGE_TP,
) -> dict:
    """
    Compute TP and SL levels using ATR-based stops.

    Args:
        entry_price: Entry price.
        direction: 1 for long, -1 for short.
        atr: Current ATR value (absolute, not normalized).
        stop_loss_atr_mult: ATR multiplier for stop loss.
        take_profit_rr: Risk:Reward ratio for take profit.
        take_profit_pct: Percentage-based TP (if use_percentage_tp=True).
        use_percentage_tp: If True, use percentage TP; else use RR-based TP.

    Returns:
        dict with 'stop_loss' and 'take_profit' price levels.
    """
    stop_distance = atr * stop_loss_atr_mult

    if use_percentage_tp:
        tp_distance = entry_price * take_profit_pct
    else:
        tp_distance = stop_distance * take_profit_rr

    if direction == 1:  # LONG
        return {
            "stop_loss": entry_price - stop_distance,
            "take_profit": entry_price + tp_distance,
        }
    else:  # SHORT
        return {
            "stop_loss": entry_price + stop_distance,
            "take_profit": entry_price - tp_distance,
        }


def check_exit(
    position: dict,
    current_high: float,
    current_low: float,
    current_close: float,
    current_atr: float,
    bars_held: int,
    hold_timeout: int = HOLD_TIMEOUT,
    use_trailing: bool = USE_TRAILING_STOP,
    trailing_atr_mult: float = TRAILING_STOP_ATR_MULT,
) -> tuple:
    """
    Check if a position should be exited.

    Exit conditions (checked in order):
    1. Stop loss hit
    2. Take profit hit
    3. Trailing stop hit (if enabled and position in profit)
    4. Hold timeout exceeded

    Args:
        position: Position dict with direction, stop_loss, take_profit, entry_price.
        current_high: Current candle high.
        current_low: Current candle low.
        current_close: Current candle close.
        current_atr: Current ATR value.
        bars_held: Number of bars position has been held.
        hold_timeout: Max bars to hold.
        use_trailing: Enable trailing stops.
        trailing_atr_mult: ATR multiplier for trailing stop distance.

    Returns:
        (should_exit: bool, exit_price: float, exit_reason: str, updated_position: dict)
    """
    direction = position["direction"]
    sl = position["stop_loss"]
    tp = position["take_profit"]
    entry = position["entry_price"]

    # Update trailing stop if enabled
    if use_trailing:
        trailing_distance = current_atr * trailing_atr_mult

        if direction == 1:  # LONG
            # Update trailing stop if price moved favorably
            new_trailing_sl = current_close - trailing_distance
            if "trailing_stop" not in position:
                position["trailing_stop"] = sl
            position["trailing_stop"] = max(position["trailing_stop"], new_trailing_sl)

        else:  # SHORT
            # Update trailing stop if price moved favorably
            new_trailing_sl = current_close + trailing_distance
            if "trailing_stop" not in position:
                position["trailing_stop"] = sl
            position["trailing_stop"] = min(position["trailing_stop"], new_trailing_sl)

    # Check exits
    if direction == 1:  # LONG
        # Stop loss
        if current_low <= sl:
            return True, sl, "stop_loss", position

        # Take profit
        if current_high >= tp:
            return True, tp, "take_profit", position

        # Trailing stop
        if use_trailing and "trailing_stop" in position:
            if current_low <= position["trailing_stop"]:
                return True, position["trailing_stop"], "trailing_stop", position

    else:  # SHORT
        # Stop loss
        if current_high >= sl:
            return True, sl, "stop_loss", position

        # Take profit
        if current_low <= tp:
            return True, tp, "take_profit", position

        # Trailing stop
        if use_trailing and "trailing_stop" in position:
            if current_high >= position["trailing_stop"]:
                return True, position["trailing_stop"], "trailing_stop", position

    # Timeout
    if bars_held >= hold_timeout:
        return True, current_close, "timeout", position

    return False, 0.0, "", position


def print_current_signals(df: pd.DataFrame, symbol: str = ""):
    """Print the latest signal for a symbol."""
    last = df.iloc[-1]

    prefix = f"  {symbol:<12}" if symbol else "  "

    if last["signal"] != "NO TRADE":
        strategy = last.get("strategy_name", "UNKNOWN")
        prob = last.get("probability", 0.0)
        n_agree = int(last.get("n_agree", 0))
        rsi = last.get("rsi_14", 0)
        adx = last.get("adx", 0)
        rel_vol = last.get("relative_volume", 0)

        print(
            f"{prefix}{last['signal']:<8} "
            f"[{strategy[:12]:<12}] "
            f"prob={prob:.4f} "
            f"agree={n_agree}/3 "
            f"RSI={rsi:.1f} ADX={adx:.1f} Vol={rel_vol:.2f}x"
        )
    else:
        # Show current market state even with no trade
        breakout_up = last.get("donchian_breakout_up", 0)
        breakout_down = last.get("donchian_breakout_down", 0)
        state = "↑BreakUp" if breakout_up else ("↓BreakDn" if breakout_down else "Range")
        adx = last.get("adx", 0)
        print(f"{prefix}NO TRADE  {state} ADX={adx:.1f}")