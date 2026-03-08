"""
strategy.py
Entry/exit logic for trend following trading.
Combines trend detection, momentum, and breakouts with ML signals.
"""

import numpy as np
import pandas as pd

from features import FEATURE_COLS
from config import (
    BUY_THRESHOLD,
    MIN_AGREE,
    EMA_FAST,
    EMA_SLOW,
    EMA_TREND,
    ADX_MIN,
    RSI_LONG_MIN,
    RSI_LONG_MAX,
    RSI_SHORT_MIN,
    RSI_SHORT_MAX,
    ATR_MIN_PERCENTILE,
    ATR_MAX_PERCENTILE,
    VOLUME_THRESHOLD,
    BREAKOUT_PERIOD,
    REQUIRE_TREND_ALIGNMENT,
    REQUIRE_EMA_CROSSOVER,
    REQUIRE_BREAKOUT,
    REQUIRE_VOLUME_CONFIRMATION,
    LONG_ONLY,
    STOP_LOSS_ATR_MULT,
    TAKE_PROFIT_RR,
    HOLD_TIMEOUT,
    USE_TRAILING_STOP,
    TRAILING_STOP_ATR_MULT,
)


def generate_signals(
    df: pd.DataFrame,
    model,
    buy_threshold: float = BUY_THRESHOLD,
    min_agree: int = MIN_AGREE,
    feature_cols: list = FEATURE_COLS,
    adx_min: float = ADX_MIN,
    rsi_long_min: float = RSI_LONG_MIN,
    rsi_long_max: float = RSI_LONG_MAX,
    rsi_short_min: float = RSI_SHORT_MIN,
    rsi_short_max: float = RSI_SHORT_MAX,
    atr_min_percentile: float = ATR_MIN_PERCENTILE,
    atr_max_percentile: float = ATR_MAX_PERCENTILE,
    volume_threshold: float = VOLUME_THRESHOLD,
    long_only: bool = LONG_ONLY,
    require_trend_alignment: bool = REQUIRE_TREND_ALIGNMENT,
    require_ema_crossover: bool = REQUIRE_EMA_CROSSOVER,
    require_breakout: bool = REQUIRE_BREAKOUT,
    require_volume: bool = REQUIRE_VOLUME_CONFIRMATION,
) -> pd.DataFrame:
    """
    Generate trend following trading signals.

    Strategy logic:
    1. Identify trend direction (EMA alignment, ADX strength)
    2. Apply momentum filters (RSI, DI+/DI-)
    3. Check volatility regime (ATR percentile)
    4. Optional: require breakout or EMA crossover
    5. Get ML probability for trend continuation
    6. Enter if all conditions met

    Args:
        df: DataFrame with features computed.
        model: Trained ensemble model.
        buy_threshold: Min ML probability for entry.
        min_agree: Min number of models that must agree.
        feature_cols: Feature columns for model input.
        adx_min: Min ADX for trend strength.
        rsi_long_min/max: RSI bounds for LONG entries.
        rsi_short_min/max: RSI bounds for SHORT entries.
        atr_min/max_percentile: Acceptable ATR range.
        volume_threshold: Min relative volume multiplier.
        long_only: If True, only LONG trades.
        require_trend_alignment: Require price aligned with EMA200.
        require_ema_crossover: Require fresh EMA crossover.
        require_breakout: Require breakout of recent high/low.
        require_volume: Require volume confirmation.

    Returns:
        DataFrame with signal, probability, n_agree, signal_direction columns.
    """
    df = df.copy()

    # Initialize signal columns
    df["signal"] = "NO TRADE"
    df["probability"] = 0.0
    df["n_agree"] = 0
    df["signal_direction"] = 0  # 1=long, -1=short

    # Get required columns
    required = [
        "ema_20", "ema_50", "ema_200", "adx", "rsi_14",
        "atr_percentile", "relative_volume", "di_plus", "di_minus",
        "close", "high", "low"
    ]
    if not all(col in df.columns for col in required):
        print("Warning: Missing required columns for strategy")
        return df

    # Pre-compute highs/lows for breakout detection
    df["high_20"] = df["high"].rolling(BREAKOUT_PERIOD).max()
    df["low_20"] = df["low"].rolling(BREAKOUT_PERIOD).min()

    # Detect EMA crossovers (fresh signals)
    df["ema_cross_bull"] = (
        (df["ema_20"] > df["ema_50"]) &
        (df["ema_20"].shift(1) <= df["ema_50"].shift(1))
    )
    df["ema_cross_bear"] = (
        (df["ema_20"] < df["ema_50"]) &
        (df["ema_20"].shift(1) >= df["ema_50"].shift(1))
    )

    # Find candidate rows (basic filters applied first)
    candidates_mask = (
        # Trend strength
        (df["adx"] >= adx_min) &
        # Volatility regime
        (df["atr_percentile"] >= atr_min_percentile) &
        (df["atr_percentile"] <= atr_max_percentile) &
        # Not NaN
        df[feature_cols].notna().all(axis=1)
    )

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

    # Generate directional signals
    for i, idx in enumerate(candidate_idx):
        prob = avg_proba[i]

        # Skip if probability too low
        if prob < buy_threshold:
            continue

        # Get current values
        ema_20 = df.loc[idx, "ema_20"]
        ema_50 = df.loc[idx, "ema_50"]
        ema_200 = df.loc[idx, "ema_200"]
        adx = df.loc[idx, "adx"]
        rsi = df.loc[idx, "rsi_14"]
        di_plus = df.loc[idx, "di_plus"]
        di_minus = df.loc[idx, "di_minus"]
        rel_volume = df.loc[idx, "relative_volume"]
        close = df.loc[idx, "close"]
        high_20 = df.loc[idx, "high_20"]
        low_20 = df.loc[idx, "low_20"]
        ema_cross_bull = df.loc[idx, "ema_cross_bull"]
        ema_cross_bear = df.loc[idx, "ema_cross_bear"]

        # ── LONG SIGNAL ──────────────────────────────────────

        long_conditions = True

        # Trend: EMAs aligned bullish
        if ema_20 <= ema_50:
            long_conditions = False

        # Trend alignment: price above 200 EMA
        if require_trend_alignment and close <= ema_200:
            long_conditions = False

        # Fresh crossover required
        if require_ema_crossover and not ema_cross_bull:
            long_conditions = False

        # Breakout: price breaking above recent high
        if require_breakout and close < high_20:
            long_conditions = False

        # Momentum: DI+ > DI- (bullish directional movement)
        if di_plus <= di_minus:
            long_conditions = False

        # RSI filter (avoid weak momentum or extreme overbought)
        if rsi < rsi_long_min or rsi > rsi_long_max:
            long_conditions = False

        # Volume confirmation
        if require_volume and rel_volume < volume_threshold:
            long_conditions = False

        # ML agreement
        n_agree_long = sum(
            1 for name in individual if individual[name][i] > buy_threshold
        )
        df.loc[idx, "n_agree"] = n_agree_long

        if long_conditions and n_agree_long >= min_agree:
            df.loc[idx, "signal"] = "LONG"
            df.loc[idx, "signal_direction"] = 1
            continue  # Skip SHORT check

        # ── SHORT SIGNAL ─────────────────────────────────────

        if long_only:
            continue

        short_conditions = True

        # Trend: EMAs aligned bearish
        if ema_20 >= ema_50:
            short_conditions = False

        # Trend alignment: price below 200 EMA
        if require_trend_alignment and close >= ema_200:
            short_conditions = False

        # Fresh crossover required
        if require_ema_crossover and not ema_cross_bear:
            short_conditions = False

        # Breakout: price breaking below recent low
        if require_breakout and close > low_20:
            short_conditions = False

        # Momentum: DI- > DI+ (bearish directional movement)
        if di_minus <= di_plus:
            short_conditions = False

        # RSI filter (avoid weak momentum or extreme oversold)
        if rsi < rsi_short_min or rsi > rsi_short_max:
            short_conditions = False

        # Volume confirmation
        if require_volume and rel_volume < volume_threshold:
            short_conditions = False

        # ML agreement
        n_agree_short = sum(
            1 for name in individual if individual[name][i] > buy_threshold
        )
        df.loc[idx, "n_agree"] = n_agree_short

        if short_conditions and n_agree_short >= min_agree:
            df.loc[idx, "signal"] = "SHORT"
            df.loc[idx, "signal_direction"] = -1

    return df


def compute_exit_levels(
    entry_price: float,
    direction: int,
    atr: float,
    stop_loss_atr_mult: float = STOP_LOSS_ATR_MULT,
    take_profit_rr: float = TAKE_PROFIT_RR,
) -> dict:
    """
    Compute TP and SL levels using ATR-based stops.

    Args:
        entry_price: Entry price.
        direction: 1 for long, -1 for short.
        atr: Current ATR value (absolute, not normalized).
        stop_loss_atr_mult: ATR multiplier for stop loss.
        take_profit_rr: Risk:Reward ratio for take profit.

    Returns:
        dict with 'stop_loss' and 'take_profit' price levels.
    """
    stop_distance = atr * stop_loss_atr_mult
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
        trend = "↑" if last.get("ema_20", 0) > last.get("ema_50", 0) else "↓"
        adx = last.get("adx", 0)
        rsi = last.get("rsi_14", 0)

        print(
            f"{prefix}{last['signal']:<8} "
            f"prob={last['probability']:.4f} "
            f"agree={int(last['n_agree'])}/3 "
            f"trend={trend} ADX={adx:.1f} RSI={rsi:.1f}"
        )
    else:
        trend = "↑" if last.get("ema_20", 0) > last.get("ema_50", 0) else "↓"
        adx = last.get("adx", 0)
        print(f"{prefix}NO TRADE  trend={trend} ADX={adx:.1f}")