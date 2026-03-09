"""
config.py
Central configuration for breakout momentum trading system.
Optimized for explosive moves in BNB, SOL, AVAX, and volatile altcoins.

Combines 3 complementary strategies:
1. Volatility Breakout (Donchian + ATR expansion)
2. Volume Surge Breakout (Price + volume confirmation)
3. Range Compression → Expansion (Coiling → explosion)
"""

# ── Data ─────────────────────────────────────────────────────
TIMEFRAME = "1h"
DAYS = 1825  # 3 years
DEFAULT_SYMBOLS = [
    "ETH/USDT",
    "BNB/USDT",
    # "NEAR/USDT",


    # no muy buenos
    # "APE/USDT",
    # "ATOM/USDT",
    # "SHIB/USDT",
    # "PEPE/USDT",
    # "FTM/USDT",
    # "POL/USDT",
    # "SOL/USDT",
    # "AVAX/USDT",

    # Test candidates
]

# ── Breakout Detection ───────────────────────────────────────
BREAKOUT_PERIOD = 20                # Donchian channel lookback
BREAKOUT_THRESHOLD = 0.99           # Close must be >= 99% of high_N for valid breakout

# ── Strategy 1: Volatility Breakout ──────────────────────────
VOLATILITY_ENABLED = True
VOL_ATR_EXPANSION_MIN = 1.3         # ATR must be > 1.3x MA(ATR_20) ATR ahora / ATR avg 20 velas
VOL_ATR_MAX_PERCENTILE = 0.95       # Avoid extreme volatility
VOL_VOLUME_MIN = 1.2                # Min relative volume
VOL_PROBABILITY_THRESHOLD = 0.60    # ML threshold

# ── Strategy 2: Volume Surge Breakout ────────────────────────
VOLUME_SURGE_ENABLED = True
SURGE_VOLUME_MIN = 2.0              # Volume must be > 2x average
SURGE_VOLUME_ZSCORE_MIN = 2.0       # Volume Z-score > 2 (unusual activity)
SURGE_RSI_MIN = 50.0                # RSI > 50 (momentum)
SURGE_RSI_MAX = 75.0                # RSI < 75 (not extreme)
SURGE_PROBABILITY_THRESHOLD = 0.65  # Higher ML threshold (quality over quantity)

# ── Strategy 3: Range Compression → Expansion ────────────────
COMPRESSION_ENABLED = True
COMP_RANGE_THRESHOLD = 0.6          # range_20 / range_50 < 0.6 (tight consolidation)
COMP_BB_WIDTH_PERCENTILE = 0.20     # BB width < 20th percentile
COMP_ADX_MAX = 20.0                 # ADX < 20 (not trending = coiling)
COMP_ATR_EXPANSION_MIN = 1.3        # ATR must expand after consolidation
COMP_PROBABILITY_THRESHOLD = 0.55   # Lower threshold (early in move)

# ── Risk Management ─────────────────────────────────────────
STOP_LOSS_ATR_MULT = 2.0            # Wider SL for volatile breakouts
TAKE_PROFIT_PCT = 0.05              # 5% TP (or use RR)
TAKE_PROFIT_RR = 1.8                # Risk:Reward ratio
USE_PERCENTAGE_TP = False           # If True, use PCT; else use RR

HOLD_TIMEOUT = 48                   # Max 48h (2 days) - breakouts are fast

# Trailing stop (optional)
USE_TRAILING_STOP = True
TRAILING_STOP_ATR_MULT = 2.5        # Lock in profits

# Target labeling (for ML training)
TARGET_TP_PCT = 0.04                # 4% TP for target labeling
TARGET_SL_ATR_MULT = 2.5            # 2.5 ATR SL for target labeling
TARGET_HORIZON = 24                 # Candles to simulate forward

# ── Entry Filters ────────────────────────────────────────────
REQUIRE_VOLUME_CONFIRMATION = True  # All strategies require volume > threshold
REQUIRE_BTC_ALIGNMENT = False       # If True, require BTC trending up for LONG
MIN_AGREE = 2                       # Min models that must agree (ensemble voting)

# ── Trading Mode ─────────────────────────────────────────────
LONG_ONLY = False                   # True = spot, False = futures (LONG+SHORT)

# ── Position Sizing ──────────────────────────────────────────
POSITION_PCT = 0.15                 # 15% per trade
MAX_POSITIONS = 3
INITIAL_CAPITAL = 10_000.0
FEE_PCT = 0.001                     # 0.1% per side

# ── Model ────────────────────────────────────────────────────
BUY_THRESHOLD = 0.60                # Base ML probability threshold
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
# test = 1 - train - val = 0.15

# ── Optuna ───────────────────────────────────────────────────
OPTUNA_TRIALS = 30
OPTUNA_CV_SPLITS = 5