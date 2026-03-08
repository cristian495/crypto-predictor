"""
config.py
Central configuration for the trend following trading system.
Optimized for trending assets like BNB, ETH, BTC.
"""

# ── Data ─────────────────────────────────────────────────────
TIMEFRAME = "1h"  # Options: "1h", "4h" (ChatGPT recommends 4h for ETH)
DAYS = 1095       # History to download (3 years)
DEFAULT_SYMBOLS = [
    "POL/USDT",
    "BTC/USDT",


    # NO VA BIEN
    # "ETH/USDT",
    # "SOL/USDT",

    # Good trend followers (add later)
    # "LINK/USDT",
    # "AVAX/USDT",
    # "MATIC/USDT",
]

# ── Trend Following ──────────────────────────────────────────
# EMA crossover settings
EMA_FAST = 20                       # Fast EMA for trend detection
EMA_SLOW = 50                       # Slow EMA for trend detection
EMA_TREND = 200                     # Long-term trend filter

# Breakout settings
BREAKOUT_PERIOD = 20                # Lookback for highs/lows
VOLUME_THRESHOLD = 1.3              # Min volume multiplier for valid breakout

# Momentum filters
ADX_MIN = 20.0                      # Min ADX for trend strength (moderate to strong trends)
RSI_LONG_MIN = 45.0                 # LONG only when RSI > 45 (avoid weak momentum)
RSI_LONG_MAX = 75.0                 # LONG only when RSI < 75 (avoid extreme overbought)
RSI_SHORT_MIN = 25.0                # SHORT only when RSI > 25 (avoid extreme oversold)
RSI_SHORT_MAX = 55.0                # SHORT only when RSI < 55 (avoid weak momentum)

# ATR-based volatility filter
ATR_MIN_PERCENTILE = 0.20           # Only trade when ATR > 20th percentile (avoid low vol)
ATR_MAX_PERCENTILE = 0.95           # Avoid extreme volatility (> 95th percentile)

# ── Risk Management ─────────────────────────────────────────
STOP_LOSS_ATR_MULT = 1.5        # Cambiar de 2.0
TAKE_PROFIT_RR = 1.8            # Cambiar de 2.5
TRAILING_STOP_ATR_MULT = 2.5    # Cambiar de 3.0
USE_TRAILING_STOP = True            # Enable trailing stops for trend following

HOLD_TIMEOUT = 72                   # Max candles to hold (72h at 1h, 12 days at 4h)
TARGET_SL_ATR_MULT = 2.5            # ATR multiplier for ML target labeling
                                    # (slightly wider than trading SL)

# ── Timeframe-specific adjustments ──────────────────────────
# For 4h timeframe (ChatGPT recommendation for ETH):
# - Trends last longer → increase HOLD_TIMEOUT
# - Less noise → can use tighter thresholds
# - Recommended: TIMEFRAME="4h", HOLD_TIMEOUT=48 (8 days)

POSITION_PCT = 0.15                 # 15% of capital per trade
MAX_POSITIONS = 3
INITIAL_CAPITAL = 10_000.0

# ── Entry Filters ────────────────────────────────────────────
REQUIRE_TREND_ALIGNMENT = True      # Require price > EMA200 for LONG, < EMA200 for SHORT
REQUIRE_EMA_CROSSOVER = False       # If True, only enter on fresh EMA crossovers
REQUIRE_BREAKOUT = False            # If True, require breakout + trend alignment
REQUIRE_VOLUME_CONFIRMATION = True  # Require volume > threshold

# ── Trading Mode ─────────────────────────────────────────────
LONG_ONLY = False                   # True = spot (LONG only), False = futures (LONG+SHORT)

# ── Fees ─────────────────────────────────────────────────────
FEE_PCT = 0.001                     # 0.1% per side (Binance spot)

# ── Model ────────────────────────────────────────────────────
BUY_THRESHOLD = 0.60                # Min probability for entry signal
MIN_AGREE = 2                       # 2/3 ensemble models must agree
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
# test = 1 - train - val = 0.15

# ── Optuna ───────────────────────────────────────────────────
OPTUNA_TRIALS = 30
OPTUNA_CV_SPLITS = 5