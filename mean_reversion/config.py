"""
config.py
Central configuration for the mean reversion trading system.
All tunable parameters in one place.
"""

# ── Data ─────────────────────────────────────────────────────
TIMEFRAME = "1h"
DAYS = 1095
DEFAULT_SYMBOLS = [
    "ETH/USDT",
    "DOGE/USDT",
    "LINK/USDT",
    "XRP/USDT",
    
    ## no tan buenos
    # "POL/USDT",   # WR 41% — no hereda comportamiento de MATIC
    # "LTC/USDT",
    # "DOT/USDT",
    # "ATOM/USDT",
    # "ADA/USDT",
    # "BNB/USDT",   # AUC 0.49 — peor que random
    # "SOL/USDT",   # WR 24% en test — catastrófico
    # "IOTA/USDT",
    # "PAXG/USDT",  # oro en tendencia — no mean-reverts
    # "TRX/USDT",

    # "BTC/USDT",   # trending
]

# ── Mean Reversion ───────────────────────────────────────────
ZSCORE_WINDOWS = [20, 50, 100]
ZSCORE_ENTRY_THRESHOLD = 2.0       # kept for compatibility / target labeling
ZSCORE_LONG_THRESHOLD = 2.5        # LONG: needs stronger oversold (falls continue longer)
ZSCORE_SHORT_THRESHOLD = 2.0       # SHORT: standard threshold
REVERSION_TARGET_PCT = 0.5         # unused in new target — kept for compatibility
REVERSION_HORIZON = 24             # candles to simulate forward (24h at 1H)

# ── Risk Management ─────────────────────────────────────────
STOP_LOSS_PCT = 0.020              # 2.0% stop loss (trading)
TAKE_PROFIT_PCT = 0.050            # 5.0% take profit (RR 1:2.5)
HOLD_TIMEOUT = 48                  # max candles to hold (48h at 1H)
TARGET_SL_PCT = 0.035              # 3.5% SL used only for ML target labeling
                                   # wider than trading SL to account for crypto
                                   # volatility and get more positive labels
POSITION_PCT = 0.15                # 15% of capital per trade
MAX_POSITIONS = 3
INITIAL_CAPITAL = 10_000.0

# ── Entry Filters ────────────────────────────────────────────
HURST_FILTER = False               # disabled — Z>2.0 + RSI filter is enough
RSI_LONG_MAX = 45.0                # LONG only when RSI < 45 (confirmed oversold)
RSI_SHORT_MIN = 55.0               # SHORT only when RSI > 55 (confirmed overbought)
ADX_MAX_THRESHOLD = 40.0           # Max ADX for mean reversion entry (40 = very strong trend)
                                   # Note: crypto markets trend more than traditional markets
                                   # 25 is too restrictive (blocks 50%+ of opportunities)
                                   # 40 blocks only the strongest trends

# ── Scoring System ───────────────────────────────────────────
USE_SCORING_SYSTEM = False         # If True, use 0-100 scoring instead of binary votes
                                   # NOTE: Scoring system is implemented but performs worse than
                                   # legacy voting system (42% WR vs 62% WR). Leave disabled.
                                   # The real improvements come from ADX/VWAP features, not scoring.
MIN_SCORE_THRESHOLD = 50.0         # Minimum score (0-100) to enter trade
                                   # 70 was too high (0 trades)
                                   # 50 is balanced for crypto volatility

# ── Trading Mode ─────────────────────────────────────────────
LONG_ONLY = False                   # True = spot (LONG only), False = futures (LONG+SHORT)

# ── Fees ─────────────────────────────────────────────────────
FEE_PCT = 0.001                    # 0.1% per side (Binance spot)

# ── Model ────────────────────────────────────────────────────
BUY_THRESHOLD = 0.60               # min probability for entry signal
MIN_AGREE = 2                      # 2/3 ensemble models must agree
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
# test = 1 - train - val = 0.15

# ── Optuna ───────────────────────────────────────────────────
OPTUNA_TRIALS = 30
OPTUNA_CV_SPLITS = 5
