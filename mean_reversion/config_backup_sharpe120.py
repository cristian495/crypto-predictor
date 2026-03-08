"""
config_backup_sharpe120.py
BACKUP — configuración que dio Sharpe 1.20 / Return +4.20% / Max DD -2.73%
5 símbolos, LONG+SHORT, Z>2.0

Para restaurar: copiar estos valores a config.py
"""

# ── Data ─────────────────────────────────────────────────────
TIMEFRAME = "1h"
DAYS = 1095
DEFAULT_SYMBOLS = [
    "ETH/USDT",
    "DOGE/USDT",
    "LINK/USDT",
    "XRP/USDT",
    "MATIC/USDT",
]

# ── Mean Reversion ───────────────────────────────────────────
ZSCORE_WINDOWS = [20, 50, 100]
ZSCORE_ENTRY_THRESHOLD = 2.0
REVERSION_TARGET_PCT = 0.5
REVERSION_HORIZON = 24

# ── Risk Management ─────────────────────────────────────────
STOP_LOSS_PCT = 0.020              # 2.0%
TAKE_PROFIT_PCT = 0.050            # 5.0%
HOLD_TIMEOUT = 48
TARGET_SL_PCT = 0.035
POSITION_PCT = 0.15
MAX_POSITIONS = 3
INITIAL_CAPITAL = 10_000.0

# ── Entry Filters ────────────────────────────────────────────
HURST_FILTER = False
RSI_LONG_MAX = 45.0
RSI_SHORT_MIN = 55.0

# ── Trading Mode ─────────────────────────────────────────────
LONG_ONLY = False

# ── Fees ─────────────────────────────────────────────────────
FEE_PCT = 0.001

# ── Model ────────────────────────────────────────────────────
BUY_THRESHOLD = 0.60
MIN_AGREE = 2
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

# ── Optuna ───────────────────────────────────────────────────
OPTUNA_TRIALS = 30
OPTUNA_CV_SPLITS = 5

# ── Exit logic (strategy.py) ─────────────────────────────────
# check_exit: sale cuando current_zscore >= 0 (LONG) o <= 0 (SHORT)
# Esta es la lógica actual ANTES del experimento con exit_z = -0.5 / +0.5

# ── Resultados del backtest con esta configuración ───────────
# Sharpe:        1.20
# Return:        +4.20%
# Max DD:        -2.73%
# Calmar:        3.41
# Win rate:      55.11%
# Profit factor: 1.30
# Trades:        176 (40L / 136S)
# Expectancy:    0.2792%
# Per symbol:
#   DOGE:  39 trades  WR 64%  Total +28.35%
#   LINK:  41 trades  WR 51%  Total +12.69%
#   ETH:   53 trades  WR 55%  Total  +4.58%
#   XRP:   43 trades  WR 51%  Total  +3.52%
#   MATIC: (no incluido en ese run, Sharpe 1.84 solo)
