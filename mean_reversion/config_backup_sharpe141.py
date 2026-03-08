"""
config_backup_sharpe141.py
MEJOR CONFIGURACION ACTUAL — Sharpe 1.41 / Return +4.48% / Max DD -2.89%
4 símbolos, LONG+SHORT, Z>2.0, exit_z=-0.5

Para restaurar config.py: copiar DEFAULT_SYMBOLS y parámetros de aquí.
Para restaurar strategy.py: exit_z_threshold default = -0.5 (ya está aplicado).
"""

# ── Data ─────────────────────────────────────────────────────
TIMEFRAME = "1h"
DAYS = 1095
DEFAULT_SYMBOLS = [
    "ETH/USDT",
    "DOGE/USDT",
    "LINK/USDT",
    "XRP/USDT",
]

# ── Mean Reversion ───────────────────────────────────────────
ZSCORE_ENTRY_THRESHOLD = 2.0
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

# ── Exit logic (strategy.py) ─────────────────────────────────
# check_exit: exit_z_threshold = -0.5
# LONG sale cuando Z >= -0.5 (antes de llegar a la media)
# SHORT sale cuando Z <= +0.5

# ── Resultados del backtest ───────────────────────────────────
# Sharpe:        1.41
# Return:        +4.48%
# Max DD:        -2.89%
# Calmar:        3.45
# Win rate:      60.11%
# Profit factor: 1.33
# Trades:        183 (42L / 141S)
# Expectancy:    0.2723%
# Per symbol:
#   DOGE:  37 trades  WR 70%  Total +24.75%
#   LINK:  42 trades  WR 60%  Total +17.62%
#   XRP:   46 trades  WR 59%  Total  +5.48%
#   ETH:   58 trades  WR 55%  Total  +1.98%
