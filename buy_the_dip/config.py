"""
config.py
Configuration for Buy-the-Dip strategy.

VALIDATED EDGE (Walk-Forward Tested):
- Buy after 7%+ drops in 24h
- BNB: Stability 76.4/100 (EXCELLENT), +25% backtest return
- DOGE: Stability 62.7/100 (GOOD), +7.6% backtest return
- Average Win Rate: 70%
- Time-based exit (24h hold)

This strategy passed walk-forward validation with 82% of
windows showing positive returns for BNB.
"""

# ── Data ─────────────────────────────────────────────────────
TIMEFRAME = "1h"
DAYS = 1825  # 5 years of data

# VALIDATED symbols (passed walk-forward)
DEFAULT_SYMBOLS = [
    # Excelentes
    "SUI/USDT",    # Stability 73.6/100 - EXCELLENT
    "ADA/USDT",    # Stability 63.8/100 - GOOD
    "XRP/USDT",    # Stability 56.4/100 - GOOD
    "INJ/USDT",    # Stability 60/100 - GOOD

    # Buenos
    "LINK/USDT",   # Stability 74.1/100 - EXCELLENT
    "BNB/USDT",    # Stability 76.4/100 - EXCELLENT
    "POL/USDT",    # Stability 66.7/100 - GOOD

    # Neutrales
    "DOGE/USDT",   # Stability 62.7/100 - GOOD
    "BTC/USDT",    # Stability 43.8/100 - MODERATE
    "AVAX/USDT",   # Stability 55.6/100 - GOOD
    "SOL/USDT",   # Stability 51.5/100 - MODERATE

    # Eliminar
    # "APT/USDT",    # Stability 54.7/100 - GOOD
    # "ARB/USDT",    # Stability 48/100 - GOOD
    # "ETH/USDT",   # Stability 45/100 - MODERATE
    # "PAXG/USDT",    # Stability 35/100 - POOR
]

# ── Entry Rules ─────────────────────────────────────────────
# The core edge: buy after big drops
DIP_THRESHOLD_PCT = -0.05       # Enter when 24h return < -5%
DIP_THRESHOLD_STRONG = -0.07    # Stronger edge at -7%
USE_STRONG_THRESHOLD = True     # Use -7% for higher quality signals

# Lookback for calculating drop
DIP_LOOKBACK_HOURS = 24

# ── Exit Rules ──────────────────────────────────────────────
# Based on observed mean reversion behavior
# Key insight: After 7% drops, price bounces ~1.5-2% in 24h
# CRITICAL: Avg intra-period drawdown is -6%, so NO tight SL
# Using TIME-BASED exit to capture the natural bounce
TAKE_PROFIT_PCT = 0.04          # 4% TP (captures ~36% of trades)
STOP_LOSS_PCT = 0.20            # 20% SL (catastrophic only - should never hit)
HOLD_TIMEOUT_HOURS = 24         # Exit after 24h regardless (optimal timing)

# ── Position Sizing ─────────────────────────────────────────
POSITION_PCT = 0.20             # 20% of capital per trade (higher conviction)
MAX_POSITIONS = 2               # Max simultaneous positions
INITIAL_CAPITAL = 10_000.0
FEE_PCT = 0.001                 # 0.1% per side

# ── Filters (optional, can disable) ─────────────────────────
# Volume filter - ensure there's liquidity
MIN_VOLUME_MULT = 0.5           # Volume should be at least 50% of average
USE_VOLUME_FILTER = False       # Disabled by default (edge works without it)

# RSI filter - don't buy extreme oversold (may be dead cat)
RSI_MIN = 10                    # Avoid RSI < 10 (extreme panic)
RSI_MAX = 40                    # Only buy when RSI < 40 (oversold)
USE_RSI_FILTER = False          # Disabled by default

# BTC alignment - optional
USE_BTC_FILTER = False          # Don't require BTC alignment

# ── Trading Mode ────────────────────────────────────────────
LONG_ONLY = True                # Buy the dip = LONG only

# ── Per-Symbol Signal Overrides ─────────────────────────────
# These values come from matrix tests (matrix_runs/*_matrix.csv).
DEFAULT_DIP_THRESHOLD = DIP_THRESHOLD_STRONG if USE_STRONG_THRESHOLD else DIP_THRESHOLD_PCT

SYMBOL_SIGNAL_OVERRIDES = {
    "SUI/USDT": {"dip_threshold": -0.07, "use_rsi_filter": True, "use_volume_filter": False},
    "BNB/USDT": {"dip_threshold": -0.05, "use_rsi_filter": False, "use_volume_filter": True},
    "DOGE/USDT": {"dip_threshold": -0.05, "use_rsi_filter": True, "use_volume_filter": True},
    "ADA/USDT": {"dip_threshold": -0.06, "use_rsi_filter": False, "use_volume_filter": True},
    "XRP/USDT": {"dip_threshold": -0.06, "use_rsi_filter": False, "use_volume_filter": True},
    "LINK/USDT": {"dip_threshold": -0.07, "use_rsi_filter": False, "use_volume_filter": False},
    "POL/USDT": {"dip_threshold": -0.08, "use_rsi_filter": False, "use_volume_filter": True},
    "AVAX/USDT": {"dip_threshold": -0.08, "use_rsi_filter": False, "use_volume_filter": True},
    "SOL/USDT": {"dip_threshold": -0.08, "use_rsi_filter": False, "use_volume_filter": True},
    "INJ/USDT": {"dip_threshold": -0.08, "use_rsi_filter": False, "use_volume_filter": True},
    "BTC/USDT": {"dip_threshold": -0.07, "use_rsi_filter": True, "use_volume_filter": False},
}


def get_symbol_signal_params(symbol: str, dip_threshold_override: float = None) -> dict:
    """
    Resolve signal parameters for a symbol with fallback to global defaults.

    Args:
        symbol: Trading pair (e.g., BTC/USDT)
        dip_threshold_override: Optional CLI override applied to all symbols

    Returns:
        Dict with dip_threshold, use_volume_filter, use_rsi_filter
    """
    params = {
        "dip_threshold": DEFAULT_DIP_THRESHOLD,
        "use_volume_filter": USE_VOLUME_FILTER,
        "use_rsi_filter": USE_RSI_FILTER,
    }
    params.update(SYMBOL_SIGNAL_OVERRIDES.get(symbol, {}))

    if dip_threshold_override is not None:
        params["dip_threshold"] = dip_threshold_override

    return params

# ── Walk-Forward Validation ─────────────────────────────────
WF_TRAIN_MONTHS = 6
WF_TEST_MONTHS = 2
WF_STEP_MONTHS = 2
