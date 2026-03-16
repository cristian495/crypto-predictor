# Sell-the-Rip (Simple)

Short-only mirror of `buy_the_dip`:
- Core entry: short when 24h move is strongly positive (`past_return > threshold`).
- Optional filters: RSI, relative volume, bearish trend regime.
- Exits: fixed `%` TP/SL + timeout.

## Commands

- Scan:
  - `python main.py --scan`
  - `python main.py --scan --symbols LINK/USDT SUI/USDT --days 365`
- Walk-forward:
  - `python main.py --walk-forward --days 365`
- Matrix tuning:
  - `python main.py --matrix --days 365`

## Output

- Current signals table (with PASS/FAIL per condition)
- Backtest summary per symbol
- Walk-forward stability score per symbol
- Matrix files in `sell_the_rip/matrix_runs/`
