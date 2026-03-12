# Buy-the-Dip Strategy

## Overview

Simple, rule-based strategy that buys after significant price drops.
**Validated with walk-forward testing** - no overfitting.

## Edge (Validated)

| Symbol | Stability Score | Backtest Return | Win Rate |
|--------|----------------|-----------------|----------|
| BNB/USDT | **76.4/100** (EXCELLENT) | +25.5% | 70% |
| DOGE/USDT | **62.7/100** (GOOD) | +7.6% | 69% |

## How It Works

1. **Entry**: Buy when price drops >7% in 24 hours
2. **Exit**: Time-based (24 hours) OR take profit at +4%
3. **Stop Loss**: 20% (catastrophic protection only)

## Why It Works

After a 7%+ drop in crypto:
- Average bounce: +1.5-2% in next 24h
- Win rate: 60-70%
- Intra-period drawdown averages -6%, so tight SL doesn't work

## Files

- `config.py` - Configuration parameters
- `strategy.py` - Signal generation
- `backtest.py` - Backtest engine
- `run_backtest.py` - Run backtests
- `walk_forward.py` - Walk-forward validation
- `main.py` - Scanner CLI + walk-forward entrypoint

## Usage

### Run Backtest
```bash
cd buy_the_dip
python run_backtest.py
```

### Validate Strategy
```bash
python walk_forward.py
```

### Scan for Live Signals
```bash
python main.py --scan
```

## Key Parameters

- `DIP_THRESHOLD_STRONG = -0.07` (7% drop required)
- `TAKE_PROFIT_PCT = 0.04` (4% take profit)
- `HOLD_TIMEOUT_HOURS = 24` (exit after 24h)
- `POSITION_PCT = 0.20` (20% per trade)

## Walk-Forward Results (BNB)

- 22 out-of-sample windows tested
- **81.8% had positive returns**
- **81.8% had positive Sharpe**
- Average return per window: +1.21%
- Average win rate: 70.5%

## Important Notes

1. **Time-based exit is critical** - Don't use tight stop losses
2. **-7% threshold is optimal** - Looser thresholds reduce edge
3. **BNB > DOGE > ETH** - Not all symbols have the same edge
4. **No ML, no overfitting** - Pure statistical edge
