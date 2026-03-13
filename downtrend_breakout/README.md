# downtrend_breakout v1

No-ML futures strategy focused on robust walk-forward behavior.

## Core Idea

- SHORT in bearish momentum breakdowns (primary edge).
- LONG in bullish momentum breakouts (to improve frequency/robustness).
- Deterministic rules only (no ML fitting).

## Rules

Features used:
- `ret_6h`, `ret_24h`
- `ema50`, `ema200`, `ema200_slope_24h`
- `donchian_high/low`
- `adx14`, `atr14`, `atr_pct_rank_90d`, `relative_volume_20`

Entry logic:
- SHORT: bearish regime + breakdown + momentum + ADX + volatility/liquidity filters.
- LONG: bullish regime + breakout + momentum + ADX + volatility/liquidity filters.
- Cooldown: 6 bars after each entry.

Risk logic:
- SL = `1.8 * ATR14`
- TP = `1.6R`
- Timeout = `36` bars
- Trailing stop enabled after `+1.0R`, distance `1.2 * ATR14`
- Fees `0.10%` per side
- Slippage `0.05%` per side

## Commands

From `downtrend_breakout/`:

```bash
python main.py --scan
python main.py --scan --symbols SUI/USDT --days 365
python main.py --walk-forward --symbols DOGE/USDT LINK/USDT SUI/USDT BNB/USDT SOL/USDT ADA/USDT --days 1825
python main.py --matrix --symbols DOGE/USDT LINK/USDT SUI/USDT BNB/USDT SOL/USDT ADA/USDT
```

Optional:
- `--mode long_short|short_only` (default `short_only`)
- `--signals-json /path/file.json`

## Matrix Overrides

`matrix_test.py` writes selected per-symbol params to:
- `downtrend_breakout/matrix_overrides.json`

These overrides are auto-loaded by `config.get_symbol_params(...)`.

## Current Practical Setting

For current behavior, use `short_only` as baseline mode and keep matrix overrides enabled.
With the current tuned override for `SOL/USDT`, a 365-day 6-symbol walk-forward reached:

- Average stability: `70.6/100`
- Symbols >= 55 stability: `6/6`
