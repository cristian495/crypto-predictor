# Veredicto Final (Buy-the-Dip Matrix) - Post Mejoras

Este archivo separa claramente dos lecturas distintas:

1. `1825d`: robustez estructural (walk-forward, horizonte largo).
2. `365d`: estado tactico reciente (scan/backtest operativo).

No se contradicen; responden preguntas distintas.

## Seccion A - Robustez estructural (1825d, walk-forward)

### Objetivo

Medir robustez estructural de la estrategia en horizonte largo.

### Fuente y metodo

- Walk-forward completo multi-simbolo.
- Parametros alineados por simbolo (sin hardcode global `-7%`).
- Data loader con retries/backoff en descargas.
- Metricas robustas: medianas, `num_trades`, y PF filtrado por `trades>=5`.

### Fecha de referencia

`2026-03-12` (UTC)

### Resumen global

Promedio de estabilidad: `66.5/100`

### Ranking por simbolo

1. `DOGE/USDT` | score `73.0` | `threshold=-0.05`, `RSI=ON`, `Volume=ON`
2. `LINK/USDT` | score `71.9` | `threshold=-0.07`, `RSI=OFF`, `Volume=OFF`
3. `SUI/USDT` | score `71.4` | `threshold=-0.07`, `RSI=ON`, `Volume=OFF`
4. `BNB/USDT` | score `70.8` | `threshold=-0.05`, `RSI=OFF`, `Volume=ON`
5. `SOL/USDT` | score `70.8` | `threshold=-0.08`, `RSI=OFF`, `Volume=ON`
6. `ADA/USDT` | score `70.4` | `threshold=-0.06`, `RSI=OFF`, `Volume=ON`
7. `POL/USDT` | score `66.7` | `threshold=-0.08`, `RSI=OFF`, `Volume=ON`
8. `XRP/USDT` | score `65.8` | `threshold=-0.06`, `RSI=OFF`, `Volume=ON`
9. `AVAX/USDT` | score `63.8` | `threshold=-0.08`, `RSI=OFF`, `Volume=ON`
10. `INJ/USDT` | score `60.0` | `threshold=-0.08`, `RSI=OFF`, `Volume=ON`
11. `BTC/USDT` | score `47.1` | `threshold=-0.07`, `RSI=ON`, `Volume=OFF`

### Tiering sugerido

- Top tier (`EXCELLENT`): `DOGE`, `LINK`, `SUI`, `BNB`, `SOL`, `ADA`
- Mid tier (`GOOD`): `POL`, `XRP`, `AVAX`, `INJ`
- Weak tier (`MODERATE`): `BTC`

### Interpretacion

- `1825d` es la referencia principal de robustez.
- `SUI` y `POL` tienen menos historial efectivo; interpretar con cautela.
- `BTC` muestra inestabilidad estructural en este setup.

### Uso recomendado

Usar esta seccion para definir el universo base y los parametros por simbolo.

## Seccion B - Estado tactico reciente (365d, monitor/scan)

### Objetivo

Medir estado tactico reciente para ajustar exposicion operativa.

### Fuente y metodo

- Corrida de `python monitor.py` con `History: 365 days`.
- Metrica observada por simbolo: Return, Sharpe, WR, PF, Trades.
- Lectura de corto plazo (no reemplaza walk-forward largo).

### Fecha de referencia

`2026-03-13` (UTC)

### Resumen global

Average: `5.83%` retorno, Sharpe `0.54`, WR `61.5%`.

### Ranking por simbolo

1. `XRP/USDT` | Return `16.75%` | Sharpe `1.44` | WR `68.0%`
2. `POL/USDT` | Return `13.18%` | Sharpe `1.16` | WR `66.7%`
3. `ADA/USDT` | Return `11.08%` | Sharpe `0.89` | WR `63.4%`
4. `SOL/USDT` | Return `7.19%` | Sharpe `0.81` | WR `72.0%`
5. `DOGE/USDT` | Return `5.97%` | Sharpe `0.46` | WR `58.1%`
6. `LINK/USDT` | Return `3.90%` | Sharpe `0.38` | WR `53.6%`
7. `SUI/USDT` | Return `3.60%` | Sharpe `0.33` | WR `61.5%`
8. `BNB/USDT` | Return `3.15%` | Sharpe `0.42` | WR `62.1%`
9. `AVAX/USDT` | Return `0.55%` | Sharpe `0.11` | WR `59.7%`
10. `BTC/USDT` | Return `-0.07%` | Sharpe `0.00` | WR `53.3%`
11. `INJ/USDT` | Return `-1.21%` | Sharpe `-0.03` | WR `58.5%`

### Tiering sugerido

- Top tactico: `XRP`, `POL`, `ADA`, `SOL`
- Mid tactico: `DOGE`, `LINK`, `SUI`, `BNB`
- Weak tactico: `AVAX`, `BTC`, `INJ`

### Interpretacion

- `365d` da pulso reciente de mercado.
- Puede diferir del ranking `1825d` sin invalidarlo.
- Si difieren, ajustar tamano/riesgo antes que reescribir toda la estrategia.

### Uso recomendado

Usar esta seccion para modular exposicion tactica en el monitor (subir/bajar peso por simbolo).

## Regla de uso combinada

1. `1825d` define universo base y permanencia de la estrategia.
2. `365d` define ajustes tacticos de riesgo y peso por simbolo.
3. Recalibrar matriz por simbolo periodicamente y registrar fecha de corrida.
