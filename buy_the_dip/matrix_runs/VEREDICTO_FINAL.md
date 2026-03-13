# Veredicto Final (Buy-the-Dip Matrix) - Post Mejoras

Base de este veredicto:
- Walk-forward completo multi-simbolo ejecutado despues de mejoras.
- Parametros alineados por simbolo (sin hardcode global `-7%`).
- Data loader con retries/backoff en descargas.
- Metricas robustas: medianas, `num_trades`, y PF filtrado por `trades>=5`.

Fecha de referencia de corrida: 2026-03-12 (UTC).

Promedio de estabilidad global: `66.5/100`

Ranking consolidado por estabilidad (mejor config por moneda):

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

## Tiering sugerido (actualizado)

- Top tier (`EXCELLENT`): `DOGE`, `LINK`, `SUI`, `BNB`, `SOL`, `ADA`
- Mid tier (`GOOD`): `POL`, `XRP`, `AVAX`, `INJ`
- Weak tier (`MODERATE`): `BTC`

## Nota de interpretacion

- `SUI` y `POL` tienen menos historial efectivo que monedas con ~5 anos completos; interpretar con cautela adicional.
- `BTC` muestra inestabilidad estructural en este setup (muchas ventanas con pocos trades), por lo que no se recomienda como core.
