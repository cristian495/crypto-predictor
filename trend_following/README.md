# Trend Following Trading System

Sistema de trading basado en **Trend Following** combinando análisis técnico, momentum e inteligencia artificial.

## Estrategia

A diferencia de Mean Reversion (que compra en sobreventa esperando reversión), esta estrategia:

1. **Identifica tendencias fuertes** usando:
   - Cruces de EMAs (20/50/200)
   - ADX > 20 (fuerza de tendencia)
   - Indicadores direccionales (DI+/DI-)

2. **Confirma momentum** con:
   - RSI en rangos saludables (45-75 para LONG, 25-55 para SHORT)
   - Volumen por encima del promedio
   - Supertrend alignment

3. **Usa Machine Learning** (ensemble XGBoost + LightGBM + CatBoost):
   - Predice probabilidad de continuación de tendencia
   - Requiere 2/3 modelos de acuerdo

4. **Gestión de riesgo avanzada**:
   - Stop Loss dinámico basado en ATR (2×ATR)
   - Take Profit con R:R de 1:2.5
   - Trailing stops para capturar tendencias largas
   - Máximo 3 posiciones simultáneas

## Ideal para

- **Activos en tendencia**: BNB, ETH, BTC, SOL
- **Mercados con impulso claro**
- **Timeframes medianos/largos** (1h, 4h)

## NO usar en

- Activos laterales (mejor usar mean_reversion)
- Mercados choppy sin tendencia clara
- Activos con ADX < 20

## Instalación

```bash
cd trend_following
pip install -r requirements.txt
```

## Uso

### Modo Single (Un símbolo)

```bash
# Test básico con BNB (usa config default)
python main.py

# Específico símbolo
python main.py --symbol ETH/USDT

# Con optimización Optuna (GUARDA parámetros automáticamente)
python main.py --symbol BTC/USDT --optimize

# Reusar parámetros optimizados guardados
python main.py --symbol BTC/USDT
# Automáticamente carga optimized_params_BTC_USDT.json si existe

# Forzar re-optimización (sobrescribe parámetros guardados)
python main.py --symbol BTC/USDT --force-optimize
```

### Sistema de Parámetros Guardados 💾

Cuando usas `--optimize`, los parámetros óptimos se guardan en:
- `optimized_params_BTC_USDT.json`
- `optimized_params_ETH_USDT.json`
- etc.

**Ventajas:**
- ✅ No necesitas re-optimizar cada vez (ahorra 2-5 minutos)
- ✅ Resultados consistentes entre ejecuciones
- ✅ Puedes compartir parámetros optimizados con tu equipo

**Archivo JSON incluye:**
```json
{
  "symbol": "BTC/USDT",
  "params": {
    "xgboost": {...},
    "lightgbm": {...},
    "catboost": {...}
  },
  "metrics": {
    "val_auc": 0.5710,
    "sharpe": 2.14,
    "win_rate": 0.857
  },
  "timestamp": "2026-03-08T..."
}
```

### Modo Scanner (Multi-crypto)

```bash
# Scanner con símbolos por defecto
python main.py --scan

# Scanner con símbolos custom
python main.py --scan --symbols BNB/USDT ETH/USDT SOL/USDT

# Scanner + optimización
python main.py --scan --optimize
```

## Configuración

Edita [config.py](config.py) para ajustar:

```python
# Trend detection
EMA_FAST = 20                    # EMA rápida
EMA_SLOW = 50                    # EMA lenta
EMA_TREND = 200                  # Filtro de tendencia largo plazo
ADX_MIN = 20.0                   # Mínimo ADX para tendencia fuerte

# Risk management
STOP_LOSS_ATR_MULT = 2.0         # SL = 2×ATR
TAKE_PROFIT_RR = 2.5             # TP = 2.5×SL (R:R 1:2.5)
USE_TRAILING_STOP = True         # Activar trailing stops
TRAILING_STOP_ATR_MULT = 3.0     # Trailing = 3×ATR

# Entry filters
REQUIRE_TREND_ALIGNMENT = True   # Requiere precio > EMA200 (LONG)
REQUIRE_VOLUME_CONFIRMATION = True  # Requiere volumen > 1.3× promedio
```

## Resultados Esperados

**BNB/USDT** (ejemplo):
- Sharpe > 1.0 (esperado en trending markets)
- Win Rate: 40-50% (normal en trend following)
- Profit Factor: > 1.5
- Avg Win > 2× Avg Loss (gracias al R:R favorable)

## Comparación vs Mean Reversion

| Característica | Mean Reversion | Trend Following |
|----------------|----------------|-----------------|
| **Entrada** | Sobrecompra/sobreventa | Momentum + tendencia |
| **Ideal para** | Mercados laterales | Mercados trending |
| **Win Rate** | 50-60% | 40-50% |
| **R:R** | 1:2.5 | 1:2.5 |
| **Trailing Stop** | No | Sí |
| **Stops** | Fijos % | Dinámicos ATR |
| **BNB** | ❌ No funciona | ✅ Funciona bien |

## Archivos

- `config.py` - Configuración de parámetros
- `features.py` - 55+ indicadores de tendencia/momentum
- `strategy.py` - Lógica de entrada/salida
- `target.py` - Etiquetado ML (TP antes de SL)
- `model.py` - Ensemble XGB+LGBM+CatBoost
- `backtest.py` - Engine de backtest con ATR stops
- `main.py` - CLI principal
- `data_loader.py` - Descarga datos de Binance
- `metrics.py` - Métricas de performance

## Notas Técnicas

### Features Clave

1. **Trend Detection (8 features)**:
   - EMA diffs, price distance from EMAs
   - Trend alignment flags
   - EMA slopes

2. **Momentum (13 features)**:
   - RSI, ROC, MACD, Stochastic
   - ADX + Directional Indicators
   - Supertrend

3. **Volatility (12 features)**:
   - ATR, ATR percentile
   - Bollinger Bands
   - Realized volatility

4. **Breakout (4 features)**:
   - Distance to recent highs/lows
   - Breakout flags

5. **Volume (3 features)**:
   - Relative volume
   - OBV divergence

### Target Labeling

El target es binario:
- `1` = TP alcanzado antes que SL (dentro de HOLD_TIMEOUT)
- `0` = SL alcanzado primero o timeout

Esto alinea el ML con el resultado real del backtest.

### Filtros de Entrada

Para evitar señales falsas, se requiere:
- ADX >= 20 (tendencia real)
- EMA20 > EMA50 (LONG) o EMA20 < EMA50 (SHORT)
- RSI en rango saludable (no extremos)
- Volumen > 1.3× promedio (confirmación)
- ML prob > 0.60 + 2/3 modelos de acuerdo

## Troubleshooting

**"Insufficient data"**:
- Aumenta `DAYS` en config.py
- Reduce `ADX_MIN` si el asset es menos volátil

**"No signals"**:
- Verifica que el asset esté en tendencia (no lateral)
- Ajusta `BUY_THRESHOLD` (ej: 0.55)
- Revisa filtros en config (REQUIRE_*)

**Sharpe negativo**:
- El asset puede no ser adecuado para trend following
- Prueba mean_reversion en su lugar
- Ajusta stops (ej: SL=1.5×ATR en vez de 2×ATR)

## License

MIT