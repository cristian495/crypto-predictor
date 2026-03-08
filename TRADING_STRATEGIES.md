# Guía de Estrategias de Trading

Este proyecto contiene **DOS estrategias complementarias** de trading algorítmico con Machine Learning.

## 📊 Resumen Rápido

| Estrategia | Mejor para | BNB | ETH | BTC | Altcoins Laterales |
|------------|------------|-----|-----|-----|--------------------|
| **Mean Reversion** | Mercados laterales/oscilantes | ❌ | ⚠️ | ⚠️ | ✅ |
| **Trend Following** | Mercados con tendencia clara | ✅ | ✅ | ✅ | ❌ |

---

## 1️⃣ Mean Reversion (Reversión a la Media)

📁 **Directorio**: `mean_reversion/`

### ¿Qué hace?

Compra cuando el precio cae **demasiado** (oversold) y vende cuando sube **demasiado** (overbought), apostando a que volverá a la media.

### Señales de Entrada

- **Z-score > 2.0** (precio muy alejado de la media)
- RSI extremo (< 45 para LONG, > 55 para SHORT)
- ML predice reversión con prob > 60%

### Características

- ✅ Win Rate alto (50-60%)
- ✅ Funciona en mercados laterales
- ❌ No funciona en tendencias fuertes
- ❌ Stops fijos en porcentaje

### Uso

```bash
cd mean_reversion

# Test single
python main.py --symbol LINK/USDT

# Scanner multi-crypto
python main.py --scan
```

### Activos Ideales

- LINK/USDT
- ATOM/USDT
- DOT/USDT
- Cualquier crypto lateral

### Activos a EVITAR

- BNB (tendencia alcista fuerte) ❌
- SOL (trending) ❌
- PAXG (oro = trending) ❌

---

## 2️⃣ Trend Following (Seguimiento de Tendencia)

📁 **Directorio**: `trend_following/`

### ¿Qué hace?

Identifica tendencias fuertes y sigue el impulso, usando trailing stops para maximizar ganancias.

### Señales de Entrada

- **EMA(20) > EMA(50)** (tendencia alcista) o viceversa
- **ADX > 20** (tendencia fuerte, no lateral)
- RSI saludable (45-75 para LONG)
- Volumen confirmando
- ML predice continuación con prob > 60%

### Características

- ✅ Captura tendencias largas
- ✅ Stops dinámicos (ATR-based)
- ✅ Trailing stops para maximizar
- ✅ R:R 1:2.5
- ⚠️ Win Rate menor (40-50%) pero ganancias grandes

### Uso

```bash
cd trend_following

# Test con BNB
python main.py --symbol BNB/USDT

# Scanner multi-crypto
python main.py --scan --symbols BNB/USDT ETH/USDT SOL/USDT
```

### Activos Ideales

- **BNB/USDT** ✅ (perfecto!)
- ETH/USDT ✅
- BTC/USDT ✅
- SOL/USDT ✅

### Activos a EVITAR

- Cryptos laterales sin tendencia clara ❌
- Activos con ADX < 20 ❌

---

## 🔍 ¿Cuál estrategia usar?

### Usa **Mean Reversion** si:

- El asset está en rango lateral (precio oscila sin tendencia clara)
- ADX < 20 (baja fuerza de tendencia)
- Buscas win rate alto

### Usa **Trend Following** si:

- El asset tiene tendencia clara (alcista o bajista)
- ADX > 20 (tendencia fuerte)
- Estás dispuesto a menor win rate pero mayores ganancias por trade

### Para BNB específicamente:

**USAR TREND FOLLOWING** ✅

Tu resultado actual con Mean Reversion en BNB:
```
AUC: 0.48 (peor que azar!)
Sharpe: -1.73 (destruye capital)
Win rate: 53% (pero pérdidas > ganancias)
```

Esto confirma que BNB **NO es mean-reverting**, está en tendencia.

---

## 📈 Cómo Probar Trend Following con BNB

```bash
cd trend_following

# Test básico
python main.py --symbol BNB/USDT

# Con optimización (tarda ~15 min pero mejores resultados)
python main.py --symbol BNB/USDT --optimize
```

Métricas esperadas (mejores que mean reversion):
- Sharpe > 1.0
- AUC > 0.55
- Win Rate: 40-50% (normal)
- Avg Win / Avg Loss > 2.0 (lo importante!)

---

## ⚙️ Configuración Avanzada

### Mean Reversion

Edita `mean_reversion/config.py`:

```python
ZSCORE_ENTRY_THRESHOLD = 2.0    # Más alto = menos señales pero mejor calidad
RSI_LONG_MAX = 45.0             # Filtro de oversold
STOP_LOSS_PCT = 0.020           # 2% stop loss
```

### Trend Following

Edita `trend_following/config.py`:

```python
ADX_MIN = 20.0                  # Mínimo para considerar tendencia
STOP_LOSS_ATR_MULT = 2.0        # SL = 2×ATR (dinámico)
USE_TRAILING_STOP = True        # Activar trailing
TAKE_PROFIT_RR = 2.5            # R:R 1:2.5
```

---

## 🎯 Recomendación Final

Para tu cartera, te sugiero:

1. **Trend Following**: BNB, ETH, BTC, SOL
2. **Mean Reversion**: LINK, ATOM, DOT, ADA (cuando estén laterales)

Puedes correr ambos scanners en paralelo:

```bash
# Terminal 1 - Mean Reversion
cd mean_reversion
python main.py --scan --symbols LINK/USDT ATOM/USDT DOT/USDT

# Terminal 2 - Trend Following
cd trend_following
python main.py --scan --symbols BNB/USDT ETH/USDT BTC/USDT
```

Combinar ambas estrategias te da **diversificación** y captura diferentes condiciones de mercado.

---

## 📚 Más Info

- [Mean Reversion README](mean_reversion/notes.md)
- [Trend Following README](trend_following/README.md)

## 🤝 Contribuir

Mejoras sugeridas:
- [ ] Agregar más estrategias (breakout, pairs trading)
- [ ] Implementar stop loss adaptativo
- [ ] Dashboard en tiempo real
- [ ] Integración con exchanges para trading automático