# Guía Completa de Trading Algorítmico

Documento maestro con todo lo aprendido en el desarrollo de las estrategias Mean Reversion y Trend Following.

---

## 📊 Resumen Ejecutivo

### Estrategias Implementadas

| Estrategia | Directorio | Mejor Para | Activos Probados |
|------------|-----------|------------|------------------|
| **Mean Reversion** | `mean_reversion/` | Mercados laterales | LINK, ATOM, DOT ✅ / BNB ❌ |
| **Trend Following** | `trend_following/` | Mercados trending | BTC ✅ / ETH ❌, SOL ❌, BNB ❌ |

### Resultados Clave

#### Mean Reversion con BNB
- ❌ **AUC**: 0.48 (peor que azar)
- ❌ **Sharpe**: -1.73 (destruye capital)
- ❌ **Win Rate**: 53% (pero pérdidas > ganancias)
- **Conclusión**: BNB está en tendencia, NO mean-reverts

#### Trend Following con BTC (OPTIMIZADO)
- ✅ **AUC**: 0.57 (predecible)
- ✅ **Sharpe**: 2.14 (excelente!)
- ✅ **Win Rate**: 86% (solo 1 de 7 perdió)
- ✅ **Profit Factor**: 9.09 (por cada $1 perdido, ganas $9)
- ✅ **Max DD**: -0.25% (riesgo mínimo)
- **Conclusión**: BTC funciona PERFECTAMENTE con optimize

#### Trend Following con ETH/SOL
- ❌ **AUC**: 0.45-0.49 (no predecibles)
- ❌ **Trades**: 0 (Optuna es ultra-conservador)
- **Conclusión**: Necesitan otra estrategia o mejor target

---

## 🎯 Qué Estrategia Usar y Cuándo

### Mean Reversion

**✅ Usar cuando:**
- Activo oscila en rango sin tendencia clara
- ADX < 20 (baja fuerza de tendencia)
- Gráfico muestra soporte/resistencia horizontales
- Buscas win rate alto (55-65%)

**❌ NO usar cuando:**
- Activo en tendencia alcista/bajista fuerte
- Breakouts frecuentes
- ADX > 25

**Activos Ideales:**
- LINK/USDT ✅
- ATOM/USDT ✅
- DOT/USDT ✅
- Cualquier altcoin lateral

**Activos a EVITAR:**
- BNB/USDT (trending)
- SOL/USDT (trending)
- PAXG/USDT (oro = trending)

### Trend Following

**✅ Usar cuando:**
- Activo con tendencia clara
- ADX > 20 (tendencia presente)
- Gráfico muestra higher highs/lower lows
- Aceptas menor win rate (40-50%) pero R:R alto

**❌ NO usar cuando:**
- Mercado lateral/choppy
- Muchos falsos breakouts
- ADX < 15

**Activos Probados:**
- **BTC/USDT** ✅ (Sharpe 2.14 con optimize)
- ETH/USDT ❌ (AUC 0.44, no predecible)
- SOL/USDT ❌ (AUC 0.49, no predecible)
- BNB/USDT ❌ (tampoco funciona!)

---

## 🔧 Sistema de Parámetros Optimizados

### Cómo Funciona

Implementamos un sistema de guardado automático de parámetros Optuna:

```bash
# 1. Primera vez - optimiza y GUARDA
python main.py --symbol BTC/USDT --optimize
# Crea: optimized_params_BTC_USDT.json

# 2. Usos posteriores - CARGA automáticamente
python main.py --symbol BTC/USDT
# Usa parámetros guardados (consistente!)

# 3. Re-optimizar solo si es necesario
python main.py --symbol BTC/USDT --force-optimize
# Sobrescribe archivo guardado
```

### Archivo JSON Generado

```json
{
  "symbol": "BTC/USDT",
  "params": {
    "xgboost": {
      "n_estimators": 85,
      "learning_rate": 0.045,
      "subsample": 0.67,
      "colsample_bytree": 0.58,
      "min_child_weight": 22,
      "gamma": 4.8,
      "reg_alpha": 6.2,
      "reg_lambda": 12.5
    },
    "lightgbm": {...},
    "catboost": {...}
  },
  "metrics": {
    "val_auc": 0.5710,
    "sharpe": 2.14,
    "win_rate": 0.857,
    "profit_factor": 9.09
  },
  "timestamp": "2026-03-08T12:34:56"
}
```

### Ventajas

- ✅ **Consistencia**: Mismos resultados siempre
- ✅ **Velocidad**: Evita re-optimizar (ahorra 2-10 min)
- ✅ **Compartible**: Puedes compartir params con tu equipo
- ✅ **Auditable**: Sabes cuándo y con qué métricas se optimizó

---

## 📈 Mejores Prácticas

### 1. Workflow de Optimización

```bash
# Paso 1: Test rápido sin optimize
python main.py --symbol BTC/USDT

# Paso 2: Si AUC > 0.53 y Sharpe > 0, optimizar
python main.py --symbol BTC/USDT --optimize

# Paso 3: Usar parámetros guardados diariamente
python main.py --symbol BTC/USDT  # Carga params
```

### 2. Cuándo Re-Optimizar

Re-optimiza solo si:
- ✅ Pasaron 3+ meses (régimen de mercado cambió)
- ✅ Sharpe cayó >50% vs guardado
- ✅ Win rate bajó >20 puntos
- ❌ NO re-optimices por variaciones pequeñas

### 3. Interpretación de Métricas

#### AUC (Area Under Curve)
- **< 0.52**: ML no puede predecir → Descartar activo
- **0.52-0.55**: Marginal → Probar con más trials
- **0.55-0.60**: Bueno → Optimizar vale la pena
- **> 0.60**: Excelente → Activo ideal

#### Sharpe Ratio
- **< 0**: Pierde dinero → No tradear
- **0-1**: Marginalmente rentable
- **1-2**: Bueno
- **> 2**: Excelente (como BTC con 2.14)

#### Win Rate vs Profit Factor
- **Win Rate alto (>60%)** con **PF bajo (<1.5)** → Muchos trades pequeños
- **Win Rate bajo (<50%)** con **PF alto (>2)** → Pocos trades pero grandes
- **BTC**: WR 86% + PF 9.09 = **IDEAL** (mejor de ambos mundos)

### 4. Gestión de Riesgo

#### Stops Dinámicos (Trend Following)
```python
# Mejor que stops fijos
STOP_LOSS_ATR_MULT = 1.5  # Se adapta a volatilidad
TAKE_PROFIT_RR = 1.8      # R:R 1:1.8
USE_TRAILING_STOP = True  # Captura tendencias largas
```

#### Position Sizing
```python
POSITION_PCT = 0.15  # 15% por trade
MAX_POSITIONS = 3    # Máximo 3 simultáneas = 45% exposición
```

---

## 🚨 Troubleshooting

### Problema: AUC < 0.50

**Causa**: El ML no puede aprender patrones
**Solución**:
1. Probar otro activo
2. Cambiar el target (más fácil de predecir)
3. Agregar más features específicas del activo

### Problema: 0 Trades después de Optimize

**Causa**: Optuna sobre-regulariza, modelo ultra-conservador
**Solución**:
```bash
# Opción 1: Más trials
OPTUNA_TRIALS = 100  # En config.py

# Opción 2: Usar parámetros de activo similar
cp optimized_params_BTC_USDT.json optimized_params_ETH_USDT.json

# Opción 3: Probar sin optimize
python main.py --symbol ETH/USDT  # Sin --optimize
```

### Problema: Win Rate alto pero Sharpe negativo

**Causa**: Ganancias pequeñas, pérdidas grandes
**Solución**:
- Aumentar TAKE_PROFIT_RR
- Reducir STOP_LOSS_ATR_MULT
- Activar trailing stops

### Problema: Muchos trades pero rentabilidad baja

**Causa**: Señales de baja calidad
**Solución**:
```python
# Ser más estricto
BUY_THRESHOLD = 0.70  # De 0.60
MIN_AGREE = 3         # De 2 (requiere 3/3 modelos)
ADX_MIN = 25.0        # De 20.0
```

---

## 📚 Comparación Profunda

### Mean Reversion vs Trend Following

| Aspecto | Mean Reversion | Trend Following |
|---------|---------------|-----------------|
| **Filosofía** | "Lo que sube, baja" | "The trend is your friend" |
| **Entrada** | Extremos (Z>2, RSI<30) | Momentum (EMA cross, ADX>20) |
| **Salida** | Reversión a media | TP o trailing stop |
| **Stops** | Fijos (%) | Dinámicos (ATR) |
| **Win Rate** | 55-65% | 40-50% |
| **R:R** | 1:2.5 | 1:1.8 (pero trailing mejora) |
| **Trades/mes** | 10-30 | 5-15 |
| **Mejor en** | Lateral, oscilante | Tendencia clara |
| **BTC** | ⚠️ Marginal | ✅ Excelente |
| **BNB** | ❌ No funciona | ❌ No funciona |
| **LINK** | ✅ Bueno | ⚠️ Depende |

---

## 🎓 Lecciones Aprendidas

### 1. No Todos los Activos Son Predecibles

- **BTC**: Institucional, líquido → Predecible ✅
- **BNB**: Exchange token, influencias internas → Impredecible ❌
- **ETH/SOL**: DeFi/NFT dependencies → Difíciles

### 2. Optimización ≠ Siempre Mejor

- Con 30 trials: BTC funciona, ETH no
- Con 100 trials: SOL sigue sin funcionar
- **Conclusión**: Si AUC base < 0.52, más trials no ayudarán

### 3. "Menos es Más" en Trading

BTC optimizado:
- Solo 7 trades en 5 meses
- Pero WR 86% y Sharpe 2.14
- Mejor que 50 trades con WR 45%

### 4. Overfitting es Real

Sin optimize:
- Train AUC: 0.76 (muy alto)
- Val AUC: 0.59
- Test Sharpe: -0.54 (FALLA!)

Con optimize:
- Train AUC: 0.70 (regularizado)
- Val AUC: 0.57
- Test Sharpe: 2.14 (ÉXITO!)

---

## 🔮 Próximos Pasos Sugeridos

### Corto Plazo

1. **Optimizar más activos**
   ```bash
   for symbol in AVAX MATIC LINK ADA; do
       python trend_following/main.py --symbol ${symbol}/USDT --optimize
   done
   ```

2. **Scanner automatizado**
   ```bash
   python trend_following/main.py --scan --optimize --symbols BTC/USDT ETH/USDT AVAX/USDT
   ```

### Medio Plazo

3. **Mejorar el target**
   - Actual: "TP antes que SL" (difícil)
   - Alternativa: "Precio sube X% en Y horas" (más simple)

4. **Agregar más features**
   - Order flow (bid/ask imbalance)
   - On-chain metrics (solo para BTC/ETH)
   - Funding rate arbitrage

### Largo Plazo

5. **Estrategias adicionales**
   - Grid Trading (para BNB lateral)
   - Pairs Trading (BTC vs ETH)
   - Market Making

6. **Automatización completa**
   - Integración con exchange API
   - Auto-ejecución de señales
   - Monitoreo 24/7

---

## 📞 Quick Reference

### Mean Reversion
```bash
cd mean_reversion
python main.py --scan --symbols LINK/USDT ATOM/USDT DOT/USDT
```

### Trend Following (BTC - el que funciona)
```bash
cd trend_following

# Primera vez
python main.py --symbol BTC/USDT --optimize

# Uso diario
python main.py --symbol BTC/USDT
```

### Comandos Útiles
```bash
# Ver parámetros guardados
cat optimized_params_BTC_USDT.json | jq '.metrics'

# Limpiar parámetros viejos
rm optimized_params_*.json

# Probar múltiples activos
for s in BTC ETH SOL; do
    python main.py --symbol ${s}/USDT
done
```

---

## ✅ Checklist para Nuevo Activo

Antes de tradear un activo nuevo:

- [ ] Probar sin optimize
- [ ] Verificar AUC > 0.53
- [ ] Verificar Sharpe > 0.5
- [ ] Verificar Win Rate > 45%
- [ ] Optimizar con 100 trials
- [ ] Backtest en test set
- [ ] Verificar >5 trades en test
- [ ] Paper trading 1 semana
- [ ] Solo entonces: trading real

---

**Última actualización**: 2026-03-08
**Versión**: 1.0
**Autor**: Sistema de Trading ML con Optuna
