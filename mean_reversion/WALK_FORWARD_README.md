# Walk-Forward Validation Guide

## ¿Qué es Walk-Forward Validation?

Walk-forward validation es una técnica **crítica** para validar estrategias de trading antes de operar en vivo. Es más robusta que un simple backtest porque:

1. **Detecta overfitting**: Si tu modelo solo funciona en un período específico, walk-forward lo revelará
2. **Simula trading real**: En la vida real, entrenas con datos pasados y operas en el futuro
3. **Mide estabilidad**: Ver cómo la estrategia performa en diferentes regímenes de mercado

## Cómo Funciona

```
Timeline:
[------- Train 6 months -------][- Test 1 month -]
   ^                                ^
   Train Model                    Test Model

   [------- Train 6 months -------][- Test 1 month -]
      ^                                ^
      Train Model                    Test Model

      [------- Train 6 months -------][- Test 1 month -]
         ^                                ^
         Train Model                    Test Model
```

- **Train window**: 6 meses de datos históricos
- **Test window**: 1 mes out-of-sample
- **Step**: Mover ventana 1 mes hacia adelante
- **Repetir**: Hasta cubrir todo el período disponible

## Uso

### Básico - 2 símbolos (rápido)
```bash
python walk_forward.py
```

Esto ejecuta con:
- ETH/USDT y DOGE/USDT
- Train: 6 meses
- Test: 1 mes
- ~10-15 ventanas por símbolo

### Todos los símbolos (completo, ~30 min)
```python
# Editar walk_forward.py línea final:
results, scores = run_multi_symbol_walk_forward(
    symbols=DEFAULT_SYMBOLS,  # Todos los símbolos
    train_months=6,
    test_months=1,
)
```

### Custom configuration
```python
from walk_forward import walk_forward_validation

results = walk_forward_validation(
    symbol="BTC/USDT",
    train_months=12,     # Ventana de entrenamiento más larga
    test_months=2,       # Test más largo
    step_months=2,       # Saltos más grandes
    total_days=1095,     # Datos históricos
)
```

## Interpretación de Resultados

### Stability Score (0-100)

El score de estabilidad combina 3 métricas:

```
Stability = 40% × (% ventanas con return positivo)
          + 30% × (% ventanas con Sharpe > 1.0)
          + 30% × (% ventanas con WR > 50%)
```

**Interpretación**:
- **75-100**: ✅ EXCELENTE - Muy estable, estrategia robusta
- **60-74**: 🟢 BUENA - Razonablemente estable, proceder con cautela
- **45-59**: 🟡 MODERADA - Inestabilidad significativa, requiere mejoras
- **0-44**: ❌ POBRE - Altamente inestable, probablemente overfitting

### Ejemplo de Output

```
==================================================================
  WALK-FORWARD ANALYSIS: ETH/USDT
==================================================================

  Summary Statistics (12 windows):
  ─────────────────────────────────────────

  Return (%):
    Mean:      4.23
    Median:    3.87
    Std:       5.12
    Min:      -2.45
    Max:      15.33

  Sharpe:
    Mean:      1.78
    Median:    1.65
    Std:       0.89
    Min:       0.12
    Max:       3.21

  Consistency Metrics:
  ─────────────────────────────────────────
    Positive returns:  10/12 (83.3%)
    Sharpe > 1.0:       9/12 (75.0%)
    Win rate > 50%:    11/12 (91.7%)

  ⭐ STABILITY SCORE: 83.1/100
     ✅ EXCELLENT - Very stable across regimes
```

## ¿Qué Buscar?

### 🟢 Señales Positivas
- Stability score > 60
- Mayoría de ventanas con return positivo (>70%)
- Sharpe consistentemente > 1.0
- Win rate estable (50-65%)
- Desviación estándar baja (<6%)

### ⚠️ Señales de Advertencia
- Stability score < 45
- Muchas ventanas con pérdidas
- Sharpe volátil (varía de -2 a +3)
- Win rate inconsistente (20% a 80%)
- Desviación estándar alta (>10%)

### ❌ Red Flags (NO OPERAR)
- Stability score < 30
- >50% ventanas negativas
- Sharpe promedio < 0.5
- Max drawdown > 15% en alguna ventana
- Performance se degrada con el tiempo

## Comparación con Backtest Simple

| Métrica | Backtest Simple | Walk-Forward |
|---------|----------------|--------------|
| Validez | Una sola prueba | Múltiples pruebas |
| Overfitting | No detecta | Sí detecta |
| Realismo | Bajo | Alto |
| Tiempo | Rápido (2 min) | Lento (30 min) |
| Confianza | Baja | Alta |

## Próximos Pasos

### Si Stability > 70 ✅
1. Ejecutar con TODOS los símbolos
2. Si mantiene estabilidad → Paper trading
3. Monitorear 1-2 meses en papel
4. Si funciona → Operar con capital pequeño

### Si Stability 50-70 🟡
1. Optimizar hiperparámetros con Optuna
2. Agregar más features (wicks, funding momentum)
3. Probar diferentes thresholds
4. Re-ejecutar walk-forward
5. Si mejora → Paper trading

### Si Stability < 50 ❌
1. **NO OPERAR**
2. Revisar estrategia fundamental
3. Considerar diferentes símbolos
4. Buscar otros regímenes de mercado
5. Tal vez esta estrategia no es robusta

## Tips y Notas

### Performance Tips
- Usa menos símbolos para pruebas rápidas (2 símbolos = 5 min)
- Walk-forward completo toma ~30 min para 4 símbolos
- Puedes ejecutar en paralelo símbolos individuales

### Interpretación Avanzada
- **Trending degradation**: Si performance cae con el tiempo → mercado cambió
- **High variance**: Si resultados muy variables → estrategia inestable
- **Consistent negative**: Si mayoría negativo → estrategia rota

### Limitaciones
- Walk-forward NO detecta:
  - Slippage real
  - Latencia de ejecución
  - Cambios de liquidez
  - Eventos de cisne negro

- Por eso **paper trading es obligatorio** después de walk-forward exitoso

## Preguntas Frecuentes

### ¿Por qué 6 meses train / 1 mes test?
- 6 meses: Suficiente data para entrenar (>4000 barras 1h)
- 1 mes: Test realista de out-of-sample
- Puedes ajustar según tu estrategia

### ¿Cuántas ventanas necesito?
- Mínimo: 8-10 ventanas
- Ideal: 12-15 ventanas
- Más ventanas = más confianza estadística

### ¿Qué hago si algunos símbolos son estables y otros no?
- Opera solo los símbolos estables (stability > 60)
- Investiga por qué otros fallan
- Considera features específicos por símbolo

### ¿Walk-forward garantiza éxito en live trading?
- **NO**. Nada garantiza éxito en trading.
- Walk-forward reduce significativamente el riesgo de overfitting
- Pero aún necesitas paper trading y gestión de riesgo

## Conclusión

Walk-forward validation es **obligatorio** antes de operar dinero real. Si tu estrategia no pasa walk-forward, **no la operes**.

Un backtest perfecto sin walk-forward es prácticamente inútil. Un backtest bueno CON walk-forward estable es una base sólida para paper trading.

Recuerda: **Backtest → Walk-Forward → Paper → Live**. No saltes pasos.