  python main.py --scan --buy-threshold 0.70 --symbols BNB/USDT XRP/USDT TRX/USDT --optimize



## Thereshold
- Probabilidad de que suba. Ejm si > 0.7 = hay probabilidad de 70% de que suba

AUC (Area Under Curve) 
- mide qué tan bien el modelo distingue entre oportunidades buenas y malas.
| AUC  | Calidad   |
| ---- | --------- |
| 0.5  | azar      |
| 0.55 | muy débil |
| 0.6  | aceptable |
| 0.65 | bueno     |
| 0.7+ | muy bueno |


## Profit factor 
- ventaja estadística
Profit Factor = total ganancias / total pérdidas
| Profit factor | Significado    |
| ------------- | -------------- |
| <1            | pierdes dinero |
| 1.1           | muy débil      |
| 1.2–1.4       | aceptable      |
| 1.5+          | bueno          |
| 2+            | excelente      |


## DD Drowdown
- es la caída desde un máximo histórico del capital hasta el mínimo posterior antes de recuperarse
Drawdown = (minimo actual - Máximo anterior) / Máximo anterior
12000 |        /\ 
11000 |       /  \__
10000 |______/      \__
 9000 |               \__
        tiempo →
- El drawdown es la distancia vertical desde el pico hasta el valle.
- si la caida fue poca entonces el drawdown sera un valor bajo


## RR Risk Reward Ratio
- Es una forma de medir cuánto arriesgas vs cuánto esperas ganar en cada trade
Supongamos este trade:
Stop loss: −2%
Take profit: +4%
=> RR = 4% / 2% = 2 => Seria RR = 1:2
- Significa arriesga 1 para ganar 2

## Z
- Distancia de la media
- Indica desviaciones estándar arriba o abajo de la media.
Ejemplo de interpretación:
Z > +2 Precio muy alto posible caida
Z < -2 Precio muy bajo posible subida

## Sharpe
- indica que tan consistentes son las ganancias

+1.9%
+2.1%
+2.0%
+1.8%
+2.2%
Ganancias consistentes -> Sharpe alto

+8%
-6%
+10%
-7%
+5%
Ganancias inconsistentes -> Sharpe bajo

## RSI Relative Strength Index (Mean Reversion)
- Mide la fuerza de una tendencia
- va de 0 a 100
- RSI < 45 -> la gente ha sobrevendido (posible subida)
- RSI > 70 -> la gente ha sobrecomprado (posible bajada)

## ADX Average Directional Index (Trend Following)
- Mide la fuerza de una tendencia 0 - 100
ADX < 20  → Mercado lateral (sin tendencia)
ADX 20-25 → Tendencia débil
ADX 25-50 → Tendencia fuerte ✅
ADX > 50  → Tendencia muy fuerte ✅✅

## ATR (Average True Range) - Stops Dinámicos
- Mide la volatilidad del activo entre un minimo y maximo
- parecido a desviacio estandar pero con rangos
- mide cuánto se mueve el precio en promedio por vela.
- Usado para stops que se adaptan a las condiciones del mercado
- 5 mercado tranquilo
- 20 mercado muy volatil

 ## Trailing Stops 
 - SL que "siguen" al precio cuando va a favor

## DI+ y DI- (Directional Indicators)
- DI+ = Fuerza del movimiento alcista
- DI- = Fuerza del movimiento bajista


## ATR Expansion
- el mercado está pasando de calma → movimiento fuerte

## Percentile
- va de 1 a 100
- indica la posición de un valor dentro de una distribución.

## ATR max percentile
- limite de volatilidad permitido
Ejem
MAX = 90
HISTORIAL ATR = 10 11 12 13 14 15 16 17 20 25 30
ACTUAL ATR = 25 => ubicacion 90 del percentile
=>  ya se llego al maximo de volatilidad (evitar trades)


## Volume surge breakout
- tecnica para detectar que se rompio un limite usando el volumen como referencia
SURGE_VOLUME_MIN = current_volume / volume_mean
SURGE_VOLUME_ZSCORE_MIN = indica cuantas desviaciones estandar esta por encima de su media

## Donchian
- mide los extremos del precio (techo y suelo)

breakout + volume
btc correlation
volatility breakout




### IMPLEMENTAR
Antes de usar dinero real:
	•	walk forward test
	•	monte carlo
	•	slippage simulation
	•	fee stress test
	•	forward test en paper trading


📋 TODO List Actualizado
Quieres que implemente esto en orden de prioridad?

[CRÍTICO] ADX + Regime Filter
[ALTO] VWAP Distance feature
[ALTO] Sistema de Scoring (0-100)
[ALTO] Bollinger Distance mejorado
[MEDIO] Wick ratios
[MEDIO] Walk-forward validation
[BAJO] Optimización per-symbol con Optuna
