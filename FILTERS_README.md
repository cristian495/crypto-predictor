# 🚀 Sistema de Filtros - Guía de Uso

## ✅ Implementación Completa

Acabas de implementar un sistema modular de filtros para post-procesar señales de trading. Aquí está todo lo que necesitas saber.

---

## 📁 Estructura Creada

```
Predictor/
├── filters/                          # ← NUEVO
│   ├── __init__.py                  # Pipeline manager
│   ├── base_filter.py               # Clase base abstracta
│   ├── volatility_filter.py         # Filtro de volatilidad
│   └── liquidity_filter.py          # Filtro de liquidez
│
├── filter_config.yaml                # ← NUEVO - Configuración
├── monitor.py                        # ← MODIFICADO - Integrado con filtros
└── test_filters_simple.py           # Script de verificación
```

---

## 🎯 ¿Qué Hace?

### **Filtros Implementados:**

#### **1. Liquidity Filter** ✅
- **Qué hace**: Veta trades cuando el volumen 24h es muy bajo
- **Por qué**: Evita slippage alto y posible manipulación
- **Configuración actual**: Min 50% del volumen promedio de 30 días

#### **2. Volatility Filter** ✅
- **Qué hace**: Veta trades cuando la volatilidad es extrema
- **Por qué**: El modelo fue entrenado con condiciones normales
- **Configuración actual**: Max 10% volatilidad realizada (24h)

---

## 🔧 Cómo Usar

### **Opción 1: Con monitor.py (Recomendado)**

```bash
# El monitor.py ya está integrado con los filtros
python monitor.py

# Verás en la salida:
# ──────────────────────────────────────────────────────────────
# FILTER CONFIGURATION
# ──────────────────────────────────────────────────────────────
# ✅ Loaded 2 filter(s):
#    - Liquidity Filter (enabled)
#    - Volatility Filter (enabled)
# ──────────────────────────────────────────────────────────────
```

### **Opción 2: Probar filtros manualmente**

Necesitarás estar en un entorno con las dependencias instaladas:

```python
from filters import FilterPipeline

# Configuración
config = {
    'filters': {
        'liquidity': {'enabled': True, 'min_volume_ratio': 0.5},
        'volatility': {'enabled': True, 'max_volatility': 0.10}
    }
}

# Crear pipeline
pipeline = FilterPipeline(config)

# Señal de ejemplo
signal = {
    'symbol': 'LINK/USDT',
    'signal': 'SHORT',
    'prob': 0.67,
    'strategy': 'mean_reversion'
}

# Filtrar
filtered = pipeline.filter_signals([signal])
```

---

## ⚙️ Configuración (filter_config.yaml)

El archivo `filter_config.yaml` controla todo:

```yaml
filters:
  liquidity:
    enabled: true              # true/false
    min_volume_ratio: 0.5      # 50% del promedio
    lookback_days: 30          # Calcular promedio en 30 días

  volatility:
    enabled: true
    max_volatility: 0.10       # 10% máximo
    lookback_hours: 24         # Última 24 horas
```

### **Activar/Desactivar Filtros:**

```yaml
# Desactivar todos los filtros
filters:
  liquidity:
    enabled: false
  volatility:
    enabled: false

# Solo volatilidad
filters:
  liquidity:
    enabled: false
  volatility:
    enabled: true
```

---

## 📊 Ejemplo de Salida

### **Sin Filtros:**
```
CURRENT SIGNALS — Mean Reversion
══════════════════════════════════════════════
  LINK/USDT    SHORT    prob=0.6734 agree=3/3 Z=2.12
══════════════════════════════════════════════

SUMMARY: Found 1 trading signal(s)
```

### **Con Filtros Activos:**
```
FILTER CONFIGURATION
──────────────────────────────────────────────
✅ Loaded 2 filter(s):
   - Liquidity Filter (enabled)
   - Volatility Filter (enabled)
──────────────────────────────────────────────

CURRENT SIGNALS — Mean Reversion
══════════════════════════════════════════════
  LINK/USDT    SHORT    prob=0.6734 agree=3/3 Z=2.12
══════════════════════════════════════════════

APPLYING FILTERS (1 signal(s))
══════════════════════════════════════════════
  Result: 1/1 signal(s) passed filters
══════════════════════════════════════════════

SUMMARY: Found 1 trading signal(s)
```

### **Si un Filtro Veta:**
```
APPLYING FILTERS (1 signal(s))
══════════════════════════════════════════════
  ❌ Filtered out LINK/USDT SHORT: Volatility Filter: Volatility too high: 15.23% > 10.00%

  Result: 0/1 signal(s) passed filters
══════════════════════════════════════════════

SUMMARY: No signals found across all strategies
```

---

## 🧪 Testing

### **Verificar Estructura:**

```bash
python test_filters_simple.py
```

### **Test Completo (requiere entorno con dependencias):**

El mejor test es ejecutar el monitor completo:

```bash
# Asegúrate de estar en el entorno de Python donde instalaste las dependencias
# (el mismo que usas para mean_reversion/main.py)

python monitor.py
```

---

## 🔄 Flujo Completo

```
1. Estrategia genera señal
   LINK/USDT SHORT prob=0.67
        ↓
2. Monitor.py recibe señal
        ↓
3. FilterPipeline procesa:
   ├─ Liquidity Filter → ✅ Pass (volume OK)
   └─ Volatility Filter → ✅ Pass (vol 3.2% < 10%)
        ↓
4. Señal aprobada
   └─ Discord notification enviada
```

---

## 📈 Monitoreo de Resultados

### **Qué Vigilar:**

1. **False Positives** (filtros vetar señales buenas):
   - Busca en logs: `❌ Filtered out`
   - Pregunta: ¿Era realmente mala esa señal?

2. **Efectividad**:
   - Compara resultados con/sin filtros
   - Métricas: Sharpe ratio, Win rate, Max DD

3. **Ajuste de Parámetros**:
   ```yaml
   # Si muchos falsos positivos, relaja límites
   volatility:
     max_volatility: 0.15  # era 0.10

   # Si pasando señales malas, endurece
   liquidity:
     min_volume_ratio: 0.7  # era 0.5
   ```

---

## 🚀 Próximos Pasos

### **Fase 1: Testing (Semana 1-2)**
- [ ] Ejecutar `python monitor.py` diariamente
- [ ] Revisar señales filtradas en logs
- [ ] Comparar con trades reales

### **Fase 2: Ajuste (Semana 3-4)**
- [ ] Ajustar thresholds en `filter_config.yaml`
- [ ] Medir impacto en Sharpe ratio
- [ ] Documentar mejoras

### **Fase 3: Expansión (Mes 2+)**
Si los filtros básicos funcionan bien, considera agregar:

- **BTC Correlation Filter** (detectar cuando correlación rompe)
- **AI Sentiment Filter** (usar Claude API para noticias)
- **Time Filter** (evitar ciertos horarios)
- **News Filter** (detectar eventos extremos)

---

## ❓ Troubleshooting

### **"No module named 'ccxt'"**
- Los filtros usan `ccxt` para consultar datos de mercado
- Asegúrate de ejecutar en el mismo entorno Python donde instalaste las dependencias de `mean_reversion`

### **"No filter configuration found"**
- Verifica que `filter_config.yaml` esté en la raíz del proyecto
- Debe estar al mismo nivel que `monitor.py`

### **"Failed to initialize filters"**
- Revisa que `filter_config.yaml` tiene formato YAML válido
- Verifica que PyYAML esté instalado: `python -c "import yaml"`

### **Filtros no se aplican**
- Verifica que `enabled: true` en `filter_config.yaml`
- Busca en logs: "✅ Loaded X filter(s)"
- Si dice "⚠️ No filters enabled", revisa la configuración

---

## 📚 Arquitectura

```python
# filters/base_filter.py
class BaseFilter(ABC):
    @abstractmethod
    def filter_signal(self, signal: Dict) -> Dict:
        """
        Returns:
            {
                'approved': True/False,
                'confidence': 0.0-1.0,
                'reason': 'explanation',
                'metadata': {...}
            }
        """
        pass

# filters/volatility_filter.py
class VolatilityFilter(BaseFilter):
    def filter_signal(self, signal):
        vol = self._calculate_realized_volatility(signal['symbol'])
        if vol > self.max_volatility:
            return {'approved': False, ...}
        return {'approved': True, ...}

# filters/__init__.py
class FilterPipeline:
    def filter_signals(self, signals: List[Dict]) -> List[Dict]:
        # Run each signal through all filters
        # Return only approved signals
```

---

## ✅ Verificación de Instalación

### **Checklist:**

- [x] Carpeta `filters/` creada
- [x] 4 archivos Python en `filters/`
- [x] `filter_config.yaml` en raíz
- [x] `monitor.py` modificado
- [ ] Ejecutar `python monitor.py` sin errores
- [ ] Ver output de filtros en logs
- [ ] Primera señal procesada con filtros

---

## 🎯 Resultado Esperado

Cuando ejecutes `python monitor.py`, deberías ver:

```
# PREDICTOR SIGNAL MONITOR
# 2026-03-09 20:30:00
──────────────────────────────────────────────

✅ Discord notifications enabled

──────────────────────────────────────────────
FILTER CONFIGURATION
──────────────────────────────────────────────
✅ Loaded 2 filter(s):
   - Liquidity Filter (enabled)
   - Volatility Filter (enabled)
──────────────────────────────────────────────

══════════════════════════════════════════════
Checking mean_reversion (Mean Reversion (Z-score extremes))
Symbols: ETH/USDT, DOGE/USDT, LINK/USDT, XRP/USDT
══════════════════════════════════════════════

... [señales generadas] ...

══════════════════════════════════════════════
APPLYING FILTERS (1 signal(s))
══════════════════════════════════════════════

  Result: 1/1 signal(s) passed filters

══════════════════════════════════════════════

SUMMARY: Found 1 trading signal(s)
```

---

¡Filtros implementados exitosamente! 🎉