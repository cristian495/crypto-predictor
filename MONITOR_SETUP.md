# 🤖 Trading Signal Monitor - Setup Guide

Sistema de monitoreo automático que ejecuta tus estrategias periódicamente y envía notificaciones a Discord cuando encuentra señales de trading.

## 📋 Opciones de Deployment

### Opción 1: GitHub Actions (RECOMENDADO - 100% GRATIS)

**Ventajas:**
- ✅ Completamente gratis (2,000 minutos/mes)
- ✅ No requiere servidor
- ✅ Ejecución automática cada hora
- ✅ Logs accesibles desde GitHub UI

**Limitaciones:**
- ⚠️ Solo ejecuta cada hora como máximo (no cada 5-15 min)
- ⚠️ Máximo 6 horas de ejecución continua

#### Setup en GitHub Actions:

1. **Crear Discord Webhook:**
   - Ve a tu servidor Discord → Server Settings → Integrations → Webhooks
   - Click "New Webhook"
   - Copia la URL (ejemplo: `https://discord.com/api/webhooks/123456789/abcdef...`)

2. **Configurar GitHub Secret:**
   - Ve a tu repositorio en GitHub
   - Settings → Secrets and variables → Actions → New repository secret
   - Name: `DISCORD_WEBHOOK`
   - Value: Pega la URL del webhook
   - Click "Add secret"

3. **Push el código a GitHub:**
   ```bash
   git add .
   git commit -m "Add signal monitor"
   git push origin main
   ```

4. **Verificar que funciona:**
   - Ve a Actions tab en GitHub
   - Deberías ver "Trading Signal Monitor" workflow
   - Click "Run workflow" para ejecutar manualmente
   - Espera 5-10 minutos y verifica notificaciones en Discord

5. **Configurar frecuencia (opcional):**
   - Edita `.github/workflows/monitor.yml`
   - Cambia la línea `cron: '0 * * * *'`:
     - `0 * * * *` = cada hora
     - `*/30 * * * *` = cada 30 minutos (recomendado para 1h timeframe)
     - `0 */4 * * *` = cada 4 horas (para 4h timeframe)
     - `0 9,15,21 * * *` = a las 9am, 3pm, 9pm UTC

---

### Opción 2: Render.com (GRATIS con limitaciones)

**Ventajas:**
- ✅ Plan gratuito disponible
- ✅ Cron jobs nativos (cada 15 min mínimo en free tier)
- ✅ Fácil configuración

**Limitaciones:**
- ⚠️ Free tier: 750 horas/mes (suficiente para cron jobs)
- ⚠️ Servicio se "duerme" después de 15 min inactividad (no aplica a cron jobs)

#### Setup en Render:

1. **Crear cuenta en [Render.com](https://render.com)**

2. **Crear nuevo Cron Job:**
   - Dashboard → New → Cron Job
   - Conecta tu repositorio GitHub
   - Name: `trading-signal-monitor`
   - Environment: `Python 3`
   - Build Command: `pip install -r requirements.txt`
   - Command: `python monitor.py`
   - Schedule: `0 * * * *` (cada hora)

3. **Configurar Environment Variables:**
   - En la página del Cron Job → Environment
   - Add Environment Variable:
     - Key: `DISCORD_WEBHOOK`
     - Value: Tu webhook URL de Discord

4. **Deploy:**
   - Click "Create Cron Job"
   - Render automáticamente ejecutará según el schedule

---

### Opción 3: Railway.app (GRATIS $5/mes de crédito)

**Ventajas:**
- ✅ $5 gratis cada mes
- ✅ Cron jobs flexibles
- ✅ Mejor performance que Render free tier

**Limitaciones:**
- ⚠️ $5/mes generalmente solo alcanza para ~100-200 horas
- ⚠️ Requiere tarjeta de crédito (pero no cobra)

#### Setup en Railway:

1. **Crear cuenta en [Railway.app](https://railway.app)**

2. **Deploy desde GitHub:**
   - New Project → Deploy from GitHub repo
   - Selecciona tu repositorio
   - Railway detecta Python automáticamente

3. **Configurar Cron:**
   - Settings → Cron Schedule
   - Schedule: `0 * * * *`
   - Command: `python monitor.py`

4. **Variables de entorno:**
   - Variables → New Variable
   - `DISCORD_WEBHOOK` = tu webhook URL

---

### Opción 4: Local con Cron (Linux/Mac)

**Para ejecutar en tu propia máquina:**

1. **Configurar webhook:**
   ```bash
   export DISCORD_WEBHOOK="https://discord.com/api/webhooks/..."
   ```

2. **Editar crontab:**
   ```bash
   crontab -e
   ```

3. **Agregar job (cada hora):**
   ```
   0 * * * * cd /Users/cristiancruzbenel/Proyectos/Predictor && /usr/bin/python3 monitor.py >> monitor.log 2>&1
   ```

4. **Verificar:**
   ```bash
   tail -f monitor.log
   ```

---

## 🧪 Testing Manual

Antes de configurar ejecución automática, prueba manualmente:

```bash
# Opción 1: Con webhook en CLI
python monitor.py --discord-webhook "https://discord.com/api/webhooks/..."

# Opción 2: Con variable de entorno
export DISCORD_WEBHOOK="https://discord.com/api/webhooks/..."
python monitor.py

# Opción 3: Con archivo de configuración
echo '{"discord_webhook": "https://discord.com/api/webhooks/..."}' > monitor_config.json
python monitor.py --config monitor_config.json
```

**Verificaciones:**
1. El script debe ejecutarse sin errores
2. Deberías ver output en consola mostrando las estrategias escaneadas
3. Si hay señales, recibirás notificación en Discord
4. Si no hay señales, verás "SUMMARY: No signals found"

---

## 📊 Configuración de Estrategias

Edita [`monitor.py`](monitor.py) para ajustar qué estrategias monitorear:

```python
STRATEGIES = {
    "mean_reversion": {
        "path": "mean_reversion",
        "symbols": ["ETH/USDT", "DOGE/USDT", "LINK/USDT", "XRP/USDT"],
        "description": "Mean Reversion (Z-score extremes)",
        "optimize": False
    },
    "breakout_momentum": {
        "path": "breakout_momentum",
        "symbols": ["ETH/USDT", "BNB/USDT"],
        "description": "Breakout Momentum (Volatility + Volume)",
        "optimize": True  # Requiere --optimize para encontrar mejor threshold
    },
}
```

**Parámetros:**
- `symbols`: Lista de pares a monitorear
- `optimize`: Si `True`, ejecuta con `--optimize` para encontrar mejor `buy_threshold`
- `description`: Descripción que aparece en Discord

---

## 🔔 Formato de Notificaciones Discord

Cuando se detectan señales, recibirás un mensaje como:

```
🤖 Trading Signal Detected

**Mean Reversion (Z-score extremes)**
🟢 ETH/USDT LONG (prob=0.72, agree=3/3 Z=-2.89)
🔴 DOGE/USDT SHORT (prob=0.68, agree=2/3 Z=2.15)

**Breakout Momentum (Volatility + Volume)**
🟢 BNB/USDT LONG (prob=0.81, strategy=volatility ATR=1.5x)
```

---

## ⚙️ Ajustes Avanzados

### Cambiar frecuencia de escaneo

**GitHub Actions** (`.github/workflows/monitor.yml`):
```yaml
schedule:
  - cron: '*/30 * * * *'  # Cada 30 minutos
```

**Render/Railway**:
- Usar sintaxis cron estándar en dashboard

### Notificar cuando NO hay señales

Edita `monitor.py`, descomenta estas líneas:

```python
if webhook_url:
    send_discord_notification(
        webhook_url,
        "No trading signals detected in this scan.",
        color=0xaaaaaa
    )
```

### Agregar más estrategias

```python
STRATEGIES = {
    # ... existing strategies ...
    "trend_following": {
        "path": "trend_following",
        "symbols": ["POL/USDT", "BTC/USDT"],
        "description": "Trend Following (EMA crossovers)",
        "optimize": False
    }
}
```

---
