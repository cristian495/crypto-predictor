#!/usr/bin/env python3
"""
monitor.py
Periodic signal monitoring for all strategies.
Sends Discord notifications when trading opportunities are found.

Usage:
    python monitor.py --discord-webhook "https://discord.com/api/webhooks/..."
    python monitor.py --config monitor_config.json

    # Or use environment variable:
    export DISCORD_WEBHOOK="https://discord.com/api/webhooks/..."
    python monitor.py
"""

import sys
import json
import argparse
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime, timezone
import time
import select
import requests
import os
import yaml
from dotenv import load_dotenv

MONITOR_SUBPROCESS_TIMEOUT_SEC = 1200


# ── Configuration ────────────────────────────────────────────────────
STRATEGIES = {
    # "mean_reversion": {
    #     "path": "mean_reversion",
    #     "symbols": ["ETH/USDT", "DOGE/USDT", "LINK/USDT", "XRP/USDT"],
    #     "description": "Mean Reversion (Z-score extremes)",
    #     "optimize": False,  # No optimization needed
    # },
    # "breakout_momentum": {
    #     "path": "breakout_momentum",
    #     "symbols": ["ETH/USDT", "BNB/USDT"],
    #     "description": "Breakout Momentum (Volatility + Volume)",
    #     "optimize": True  # Requires --optimize flag
    # },
    "buy_the_dip": {
        "path": "buy_the_dip",
        "description": "Buy the Dip",
    },
    "downtrend_breakout": {
        "path": "downtrend_breakout",
        "description": "Downtrend Breakout v1",
    },
    "sell_the_rip": {
        "path": "sell_the_rip",
        "description": "Sell the Rip",
    },
    # trend_following requires 4h timeframe, slower signals
    # "trend_following": {
    #     "path": "trend_following",
    #     "symbols": ["POL/USDT", "BTC/USDT"],
    #     "description": "Trend Following (EMA crossovers)",
    #     "optimize": False
    # }
}


# ── Discord Notification ─────────────────────────────────────────────
def send_discord_notification(webhook_url: str, message: str, color: int = 0x00ff00):
    """
    Send rich embed notification to Discord webhook.

    Args:
        webhook_url: Discord webhook URL
        message: Message content (supports markdown)
        color: Embed color (0x00ff00 = green, 0xff0000 = red, 0xffaa00 = orange)
    """
    if not webhook_url:
        print("⚠️  No Discord webhook configured, skipping notification")
        return

    embed = {
        "title": "🤖 Trading Signal Scan",
        "description": message,
        "color": color,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "footer": {
            "text": "Predictor Signal Monitor"
        }
    }

    payload = {
        "embeds": [embed]
    }

    try:
        response = requests.post(webhook_url, json=payload, timeout=10)
        response.raise_for_status()
        print("✅ Discord notification sent successfully")
    except Exception as e:
        print(f"❌ Failed to send Discord notification: {e}")


def send_error_notification(webhook_url: str, strategy: str, error: str):
    """Send error notification to Discord."""
    message = f"**Strategy:** {strategy}\n**Error:** ```{error[:500]}```"
    send_discord_notification(webhook_url, message, color=0xff0000)


# ── Signal Detection ─────────────────────────────────────────────────
def check_strategy_signals(strategy_name: str, config: dict) -> list:
    """
    Run a strategy and parse signals from JSON output.

    Returns:
        List of dicts with signal info: [{"symbol": "ETH/USDT", "signal": "LONG", "prob": 0.75, ...}]
    """
    strategy_path = Path(config["path"])
    timeout_sec = MONITOR_SUBPROCESS_TIMEOUT_SEC

    print(f"\n{'=' * 70}")
    print(f"Checking {strategy_name} ({config['description']})")
    print("Using strategy defaults from config.py")
    print(f"Timeout: {timeout_sec}s")
    print(f"{'=' * 70}")

    if not strategy_path.exists():
        print(f"❌ Strategy path not found: {strategy_path}")
        return []

    # Build command
    cmd = [
        sys.executable,
        "-u",
        "main.py",
        "--scan"
    ]
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            signals_path = Path(tmpdir) / f"{strategy_name}_signals.json"
            cmd_with_json = cmd + ["--signals-json", str(signals_path)]

            print(f"\n--- {strategy_name} live logs ---")
            child_env = os.environ.copy()
            child_env["PYTHONUNBUFFERED"] = "1"
            process = subprocess.Popen(
                cmd_with_json,
                cwd=strategy_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,
                env=child_env,
            )
            start_ts = time.monotonic()
            last_output_ts = start_ts
            last_heartbeat_ts = start_ts
            heartbeat_interval_s = 5.0
            stdout_pipe = process.stdout
            if stdout_pipe is None:
                process.kill()
                print("❌ Failed to capture strategy stdout")
                return []
            fd = stdout_pipe.fileno()

            while True:
                now = time.monotonic()
                if now - start_ts > timeout_sec:
                    process.kill()
                    raise subprocess.TimeoutExpired(cmd_with_json, timeout_sec)

                ready, _, _ = select.select([fd], [], [], 0.5)
                if ready:
                    chunk = os.read(fd, 4096)
                    if chunk:
                        last_output_ts = time.monotonic()
                        sys.stdout.buffer.write(chunk)
                        sys.stdout.flush()
                        continue

                if process.poll() is not None:
                    break

                now = time.monotonic()
                if (
                    now - last_output_ts >= heartbeat_interval_s
                    and now - last_heartbeat_ts >= heartbeat_interval_s
                ):
                    print(f"[{strategy_name}] running... {int(now - start_ts)}s elapsed", flush=True)
                    last_heartbeat_ts = now

            returncode = process.wait()

            if returncode != 0:
                print(f"❌ Strategy failed with code {returncode}")
                return []

            if not signals_path.exists():
                print("❌ Strategy did not generate signals JSON")
                return []

            signals = parse_signals_json(signals_path, strategy_name)

        if signals:
            print(f"✅ Found {len(signals)} signal(s)")
            for sig in signals:
                print(f"  {sig['symbol']:<12} {sig['signal']:<8} prob={sig['prob']:.3f}")
        else:
            print("  No signals found")

        return signals

    except subprocess.TimeoutExpired:
        print(f"❌ Strategy timed out after {timeout_sec} seconds")
        return []
    except Exception as e:
        print(f"❌ Error running strategy: {e}")
        return []


def parse_signals_json(path: Path, strategy: str) -> list:
    """
    Parse trading signals from a JSON file produced by strategy main.py.

    Accepts either:
      - {"signals": [...]} or
      - [...] (list of signals)
    """
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"⚠️  Failed to read signals JSON: {e}")
        return []

    if isinstance(data, dict):
        signals = data.get("signals", [])
    elif isinstance(data, list):
        signals = data
    else:
        return []

    # Normalize and ensure strategy field
    normalized = []
    for sig in signals:
        if not isinstance(sig, dict):
            continue
        if sig.get("signal") not in ["LONG", "SHORT"]:
            continue
        sig = sig.copy()
        sig.setdefault("strategy", strategy)
        normalized.append(sig)

    return normalized


# ── Load Filter Configuration ────────────────────────────────────────
def load_filter_config():
    """Load filter configuration from YAML file."""
    filter_config_path = Path("filter_config.yaml")
    if filter_config_path.exists():
        try:
            with open(filter_config_path) as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"⚠️  Failed to load filter config: {e}")
            return {}
    return {}


# ── Main Monitor Loop ────────────────────────────────────────────────
def run_monitor(webhook_url: str = None, config_file: str = None):
    """
    Run signal monitoring for all strategies.
    """
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)

    print(f"\n{'#' * 70}")
    print(f"# PREDICTOR SIGNAL MONITOR")
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 70}\n")

    load_dotenv(override=False)

    # Load config from file if provided
    if config_file and Path(config_file).exists():
        with open(config_file) as f:
            config_data = json.load(f)
            webhook_url = webhook_url or config_data.get("discord_webhook")

    # Try environment variable if no webhook URL provided
    if not webhook_url:
        webhook_url = os.getenv("DISCORD_WEBHOOK")

    if webhook_url:
        print(f"✅ Discord notifications enabled")
    else:
        print(f"⚠️  Discord notifications disabled (no webhook URL)")

    # Load filter configuration
    filter_config = load_filter_config()
    filter_pipeline = None

    if filter_config.get('filters'):
        print(f"\n{'─' * 70}")
        print("FILTER CONFIGURATION")
        print(f"{'─' * 70}")
        try:
            from filters import FilterPipeline
            filter_pipeline = FilterPipeline(filter_config)
        except Exception as e:
            print(f"⚠️  Failed to initialize filters: {e}")
            print("   Continuing without filters...")
        print(f"{'─' * 70}\n")
    else:
        print(f"⚠️  No filter configuration found (filter_config.yaml)\n")

    all_signals = []

    # Check each strategy
    for strategy_name, strategy_config in STRATEGIES.items():
        try:
            signals = check_strategy_signals(strategy_name, strategy_config)

            # Apply filters to signals from this strategy
            if signals and filter_pipeline:
                print(f"\nApplying filters to {len(signals)} signal(s) from {strategy_name}...")
                original_count = len(signals)
                filtered = filter_pipeline.filter_signals(signals)
                if isinstance(filtered, tuple):
                    signals, _rejected = filtered
                else:
                    signals = filtered
                filtered_count = len(signals)
                print(f"  Result: {filtered_count}/{original_count} signal(s) passed filters")

            # Send notification for this strategy (always, regardless of signals)
            if webhook_url:
                if signals:
                    # Send signals notification
                    message = format_signal_summary(signals)
                    send_discord_notification(webhook_url, message, color=0x00ff00)
                    print(f"✅ Sent Discord notification for {len(signals)} signal(s)")
                else:
                    # Send "no signals" notification for this strategy
                    desc = strategy_config.get("description", strategy_name)
                    message = f"**{desc}**\nNo signals found"
                    send_discord_notification(webhook_url, message, color=0x808080)
                    print(f"✅ Sent Discord notification: No signals")

            all_signals.extend(signals)
        except Exception as e:
            error_msg = f"Failed to check {strategy_name}: {e}"
            print(f"❌ {error_msg}")
            if webhook_url:
                send_error_notification(webhook_url, strategy_name, str(e))

    # Print summary
    print(f"\n{'=' * 70}")
    if all_signals:
        print(f"SUMMARY: Found {len(all_signals)} trading signal(s) total")
    else:
        print(f"SUMMARY: No signals found across all strategies")
    print(f"{'=' * 70}\n")


def format_signal_summary(signals: list) -> str:
    """
    Format signals into Discord markdown message.

    Example:
        **Mean Reversion**
        • ETH/USDT LONG (prob=0.72, agree=3/3, Z=-2.89)
        • DOGE/USDT SHORT (prob=0.68, agree=2/3, Z=2.45)

        **Breakout Momentum**
        • BNB/USDT LONG (prob=0.81, ATR=1.5x)
    """
    grouped = {}
    for sig in signals:
        strategy = sig["strategy"]
        if strategy not in grouped:
            grouped[strategy] = []
        grouped[strategy].append(sig)

    lines = []
    for strategy, sigs in grouped.items():
        # Get strategy description
        desc = STRATEGIES.get(strategy, {}).get("description", strategy)
        lines.append(f"**{desc}**")

        for sig in sigs:
            emoji = "🟢" if sig["signal"] == "LONG" else "🔴"
            lines.append(
                f"{emoji} {sig['symbol']} **{sig['signal']}** "
                f"(prob={sig['prob']:.2f}, {sig['extra']})"
            )

        lines.append("")  # Blank line between strategies

    return "\n".join(lines)


# ── CLI ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Monitor trading strategies and send Discord notifications"
    )
    parser.add_argument(
        "--discord-webhook",
        type=str,
        help="Discord webhook URL for notifications"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to JSON config file with webhook URL and settings"
    )

    args = parser.parse_args()

    run_monitor(
        webhook_url=args.discord_webhook,
        config_file=args.config
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nMonitor interrupted by user")
        sys.exit(130)
