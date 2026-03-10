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
from pathlib import Path
from datetime import datetime, timezone
import requests
import os
import yaml


# ── Configuration ────────────────────────────────────────────────────
STRATEGIES = {
    "mean_reversion": {
        "path": "mean_reversion",
        "symbols": [ "XRP/USDT"],
        "description": "Mean Reversion (Z-score extremes)",
        "optimize": False,  # No optimization needed
    },
    # "breakout_momentum": {
    #     "path": "breakout_momentum",
    #     "symbols": ["ETH/USDT", "BNB/USDT"],
    #     "description": "Breakout Momentum (Volatility + Volume)",
    #     "optimize": True  # Requires --optimize flag
    # },
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
        "title": "🤖 Trading Signal Detected",
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
    Run a strategy and parse signals from stdout.

    Returns:
        List of dicts with signal info: [{"symbol": "ETH/USDT", "signal": "LONG", "prob": 0.75, ...}]
    """
    strategy_path = Path(config["path"])
    symbols = config["symbols"]
    optimize = config.get("optimize", False)

    print(f"\n{'=' * 70}")
    print(f"Checking {strategy_name} ({config['description']})")
    print(f"Symbols: {', '.join(symbols)}")
    if optimize:
        print(f"Optimize: enabled (finding best buy_threshold)")
    print(f"{'=' * 70}")

    # Build command
    cmd = [
        sys.executable,
        "main.py",
        "--symbols",
        *symbols,
        "--scan"
    ]

    # Add --optimize flag if needed
    if optimize:
        cmd.append("--optimize")

    # Add --buy-threshold for testing (forces signals with low threshold)
    test_threshold = config.get("test_threshold")
    if test_threshold:
        cmd.extend(["--buy-threshold", str(test_threshold)])

    try:
        result = subprocess.run(
            cmd,
            cwd=strategy_path,
            capture_output=True,
            text=True,
            timeout=600  # 10 min max
        )

        if result.returncode != 0:
            print(f"❌ Strategy failed with code {result.returncode}")
            print(f"stderr: {result.stderr[:500]}")
            return []

        # Parse signals from output
        signals = parse_signals(result.stdout, strategy_name)

        if signals:
            print(f"✅ Found {len(signals)} signal(s)")
            for sig in signals:
                print(f"  {sig['symbol']:<12} {sig['signal']:<8} prob={sig['prob']:.3f}")
        else:
            print("  No signals found")

        return signals

    except subprocess.TimeoutExpired:
        print(f"❌ Strategy timed out after 10 minutes")
        return []
    except Exception as e:
        print(f"❌ Error running strategy: {e}")
        return []


def parse_signals(stdout: str, strategy: str) -> list:
    """
    Parse trading signals from main.py stdout.

    Expected format (in CURRENT SIGNALS section):
      ETH/USDT    LONG     prob=0.7234 agree=3/3 Z=-2.89
      BTC/USDT    NO TRADE  Z=0.45

    Returns:
        [{"symbol": "ETH/USDT", "signal": "LONG", "prob": 0.72, "extra": "agree=3/3 Z=-2.89"}]
    """
    signals = []
    in_signals_section = False

    for line in stdout.splitlines():
        line_stripped = line.strip()

        # Detect start of CURRENT SIGNALS section
        if "CURRENT SIGNALS" in line:
            in_signals_section = True
            continue

        # Detect end of CURRENT SIGNALS section (next section starts with "[")
        if in_signals_section and line_stripped.startswith("["):
            break

        # Only parse lines within CURRENT SIGNALS section
        if not in_signals_section:
            continue

        # Skip empty lines and separator lines (but not signal lines that contain "prob=")
        if not line_stripped:
            continue
        if "=" in line_stripped and "prob=" not in line_stripped:
            continue

        # Parse signal line: "  ETH/USDT    LONG     prob=0.7234 agree=3/3 Z=-2.89"
        parts = line_stripped.split()
        if len(parts) < 2:
            continue

        symbol = parts[0]
        signal_type = parts[1]

        # Only keep LONG/SHORT signals (skip NO TRADE)
        if signal_type not in ["LONG", "SHORT"]:
            continue

        # Extract probability
        prob = 0.0
        extra_info = []
        for part in parts[2:]:
            if part.startswith("prob="):
                try:
                    prob = float(part.split("=")[1])
                except ValueError:
                    pass
            else:
                extra_info.append(part)

        signals.append({
            "symbol": symbol,
            "signal": signal_type,
            "prob": prob,
            "extra": " ".join(extra_info),
            "strategy": strategy
        })

    return signals


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
    print(f"\n{'#' * 70}")
    print(f"# PREDICTOR SIGNAL MONITOR")
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 70}\n")

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
            rejected_signals = []

            # Apply filters to signals from this strategy
            if signals and filter_pipeline:
                print(f"\nApplying filters to {len(signals)} signal(s) from {strategy_name}...")
                original_count = len(signals)
                signals, rejected_signals = filter_pipeline.filter_signals(signals)
                filtered_count = len(signals)
                print(f"  Result: {filtered_count}/{original_count} signal(s) passed filters")

            # Send notification for this strategy (always, regardless of signals)
            if webhook_url:
                if signals:
                    # Send approved signals notification
                    message = format_signal_summary(signals)
                    send_discord_notification(webhook_url, message, color=0x00ff00)
                    print(f"✅ Sent Discord notification for {len(signals)} signal(s)")
                else:
                    # Send "no signals" notification for this strategy
                    desc = strategy_config.get("description", strategy_name)
                    message = f"**{desc}**\nNo signals found"
                    send_discord_notification(webhook_url, message, color=0x808080)
                    print(f"✅ Sent Discord notification: No signals")

                # Send rejected signals notification if any
                if rejected_signals:
                    message = format_rejected_signals_summary(rejected_signals, strategy_config)
                    send_discord_notification(webhook_url, message, color=0xFFA500)  # Orange
                    print(f"⚠️  Sent Discord notification for {len(rejected_signals)} rejected signal(s)")

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


def format_rejected_signals_summary(rejected_signals: list, strategy_config: dict) -> str:
    """
    Format rejected signals into Discord markdown message.

    Example:
        **⚠️ Mean Reversion - Filtered Signals**
        🚫 XRP/USDT SHORT (prob=0.65)
        Reason: Low liquidity: volume 11.5% of 30d average (min 50.0%)
    """
    desc = strategy_config.get("description", "Strategy")
    lines = [f"**⚠️ {desc} - Filtered Signals**"]

    for sig in rejected_signals:
        emoji = "🟢" if sig["signal"] == "LONG" else "🔴"
        lines.append(
            f"🚫 {emoji} {sig['symbol']} **{sig['signal']}** "
            f"(prob={sig['prob']:.2f}, {sig['extra']})"
        )
        # Add rejection reason on next line with indent
        reason = sig.get('rejection_reason', 'Unknown reason')
        lines.append(f"   └ *{reason}*")

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
    main()