"""
Quick diagnostic script to check ADX values
"""
import pandas as pd
import numpy as np
from data_loader import fetch_ohlcv
from features import add_features

# Test with ETH
symbol = "ETH/USDT"
print(f"Analyzing {symbol}...")

df = fetch_ohlcv(symbol, timeframe="1h", days=1095)
# Add funding/OI
df["funding_rate"] = 0.0
df["open_interest"] = 0.0
df = add_features(df)

# Check last 100 rows
recent = df.tail(100)

print(f"\nADX Statistics (last 100 bars):")
print(f"  Mean: {recent['adx_14'].mean():.2f}")
print(f"  Median: {recent['adx_14'].median():.2f}")
print(f"  Min: {recent['adx_14'].min():.2f}")
print(f"  Max: {recent['adx_14'].max():.2f}")
print(f"  % below 25: {(recent['adx_14'] < 25).sum() / len(recent) * 100:.1f}%")
print(f"  % below 30: {(recent['adx_14'] < 30).sum() / len(recent) * 100:.1f}%")

print(f"\nLast 10 values:")
print(recent[['timestamp', 'close', 'adx_14', 'zscore_50', 'rsi_14']].tail(10).to_string())

# Check distribution in test period
test_rows = df.iloc[-557:]  # last 557 rows (test set)
print(f"\nTest set ADX distribution:")
print(f"  Mean: {test_rows['adx_14'].mean():.2f}")
print(f"  % below 25: {(test_rows['adx_14'] < 25).sum() / len(test_rows) * 100:.1f}%")
print(f"  % below 30: {(test_rows['adx_14'] < 30).sum() / len(test_rows) * 100:.1f}%")
print(f"  % below 35: {(test_rows['adx_14'] < 35).sum() / len(test_rows) * 100:.1f}%")