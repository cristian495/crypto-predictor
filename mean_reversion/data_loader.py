"""
data_loader.py
Data ingestion: OHLCV from Binance spot + funding rates and open interest
from Binance futures via ccxt.
"""

import time
import pandas as pd
import ccxt


def fetch_ohlcv(symbol: str = "BTC/USDT",
                timeframe: str = "1h",
                days: int = 730) -> pd.DataFrame:
    """
    Download historical candles from Binance spot.

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume.
    """
    exchange = ccxt.binance({"enableRateLimit": True})

    since = exchange.parse8601(
        (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days))
        .strftime("%Y-%m-%dT%H:%M:%SZ")
    )

    all_candles = []
    limit = 1000

    print(f"  Downloading {symbol} {timeframe} — last {days} days...")

    while True:
        candles = exchange.fetch_ohlcv(
            symbol, timeframe, since=since, limit=limit
        )
        if not candles:
            break

        all_candles.extend(candles)
        since = candles[-1][0] + 1

        if len(candles) < limit:
            break

        time.sleep(exchange.rateLimit / 1000)

    df = pd.DataFrame(
        all_candles,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp")
    df = df.reset_index(drop=True)

    print(f"  Downloaded {len(df)} candles "
          f"({df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]})")

    return df


def _to_futures_symbol(symbol: str) -> str:
    """Convert spot symbol (BTC/USDT) to futures format (BTC/USDT:USDT)."""
    if ":USDT" not in symbol:
        return symbol + ":USDT"
    return symbol


def fetch_funding_rates(symbol: str = "BTC/USDT",
                        days: int = 730) -> pd.DataFrame:
    """
    Download historical funding rates from Binance futures.
    Funding is paid every 8h, so ~3 data points per day.

    Returns:
        DataFrame with columns: timestamp, funding_rate.
        Empty DataFrame if futures data is unavailable for this symbol.
    """
    try:
        exchange = ccxt.binanceusdm({"enableRateLimit": True})
        futures_symbol = _to_futures_symbol(symbol)

        since = exchange.parse8601(
            (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days))
            .strftime("%Y-%m-%dT%H:%M:%SZ")
        )

        all_rates = []
        limit = 1000

        print(f"  Downloading funding rates for {futures_symbol}...")

        while True:
            rates = exchange.fetch_funding_rate_history(
                futures_symbol, since=since, limit=limit
            )
            if not rates:
                break

            all_rates.extend(rates)
            since = rates[-1]["timestamp"] + 1

            if len(rates) < limit:
                break

            time.sleep(exchange.rateLimit / 1000)

        if not all_rates:
            print(f"  No funding rate data for {symbol}")
            return pd.DataFrame(columns=["timestamp", "funding_rate"])

        df = pd.DataFrame([
            {
                "timestamp": pd.to_datetime(r["timestamp"], unit="ms", utc=True),
                "funding_rate": r["fundingRate"],
            }
            for r in all_rates
        ])
        df = df.drop_duplicates(subset="timestamp").sort_values("timestamp")
        df = df.reset_index(drop=True)

        print(f"  Downloaded {len(df)} funding rate records")
        return df

    except Exception as e:
        print(f"  Funding rates unavailable for {symbol}: {e}")
        return pd.DataFrame(columns=["timestamp", "funding_rate"])


def fetch_open_interest(symbol: str = "BTC/USDT",
                        timeframe: str = "1h",
                        days: int = 730) -> pd.DataFrame:
    """
    Download historical open interest from Binance futures.

    Returns:
        DataFrame with columns: timestamp, open_interest.
        Empty DataFrame if futures data is unavailable.
    """
    try:
        exchange = ccxt.binanceusdm({"enableRateLimit": True})
        futures_symbol = _to_futures_symbol(symbol)

        since = exchange.parse8601(
            (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days))
            .strftime("%Y-%m-%dT%H:%M:%SZ")
        )

        all_oi = []
        limit = 500

        print(f"  Downloading open interest for {futures_symbol}...")

        # Binance OI history has limited lookback — fetch in smaller chunks
        # and handle API errors gracefully
        chunk_days = 30
        chunk_since = since
        chunk_end = exchange.parse8601(
            pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
        )

        while chunk_since < chunk_end:
            try:
                oi_data = exchange.fetch_open_interest_history(
                    futures_symbol, timeframe=timeframe,
                    since=chunk_since, limit=limit,
                )
            except Exception:
                # Some pairs or date ranges don't support OI
                break

            if not oi_data:
                break

            all_oi.extend(oi_data)
            chunk_since = oi_data[-1]["timestamp"] + 1

            if len(oi_data) < limit:
                break

            time.sleep(exchange.rateLimit / 1000)

        if not all_oi:
            print(f"  No open interest data for {symbol}")
            return pd.DataFrame(columns=["timestamp", "open_interest"])

        df = pd.DataFrame([
            {
                "timestamp": pd.to_datetime(r["timestamp"], unit="ms", utc=True),
                "open_interest": r["openInterestAmount"],
            }
            for r in all_oi
        ])
        df = df.drop_duplicates(subset="timestamp").sort_values("timestamp")
        df = df.reset_index(drop=True)

        print(f"  Downloaded {len(df)} open interest records")
        return df

    except Exception as e:
        print(f"  Open interest unavailable for {symbol}: {e}")
        return pd.DataFrame(columns=["timestamp", "open_interest"])


def load_all_data(symbol: str = "BTC/USDT",
                  timeframe: str = "1h",
                  days: int = 730) -> pd.DataFrame:
    """
    Load OHLCV + funding rates + open interest, merged by timestamp.
    Derivatives data is forward-filled to match OHLCV frequency.
    Missing derivatives data is filled with 0.
    """
    ohlcv = fetch_ohlcv(symbol, timeframe, days)

    # Funding rates (every 8h → forward-fill to 1h)
    funding = fetch_funding_rates(symbol, days)
    if len(funding) > 0:
        ohlcv = pd.merge_asof(
            ohlcv.sort_values("timestamp"),
            funding.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
        )
    else:
        ohlcv["funding_rate"] = 0.0

    # Open interest
    oi = fetch_open_interest(symbol, timeframe, days)
    if len(oi) > 0:
        ohlcv = pd.merge_asof(
            ohlcv.sort_values("timestamp"),
            oi.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
        )
    else:
        ohlcv["open_interest"] = 0.0

    # Fill any remaining NaN in derivatives columns
    ohlcv["funding_rate"] = ohlcv["funding_rate"].fillna(0.0)
    ohlcv["open_interest"] = ohlcv["open_interest"].fillna(0.0)

    ohlcv = ohlcv.reset_index(drop=True)
    print(f"  Final dataset: {len(ohlcv)} rows, "
          f"cols: {list(ohlcv.columns)}")

    return ohlcv


if __name__ == "__main__":
    df = load_all_data("BTC/USDT", "1h", 30)
    print(df.tail())
    print(f"\nShape: {df.shape}")
