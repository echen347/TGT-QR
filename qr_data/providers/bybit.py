"""Bybit data provider implementation."""

import os
import time
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from pybit.unified_trading import HTTP

from .base import DataProvider
from ..types import OHLCVData, timeframe_to_seconds


# Bybit interval mapping (API uses string intervals)
BYBIT_INTERVALS = {
    "1m": "1",
    "3m": "3",
    "5m": "5",
    "15m": "15",
    "30m": "30",
    "1h": "60",
    "2h": "120",
    "4h": "240",
    "6h": "360",
    "12h": "720",
    "1d": "D",
    "1w": "W",
}


class BybitProvider(DataProvider):
    """
    Bybit USDT-perpetual futures data provider.

    Uses the Bybit V5 API to fetch historical kline data.
    Handles pagination and rate limiting automatically.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: bool = False,
        rate_limit_delay: float = 0.1,
    ):
        """
        Initialize Bybit provider.

        Args:
            api_key: Bybit API key (optional for public data)
            api_secret: Bybit API secret (optional for public data)
            testnet: Use testnet API
            rate_limit_delay: Delay between API calls in seconds
        """
        self.api_key = api_key or os.getenv("BYBIT_API_KEY", "")
        self.api_secret = api_secret or os.getenv("BYBIT_API_SECRET", "")
        self.testnet = testnet
        self.rate_limit_delay = rate_limit_delay

        self.session = HTTP(
            testnet=testnet,
            api_key=self.api_key if self.api_key else None,
            api_secret=self.api_secret if self.api_secret else None,
        )

        # Cache available symbols
        self._symbols_cache: Optional[list[str]] = None
        self._symbols_cache_time: float = 0
        self._symbols_cache_ttl: float = 3600  # 1 hour

    @property
    def name(self) -> str:
        return "bybit"

    def get_available_symbols(self) -> list[str]:
        """Get list of available USDT perpetual trading pairs."""
        now = time.time()

        # Return cached if fresh
        if self._symbols_cache and (now - self._symbols_cache_time) < self._symbols_cache_ttl:
            return self._symbols_cache

        try:
            response = self.session.get_instruments_info(category="linear")

            if response["retCode"] != 0:
                raise ConnectionError(f"Bybit API error: {response['retMsg']}")

            symbols = [
                item["symbol"]
                for item in response["result"]["list"]
                if item["symbol"].endswith("USDT")
                and item.get("status") == "Trading"
            ]

            self._symbols_cache = sorted(symbols)
            self._symbols_cache_time = now

            return self._symbols_cache

        except Exception as e:
            raise ConnectionError(f"Failed to fetch symbols: {e}")

    def get_available_timeframes(self) -> list[str]:
        """Get list of supported timeframes."""
        return list(BYBIT_INTERVALS.keys())

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> OHLCVData:
        """
        Fetch OHLCV data from Bybit.

        Args:
            symbol: Trading pair (e.g., "ETHUSDT")
            timeframe: Candle timeframe (e.g., "15m")
            start: Start datetime (UTC)
            end: End datetime (UTC)

        Returns:
            OHLCVData instance
        """
        # Convert timeframe to Bybit interval
        if timeframe not in BYBIT_INTERVALS:
            raise ValueError(f"Invalid timeframe: {timeframe}. Supported: {list(BYBIT_INTERVALS.keys())}")

        bybit_interval = BYBIT_INTERVALS[timeframe]

        # Ensure UTC timezone
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)

        # Convert to milliseconds
        start_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)

        # Fetch in chunks (Bybit returns max 1000 candles per request)
        interval_ms = timeframe_to_seconds(timeframe) * 1000
        chunk_size = 1000 * interval_ms

        all_klines = []
        current_start_ms = start_ms

        while current_start_ms < end_ms:
            current_end_ms = min(current_start_ms + chunk_size, end_ms)

            try:
                response = self.session.get_kline(
                    category="linear",
                    symbol=symbol.upper(),
                    interval=bybit_interval,
                    start=current_start_ms,
                    end=current_end_ms,
                    limit=1000,
                )

                if response["retCode"] != 0:
                    raise ConnectionError(f"Bybit API error: {response['retMsg']}")

                klines = response["result"]["list"]
                if not klines:
                    # No more data
                    break

                all_klines.extend(klines)
                current_start_ms = current_end_ms

                # Rate limiting
                time.sleep(self.rate_limit_delay)

            except Exception as e:
                raise ConnectionError(f"Failed to fetch data for {symbol}: {e}")

        if not all_klines:
            # Return empty OHLCVData
            return OHLCVData(
                symbol=symbol.upper(),
                timeframe=timeframe,
                df=pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"]),
            )

        # Convert to DataFrame
        # Bybit returns: [timestamp, open, high, low, close, volume, turnover]
        df = pd.DataFrame(
            all_klines,
            columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"],
        )

        # Process columns
        df["timestamp"] = pd.to_datetime(pd.to_numeric(df["timestamp"]), unit="ms", utc=True)
        df["open"] = pd.to_numeric(df["open"], errors="coerce")
        df["high"] = pd.to_numeric(df["high"], errors="coerce")
        df["low"] = pd.to_numeric(df["low"], errors="coerce")
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

        # Drop turnover column and duplicates
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        df = df.drop_duplicates(subset=["timestamp"], keep="last")

        # Sort by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Filter to requested range
        df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]

        return OHLCVData(
            symbol=symbol.upper(),
            timeframe=timeframe,
            df=df.reset_index(drop=True),
        )

    def fetch_recent(
        self,
        symbol: str,
        timeframe: str,
        num_bars: int = 200,
    ) -> OHLCVData:
        """
        Fetch recent OHLCV data.

        Args:
            symbol: Trading pair
            timeframe: Candle timeframe
            num_bars: Number of recent bars to fetch

        Returns:
            OHLCVData instance
        """
        now = datetime.now(timezone.utc)
        interval_seconds = timeframe_to_seconds(timeframe)
        start = now - pd.Timedelta(seconds=interval_seconds * num_bars * 1.1)  # 10% buffer

        return self.fetch_ohlcv(symbol, timeframe, start, now)
