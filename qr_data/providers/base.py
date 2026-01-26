"""Abstract base class for data providers."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

from ..types import OHLCVData


class DataProvider(ABC):
    """
    Abstract base class for market data providers.

    Implementations should handle:
    - Rate limiting
    - Pagination for large requests
    - Error handling and retries
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return provider name."""
        ...

    @abstractmethod
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> OHLCVData:
        """
        Fetch OHLCV data for a symbol.

        Args:
            symbol: Trading pair (e.g., "ETHUSDT")
            timeframe: Candle timeframe (e.g., "15m")
            start: Start datetime (UTC)
            end: End datetime (UTC)

        Returns:
            OHLCVData instance with the requested data

        Raises:
            ValueError: If symbol or timeframe is invalid
            ConnectionError: If API request fails
        """
        ...

    @abstractmethod
    def get_available_symbols(self) -> list[str]:
        """
        Get list of available trading symbols.

        Returns:
            List of symbol strings (e.g., ["BTCUSDT", "ETHUSDT", ...])
        """
        ...

    @abstractmethod
    def get_available_timeframes(self) -> list[str]:
        """
        Get list of supported timeframes.

        Returns:
            List of timeframe strings (e.g., ["1m", "5m", "15m", ...])
        """
        ...

    def is_symbol_available(self, symbol: str) -> bool:
        """
        Check if a symbol is available.

        Args:
            symbol: Trading pair to check

        Returns:
            True if symbol is available
        """
        return symbol.upper() in self.get_available_symbols()

    def is_timeframe_available(self, timeframe: str) -> bool:
        """
        Check if a timeframe is supported.

        Args:
            timeframe: Timeframe to check

        Returns:
            True if timeframe is supported
        """
        return timeframe in self.get_available_timeframes()

    def validate_request(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> None:
        """
        Validate a data request.

        Args:
            symbol: Trading pair
            timeframe: Candle timeframe
            start: Start datetime
            end: End datetime

        Raises:
            ValueError: If any parameter is invalid
        """
        if not self.is_symbol_available(symbol):
            raise ValueError(
                f"Symbol '{symbol}' not available. "
                f"Use get_available_symbols() to see options."
            )

        if not self.is_timeframe_available(timeframe):
            raise ValueError(
                f"Timeframe '{timeframe}' not supported. "
                f"Available: {self.get_available_timeframes()}"
            )

        if start >= end:
            raise ValueError(f"Start time ({start}) must be before end time ({end})")

        if end > datetime.now(end.tzinfo if end.tzinfo else None):
            raise ValueError(f"End time ({end}) cannot be in the future")
