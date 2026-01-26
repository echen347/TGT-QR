"""
qr_data - Data ingestion and caching for quantitative research.

This package provides:
- Data providers for fetching OHLCV data from exchanges
- Parquet-based caching for efficient storage
- Data validation and quality checks

Example usage:
    from qr_data import BybitProvider, DataCache
    from pathlib import Path
    from datetime import datetime, timedelta

    # Initialize provider and cache
    provider = BybitProvider()
    cache = DataCache(Path("./data_cache"), provider)

    # Fetch data (auto-caches)
    data = cache.get(
        symbol="ETHUSDT",
        timeframe="15m",
        start=datetime.now() - timedelta(days=30),
        end=datetime.now()
    )

    print(f"Loaded {len(data)} bars")
    print(f"Valid: {data.is_valid()}")
"""

from .types import OHLCVData, timeframe_to_seconds, timeframe_to_minutes, TIMEFRAMES
from .cache import DataCache
from .providers import DataProvider, BybitProvider
from .validators import (
    validate_ohlcv,
    check_data_quality,
    fill_gaps,
    remove_duplicates,
    ensure_timezone,
)

__version__ = "0.1.0"

__all__ = [
    # Core types
    "OHLCVData",
    "TIMEFRAMES",
    "timeframe_to_seconds",
    "timeframe_to_minutes",
    # Cache
    "DataCache",
    # Providers
    "DataProvider",
    "BybitProvider",
    # Validators
    "validate_ohlcv",
    "check_data_quality",
    "fill_gaps",
    "remove_duplicates",
    "ensure_timezone",
]
