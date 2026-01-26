"""Parquet-based caching layer for qr_data package."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from .providers.base import DataProvider
from .types import OHLCVData
from .validators import validate_ohlcv


class DataCache:
    """
    Parquet-based caching layer for OHLCV data.

    Caches data locally to avoid repeated API calls. Automatically fetches
    missing data from the provider when needed.
    """

    def __init__(
        self,
        cache_dir: Path,
        provider: DataProvider,
        validate_on_load: bool = True,
    ):
        """
        Initialize cache.

        Args:
            cache_dir: Directory to store cached data
            provider: Data provider for fetching new data
            validate_on_load: Validate data when loading from cache
        """
        self.cache_dir = Path(cache_dir)
        self.provider = provider
        self.validate_on_load = validate_on_load

        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, symbol: str, timeframe: str) -> Path:
        """Get cache file path for a symbol/timeframe."""
        return self.cache_dir / f"{symbol}_{timeframe}.parquet"

    def _get_meta_path(self, symbol: str, timeframe: str) -> Path:
        """Get metadata file path."""
        return self.cache_dir / f"{symbol}_{timeframe}.meta.json"

    def _load_metadata(self, symbol: str, timeframe: str) -> Optional[dict]:
        """Load cache metadata."""
        meta_path = self._get_meta_path(symbol, timeframe)
        if meta_path.exists():
            with open(meta_path) as f:
                return json.load(f)
        return None

    def _save_metadata(self, symbol: str, timeframe: str, data: OHLCVData) -> None:
        """Save cache metadata."""
        meta_path = self._get_meta_path(symbol, timeframe)
        meta = {
            "symbol": symbol,
            "timeframe": timeframe,
            "start_time": data.start_time.isoformat() if data.start_time else None,
            "end_time": data.end_time.isoformat() if data.end_time else None,
            "num_bars": len(data),
            "cached_at": datetime.now(timezone.utc).isoformat(),
            "provider": self.provider.name,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    def _load_from_cache(self, symbol: str, timeframe: str) -> Optional[OHLCVData]:
        """Load data from cache."""
        cache_path = self._get_cache_path(symbol, timeframe)

        if not cache_path.exists():
            return None

        try:
            df = pd.read_parquet(cache_path, engine="pyarrow")

            # Ensure timestamp is datetime with UTC
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

            data = OHLCVData(
                symbol=symbol,
                timeframe=timeframe,
                df=df,
            )

            if self.validate_on_load:
                is_valid, errors = data.validate()
                if not is_valid:
                    print(f"Warning: Cached data has validation errors: {errors}")

            return data

        except Exception as e:
            print(f"Warning: Failed to load cache for {symbol}_{timeframe}: {e}")
            return None

    def _save_to_cache(self, data: OHLCVData) -> None:
        """Save data to cache."""
        cache_path = self._get_cache_path(data.symbol, data.timeframe)

        try:
            data.df.to_parquet(
                cache_path,
                index=False,
                engine="pyarrow",
                compression="snappy",
            )
            self._save_metadata(data.symbol, data.timeframe, data)
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")

    def _merge_data(self, existing: OHLCVData, new: OHLCVData) -> OHLCVData:
        """Merge existing and new data, removing duplicates."""
        combined = pd.concat([existing.df, new.df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["timestamp"], keep="last")
        combined = combined.sort_values("timestamp").reset_index(drop=True)

        return OHLCVData(
            symbol=existing.symbol,
            timeframe=existing.timeframe,
            df=combined,
        )

    def get(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        refresh: bool = False,
    ) -> OHLCVData:
        """
        Get OHLCV data, fetching from provider if needed.

        Args:
            symbol: Trading pair (e.g., "ETHUSDT")
            timeframe: Candle timeframe (e.g., "15m")
            start: Start datetime (UTC)
            end: End datetime (UTC)
            refresh: Force refresh from provider

        Returns:
            OHLCVData with requested data
        """
        symbol = symbol.upper()

        # Ensure UTC timezone
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)

        # Try to load from cache
        cached = None if refresh else self._load_from_cache(symbol, timeframe)

        if cached is not None:
            # Check if cache covers the requested range
            cache_start = cached.start_time
            cache_end = cached.end_time

            if cache_start and cache_end:
                # Make sure cached times are timezone-aware for comparison
                if cache_start.tzinfo is None:
                    cache_start = cache_start.replace(tzinfo=timezone.utc)
                if cache_end.tzinfo is None:
                    cache_end = cache_end.replace(tzinfo=timezone.utc)

                if cache_start <= start and cache_end >= end:
                    # Cache covers the range - slice and return
                    return cached.slice(start, end)

                # Cache is partial - fetch missing ranges
                if cache_start > start:
                    # Fetch data before cache
                    before_data = self.provider.fetch_ohlcv(symbol, timeframe, start, cache_start)
                    cached = self._merge_data(before_data, cached)

                if cache_end < end:
                    # Fetch data after cache
                    after_data = self.provider.fetch_ohlcv(symbol, timeframe, cache_end, end)
                    cached = self._merge_data(cached, after_data)

                # Save merged data
                self._save_to_cache(cached)

                return cached.slice(start, end)

        # No usable cache - fetch from provider
        data = self.provider.fetch_ohlcv(symbol, timeframe, start, end)

        # Save to cache
        if len(data) > 0:
            self._save_to_cache(data)

        return data

    def invalidate(self, symbol: Optional[str] = None, timeframe: Optional[str] = None) -> int:
        """
        Invalidate cached data.

        Args:
            symbol: Symbol to invalidate (None = all symbols)
            timeframe: Timeframe to invalidate (None = all timeframes)

        Returns:
            Number of cache files removed
        """
        count = 0

        for cache_file in self.cache_dir.glob("*.parquet"):
            stem = cache_file.stem
            parts = stem.rsplit("_", 1)

            if len(parts) != 2:
                continue

            file_symbol, file_timeframe = parts

            if symbol and file_symbol != symbol.upper():
                continue
            if timeframe and file_timeframe != timeframe:
                continue

            # Remove parquet and metadata files
            cache_file.unlink(missing_ok=True)
            meta_file = cache_file.with_suffix(".meta.json")
            meta_file.unlink(missing_ok=True)
            count += 1

        return count

    def list_cached(self) -> list[dict]:
        """
        List all cached data.

        Returns:
            List of dicts with cache info (symbol, timeframe, start, end, num_bars)
        """
        cached = []

        for meta_file in self.cache_dir.glob("*.meta.json"):
            try:
                with open(meta_file) as f:
                    meta = json.load(f)
                    cached.append({
                        "symbol": meta["symbol"],
                        "timeframe": meta["timeframe"],
                        "start": meta.get("start_time"),
                        "end": meta.get("end_time"),
                        "num_bars": meta.get("num_bars"),
                        "cached_at": meta.get("cached_at"),
                    })
            except Exception:
                continue

        return sorted(cached, key=lambda x: (x["symbol"], x["timeframe"]))

    def get_cache_stats(self) -> dict:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        cached = self.list_cached()
        total_bars = sum(c.get("num_bars", 0) for c in cached)

        # Calculate disk usage
        disk_usage = sum(
            f.stat().st_size
            for f in self.cache_dir.glob("*")
            if f.is_file()
        )

        return {
            "cache_dir": str(self.cache_dir),
            "num_symbols": len(set(c["symbol"] for c in cached)),
            "num_files": len(cached),
            "total_bars": total_bars,
            "disk_usage_mb": disk_usage / (1024 * 1024),
        }
