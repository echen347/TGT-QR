"""Core data types for qr_data package."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class OHLCVData:
    """
    OHLCV (Open, High, Low, Close, Volume) data container.

    Attributes:
        symbol: Trading pair symbol (e.g., "ETHUSDT")
        timeframe: Candle timeframe (e.g., "1m", "5m", "15m", "1h", "4h", "1d")
        df: DataFrame with columns [timestamp, open, high, low, close, volume]
    """
    symbol: str
    timeframe: str
    df: pd.DataFrame

    # Metadata
    start_time: Optional[datetime] = field(default=None)
    end_time: Optional[datetime] = field(default=None)

    def __post_init__(self):
        """Set start/end times from dataframe if not provided."""
        if self.df is not None and len(self.df) > 0:
            if self.start_time is None:
                self.start_time = pd.to_datetime(self.df["timestamp"].iloc[0])
            if self.end_time is None:
                self.end_time = pd.to_datetime(self.df["timestamp"].iloc[-1])

    def __len__(self) -> int:
        """Return number of bars."""
        return len(self.df) if self.df is not None else 0

    def __repr__(self) -> str:
        return (
            f"OHLCVData(symbol={self.symbol!r}, timeframe={self.timeframe!r}, "
            f"bars={len(self)}, start={self.start_time}, end={self.end_time})"
        )

    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate data quality.

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        from .validators import validate_ohlcv
        return validate_ohlcv(self.df, self.timeframe)

    def is_valid(self) -> bool:
        """Return True if data passes all validation checks."""
        valid, _ = self.validate()
        return valid

    def to_parquet(self, path: Path) -> None:
        """
        Save data to Parquet file with metadata.

        Args:
            path: Output file path
        """
        # Ensure directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save with metadata
        self.df.to_parquet(
            path,
            index=False,
            engine="pyarrow",
            compression="snappy",
        )

        # Save metadata as separate JSON sidecar
        import json
        meta_path = path.with_suffix(".meta.json")
        meta = {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "num_bars": len(self),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def from_parquet(cls, path: Path) -> "OHLCVData":
        """
        Load data from Parquet file.

        Args:
            path: Input file path

        Returns:
            OHLCVData instance
        """
        import json

        df = pd.read_parquet(path, engine="pyarrow")

        # Load metadata from sidecar file
        meta_path = path.with_suffix(".meta.json")
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            return cls(
                symbol=meta["symbol"],
                timeframe=meta["timeframe"],
                df=df,
                start_time=datetime.fromisoformat(meta["start_time"]) if meta.get("start_time") else None,
                end_time=datetime.fromisoformat(meta["end_time"]) if meta.get("end_time") else None,
            )

        # Fallback: extract from path naming convention (symbol_timeframe.parquet)
        import warnings
        warnings.warn(
            f"No metadata file found for {path}. "
            "Inferring symbol/timeframe from filename. "
            "Consider regenerating cache with metadata.",
            UserWarning,
        )

        stem = path.stem
        parts = stem.split("_")
        if len(parts) >= 2:
            symbol = parts[0]
            timeframe = parts[1]
        else:
            symbol = "UNKNOWN"
            timeframe = "UNKNOWN"

        return cls(symbol=symbol, timeframe=timeframe, df=df)

    def slice(self, start: datetime, end: datetime) -> "OHLCVData":
        """
        Return a new OHLCVData with data between start and end times.

        Args:
            start: Start datetime (inclusive)
            end: End datetime (inclusive)

        Returns:
            New OHLCVData instance with sliced data
        """
        mask = (self.df["timestamp"] >= start) & (self.df["timestamp"] <= end)
        return OHLCVData(
            symbol=self.symbol,
            timeframe=self.timeframe,
            df=self.df[mask].reset_index(drop=True),
        )

    def get_bar(self, idx: int) -> dict:
        """
        Get bar data at index as dictionary.

        Args:
            idx: Bar index

        Returns:
            Dictionary with bar data
        """
        row = self.df.iloc[idx]
        return {
            "timestamp": row["timestamp"],
            "open": row["open"],
            "high": row["high"],
            "low": row["low"],
            "close": row["close"],
            "volume": row["volume"],
        }


# Timeframe constants
TIMEFRAMES = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "6h": 21600,
    "12h": 43200,
    "1d": 86400,
    "1w": 604800,
}


def timeframe_to_seconds(timeframe: str) -> int:
    """Convert timeframe string to seconds."""
    if timeframe in TIMEFRAMES:
        return TIMEFRAMES[timeframe]
    raise ValueError(f"Unknown timeframe: {timeframe}. Valid: {list(TIMEFRAMES.keys())}")


def timeframe_to_minutes(timeframe: str) -> int:
    """Convert timeframe string to minutes."""
    return timeframe_to_seconds(timeframe) // 60
