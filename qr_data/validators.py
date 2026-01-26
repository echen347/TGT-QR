"""Data validation functions for qr_data package."""

from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import numpy as np

from .types import timeframe_to_seconds


def validate_ohlcv(
    df: pd.DataFrame,
    timeframe: str,
    max_gap_multiplier: float = 2.0,
) -> tuple[bool, list[str]]:
    """
    Validate OHLCV data quality.

    Checks:
    - Required columns exist
    - No duplicate timestamps
    - No gaps > max_gap_multiplier * timeframe interval
    - OHLC relationships: low <= open, close <= high
    - Volume >= 0
    - No NaN values in critical columns

    Args:
        df: DataFrame with OHLCV data
        timeframe: Timeframe string (e.g., "15m")
        max_gap_multiplier: Maximum allowed gap as multiplier of timeframe

    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []

    # Check required columns
    required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {missing}")
        return False, errors

    if len(df) == 0:
        errors.append("DataFrame is empty")
        return False, errors

    # Check for NaN values
    for col in required_cols:
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            errors.append(f"Column '{col}' has {nan_count} NaN values")

    # Check for duplicate timestamps
    dup_count = df["timestamp"].duplicated().sum()
    if dup_count > 0:
        errors.append(f"Found {dup_count} duplicate timestamps")

    # Check timestamp ordering
    if not df["timestamp"].is_monotonic_increasing:
        errors.append("Timestamps are not monotonically increasing")

    # Check for gaps
    try:
        interval_seconds = timeframe_to_seconds(timeframe)
        max_gap_seconds = interval_seconds * max_gap_multiplier

        timestamps = pd.to_datetime(df["timestamp"])
        diffs = timestamps.diff().dt.total_seconds().dropna()

        gap_count = (diffs > max_gap_seconds).sum()
        if gap_count > 0:
            max_gap = diffs.max()
            errors.append(
                f"Found {gap_count} gaps > {max_gap_multiplier}x interval "
                f"(max gap: {max_gap/60:.1f} min, expected: {interval_seconds/60:.1f} min)"
            )
    except ValueError as e:
        errors.append(f"Cannot validate gaps: {e}")

    # Check OHLC relationships
    invalid_ohlc = (
        (df["low"] > df["open"]) |
        (df["low"] > df["close"]) |
        (df["high"] < df["open"]) |
        (df["high"] < df["close"]) |
        (df["low"] > df["high"])
    ).sum()
    if invalid_ohlc > 0:
        errors.append(f"Found {invalid_ohlc} bars with invalid OHLC relationships")

    # Check volume is non-negative
    neg_volume = (df["volume"] < 0).sum()
    if neg_volume > 0:
        errors.append(f"Found {neg_volume} bars with negative volume")

    return len(errors) == 0, errors


def check_data_quality(
    df: pd.DataFrame,
    timeframe: str,
) -> dict:
    """
    Compute data quality metrics.

    Args:
        df: DataFrame with OHLCV data
        timeframe: Timeframe string

    Returns:
        Dictionary with quality metrics
    """
    is_valid, errors = validate_ohlcv(df, timeframe)

    interval_seconds = timeframe_to_seconds(timeframe)
    timestamps = pd.to_datetime(df["timestamp"])
    diffs = timestamps.diff().dt.total_seconds().dropna()

    # Compute expected number of bars
    if len(df) > 0:
        total_seconds = (timestamps.iloc[-1] - timestamps.iloc[0]).total_seconds()
        expected_bars = total_seconds / interval_seconds + 1
        completeness = len(df) / expected_bars if expected_bars > 0 else 0
    else:
        completeness = 0

    return {
        "is_valid": is_valid,
        "errors": errors,
        "num_bars": len(df),
        "completeness": completeness,
        "duplicate_count": df["timestamp"].duplicated().sum() if len(df) > 0 else 0,
        "gap_count": (diffs > interval_seconds * 2).sum() if len(diffs) > 0 else 0,
        "nan_count": df.isna().sum().sum() if len(df) > 0 else 0,
        "invalid_ohlc_count": (
            (df["low"] > df["open"]) |
            (df["low"] > df["close"]) |
            (df["high"] < df["open"]) |
            (df["high"] < df["close"]) |
            (df["low"] > df["high"])
        ).sum() if len(df) > 0 else 0,
    }


def fill_gaps(
    df: pd.DataFrame,
    timeframe: str,
    method: str = "ffill",
) -> pd.DataFrame:
    """
    Fill gaps in OHLCV data.

    Args:
        df: DataFrame with OHLCV data
        timeframe: Timeframe string
        method: Fill method - "ffill" (forward fill) or "interpolate"

    Returns:
        DataFrame with gaps filled
    """
    if len(df) == 0:
        return df.copy()

    interval_seconds = timeframe_to_seconds(timeframe)

    # Create complete timestamp index
    start = pd.to_datetime(df["timestamp"].iloc[0])
    end = pd.to_datetime(df["timestamp"].iloc[-1])

    freq_map = {
        60: "1min", 180: "3min", 300: "5min", 900: "15min", 1800: "30min",
        3600: "1h", 7200: "2h", 14400: "4h", 21600: "6h", 43200: "12h",
        86400: "1D", 604800: "1W",
    }
    freq = freq_map.get(interval_seconds, f"{interval_seconds}s")

    complete_index = pd.date_range(start=start, end=end, freq=freq)

    # Reindex with complete timestamps
    df_indexed = df.copy()
    df_indexed["timestamp"] = pd.to_datetime(df_indexed["timestamp"])
    df_indexed = df_indexed.set_index("timestamp")
    df_reindexed = df_indexed.reindex(complete_index)

    if method == "ffill":
        df_filled = df_reindexed.ffill()
    elif method == "interpolate":
        df_filled = df_reindexed.interpolate(method="linear")
    else:
        raise ValueError(f"Unknown fill method: {method}")

    # Reset index
    df_filled = df_filled.reset_index()
    df_filled = df_filled.rename(columns={"index": "timestamp"})

    return df_filled


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate timestamps, keeping the last occurrence.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)


def ensure_timezone(df: pd.DataFrame, tz: str = "UTC") -> pd.DataFrame:
    """
    Ensure timestamps are timezone-aware.

    Args:
        df: DataFrame with OHLCV data
        tz: Target timezone (default: UTC)

    Returns:
        DataFrame with timezone-aware timestamps
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(tz)
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert(tz)

    return df
