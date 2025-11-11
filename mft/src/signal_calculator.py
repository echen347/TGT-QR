"""
Unified signal calculation module.
All signal calculations should use this module to ensure consistency.
"""
import pandas as pd
from config.config import (
    MA_PERIOD, MIN_TREND_STRENGTH, VOLATILITY_THRESHOLD_HIGH, 
    VOLATILITY_THRESHOLD_LOW
)

# Thresholds from config - Phase 1 improvements
THRESHOLD_HIGH_VOL = 0.002  # 0.2% (reduced from 0.3%)
THRESHOLD_NORMAL_VOL = 0.0005  # 0.05% (reduced from 0.1%)
THRESHOLD_LOW_VOL = 0.0003  # 0.03% (reduced from 0.05%)


def calculate_signal(prices_df, ma_period=MA_PERIOD, min_trend_strength=MIN_TREND_STRENGTH,
                    vol_threshold_high=VOLATILITY_THRESHOLD_HIGH, 
                    vol_threshold_low=VOLATILITY_THRESHOLD_LOW,
                    threshold_high=THRESHOLD_HIGH_VOL,
                    threshold_normal=THRESHOLD_NORMAL_VOL,
                    threshold_low=THRESHOLD_LOW_VOL):
    """
    Unified signal calculation function.
    
    Args:
        prices_df: DataFrame with 'close' column
        ma_period: Moving average period (default from config)
        min_trend_strength: Minimum trend strength filter (default from config)
        vol_threshold_high: High volatility threshold (default from config)
        vol_threshold_low: Low volatility threshold (default from config)
        threshold_high: Price deviation threshold for high vol (default from config)
        threshold_normal: Price deviation threshold for normal vol (default from config)
        threshold_low: Price deviation threshold for low vol (default from config)
    
    Returns:
        tuple: (signal_value, signal_name, metadata_dict)
            signal_value: -1 (SELL), 0 (NEUTRAL), 1 (BUY)
            signal_name: "STRONG_SELL", "NEUTRAL", "STRONG_BUY"
            metadata: dict with price, ma, deviation, threshold, vol_category, trend_strength
    """
    if len(prices_df) < ma_period + 10:
        return 0, "NEUTRAL", {
            'price': prices_df['close'].iloc[-1] if len(prices_df) > 0 else 0,
            'ma': 0,
            'deviation_pct': 0,
            'threshold': 0,
            'vol_category': 'N/A',
            'trend_strength': 0,
            'reason': 'insufficient_data'
        }
    
    current_price = prices_df['close'].iloc[-1]
    ma = prices_df['close'].rolling(window=ma_period).mean().iloc[-1]
    
    # Calculate trend strength using MA slope
    ma_series = prices_df['close'].rolling(window=ma_period).mean()
    ma_slope = (ma_series.iloc[-1] - ma_series.iloc[-5]) / ma_period
    trend_strength = abs(ma_slope) / current_price
    
    # Filter by trend strength
    if trend_strength < min_trend_strength:
        return 0, "NEUTRAL", {
            'price': current_price,
            'ma': ma,
            'deviation_pct': ((current_price - ma) / ma) * 100,
            'threshold': 0,
            'vol_category': 'N/A',
            'trend_strength': trend_strength,
            'reason': 'trend_too_weak'
        }
    
    # Calculate volatility for threshold selection
    recent_volatility = prices_df['close'].pct_change().rolling(window=10).std().iloc[-1]
    
    # Select threshold based on volatility
    if recent_volatility > vol_threshold_high:
        threshold = threshold_high
        vol_category = "HIGH"
    elif recent_volatility > vol_threshold_low:
        threshold = threshold_normal
        vol_category = "NORMAL"
    else:
        threshold = threshold_low
        vol_category = "LOW"
    
    # Calculate price deviation
    price_deviation_pct = ((current_price - ma) / ma) * 100
    
    # Generate signal (Phase 1: no MA slope requirement)
    if current_price > ma * (1 + threshold):
        signal_value = 1
        signal_name = "STRONG_BUY"
    elif current_price < ma * (1 - threshold):
        signal_value = -1
        signal_name = "STRONG_SELL"
    else:
        signal_value = 0
        signal_name = "NEUTRAL"
    
    metadata = {
        'price': current_price,
        'ma': ma,
        'deviation_pct': price_deviation_pct,
        'threshold': threshold * 100,  # Convert to percentage
        'vol_category': vol_category,
        'trend_strength': trend_strength,
        'reason': 'signal_generated'
    }
    
    return signal_value, signal_name, metadata

