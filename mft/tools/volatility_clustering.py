#!/usr/bin/env python3
"""
Volatility Clustering Strategy
Trades based on volatility clustering patterns (high vol followed by high vol, low vol followed by low vol)
"""

import pandas as pd
import numpy as np

def volatility_clustering_signal(historical_prices):
    """
    Volatility Clustering Strategy
    Theory: Volatility clusters - periods of high volatility are followed by high volatility,
            periods of low volatility are followed by low volatility.
            We trade breakouts from low volatility periods.
    """
    if len(historical_prices) < 50:
        return 0
    
    df = historical_prices.tail(100).copy()
    
    # Calculate returns and volatility
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(window=20).std()
    
    # Identify volatility regime
    vol_ma = df['volatility'].rolling(window=50).mean().iloc[-1]
    current_vol = df['volatility'].iloc[-1]
    
    if pd.isna(vol_ma) or pd.isna(current_vol):
        return 0
    
    # Low volatility period (potential breakout coming)
    if current_vol < vol_ma * 0.7:
        # Look for price momentum
        price_change = (df['close'].iloc[-1] - df['close'].iloc[-10]) / df['close'].iloc[-10]
        
        # If price is moving up in low vol, expect breakout up
        if price_change > 0.01:  # 1% move
            return 1
        # If price is moving down in low vol, expect breakdown down
        elif price_change < -0.01:
            return -1
    
    # High volatility period - trade momentum
    elif current_vol > vol_ma * 1.3:
        # In high vol, follow momentum
        price_change = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5]
        
        if price_change > 0.005:  # 0.5% move
            return 1
        elif price_change < -0.005:
            return -1
    
    return 0

if __name__ == "__main__":
    print("Volatility Clustering Strategy")
    print("Trades based on volatility clustering patterns")

