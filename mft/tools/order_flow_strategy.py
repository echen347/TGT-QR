#!/usr/bin/env python3
"""
Order Flow Imbalance Strategy
Trades based on buy/sell volume imbalance
"""

import pandas as pd
import numpy as np

def order_flow_imbalance_signal(historical_prices):
    """
    Order Flow Imbalance Strategy
    Uses volume and price action to infer buy/sell pressure
    
    Theory: When price moves up on high volume, buyers are in control.
            When price moves down on high volume, sellers are in control.
            We trade when there's a strong imbalance.
    """
    if len(historical_prices) < 50:
        return 0
    
    df = historical_prices.tail(50).copy()
    
    # Calculate price change and volume
    df['price_change'] = df['close'].pct_change()
    df['volume_ma'] = df['volume'].rolling(window=20).mean()
    
    # Calculate buy/sell pressure
    # If price goes up with high volume = buying pressure
    # If price goes down with high volume = selling pressure
    df['buy_pressure'] = np.where(
        (df['price_change'] > 0) & (df['volume'] > df['volume_ma'] * 1.5),
        df['volume'] * df['price_change'],
        0
    )
    
    df['sell_pressure'] = np.where(
        (df['price_change'] < 0) & (df['volume'] > df['volume_ma'] * 1.5),
        df['volume'] * abs(df['price_change']),
        0
    )
    
    # Rolling sum of pressure (last 10 bars)
    df['buy_pressure_sum'] = df['buy_pressure'].rolling(window=10).sum()
    df['sell_pressure_sum'] = df['sell_pressure'].rolling(window=10).sum()
    
    # Calculate imbalance ratio
    total_pressure = df['buy_pressure_sum'].iloc[-1] + df['sell_pressure_sum'].iloc[-1]
    if total_pressure == 0:
        return 0
    
    imbalance_ratio = (df['buy_pressure_sum'].iloc[-1] - df['sell_pressure_sum'].iloc[-1]) / total_pressure
    
    # Entry thresholds
    if imbalance_ratio > 0.3:  # Strong buying pressure
        return 1
    elif imbalance_ratio < -0.3:  # Strong selling pressure
        return -1
    
    return 0

if __name__ == "__main__":
    print("Order Flow Imbalance Strategy")
    print("Trades based on buy/sell volume imbalance")

