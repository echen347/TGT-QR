#!/usr/bin/env python3
"""
Cross-Asset Momentum Strategy
Uses BTC/ETH momentum to predict altcoin direction
"""

import pandas as pd
import numpy as np

def cross_asset_momentum_signal(historical_prices, btc_data=None, eth_data=None):
    """
    Cross-Asset Momentum Strategy
    Theory: Altcoins often follow BTC/ETH momentum with a lag.
            If BTC/ETH are trending up, altcoins will likely follow.
    """
    if len(historical_prices) < 30:
        return 0
    
    # If we have BTC/ETH data, use it
    if btc_data is not None and len(btc_data) >= 20:
        btc_momentum = (btc_data['close'].iloc[-1] - btc_data['close'].iloc[-20]) / btc_data['close'].iloc[-20]
        if btc_momentum > 0.05:  # BTC up 5%+
            return 1
        elif btc_momentum < -0.05:  # BTC down 5%+
            return -1
    
    if eth_data is not None and len(eth_data) >= 20:
        eth_momentum = (eth_data['close'].iloc[-1] - eth_data['close'].iloc[-20]) / eth_data['close'].iloc[-20]
        if eth_momentum > 0.05:  # ETH up 5%+
            return 1
        elif eth_momentum < -0.05:  # ETH down 5%+
            return -1
    
    # Fallback: use own momentum if no BTC/ETH data
    df = historical_prices.tail(30).copy()
    momentum = (df['close'].iloc[-1] - df['close'].iloc[-20]) / df['close'].iloc[-20]
    
    if momentum > 0.03:  # 3% momentum
        return 1
    elif momentum < -0.03:
        return -1
    
    return 0

if __name__ == "__main__":
    print("Cross-Asset Momentum Strategy")
    print("Uses BTC/ETH momentum to predict altcoin direction")

