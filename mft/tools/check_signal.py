#!/usr/bin/env python3
"""Quick script to check current signal vs position"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv()
from pybit.unified_trading import HTTP
import pandas as pd
from config.config import MA_PERIOD, MIN_TREND_STRENGTH, VOLATILITY_THRESHOLD_HIGH, VOLATILITY_THRESHOLD_LOW

session = HTTP(testnet=False, api_key=os.getenv('BYBIT_API_KEY'), api_secret=os.getenv('BYBIT_API_SECRET'))

# Get current position
positions = session.get_positions(category='linear', settleCoin='USDT')
print('📊 CURRENT POSITION:')
if positions.get('retCode') == 0:
    for pos in positions['result']['list']:
        size = float(pos.get('size', 0))
        if size != 0:
            symbol = pos['symbol']
            side = pos.get('side', 'N/A')
            entry = float(pos.get('avgPrice', 0))
            mark = float(pos.get('markPrice', 0))
            pnl = float(pos.get('unrealisedPnl', 0))
            print(f"  {symbol}: {side} {abs(size)} @ ${entry:.2f}")
            print(f"    Current Price: ${mark:.2f}")
            print(f"    Unrealized PnL: ${pnl:.4f}")
            print(f"    {'📈 Price UP (bad for short)' if mark > entry else '📉 Price DOWN (good for short)'}")
            print()

# Get recent price data for signal calculation
print('🔍 SIGNAL ANALYSIS:')
symbol = 'SOLUSDT'
klines = session.get_kline(category='linear', symbol=symbol, interval='1', limit=100)
if klines.get('retCode') == 0:
    data = klines['result']['list']
    # Bybit returns: [timestamp, open, high, low, close, volume, turnover]
    prices = pd.DataFrame([{
        'close': float(k[4]),  # Index 4 is close price
        'timestamp': int(k[0])
    } for k in reversed(data)])
    
    ma = prices['close'].rolling(window=MA_PERIOD).mean().iloc[-1]
    current_price = prices['close'].iloc[-1]
    
    # Calculate trend strength
    ma_slope = (ma - prices['close'].rolling(window=MA_PERIOD).mean().iloc[-5]) / MA_PERIOD
    trend_strength = abs(ma_slope) / current_price
    
    # Calculate volatility
    recent_volatility = prices['close'].pct_change().rolling(window=10).std().iloc[-1]
    
    # Use unified signal calculator
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
    from signal_calculator import calculate_signal
    
    signal_value, signal_name, metadata = calculate_signal(prices)
    
    print(f"  {symbol}:")
    print(f"    Current Price: ${metadata['price']:.2f}")
    print(f"    MA({MA_PERIOD}): ${metadata['ma']:.2f}")
    print(f"    Deviation: {metadata['deviation_pct']:.3f}%")
    print(f"    Threshold: {metadata['threshold']:.3f}% ({metadata['vol_category']} vol)")
    print(f"    Trend Strength: {metadata['trend_strength']:.6f}")
    
    signal = signal_name
    if signal_name == 'NEUTRAL' and metadata.get('reason') == 'trend_too_weak':
        signal = 'NEUTRAL (trend too weak)'
    
    signal_side = 'BUY' if signal_value == 1 else 'SELL' if signal_value == -1 else 'NEUTRAL'
    
    print(f"    Signal: {signal}")
    print()
    
    # Explain relationship
    if side == 'Sell':  # They have a SHORT position
        print('💡 SIGNAL vs YOUR POSITION:')
        if signal_side == 'SELL':
            print('  ✅ STRONG_SELL signal = Price below MA = GOOD for your SHORT')
            print('  → Price should continue DOWN, your short will profit')
        elif signal_side == 'BUY':
            print('  ⚠️  STRONG_BUY signal = Price above MA = BAD for your SHORT')
            print('  → Price is rising, your short is losing money')
            print('  → Signal suggests price may continue UP (opposite of what you want)')
        else:
            print('  ➡️  NEUTRAL signal = Price near MA = No strong direction')
            print('  → Wait for clearer signal')

