#!/usr/bin/env python3
"""Set stop loss and take profit on existing positions"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv()
from pybit.unified_trading import HTTP
from config.config import STOP_LOSS_PCT, TAKE_PROFIT_PCT

session = HTTP(testnet=False, api_key=os.getenv('BYBIT_API_KEY'), api_secret=os.getenv('BYBIT_API_SECRET'))

# Get current positions
positions = session.get_positions(category='linear', settleCoin='USDT')
print('🛡️ SETTING STOP LOSS & TAKE PROFIT:')
if positions.get('retCode') == 0:
    for pos in positions['result']['list']:
        size = float(pos.get('size', 0))
        if size != 0:
            symbol = pos['symbol']
            entry = float(pos.get('avgPrice', 0))
            side = pos.get('side', 'N/A')
            
            # Calculate TP/SL based on config
            if side == 'Sell':  # Short position
                stop_loss = entry * (1 + STOP_LOSS_PCT)  # Price goes UP = loss
                take_profit = entry * (1 - TAKE_PROFIT_PCT)  # Price goes DOWN = profit
            else:  # Long position
                stop_loss = entry * (1 - STOP_LOSS_PCT)  # Price goes DOWN = loss
                take_profit = entry * (1 + TAKE_PROFIT_PCT)  # Price goes UP = profit
            
            print(f"\n  {symbol} ({side}):")
            print(f"    Entry: ${entry:.2f}")
            print(f"    Stop Loss: ${stop_loss:.2f} ({STOP_LOSS_PCT*100:.1f}%)")
            print(f"    Take Profit: ${take_profit:.2f} ({TAKE_PROFIT_PCT*100:.1f}%)")
            
            # Set TP/SL using set_trading_stop endpoint
            try:
                response = session.set_trading_stop(
                    category="linear",
                    symbol=symbol,
                    takeProfit=str(take_profit),
                    stopLoss=str(stop_loss)
                )
                if response.get('retCode') == 0:
                    print(f"    ✅ TP/SL set successfully!")
                else:
                    print(f"    ❌ Failed: {response.get('retMsg', 'Unknown error')}")
            except Exception as e:
                print(f"    ❌ Error: {e}")
else:
    print("No open positions found.")

