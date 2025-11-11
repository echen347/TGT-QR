#!/usr/bin/env python3
"""
Discover top liquid trading pairs on Bybit for research
"""
import os
import sys
from dotenv import load_dotenv
from pybit.unified_trading import HTTP

load_dotenv()

def discover_top_tickers(min_volume_usdt=1000000):
    """Discover top liquid USDT perpetual futures pairs"""
    api_key = os.getenv('BYBIT_API_KEY')
    api_secret = os.getenv('BYBIT_API_SECRET')
    
    if not api_key or not api_secret:
        print("❌ API keys not found")
        return
    
    session = HTTP(testnet=False, api_key=api_key, api_secret=api_secret)
    
    print("🔍 Discovering top liquid trading pairs on Bybit...")
    print("=" * 70)
    
    # Get all instruments
    response = session.get_instruments_info(category="linear")
    if response['retCode'] != 0:
        print(f"❌ Failed: {response['retMsg']}")
        return
    
    instruments = response['result']['list']
    
    # Get ticker data for volume
    ticker_response = session.get_tickers(category="linear")
    tickers = {t['symbol']: t for t in ticker_response['result']['list']} if ticker_response['retCode'] == 0 else {}
    
    # Filter USDT pairs with sufficient volume
    candidates = []
    for inst in instruments:
        symbol = inst['symbol']
        if not symbol.endswith('USDT'):
            continue
        
        ticker = tickers.get(symbol, {})
        volume_24h = float(ticker.get('volume24h', 0))
        
        if volume_24h >= min_volume_usdt:
            price = float(ticker.get('lastPrice', 0))
            candidates.append({
                'symbol': symbol,
                'volume_24h': volume_24h,
                'price': price,
                'status': inst.get('status', 'Unknown')
            })
    
    # Sort by volume
    candidates.sort(key=lambda x: x['volume_24h'], reverse=True)
    
    print(f"\n📊 Top {len(candidates)} liquid USDT pairs (≥${min_volume_usdt:,.0f} 24h volume):\n")
    print(f"{'Rank':<6} {'Symbol':<12} {'Price':<12} {'24h Volume (USD)':<20} {'Status':<10}")
    print("-" * 70)
    
    for i, pair in enumerate(candidates[:30], 1):  # Top 30
        print(f"{i:<6} {pair['symbol']:<12} ${pair['price']:<11,.2f} ${pair['volume_24h']:<19,.0f} {pair['status']:<10}")
    
    print("\n💡 Research Ideas:")
    print("  - Test strategies on top 10 most liquid pairs")
    print("  - Look for pairs with high volatility (good for mean reversion)")
    print("  - Consider altcoins vs BTC/ETH (pairs trading opportunities)")
    print("  - Focus on pairs with consistent volume (reduces slippage)")
    
    return [c['symbol'] for c in candidates[:20]]  # Return top 20

if __name__ == "__main__":
    top_tickers = discover_top_tickers()
    print(f"\n✅ Found {len(top_tickers)} candidates for testing")

