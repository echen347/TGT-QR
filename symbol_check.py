#!/usr/bin/env python3
"""
Check available symbols on Bybit testnet
"""

import os
from dotenv import load_dotenv
from pybit.unified_trading import HTTP

def check_available_symbols():
    """Check what symbols are available on testnet"""

    print("🔍 Checking available symbols on Bybit testnet...")
    print("=" * 60)

    # Load environment variables
    load_dotenv()

    # Check for API credentials
    api_key = os.getenv('BYBIT_API_KEY')
    api_secret = os.getenv('BYBIT_API_SECRET')
    testnet = os.getenv('BYBIT_TESTNET', 'true').lower() == 'true'

    if not api_key or not api_secret:
        print("❌ Error: BYBIT_API_KEY and BYBIT_API_SECRET not found in .env file")
        return

    try:
        # Create HTTP session
        session = HTTP(
            testnet=testnet,
            api_key=api_key,
            api_secret=api_secret
        )

        print(f"📊 Connected to {'TESTNET' if testnet else 'MAINNET'}")
        print()

        # Get all tickers
        response = session.get_tickers(category="linear")
        if response['retCode'] != 0:
            print(f"❌ Failed to get tickers: {response['retMsg']}")
            return

        tickers = response['result']['list']
        print(f"📈 Found {len(tickers)} symbols")

        # Filter for USDT pairs and sort by volume
        usdt_pairs = [t for t in tickers if t['symbol'].endswith('USDT')]
        usdt_pairs.sort(key=lambda x: float(x['volume24h']), reverse=True)

        print()
        print("🏆 Top 10 USDT pairs by 24h volume:")
        print("-" * 50)
        for i, ticker in enumerate(usdt_pairs[:10], 1):
            symbol = ticker['symbol']
            price = float(ticker['lastPrice'])
            volume = float(ticker['volume24h'])
            print(f"{i:2d}. {symbol:10s} ${price:10,.2f} Vol: {volume:12,.0f}")

        print()
        print("🔍 Looking for BTCUSDT and ETHUSDT specifically:")
        btc_ticker = next((t for t in tickers if t['symbol'] == 'BTCUSDT'), None)
        eth_ticker = next((t for t in tickers if t['symbol'] == 'ETHUSDT'), None)

        if btc_ticker:
            print(f"   ✅ BTCUSDT: ${float(btc_ticker['lastPrice']):,.2f} (Vol: {float(btc_ticker['volume24h']):,.0f})")
        else:
            print("   ❌ BTCUSDT: Not found")

        if eth_ticker:
            print(f"   ✅ ETHUSDT: ${float(eth_ticker['lastPrice']):,.2f} (Vol: {float(eth_ticker['volume24h']):,.0f})")
        else:
            print("   ❌ ETHUSDT: Not found")

        print()
        if not btc_ticker and not eth_ticker:
            print("💡 Try using one of these popular symbols instead:")
            popular_symbols = [t['symbol'] for t in usdt_pairs[:5]]
            for symbol in popular_symbols:
                print(f"   - {symbol}")

    except Exception as e:
        print(f"❌ Exception: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_available_symbols()
