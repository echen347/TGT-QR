#!/usr/bin/env python3
"""
Debug the API calls to understand the issue
"""

import os
import sys
from dotenv import load_dotenv
from pybit.unified_trading import HTTP

def debug_api_calls():
    """Debug the API calls"""

    print("🔍 Debugging API calls...")
    print("=" * 50)

    # Load environment variables
    load_dotenv()

    # Check for API credentials
    api_key = os.getenv('BYBIT_API_KEY')
    api_secret = os.getenv('BYBIT_API_SECRET')
    testnet = os.getenv('BYBIT_TESTNET', 'true').lower() == 'true'

    if not api_key or not api_secret:
        print("❌ Error: API credentials not found")
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

        # Test the exact same call as in the strategy
        print("🔧 Testing get_kline call (same as strategy)...")
        response = session.get_kline(
            category="linear",
            symbol="BTCUSDT",
            interval="1",
            limit=10
        )

        print(f"📊 Response code: {response['retCode']}")
        print(f"📊 Response message: {response['retMsg']}")
        print(f"📊 Response keys: {list(response.keys())}")

        if response['retCode'] == 0:
            result = response['result']
            print(f"📊 Result keys: {list(result.keys())}")

            klines = result['list']
            print(f"📊 Number of klines: {len(klines)}")

            if klines:
                print("📊 First kline structure:")
                first_kline = klines[0]
                print(f"   Type: {type(first_kline)}")
                print(f"   Length: {len(first_kline)}")
                print(f"   Content: {first_kline}")

                # Try to understand the kline format
                if len(first_kline) >= 6:
                    print("📊 Kline format appears to be:")
                    print(f"   [0] timestamp: {first_kline[0]}")
                    print(f"   [1] open: {first_kline[1]}")
                    print(f"   [2] high: {first_kline[2]}")
                    print(f"   [3] low: {first_kline[3]}")
                    print(f"   [4] close: {first_kline[4]}")
                    print(f"   [5] volume: {first_kline[5]}")
            else:
                print("📊 No klines returned")
        else:
            print(f"❌ API call failed: {response['retMsg']}")

        print()
        print("🔍 Let's also check the exact strategy method...")

        # Import and test the strategy method directly
        sys.path.append('src')
        from strategy import MovingAverageStrategy

        strategy = MovingAverageStrategy()
        prices = strategy.get_historical_prices("BTCUSDT", limit=10)

        print(f"📊 Strategy method returned: {type(prices)} with length {len(prices) if prices else 0}")

        if prices:
            print(f"📊 First price entry: {prices[0]}")

    except Exception as e:
        print(f"❌ Exception: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_api_calls()
