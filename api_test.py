#!/usr/bin/env python3
"""
Simple API Connection Test for Bybit
"""

import os
from dotenv import load_dotenv
from pybit.unified_trading import HTTP

def test_api_connection():
    """Test basic API connection to Bybit"""

    print("🔍 Testing Bybit API Connection...")
    print("=" * 50)

    # Load environment variables
    load_dotenv()

    # Check for API credentials
    api_key = os.getenv('BYBIT_API_KEY')
    api_secret = os.getenv('BYBIT_API_SECRET')
    testnet = os.getenv('BYBIT_TESTNET', 'true').lower() == 'true'

    if not api_key or not api_secret:
        print("❌ Error: BYBIT_API_KEY and BYBIT_API_SECRET not found in .env file")
        print("Please add your Bybit API credentials to the .env file")
        return

    print(f"📊 API Configuration:")
    print(f"   API Key: {api_key[:8]}...{api_key[-4:] if len(api_key) > 12 else api_key}")
    print(f"   API Secret: {api_secret[:8]}...{api_secret[-4:] if len(api_secret) > 12 else api_secret}")
    print(f"   Testnet: {testnet}")
    print()

    try:
        # Create HTTP session
        session = HTTP(
            testnet=testnet,
            api_key=api_key,
            api_secret=api_secret
        )

        print("🔗 Testing basic API calls...")

        # Test 1: Get server time
        print("   1. Getting server time...")
        response = session.get_server_time()
        if response['retCode'] == 0:
            print(f"   ✅ Server time: {response['result']['timeSecond']}")
        else:
            print(f"   ❌ Server time failed: {response['retMsg']}")
            return

        # Test 2: Get tickers
        print("   2. Getting tickers...")
        response = session.get_tickers(category="linear")
        if response['retCode'] == 0:
            tickers = response['result']['list']
            btc_ticker = next((t for t in tickers if t['symbol'] == 'BTCUSDT'), None)
            if btc_ticker:
                print(f"   ✅ BTCUSDT Price: ${float(btc_ticker['lastPrice']):,.2f}")
                print(f"   ✅ BTCUSDT Volume: {float(btc_ticker['volume24h']):,.0f}")
            else:
                print("   ❌ BTCUSDT ticker not found")
        else:
            print(f"   ❌ Tickers failed: {response['retMsg']}")
            return

        # Test 3: Get klines (price data)
        print("   3. Getting price data (klines)...")
        response = session.get_kline(
            category="linear",
            symbol="BTCUSDT",
            interval="1",
            limit=10
        )
        if response['retCode'] == 0:
            klines = response['result']['list']
            if klines:
                latest_kline = klines[0]  # Most recent first
                print(f"   ✅ Latest price: ${float(latest_kline[4]):.2f}")
                print(f"   ✅ Got {len(klines)} data points")
            else:
                print("   ❌ No klines returned")
        else:
            print(f"   ❌ Klines failed: {response['retMsg']}")
            return

        print()
        print("🎉 All API tests passed!")
        print("✅ Your API connection is working correctly")
        print()
        print("🚀 You can now run the full day1_test.py")

    except Exception as e:
        print(f"❌ Exception during API test: {str(e)}")
        print()
        print("🔧 Troubleshooting steps:")
        print("   1. Check that your API key and secret are correct")
        print("   2. Ensure you're using the right testnet/mainnet setting")
        print("   3. Check your internet connection")
        print("   4. Verify your Bybit account has API trading permissions")

if __name__ == "__main__":
    test_api_connection()
