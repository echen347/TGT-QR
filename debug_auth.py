#!/usr/bin/env python3
"""
Debug API authentication issues
"""

import os
import time
from dotenv import load_dotenv
from pybit.unified_trading import HTTP

def debug_authentication():
    """Test different API authentication scenarios"""

    print("🔍 Debugging API Authentication...")
    print("=" * 50)

    # Load environment variables
    load_dotenv()

    # Check credentials
    api_key = os.getenv('BYBIT_API_KEY')
    api_secret = os.getenv('BYBIT_API_SECRET')
    testnet = os.getenv('BYBIT_TESTNET', 'true').lower() == 'true'

    if not api_key or not api_secret:
        print("❌ API credentials not found")
        return

    print(f"📊 API Key: {api_key[:8]}...{api_key[-4:]}")
    print(f"📊 Testnet: {testnet}")
    print()

    try:
        # Test 1: Basic connection (no auth required for server time)
        session = HTTP(testnet=testnet)
        response = session.get_server_time()
        if response['retCode'] == 0:
            print("✅ Basic connectivity works")
        else:
            print(f"❌ Basic connectivity failed: {response['retMsg']}")

        # Test 2: Authenticated call that doesn't require trading permissions
        session_auth = HTTP(
            testnet=testnet,
            api_key=api_key,
            api_secret=api_secret
        )

        # Try getting account info (requires auth but not trading permissions)
        response = session_auth.get_wallet_balance(accountType="UNIFIED", coin="USDT")
        if response['retCode'] == 0:
            print("✅ Account authentication works")
            balance = response['result']['list'][0]['totalWalletBalance']
            print(f"   USDT Balance: ${float(balance)}")
        else:
            print(f"❌ Account authentication failed: {response['retMsg']}")
            print(f"   Error Code: {response['retCode']}")

        # Test 3: Try a trading-related call that should fail if no permissions
        if not testnet:  # Only test trading calls on live
            try:
                response = session_auth.get_positions(category="linear", symbol="ETHUSDT")
                if response['retCode'] == 0:
                    print("✅ Position query works (trading permissions OK)")
                else:
                    print(f"❌ Position query failed: {response['retMsg']}")
                    print(f"   This suggests trading permissions issue")
            except Exception as e:
                print(f"❌ Position query error: {str(e)}")

        print()
        print("🔧 If all tests pass except order placement:")
        print("   - Your API key and secret are correct")
        print("   - Basic authentication works")
        print("   - The issue is specifically with order placement permissions")
        print("   - Check if 'Orders' permission is enabled for Derivatives")

    except Exception as e:
        print(f"❌ Exception: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_authentication()
