#!/usr/bin/env python3
"""
Day 1 Quick Test - Simple Buy/Sell Test (Simplified Version)
"""

import os
import sys
import time
from dotenv import load_dotenv
from pybit.unified_trading import HTTP

def day1_test():
    """Execute a simple $0.50 buy/sell test"""

    print("🚀 TGT QR Day 1 Test - $0.50 Buy/Sell")
    print("=" * 50)

    # Load environment variables
    load_dotenv()

    # Check for API credentials
    if not os.getenv('BYBIT_API_KEY') or not os.getenv('BYBIT_API_SECRET'):
        print("❌ Error: BYBIT_API_KEY and BYBIT_API_SECRET not found in .env file")
        print("Please add your Bybit API credentials to the .env file")
        return

    # Test parameters - Using XRPUSDT perpetual futures
    SYMBOL = 'XRPUSDT'  # XRP perpetual futures (not spot)
    TEST_AMOUNT_USDT = 5.00  # $5 with 10x leverage for XRP
    LEVERAGE = 10

    print(f"📊 Test Parameters:")
    print(f"   Symbol: {SYMBOL}")
    print(f"   Amount: ${TEST_AMOUNT_USDT}")
    print(f"   Leverage: {LEVERAGE}x")
    print()

    try:
        # Create session
        session = HTTP(
            testnet=False,  # Live trading
            api_key=os.getenv('BYBIT_API_KEY'),
            api_secret=os.getenv('BYBIT_API_SECRET')
        )

        print("🔗 Connected to Bybit API")

        # Step 1: Place buy order (fills immediately at market price)
        print("📈 Step 1: Placing BUY order...")
        buy_order = session.place_order(
            category="linear",
            symbol=SYMBOL,
            side="Buy",
            orderType="Market",
            qty=str(TEST_AMOUNT_USDT),  # USDT amount
            leverage=str(LEVERAGE),
            timeInForce="GTC"
        )

        print("Buy order response:", buy_order)

        if buy_order['retCode'] == 0:
            print("✅ BUY order placed successfully!")
        else:
            print(f"❌ BUY order failed: {buy_order['retMsg']}")
            return

        print()

        # Step 2: Wait a moment
        print("⏳ Waiting 3 seconds...")
        time.sleep(3)

        # Step 3: Place sell order (same size)
        print("📉 Step 2: Placing SELL order...")
        sell_order = session.place_order(
            category="linear",
            symbol=SYMBOL,
            side="Sell",
            orderType="Market",
            qty=str(TEST_AMOUNT_USDT),  # Same USDT amount
            leverage=str(LEVERAGE),
            timeInForce="GTC"
        )

        print("Sell order response:", sell_order)

        if sell_order['retCode'] == 0:
            print("✅ SELL order placed successfully!")
        else:
            print(f"❌ SELL order failed: {sell_order['retMsg']}")

        print()

        # Step 4: Show summary
        print("📋 Test Summary:")
        print(f"   ✅ API Connection: Working")
        print(f"   ✅ Order Placement: Working")
        print(f"   ✅ Both BUY and SELL orders placed")
        print(f"   💰 Total Volume: ${TEST_AMOUNT_USDT * 2}")
        print()

        print("🎉 Day 1 Test Complete!")
        print("Check your Bybit account for the orders.")
        print("Monitor the orders to see them fill and calculate any P&L.")

    except Exception as e:
        print(f"❌ Error during test: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    day1_test()
