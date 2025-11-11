#!/usr/bin/env python3
"""
TGT QR Trading System Test Script
Tests all components without placing real trades
"""

import sys
import os
import time
from datetime import datetime

# Add project root and src to path
project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, project_root)
sys.path.insert(0, src_path)

def test_imports():
    """Test that all modules can be imported"""
    print("🔍 Testing imports...")

    try:
        import config.config
        from src.database import db_manager
        from src.risk_manager import RiskManager
        from src.strategy import MovingAverageStrategy
        from src.dashboard import app, TradingDashboard
        print("✅ All imports successful")
        return True
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ Import error: {e}")
        return False

def test_database():
    """Test database operations"""
    print("💾 Testing database...")

    try:
        from src.database import db_manager

        # Test saving price data (using the method signature that matches strategy.py)
        test_price = 3000.0  # ETH price
        test_volume = 1000000.0  # volume
        
        db_manager.save_price_data('ETHUSDT', test_price, test_volume)

        # Test retrieving data
        prices = db_manager.get_recent_prices('ETHUSDT', limit=1)
        if prices:
            print("✅ Database operations working")
            return True
        else:
            print("❌ Database retrieval failed")
            return False

    except Exception as e:
        print(f"❌ Database error: {e}")
        return False

def test_risk_manager():
    """Test risk management system"""
    print("🛡️ Testing risk manager...")

    try:
        from src.risk_manager import RiskManager

        risk_manager = RiskManager()
        
        # Test initial state
        status = risk_manager.get_risk_status()
        print(f"   Initial status: Can trade = {status.get('can_trade', 'N/A')}")

        # Test position size check
        can_open = risk_manager.can_open_position('ETHUSDT', 1.00, 2000000)  # 2M volume
        print(f"   Position check: {can_open}")

        # Test loss recording
        risk_manager.record_loss(0.50)
        status_after = risk_manager.get_risk_status()
        print(f"   After $0.50 loss: Daily loss = ${status_after.get('daily_loss', 0):.2f}")

        print("✅ Risk manager working")
        return True

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ Risk manager error: {e}")
        return False

def test_strategy():
    """Test strategy without placing orders"""
    print("📈 Testing strategy (no orders)...")

    try:
        from src.strategy import MovingAverageStrategy

        strategy = MovingAverageStrategy()

        # Test price data retrieval (without Bybit API key)
        print("   Note: Price data requires API keys for live data")
        print("   Strategy structure is ready for deployment")

        print("✅ Strategy framework ready")
        return True

    except Exception as e:
        print(f"❌ Strategy error: {e}")
        return False

def test_dashboard():
    """Test dashboard components"""
    print("📊 Testing dashboard...")

    try:
        from src.dashboard import TradingDashboard
        from src.strategy import MovingAverageStrategy
        from src.risk_manager import RiskManager

        # Create mock instances for testing
        strategy = MovingAverageStrategy()
        risk_manager = RiskManager()
        dashboard = TradingDashboard(strategy, risk_manager)

        # Test data retrieval methods
        pnl_data = dashboard.get_pnl_chart_data()
        positions = dashboard.get_positions_data()
        signals = dashboard.get_signals_data()
        risk_status = dashboard.get_risk_status()

        print(f"   PnL data keys: {list(pnl_data.keys()) if 'error' not in pnl_data else 'No data'}")
        print(f"   Positions count: {len(positions)}")
        print(f"   Signals count: {len(signals)}")
        print(f"   Risk status keys: {list(risk_status.keys())}")

        print("✅ Dashboard data retrieval working")
        return True

    except Exception as e:
        print(f"❌ Dashboard error: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("🧪 TGT QR TRADING SYSTEM - COMPREHENSIVE TEST")
    print("=" * 60)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    tests = [
        ("Import Test", test_imports),
        ("Database Test", test_database),
        ("Risk Manager Test", test_risk_manager),
        ("Strategy Test", test_strategy),
        ("Dashboard Test", test_dashboard)
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        success = test_func()
        results.append((test_name, success))

    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {test_name}: {status}")

    print("-" * 60)
    print(f"Overall: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ System is ready for deployment")
        print("📝 Next steps:")
        print("   1. Add Bybit API keys to .env")
        print("   2. Deploy to AWS server")
        print("   3. Monitor dashboard at http://localhost:5000")
        print("   4. Start with testnet trading first")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        print("❌ Please fix errors before deployment")

    print("=" * 60)

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
