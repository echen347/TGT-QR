#!/usr/bin/env python3
"""
TGT QR Trading System Startup Script
Run this to start the complete trading system
"""

import subprocess
import sys
import os
import logging
from threading import Thread

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import DASHBOARD_HOST, DASHBOARD_PORT

def run_dashboard():
    """Run the dashboard in a separate thread"""
    try:
        from dashboard import app
        print("🚀 Starting dashboard...")
        app.run(host=DASHBOARD_HOST, port=DASHBOARD_PORT, debug=False, use_reloader=False)
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")

def run_scheduler():
    """Run the trading scheduler"""
    try:
        from scheduler import TradingScheduler
        print("📈 Starting trading scheduler...")
        scheduler = TradingScheduler()
        scheduler.start()
    except Exception as e:
        print(f"❌ Error starting scheduler: {e}")

def main():
    """Main startup function"""
    print("=" * 60)
    print("🚀 TGT QR TRADING SYSTEM STARTUP")
    print("=" * 60)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("📊 Dashboard will be available at: http://localhost:5000")
    print("🛑 Press Ctrl+C to stop the system")
    print()

    try:
        # Start dashboard in background thread
        dashboard_thread = Thread(target=run_dashboard, daemon=True)
        dashboard_thread.start()

        # Give dashboard a moment to start
        import time
        time.sleep(2)

        # Run scheduler in main thread
        run_scheduler()

    except KeyboardInterrupt:
        print("\n🛑 Trading system stopped by user")
    except Exception as e:
        print(f"\n❌ Error in trading system: {e}")

if __name__ == "__main__":
    main()
