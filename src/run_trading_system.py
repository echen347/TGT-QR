#!/usr/bin/env python3
"""
TGT QR Trading System Startup Script
Run this to start the complete trading system
"""

import threading
import logging
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.dashboard import app as dashboard_app
from src.scheduler import start_scheduler
from config.config import DASHBOARD_HOST, DASHBOARD_PORT

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def run_dashboard():
    """Run the Flask dashboard."""
    logging.info(f"🚀 Starting dashboard...")
    dashboard_app.run(host=DASHBOARD_HOST, port=DASHBOARD_PORT, debug=False)

def run_scheduler():
    """Run the trading strategy scheduler."""
    logging.info("📈 Starting trading scheduler...")
    try:
        start_scheduler()
    except Exception as e:
        logging.critical(f"❌ Scheduler failed to start: {e}")

def main():
    """Main function to start both dashboard and scheduler in separate threads."""
    print("=" * 60)
    print("🚀 TGT QR TRADING SYSTEM STARTUP")
    print("=" * 60)
    print(f"📊 Dashboard will be available at: http://localhost:{DASHBOARD_PORT}")
    print("🛑 Press Ctrl+C to stop the system")

    dashboard_thread = threading.Thread(target=run_dashboard)
    scheduler_thread = threading.Thread(target=run_scheduler)

    dashboard_thread.start()
    scheduler_thread.start()

    dashboard_thread.join()
    scheduler_thread.join()

if __name__ == "__main__":
    main()
