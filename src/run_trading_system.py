#!/usr/bin/env python3
"""
TGT QR Trading System Startup Script
Run this to start the complete trading system
"""

import threading
import logging
import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import the classes, not the instances
from src.strategy import MovingAverageStrategy
from src.risk_manager import RiskManager
from src.dashboard import run_dashboard
from src.scheduler import start_scheduler
from config.config import DASHBOARD_HOST, DASHBOARD_PORT

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def main():
    """Main function to create shared instances and start both dashboard and scheduler."""
    print("=" * 60)
    print("🚀 TGT QR TRADING SYSTEM STARTUP")
    print("=" * 60)

    # --- Create the single, shared instances ---
    risk_manager = RiskManager()
    strategy = MovingAverageStrategy()
    # Give risk_manager a reference to the strategy for emergency stops
    risk_manager.set_strategy(strategy)
    
    print(f"📊 Dashboard will be available at: http://localhost:{DASHBOARD_PORT}")
    print("🛑 Press Ctrl+C to stop the system")
    print("=" * 60)

    # --- Start services in separate threads, passing the shared instances ---
    dashboard_thread = threading.Thread(target=run_dashboard, args=(strategy, risk_manager))
    scheduler_thread = threading.Thread(target=start_scheduler, args=(strategy,))

    dashboard_thread.start()
    scheduler_thread.start()

    # Keep the main thread alive to handle shutdown
    try:
        while True:
            time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        logging.info("Shutting down trading system...")
        # A more robust shutdown would signal threads to exit gracefully
        sys.exit(0)

if __name__ == "__main__":
    main()
