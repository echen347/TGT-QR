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
log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_file_path = os.path.join(log_dir, 'trading_system.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler(sys.stdout)
    ]
)

def main():
    """Main function to create shared instances and start both dashboard and scheduler."""
    print("=" * 60)
    print("🚀 TGT QR TRADING SYSTEM STARTUP")
    print("=" * 60)

    # --- Create the single, shared instances ---
    risk_manager = RiskManager()
    strategy = MovingAverageStrategy()
    # Apply exchange risk limits once at startup if supported
    try:
        from config.config import RISK_LIMIT_ENABLED, RISK_LIMIT_VALUE, SYMBOLS
        if RISK_LIMIT_ENABLED and hasattr(strategy.session, 'set_risk_limit'):
            for sym in SYMBOLS:
                try:
                    strategy.session.set_risk_limit(category="linear", symbol=sym, riskId=str(RISK_LIMIT_VALUE))
                except Exception:
                    continue
    except Exception:
        pass
    # Give risk_manager a reference to the strategy for emergency stops
    risk_manager.set_strategy(strategy)
    # Give strategy a reference to the risk_manager
    strategy.set_risk_manager(risk_manager)
    
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
