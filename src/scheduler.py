import logging
from apscheduler.schedulers.background import BackgroundScheduler
import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.strategy import MovingAverageStrategy
from config.config import STRATEGY_INTERVAL_MINUTES, LOG_FILE, LOG_LEVEL
from src.risk_manager import risk_manager

class TradingScheduler:
    """Schedule trading strategy execution"""

    def __init__(self):
        self.scheduler = BackgroundScheduler()
        self.strategy = MovingAverageStrategy()
        risk_manager.set_strategy(self.strategy)  # Link strategy to risk manager
        self.logger = logging.getLogger('TradingScheduler')

        # Setup logging
        logging.basicConfig(
            level=LOG_LEVEL,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            filename=LOG_FILE
        )

    def run_strategy_job(self):
        """Job to run the trading strategy"""
        try:
            self.logger.info("Executing scheduled strategy run...")
            self.strategy.run_strategy()
        except Exception as e:
            self.logger.error(f"Error in scheduled strategy run: {str(e)}")

    def start(self):
        """Start the scheduler"""
        self.logger.info(f"Starting trading scheduler (interval: {STRATEGY_INTERVAL_MINUTES} minutes)")

        # Schedule the strategy to run every X minutes
        # schedule.every(STRATEGY_INTERVAL_MINUTES).minutes.do(self.run_strategy_job) # This line is removed as per the new_code

        # Also run immediately on startup
        self.run_strategy_job()

        try:
            while True:
                # schedule.run_pending() # This line is removed as per the new_code
                time.sleep(1)  # Check every second
        except KeyboardInterrupt:
            self.logger.info("Scheduler stopped by user")
        except Exception as e:
            self.logger.error(f"Scheduler error: {str(e)}")

if __name__ == "__main__":
    scheduler = TradingScheduler()
    scheduler.start()
