import schedule
import time
import logging
from logging.handlers import TimedRotatingFileHandler
from apscheduler.schedulers.background import BackgroundScheduler
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from strategy import MovingAverageStrategy
from config.config import STRATEGY_INTERVAL_MINUTES, LOG_FILE, LOG_LEVEL, LOG_ROTATION, LOG_RETENTION_DAYS
from risk_manager import risk_manager

class TradingScheduler:
    """Schedule trading strategy execution"""

    def __init__(self):
        self.scheduler = BackgroundScheduler()
        self.strategy = MovingAverageStrategy()
        risk_manager.set_strategy(self.strategy)  # Link strategy to risk manager
        self.logger = self._setup_logging()

    def _setup_logging(self):
        """Setup logging with daily rotation"""
        logger = logging.getLogger('TradingScheduler')
        logger.setLevel(LOG_LEVEL)

        handler = TimedRotatingFileHandler(
            LOG_FILE,
            when='midnight' if LOG_ROTATION == 'daily' else 'h',
            interval=1,
            backupCount=LOG_RETENTION_DAYS
        )
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # Also log to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        return logger

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
        schedule.every(STRATEGY_INTERVAL_MINUTES).minutes.do(self.run_strategy_job)

        # Also run immediately on startup
        self.run_strategy_job()

        try:
            while True:
                schedule.run_pending()
                time.sleep(1)  # Check every second
        except KeyboardInterrupt:
            self.logger.info("Scheduler stopped by user")
        except Exception as e:
            self.logger.error(f"Scheduler error: {str(e)}")

if __name__ == "__main__":
    scheduler = TradingScheduler()
    scheduler.start()
