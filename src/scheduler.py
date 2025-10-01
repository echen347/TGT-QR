import schedule
import time
import logging
from strategy import MovingAverageStrategy
from config.config import STRATEGY_INTERVAL_MINUTES

class TradingScheduler:
    """Schedule trading strategy execution"""

    def __init__(self):
        self.strategy = MovingAverageStrategy()
        self.logger = logging.getLogger('TradingScheduler')

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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
