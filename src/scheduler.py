from apscheduler.schedulers.background import BackgroundScheduler
import logging
import time
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.strategy import MovingAverageStrategy
from config.config import STRATEGY_INTERVAL_MINUTES

# Basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def run_strategy_job():
    """Wrapper function to run the strategy."""
    try:
        logging.info("Scheduler is running the strategy...")
        strategy = MovingAverageStrategy()
        strategy.run_strategy()
        logging.info("Strategy run finished.")
    except Exception as e:
        logging.exception(f"Error in scheduled strategy run: {e}")

def start_scheduler():
    """Starts the background scheduler."""
    scheduler = BackgroundScheduler()
    scheduler.add_job(run_strategy_job, 'interval', minutes=STRATEGY_INTERVAL_MINUTES)
    scheduler.start()
    logging.info(f"Trading strategy scheduled to run every {STRATEGY_INTERVAL_MINUTES} minutes.")

    # Run once immediately on startup
    run_strategy_job()

    try:
        # Keep the main thread alive
        while True:
            time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        logging.info("Scheduler shut down.")

if __name__ == "__main__":
    start_scheduler()
