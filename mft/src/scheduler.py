from apscheduler.schedulers.background import BackgroundScheduler
import logging
import time
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# The strategy instance will be passed in, not imported globally
# from src.strategy import strategy 
from config.config import STRATEGY_INTERVAL_MINUTES

# Logging configured by main script - no console output needed
# Logs go to file and systemd/journalctl on AWS

def start_scheduler(strategy):
    """
    Starts a simple loop to run the strategy at a set interval.
    This replaces BackgroundScheduler for robustness.
    """
    logging.info("Starting simple scheduler...")
    logging.info(f"Strategy will run every {STRATEGY_INTERVAL_MINUTES} minutes.")
    
    while True:
        try:
            logging.info("Scheduler is triggering the strategy run...")
            strategy.run_strategy()
            logging.info("Strategy run finished. Waiting for next interval.")
        except Exception as e:
            logging.exception(f"An error occurred in the strategy execution loop: {e}")
        
        # Wait for the specified interval before the next run
        time.sleep(STRATEGY_INTERVAL_MINUTES * 60)

if __name__ == "__main__":
    # This part is for standalone testing, which needs to be updated if used
    # from src.strategy import MovingAverageStrategy
    # strategy_instance = MovingAverageStrategy()
    # start_scheduler(strategy_instance)
    pass
