import os
import sys
from pybit.unified_trading import HTTP
from dotenv import load_dotenv
from datetime import datetime, timedelta
import logging

# Basic logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load .env file from the project root
try:
    # The script is in the root, so the .env file is in the same directory
    dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path)
        logging.info(".env file loaded successfully.")
    else:
        logging.error(".env file not found at the project root. Please ensure it exists.")
        sys.exit(1)
except Exception as e:
    logging.error(f"Error loading .env file: {e}")
    sys.exit(1)

# Get API keys from environment
API_KEY = os.getenv('BYBIT_API_KEY')
API_SECRET = os.getenv('BYBIT_API_SECRET')

if not API_KEY or not API_SECRET:
    logging.error("API_KEY or API_SECRET not found in .env file.")
    sys.exit(1)

def run_api_test():
    """
    A minimal script to test the Bybit API connection and data fetching.
    """
    logging.info("--- Starting Bybit API Connection Test ---")
    
    try:
        # Initialize the HTTP session (explicitly use mainnet)
        session = HTTP(
            testnet=False,
            api_key=API_KEY,
            api_secret=API_SECRET
        )
        logging.info("Session initialized for mainnet.")

        # Define parameters for a simple kline request
        symbol = "BTCUSDT"
        interval = "1" # 1-minute interval
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(minutes=10)).timestamp() * 1000)

        logging.info(f"Fetching kline data for {symbol} from {start_time} to {end_time}...")

        # Make the API call
        response = session.get_kline(
            category="linear",
            symbol=symbol,
            interval=interval,
            start=start_time,
            end=end_time
        )

        # Print the full, raw response from Bybit
        logging.info("--- Full API Response ---")
        print(response)
        logging.info("--------------------------")

        # Analyze the response
        if response.get('retCode') == 0:
            kline_list = response.get('result', {}).get('list', [])
            logging.info(f"API call successful. Fetched {len(kline_list)} candles.")
            if not kline_list:
                logging.warning("Warning: Fetched 0 candles. This might indicate an issue with the request parameters or no data for the period.")
        else:
            logging.error(f"API call failed with retCode: {response.get('retCode')}")
            logging.error(f"Error Message: {response.get('retMsg')}")

    except Exception as e:
        logging.exception(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    run_api_test()
