#!/usr/bin/env python3
"""
Moving Average Strategy Backtester
Tests the strategy on historical data without placing real orders
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sys
import os
import logging
from dotenv import load_dotenv
from pybit.unified_trading import HTTP
import time

# Add src to path for imports
sys.path.append('src')

# Setup comprehensive logging for backtesting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - BACKTESTER - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/backtesting.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('BACKTESTER')

import pandas as pd
import numpy as np
from pybit.unified_trading import HTTP
import sys
import os
from datetime import datetime, timedelta
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure the project root is in the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Load environment variables from the project root
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path=dotenv_path)

from config.config import BYBIT_TESTNET, SYMBOLS, MA_PERIOD, TIMEFRAME, LEVERAGE
from config.config import BYBIT_API_KEY, BYBIT_API_SECRET


class Backtester:
    """
    A standalone backtester for the Moving Average Strategy.
    Simulates trading on historical data to evaluate performance.
    """

    def __init__(self, symbols, start_date, end_date):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        # Force mainnet and provide keys for historical data fetching
        self.session = HTTP(
                testnet=False,
            api_key=BYBIT_API_KEY,
            api_secret=BYBIT_API_SECRET
        )
        self.historical_data = {}

    def fetch_historical_data(self):
        """Fetches historical kline data from Bybit using a robust, chunked approach."""
        logging.info("Fetching historical data...")
        for symbol in self.symbols:
            all_klines = []
            # Define the time window for each chunk (1000 minutes = 1000 candles)
            chunk_duration_ms = 1000 * 60 * 1000 
            
            current_start_ms = int(self.start_date.timestamp() * 1000)
            end_limit_ms = int(self.end_date.timestamp() * 1000)

            while current_start_ms < end_limit_ms:
                current_end_ms = current_start_ms + chunk_duration_ms
                try:
                    response = self.session.get_kline(
                    category="linear",
                        symbol=symbol,
                        interval=TIMEFRAME,
                        start=current_start_ms,
                        end=current_end_ms,
                        limit=1000
                    )

                    if response['retCode'] == 0 and response['result']['list']:
                    klines = response['result']['list']
                        all_klines.extend(klines)
                        # The next chunk starts after the current one
                        current_start_ms = current_end_ms
                    else:
                        logging.warning(f"Stopping data fetch for {symbol}. Reason: {response.get('retMsg', 'No more data')}")
                        break
                    
                    time.sleep(0.2) # Respect API rate limits

                except Exception as e:
                    logging.error(f"Error fetching data for {symbol}: {e}")
                    break
            
            if not all_klines:
                logging.error(f"Failed to fetch any klines for {symbol}. Check API keys and symbol validity.")
                continue

            df = pd.DataFrame(all_klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df = df.astype(float).drop_duplicates()
            # Filter for the exact date range to remove any overlap
            df = df[(df.index >= self.start_date) & (df.index <= self.end_date)]
            self.historical_data[symbol] = df.sort_index()
            logging.info(f"Fetched {len(df)} candles for {symbol} from {df.index.min()} to {df.index.max()}")

    def _get_signal(self, historical_prices):
        """Calculates the MA signal. Replicates the logic from strategy.py."""
        if len(historical_prices) < MA_PERIOD:
            return 0

        ma = historical_prices['close'].rolling(window=MA_PERIOD).mean().iloc[-1]
        current_price = historical_prices['close'].iloc[-1]
        
        if current_price > ma * 1.001:
            return 1  # Buy
        elif current_price < ma * 0.999:
            return -1  # Sell
        else:
            return 0  # Neutral

    def run(self):
        """Runs the backtest and prints the results."""
        self.fetch_historical_data()
        
        all_results = {}
        for symbol in self.symbols:
            if symbol not in self.historical_data or self.historical_data[symbol].empty:
                logging.warning(f"No data for {symbol}, skipping.")
                continue

            logging.info(f"--- Running simulation for {symbol} ---")
            df = self.historical_data[symbol]
            trades = []
            position = 0  # 0: none, 1: long, -1: short
            entry_price = 0
            
            # Iterate through the historical data, simulating the strategy
            for i in range(MA_PERIOD, len(df)):
                # This ensures no lookahead bias. The strategy only sees data up to the current point in time.
                current_market_slice = df.iloc[i-MA_PERIOD:i]
                signal = self._get_signal(current_market_slice)
                current_price = current_market_slice['close'].iloc[-1]

                # --- Execute Trades ---
                if position == 0 and signal != 0: # Open a new position
                    position = signal
                    entry_price = current_price
                    trades.append({'entry_date': df.index[i], 'entry_price': entry_price, 'type': 'long' if signal == 1 else 'short'})
                
                elif position == 1 and signal == -1: # Close long, open short
                    pnl = (current_price - entry_price) / entry_price
                    trades[-1].update({'exit_date': df.index[i], 'exit_price': current_price, 'pnl': pnl})
                    position = -1
                    entry_price = current_price
                    trades.append({'entry_date': df.index[i], 'entry_price': entry_price, 'type': 'short'})

                elif position == -1 and signal == 1: # Close short, open long
                    pnl = (entry_price - current_price) / entry_price
                    trades[-1].update({'exit_date': df.index[i], 'exit_price': current_price, 'pnl': pnl})
                    position = 1
                    entry_price = current_price
                    trades.append({'entry_date': df.index[i], 'entry_price': entry_price, 'type': 'long'})

            all_results[symbol] = pd.DataFrame(trades)
        
        self.calculate_metrics(all_results)
        
    def calculate_metrics(self, all_results):
        """Calculates and prints key performance metrics for the backtest."""
        logging.info("\n--- Backtest Performance Results ---")
        
        for symbol, trades_df in all_results.items():
            if trades_df.empty or 'pnl' not in trades_df.columns:
                logging.warning(f"\nSymbol: {symbol}\nNo trades executed or no PnL data.")
                continue

            # Drop trades that were not closed
            trades_df.dropna(subset=['pnl'], inplace=True)
            
            # --- Key Metrics ---
            total_trades = len(trades_df)
            pnl_with_leverage = (trades_df['pnl'] * LEVERAGE) + 1
            cumulative_return = pnl_with_leverage.cumprod()
            total_return_pct = (cumulative_return.iloc[-1] - 1) * 100
            
            # Win Rate
            wins = trades_df[trades_df['pnl'] > 0]
            win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0
            
            # Average Win / Loss
            avg_win_pct = wins['pnl'].mean() * 100 * LEVERAGE if not wins.empty else 0
            avg_loss_pct = trades_df[trades_df['pnl'] < 0]['pnl'].mean() * 100 * LEVERAGE if total_trades > len(wins) else 0

            # Max Drawdown
            cumulative_max = cumulative_return.cummax()
            drawdown = (cumulative_return - cumulative_max) / cumulative_max
            max_drawdown_pct = drawdown.min() * 100

            # Sharpe Ratio (annualized)
            daily_returns = trades_df.set_index('exit_date')['pnl'].resample('D').sum() * LEVERAGE
            if daily_returns.std() > 0:
                sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(365)
        else:
            sharpe_ratio = 0

            # --- Print Results ---
            print("\n" + "="*50)
            print(f"SYMBOL: {symbol} | PERIOD: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}")
            print("="*50)
            print(f"Total Return:       {total_return_pct:.2f}%")
            print(f"Total Trades:       {total_trades}")
            print(f"Win Rate:           {win_rate:.2f}%")
            print(f"Average Win:        {avg_win_pct:.2f}%")
            print(f"Average Loss:       {avg_loss_pct:.2f}%")
            print(f"Max Drawdown:       {max_drawdown_pct:.2f}%")
            print(f"Sharpe Ratio:       {sharpe_ratio:.2f}")
            print("="*50)


if __name__ == "__main__":
    # Configuration for the backtest run
    BACKTEST_START_DATE = datetime.now() - timedelta(days=90) # 3 months of data
    BACKTEST_END_DATE = datetime.now()

    backtester = Backtester(
        symbols=SYMBOLS,
        start_date=BACKTEST_START_DATE,
        end_date=BACKTEST_END_DATE
    )
    
    backtester.run()
