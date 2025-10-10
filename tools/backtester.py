#!/usr/bin/env python3
"""
Moving Average Strategy Backtester
Tests the strategy on historical data without placing real orders
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Try to import matplotlib for visualizations, but handle gracefully if not available
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠️ matplotlib not available - visualizations will be skipped")
import sys
import os
import logging
import pickle
import threading
import json
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from pybit.unified_trading import HTTP
import time

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Setup comprehensive logging for backtesting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - BACKTESTER - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), '..', 'logs', 'backtesting.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('BACKTESTER')

# Ensure the project root is in the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Load environment variables from the project root
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path=dotenv_path)

from config.config import BYBIT_TESTNET, SYMBOLS, MA_PERIOD, TIMEFRAME, LEVERAGE
from config.config import BYBIT_API_KEY, BYBIT_API_SECRET
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from database import db_manager

# Set dummy API keys for backtesting if not available
if not BYBIT_API_KEY or not BYBIT_API_SECRET:
    BYBIT_API_KEY = "dummy_key"
    BYBIT_API_SECRET = "dummy_secret"
    print("Warning: Using dummy API keys for backtesting. This may not work for live data.")


class Backtester:
    """
    A standalone backtester for the Moving Average Strategy.
    Simulates trading on historical data to evaluate performance.
    """

    def __init__(self, symbols, start_date, end_date, run_name=None, run_description=None):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.run_name = run_name or f"Backtest {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        self.run_description = run_description
        self.run_id = None
        # Force mainnet and provide keys for historical data fetching
        self.session = HTTP(
                testnet=False,
            api_key=BYBIT_API_KEY,
            api_secret=BYBIT_API_SECRET
        )
        self.historical_data = {}
        self.progress = 0
        self.total_symbols = len(self.symbols)

    def _load_cached_data(self, symbol):
        """Load cached data if available and recent"""
        cache_file = os.path.join(os.path.dirname(__file__), '..', 'data', f'{symbol}_cache.pkl')
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    # Use cache if less than 1 hour old
                    if time.time() - cached_data['timestamp'] < 3600:
                        print(f"📦 Using cached data for {symbol}")
                        return cached_data['data']
            except Exception as e:
                print(f"⚠️ Cache error for {symbol}: {e}")
        return None

    def _save_cached_data(self, symbol, data):
        """Save data to cache"""
        cache_file = os.path.join(os.path.dirname(__file__), '..', 'data', f'{symbol}_cache.pkl')
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump({'data': data, 'timestamp': time.time()}, f)
        except Exception as e:
            print(f"⚠️ Cache save error for {symbol}: {e}")

    def _fetch_symbol_data(self, symbol):
        """Fetch data for a single symbol with progress tracking"""
        print(f"📊 Fetching {symbol}...")

        # Try cache first
        cached_data = self._load_cached_data(symbol)
        if cached_data is not None:
            self.progress += 1
            print(f"✅ {symbol} loaded from cache ({self.progress}/{self.total_symbols})")
            return symbol, cached_data

        # Fetch fresh data
        all_klines = []
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
                    current_start_ms = current_end_ms
                else:
                    break

                time.sleep(0.1)  # Rate limiting

            except Exception as e:
                print(f"❌ Error fetching {symbol}: {e}")
                break

        if not all_klines:
            print(f"❌ No data for {symbol}")
            return symbol, None

        df = pd.DataFrame(all_klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df.astype(float).drop_duplicates()
        df = df[(df.index >= self.start_date) & (df.index <= self.end_date)]
        df = df.sort_index()

        # Cache the data
        self._save_cached_data(symbol, df)

        self.progress += 1
        print(f"✅ {symbol} fetched ({self.progress}/{self.total_symbols})")
        return symbol, df

    def fetch_historical_data(self):
        """Fetches historical kline data with caching and parallel processing"""
        print(f"🚀 Starting data fetch for {len(self.symbols)} symbols...")

        # Use parallel processing for faster data fetching
        with ThreadPoolExecutor(max_workers=min(len(self.symbols), 3)) as executor:
            results = list(executor.map(self._fetch_symbol_data, self.symbols))

        # Process results
        for symbol, df in results:
            if df is not None:
                self.historical_data[symbol] = df
                print(f"📈 {symbol}: {len(df)} candles from {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
            else:
                print(f"❌ Failed to get data for {symbol}")

        if not self.historical_data:
            raise Exception("Failed to fetch data for any symbols")

        print("✅ Data fetch complete!")

    def _get_signal(self, historical_prices):
        """Calculates the MA signal with improved filtering. Replicates the logic from strategy.py."""
        if len(historical_prices) < MA_PERIOD + 10:
            return 0

        ma = historical_prices['close'].rolling(window=MA_PERIOD).mean().iloc[-1]
        current_price = historical_prices['close'].iloc[-1]

        # Calculate trend strength using MA slope
        ma_slope = (ma - historical_prices['close'].rolling(window=MA_PERIOD).mean().iloc[-5]) / MA_PERIOD
        trend_strength = abs(ma_slope) / current_price

        # Only trade if trend is reasonably strong (>0.1% per period)
        if trend_strength < 0.001:  # 0.1% minimum trend strength
            return 0

        # Calculate volatility for better signal filtering
        recent_volatility = historical_prices['close'].pct_change().rolling(window=10).std().iloc[-1]

        # Adjust thresholds based on volatility
        if recent_volatility > 0.02:  # High volatility
            threshold = 0.003  # 0.3% threshold
        else:  # Normal volatility
            threshold = 0.001  # 0.1% threshold

        # Improved signal logic with trend confirmation
        if current_price > ma * (1 + threshold):
            # Additional confirmation: price above MA and MA is rising
            if ma_slope > 0:
                return 1
        elif current_price < ma * (1 - threshold):
            # Additional confirmation: price below MA and MA is falling
            if ma_slope < 0:
                return -1

        return 0  # Neutral

    def _calculate_atr(self, prices, period=14):
        """Calculate Average True Range for backtester"""
        if len(prices) < period + 1:
            return 0

        high = prices['high']
        low = prices['low']
        close = prices['close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean().iloc[-1]

        return atr

    def run(self):
        """Runs the backtest and prints the results."""
        # Create backtest run record
        self.run_id = db_manager.create_backtest_run(
            name=self.run_name,
            description=self.run_description,
            symbols=self.symbols,
            start_date=self.start_date,
            end_date=self.end_date,
            parameters={'ma_period': MA_PERIOD, 'leverage': LEVERAGE, 'timeframe': TIMEFRAME}
        )

        if not self.run_id:
            print("❌ Failed to create backtest run record")
            return

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
            stop_loss = 0
            take_profit = 0

            # Iterate through the historical data, simulating the strategy
            for i in range(MA_PERIOD + 10, len(df)):  # Need more data for ATR calculation
                # This ensures no lookahead bias. The strategy only sees data up to the current point in time.
                current_market_slice = df.iloc[i-MA_PERIOD-10:i]
                signal = self._get_signal(current_market_slice)
                current_price = current_market_slice['close'].iloc[-1]

                # Check for exit conditions (stop loss/take profit)
                if position != 0:
                    should_exit = False
                    if position == 1:  # Long position
                        if current_price <= stop_loss or current_price >= take_profit:
                            should_exit = True
                            pnl = (current_price - entry_price) / entry_price
                    else:  # Short position
                        if current_price >= stop_loss or current_price <= take_profit:
                            should_exit = True
                            pnl = (entry_price - current_price) / entry_price

                    if should_exit:
                        trades[-1].update({'exit_date': df.index[i], 'exit_price': current_price, 'pnl': pnl})
                        position = 0
                        entry_price = 0
                        continue

                # --- Execute Trades ---
                if position == 0 and signal != 0: # Open a new position
                    position = signal
                    entry_price = current_price

                    # Calculate stop loss and take profit using ATR
                    atr_prices = current_market_slice.iloc[-20:] if len(current_market_slice) >= 20 else current_market_slice
                    atr = self._calculate_atr(atr_prices)

                    if atr > 0:
                        stop_distance = max(atr * 1.5, entry_price * 0.01)  # At least 1%
                    else:
                        stop_distance = entry_price * 0.02  # 2% fallback

                    take_profit_distance = stop_distance * 2

                    if signal == 1:  # Long
                        stop_loss = entry_price - stop_distance
                        take_profit = entry_price + take_profit_distance
                    else:  # Short
                        stop_loss = entry_price + stop_distance
                        take_profit = entry_price - take_profit_distance

                    trades.append({'entry_date': df.index[i], 'entry_price': entry_price, 'type': 'long' if signal == 1 else 'short'})

                elif position == 1 and signal == -1: # Close long, open short
                    pnl = (current_price - entry_price) / entry_price
                    trades[-1].update({'exit_date': df.index[i], 'exit_price': current_price, 'pnl': pnl})
                    position = -1
                    entry_price = current_price

                    # Set new stop loss and take profit for short
                    atr_prices = current_market_slice.iloc[-20:] if len(current_market_slice) >= 20 else current_market_slice
                    atr = self._calculate_atr(atr_prices)
                    if atr > 0:
                        stop_distance = max(atr * 1.5, entry_price * 0.01)
                    else:
                        stop_distance = entry_price * 0.02
                    take_profit_distance = stop_distance * 2

                    stop_loss = entry_price + stop_distance
                    take_profit = entry_price - take_profit_distance

                    trades.append({'entry_date': df.index[i], 'entry_price': entry_price, 'type': 'short'})

                elif position == -1 and signal == 1: # Close short, open long
                    pnl = (entry_price - current_price) / entry_price
                    trades[-1].update({'exit_date': df.index[i], 'exit_price': current_price, 'pnl': pnl})
                    position = 1
                    entry_price = current_price

                    # Set new stop loss and take profit for long
                    atr_prices = current_market_slice.iloc[-20:] if len(current_market_slice) >= 20 else current_market_slice
                    atr = self._calculate_atr(atr_prices)
                    if atr > 0:
                        stop_distance = max(atr * 1.5, entry_price * 0.01)
                    else:
                        stop_distance = entry_price * 0.02
                    take_profit_distance = stop_distance * 2

                    stop_loss = entry_price - stop_distance
                    take_profit = entry_price + take_profit_distance

                    trades.append({'entry_date': df.index[i], 'entry_price': entry_price, 'type': 'long'})

            all_results[symbol] = pd.DataFrame(trades)
        
        # Collect signal data for visualization
        self.signal_data = self._collect_signal_data()

        self.calculate_metrics(all_results)

    def _collect_signal_data(self):
        """Collect signal data for visualization"""
        signal_data = {}

        for symbol in self.symbols:
            if symbol not in self.historical_data:
                continue

            df = self.historical_data[symbol]
            signals = []
            prices = []
            mas = []

            # Calculate signals throughout the dataset
            for i in range(MA_PERIOD + 10, len(df)):
                current_slice = df.iloc[i-MA_PERIOD-10:i]
                signal = self._get_signal(current_slice)
                current_price = current_slice['close'].iloc[-1]
                ma = current_slice['close'].rolling(window=MA_PERIOD).mean().iloc[-1]

                signals.append({'date': df.index[i], 'signal': signal, 'price': current_price, 'ma': ma})

            if signals:
                signal_df = pd.DataFrame(signals)
                signal_data[symbol] = signal_df

        return signal_data

    def visualize_results(self, symbol=None):
        """Create visualizations for backtest results"""
        if not HAS_MATPLOTLIB:
            print("❌ matplotlib not installed - install with: pip install matplotlib")
            return

        if not self.signal_data:
            print("❌ No signal data available for visualization")
            return

        symbols_to_plot = [symbol] if symbol else list(self.signal_data.keys())

        for sym in symbols_to_plot:
            if sym not in self.signal_data:
                continue

            plt.figure(figsize=(15, 10))

            # Plot 1: Price and MA
            plt.subplot(3, 1, 1)
            data = self.signal_data[sym]
            plt.plot(data['date'], data['price'], label='Price', alpha=0.7)
            plt.plot(data['date'], data['ma'], label=f'MA({MA_PERIOD})', linewidth=2)
            plt.title(f'{sym} - Price & Moving Average')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Plot 2: Signals
            plt.subplot(3, 1, 2)
            buy_signals = data[data['signal'] == 1]
            sell_signals = data[data['signal'] == -1]

            plt.plot(data['date'], data['price'], alpha=0.5, color='gray')

            if not buy_signals.empty:
                plt.scatter(buy_signals['date'], buy_signals['price'],
                           color='green', marker='^', s=100, label='Buy Signal', zorder=5)

            if not sell_signals.empty:
                plt.scatter(sell_signals['date'], sell_signals['price'],
                           color='red', marker='v', s=100, label='Sell Signal', zorder=5)

            plt.title(f'{sym} - Trading Signals')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Plot 3: Signal Strength (MA slope)
            plt.subplot(3, 1, 3)
            ma_slopes = []
            symbol_df = self.historical_data[sym]
            for i in range(MA_PERIOD + 10, len(symbol_df)):
                current_slice = symbol_df.iloc[i-MA_PERIOD-10:i]
                ma = current_slice['close'].rolling(window=MA_PERIOD).mean().iloc[-1]
                prev_ma = current_slice['close'].rolling(window=MA_PERIOD).mean().iloc[-5]
                slope = (ma - prev_ma) / MA_PERIOD
                ma_slopes.append(slope)

            plt.plot(data['date'], ma_slopes, label='MA Slope', color='purple')
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            plt.title(f'{sym} - Trend Strength (MA Slope)')
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(os.path.dirname(__file__), '..', 'results', f'{sym}_backtest_analysis.png'),
                        dpi=300, bbox_inches='tight')
            plt.show()

            print(f"📊 Visualization saved for {sym}")

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

            # Advanced Metrics
            daily_returns = trades_df.set_index('exit_date')['pnl'].resample('D').sum() * LEVERAGE

            # Sharpe Ratio (annualized)
            if daily_returns.std() > 0:
                sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(365)
            else:
                sharpe_ratio = 0

            # Volatility (annualized)
            volatility = daily_returns.std() * np.sqrt(365)

            # Sortino Ratio (only downside volatility)
            downside_returns = daily_returns[daily_returns < 0]
            if len(downside_returns) > 0 and downside_returns.std() > 0:
                sortino_ratio = (daily_returns.mean() / downside_returns.std()) * np.sqrt(365)
            else:
                sortino_ratio = 0

            # Calmar Ratio (return / max drawdown)
            if max_drawdown_pct != 0:
                calmar_ratio = total_return_pct / abs(max_drawdown_pct)
            else:
                calmar_ratio = 0

            # Alpha and Beta (simplified - in real implementation would compare to benchmark)
            # For now, using simple approximations
            alpha = total_return_pct - 0.05  # Assuming 5% risk-free rate
            beta = 1.0  # Assuming market beta of 1.0

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
            print(f"Sortino Ratio:      {sortino_ratio:.2f}")
            print(f"Calmar Ratio:       {calmar_ratio:.2f}")
            print(f"Volatility:         {volatility:.2f}%")
            print(f"Alpha:              {alpha:.2f}%")
            print(f"Beta:               {beta:.2f}")
            print("="*50)

            # Save results to database
            metrics = {
                'total_return_pct': total_return_pct,
                'total_trades': total_trades,
                'win_rate_pct': win_rate,
                'avg_win_pct': avg_win_pct,
                'avg_loss_pct': avg_loss_pct,
                'max_drawdown_pct': max_drawdown_pct,
                'sharpe_ratio': sharpe_ratio,
                'alpha': alpha,
                'beta': beta,
                'volatility': volatility,
                'calmar_ratio': calmar_ratio,
                'sortino_ratio': sortino_ratio
            }

            db_manager.save_backtest_result(self.run_id, symbol, self.start_date, self.end_date, metrics)

        # Mark run as completed
        if self.run_id:
            db_manager.update_backtest_run_status(self.run_id, 'completed')
            print(f"✅ Backtest run {self.run_id} completed!")


if __name__ == "__main__":
    # Configuration for the backtest run - shorter period to avoid API issues
    BACKTEST_START_DATE = datetime.now() - timedelta(days=30) # 1 month of data
    BACKTEST_END_DATE = datetime.now()

    run_name = f"Terminal Backtest - {', '.join(SYMBOLS)} - {BACKTEST_START_DATE.strftime('%Y-%m-%d')} to {BACKTEST_END_DATE.strftime('%Y-%m-%d')}"
    run_description = f"Backtest run from terminal: {len(SYMBOLS)} symbols, {(BACKTEST_END_DATE - BACKTEST_START_DATE).days} days"

    backtester = Backtester(
        symbols=SYMBOLS,
        start_date=BACKTEST_START_DATE,
        end_date=BACKTEST_END_DATE,
        run_name=run_name,
        run_description=run_description
    )

    backtester.run()

    # Show visualizations
    print("\n🎨 Generating visualizations...")
    backtester.visualize_results()

    print("✅ Backtest complete! Check results/ directory for charts.")
