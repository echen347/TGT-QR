#!/usr/bin/env python3
"""
Moving Average Strategy Backtester
Tests the strategy on historical data without placing real orders
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse

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

    def __init__(self, symbols, start_date, end_date, run_name=None, run_description=None,
                 timeframe: str = None, ma_period: int = None, fee_bps: float = 5.0,
                 slippage_bps: float = 2.0, cache_ttl_sec: int = 3600, max_workers: int = 3,
                 no_db: bool = False, atr_mult: float = 2.0, min_stop_pct: float = 0.01,
                 enable_atr_gate: bool = False, atr_gate_mult: float = 0.5, cooldown_bars: int = 0):
        self.db_manager = db_manager # Initialize db_manager
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.run_name = run_name or f"Backtest {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        self.run_description = run_description
        self.run_id = None
        # Overrides / simulation params
        self.timeframe = str(timeframe or TIMEFRAME)
        self.ma_period = int(ma_period or MA_PERIOD)
        self.fee_bps = float(fee_bps)
        self.slippage_bps = float(slippage_bps)
        self.cache_ttl_sec = int(cache_ttl_sec)
        self.max_workers = int(max_workers)
        self.no_db = bool(no_db)
        self.atr_mult = float(atr_mult)
        self.min_stop_pct = float(min_stop_pct)
        self.enable_atr_gate = bool(enable_atr_gate)
        self.atr_gate_mult = float(atr_gate_mult)
        self.cooldown_bars = int(cooldown_bars)
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
                    # Use cache if within TTL
                    if time.time() - cached_data['timestamp'] < self.cache_ttl_sec:
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
        # Pull in chunks of up to 1000 candles for the selected timeframe
        try:
            tf_minutes = int(self.timeframe)
        except Exception:
            tf_minutes = 60
        chunk_duration_ms = tf_minutes * 60 * 1000 * 1000  # 1000 candles per chunk
        current_start_ms = int(self.start_date.timestamp() * 1000)
        end_limit_ms = int(self.end_date.timestamp() * 1000)

        while current_start_ms < end_limit_ms:
            current_end_ms = current_start_ms + chunk_duration_ms
            try:
                response = self.session.get_kline(
                    category="linear",
                    symbol=symbol,
                    interval=self.timeframe,
                    start=current_start_ms,
                    end=current_end_ms,
                    limit=1000
                )

                if response['retCode'] == 0 and response['result']['list']:
                    klines = response['result']['list']
                    all_klines.extend(klines)
                    current_start_ms = current_end_ms
                else:
                    # Respect rate limits / empty window
                    time.sleep(0.2)
                    break

                time.sleep(0.1)  # basic rate limiting

            except Exception as e:
                print(f"❌ Error fetching {symbol}: {e}")
                break
            
        if not all_klines:
            print(f"❌ No data for {symbol}")
            return symbol, None

        df = pd.DataFrame(all_klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
        df['timestamp'] = pd.to_datetime(pd.to_numeric(df['timestamp'], errors='coerce'), unit='ms')
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
        """Calculates the MA signal with improved filtering. MUST MATCH strategy.py exactly."""
        if len(historical_prices) < self.ma_period + 10:
            return 0

        ma = historical_prices['close'].rolling(window=self.ma_period).mean().iloc[-1]
        current_price = historical_prices['close'].iloc[-1]

        # Calculate trend strength using MA slope
        ma_slope = (ma - historical_prices['close'].rolling(window=self.ma_period).mean().iloc[-5]) / self.ma_period
        trend_strength = abs(ma_slope) / current_price

        # Use config values - MUST match strategy.py
        from config.config import MIN_TREND_STRENGTH, VOLATILITY_THRESHOLD_HIGH, VOLATILITY_THRESHOLD_LOW
        
        # Only trade if trend is reasonably strong
        if trend_strength < MIN_TREND_STRENGTH:
            return 0

        # Calculate volatility for better signal filtering
        recent_volatility = historical_prices['close'].pct_change().rolling(window=10).std().iloc[-1]

        # Adjust thresholds based on volatility - MUST match strategy.py
        if recent_volatility > VOLATILITY_THRESHOLD_HIGH:  # High volatility
            threshold = 0.003  # 0.3% threshold
        elif recent_volatility > VOLATILITY_THRESHOLD_LOW:  # Normal volatility
            threshold = 0.001  # 0.1% threshold
        else:  # Low volatility
            threshold = 0.0005  # 0.05% threshold for very calm markets

        # ATR distance gate (optional): avoid trading too close to MA (chop)
        if self.enable_atr_gate:
            atr_local = self._calculate_atr(historical_prices.tail(20), period=14)
            if atr_local > 0:
                if abs(current_price - ma) < self.atr_gate_mult * atr_local:
                    return 0

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
        if not self.no_db:
            self.run_id = self.db_manager.create_backtest_run(
            name=self.run_name,
            description=self.run_description,
            symbols=self.symbols,
            start_date=self.start_date,
            end_date=self.end_date,
            parameters={'ma_period': self.ma_period, 'leverage': LEVERAGE, 'timeframe': self.timeframe}
        )

        if not self.no_db and not self.run_id:
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
            cooldown = 0
            for i in range(self.ma_period + 10, len(df)):  # Need more data for ATR calculation
                # This ensures no lookahead bias. The strategy only sees data up to the current point in time.
                current_market_slice = df.iloc[i-self.ma_period-10:i]
                signal = self._get_signal(current_market_slice)
                current_price = current_market_slice['close'].iloc[-1]
                fee_pct = self.fee_bps / 10000.0
                slip_pct = self.slippage_bps / 10000.0

                # Apply cooldown after a closed losing trade
                if cooldown > 0:
                    cooldown -= 1
                    signal = 0

                # Check for exit conditions (stop loss/take profit)
                if position != 0:
                    should_exit = False
                    if position == 1:  # Long position
                        if current_price <= stop_loss or current_price >= take_profit:
                            should_exit = True
                            # execute sell with negative slippage
                            exit_exec = current_price * (1 - slip_pct)
                            pnl = (exit_exec - entry_price) / entry_price - (2 * fee_pct)
                    else:  # Short position
                        if current_price >= stop_loss or current_price <= take_profit:
                            should_exit = True
                            # execute buy with positive slippage
                            exit_exec = current_price * (1 + slip_pct)
                            pnl = (entry_price - exit_exec) / entry_price - (2 * fee_pct)

                    if should_exit:
                        trades[-1].update({'exit_date': df.index[i], 'exit_price': current_price, 'pnl': pnl})
                        position = 0
                        entry_price = 0
                        # set cooldown after loss
                        if pnl < 0 and self.cooldown_bars > 0:
                            cooldown = self.cooldown_bars
                        continue

                # --- Execute Trades ---
                if position == 0 and signal != 0: # Open a new position
                    position = signal
                    # execute entry with adverse slippage
                    entry_price = current_price * (1 + slip_pct) if signal == 1 else current_price * (1 - slip_pct)

                    # Calculate stop loss and take profit using ATR
                    atr_prices = current_market_slice.iloc[-20:] if len(current_market_slice) >= 20 else current_market_slice
                    atr = self._calculate_atr(atr_prices)

                    if atr > 0:
                        stop_distance = max(atr * self.atr_mult, entry_price * self.min_stop_pct)
                    else:
                        stop_distance = entry_price * max(self.min_stop_pct * 2, 0.02)  # fallback

                    take_profit_distance = stop_distance * 2

                    if signal == 1:  # Long
                        stop_loss = entry_price - stop_distance
                        take_profit = entry_price + take_profit_distance
                    else:  # Short
                        stop_loss = entry_price + stop_distance
                        take_profit = entry_price - take_profit_distance

                    trades.append({'entry_date': df.index[i], 'entry_price': entry_price, 'type': 'long' if signal == 1 else 'short',
                                   'fee_bps': self.fee_bps, 'slippage_bps': self.slippage_bps})
                
                elif position == 1 and signal == -1: # Close long, open short
                    exit_exec = current_price * (1 - slip_pct)
                    pnl = (exit_exec - entry_price) / entry_price - (2 * fee_pct)
                    trades[-1].update({'exit_date': df.index[i], 'exit_price': current_price, 'pnl': pnl})
                    position = -1
                    entry_price = current_price * (1 - slip_pct)

                    # Set new stop loss and take profit for short
                    atr_prices = current_market_slice.iloc[-20:] if len(current_market_slice) >= 20 else current_market_slice
                    atr = self._calculate_atr(atr_prices)
                    if atr > 0:
                        stop_distance = max(atr * self.atr_mult, entry_price * self.min_stop_pct)
                    else:
                        stop_distance = entry_price * max(self.min_stop_pct * 2, 0.02)
                    take_profit_distance = stop_distance * 2

                    stop_loss = entry_price + stop_distance
                    take_profit = entry_price - take_profit_distance

                    trades.append({'entry_date': df.index[i], 'entry_price': entry_price, 'type': 'short',
                                   'fee_bps': self.fee_bps, 'slippage_bps': self.slippage_bps})

                elif position == -1 and signal == 1: # Close short, open long
                    exit_exec = current_price * (1 + slip_pct)
                    pnl = (entry_price - exit_exec) / entry_price - (2 * fee_pct)
                    trades[-1].update({'exit_date': df.index[i], 'exit_price': current_price, 'pnl': pnl})
                    position = 1
                    entry_price = current_price * (1 + slip_pct)

                    # Set new stop loss and take profit for long
                    atr_prices = current_market_slice.iloc[-20:] if len(current_market_slice) >= 20 else current_market_slice
                    atr = self._calculate_atr(atr_prices)
                    if atr > 0:
                        stop_distance = max(atr * self.atr_mult, entry_price * self.min_stop_pct)
                    else:
                        stop_distance = entry_price * max(self.min_stop_pct * 2, 0.02)
                    take_profit_distance = stop_distance * 2

                    stop_loss = entry_price - stop_distance
                    take_profit = entry_price + take_profit_distance

                    trades.append({'entry_date': df.index[i], 'entry_price': entry_price, 'type': 'long',
                                   'fee_bps': self.fee_bps, 'slippage_bps': self.slippage_bps})

            # Only include completed trades (those with exit information)
            completed_trades = [trade for trade in trades if 'exit_date' in trade and 'pnl' in trade]
            # Ensure all trades have consistent structure
            for trade in completed_trades:
                # Add any missing fields with default values
                if 'exit_price' not in trade:
                    trade['exit_price'] = 0
            all_results[symbol] = pd.DataFrame(completed_trades) if completed_trades else pd.DataFrame()
        
        # Collect signal data for visualization
        self.signal_data = self._collect_signal_data()
        
        self.calculate_metrics(all_results)
        if not self.no_db:
            self.calculate_and_save_run_summary(all_results)

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
            for i in range(self.ma_period + 10, len(df)):
                current_slice = df.iloc[i-self.ma_period-10:i]
                signal = self._get_signal(current_slice)
                current_price = current_slice['close'].iloc[-1]
                ma = current_slice['close'].rolling(window=self.ma_period).mean().iloc[-1]

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
            plt.plot(data['date'], data['ma'], label=f'MA({self.ma_period})', linewidth=2)
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
            for i in range(self.ma_period + 10, len(symbol_df)):
                current_slice = symbol_df.iloc[i-self.ma_period-10:i]
                ma = current_slice['close'].rolling(window=self.ma_period).mean().iloc[-1]
                prev_ma = current_slice['close'].rolling(window=self.ma_period).mean().iloc[-5]
                slope = (ma - prev_ma) / self.ma_period
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
            if trades_df.empty or 'pnl' not in trades_df.columns or trades_df.dropna(subset=['pnl']).empty:
                logging.warning(f"\nSymbol: {symbol}\nNo trades executed or no PnL data. Saving zeroed results.")
                # Save zeroed-out metrics only if DB writes are enabled
                if not self.no_db:
                    self.db_manager.save_backtest_result(self.run_id, symbol, self.start_date, self.end_date, {
                        'total_return_pct': 0, 'total_trades': 0, 'win_rate_pct': 0,
                        'avg_win_pct': 0, 'avg_loss_pct': 0, 'max_drawdown_pct': 0,
                        'sharpe_ratio': 0, 'alpha': 0, 'beta': 1.0, 'volatility': 0,
                        'calmar_ratio': 0, 'sortino_ratio': 0
                    })
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
            loss_trades = trades_df[trades_df['pnl'] < 0]
            avg_loss_pct = loss_trades['pnl'].mean() * 100 * LEVERAGE if not loss_trades.empty else 0

            # Max Drawdown
            cumulative_max = cumulative_return.cummax()
            drawdown = (cumulative_return - cumulative_max) / cumulative_max
            max_drawdown_pct = drawdown.min() * 100 if not np.isnan(drawdown.min()) else 0

            # Advanced Metrics
            daily_returns = trades_df.set_index('exit_date')['pnl'].resample('D').sum() * LEVERAGE

            # Sharpe Ratio (annualized)
            if daily_returns.std() > 0:
                sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(365)
            else:
                sharpe_ratio = 0

            # Volatility (annualized)
            daily_std = daily_returns.std()
            volatility = daily_std * np.sqrt(365) if not np.isnan(daily_std) and daily_std > 0 else 0

            # Sortino Ratio (only downside volatility)
            downside_returns = daily_returns[daily_returns < 0]
            downside_std = downside_returns.std()
            if len(downside_returns) > 0 and not np.isnan(downside_std) and downside_std > 0:
                sortino_ratio = (daily_returns.mean() / downside_std) * np.sqrt(365)
            else:
                sortino_ratio = 0

            # Calmar Ratio (return / max drawdown)
            if not np.isnan(max_drawdown_pct) and max_drawdown_pct != 0:
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

            if not self.no_db:
                self.db_manager.save_backtest_result(self.run_id, symbol, self.start_date, self.end_date, metrics)

            # Save detailed trades for visualization
            if not self.no_db:
                self.db_manager.save_backtest_trades(self.run_id, symbol, trades_df)

        # Mark run as completed
        if self.run_id:
            self.db_manager.update_backtest_run_status(self.run_id, 'completed')
            print(f"✅ Backtest run {self.run_id} completed!")

    def calculate_and_save_run_summary(self, all_results):
        """Calculates the overall summary for the entire run and saves it."""
        logging.info(f"Calculating and saving summary for Run ID: {self.run_id}...")
        
        # Ensure there are results to process
        if not all_results:
            logging.warning("No results found to generate a summary.")
            zero_summary = {'total_pnl': 0, 'total_trades': 0, 'win_rate': 0, 'sharpe_ratio': 0, 'max_drawdown': 0, 'avg_return': 0}
            self.db_manager.update_backtest_run_summary(self.run_id, zero_summary)
            return

        # Filter out empty DataFrames before concatenation
        valid_results = [df for df in all_results.values() if not df.empty]

        if not valid_results:
            logging.info("Backtest resulted in no trades across all symbols.")
            zero_summary = {'total_pnl': 0, 'total_trades': 0, 'win_rate': 0, 'sharpe_ratio': 0, 'max_drawdown': 0, 'avg_return': 0}
            self.db_manager.update_backtest_run_summary(self.run_id, zero_summary)
            return

        all_trades = pd.concat(valid_results)

        # Final check if the concatenated frame is empty or lacks the 'pnl' column
        if all_trades.empty or 'pnl' not in all_trades.columns:
            logging.info("Concatenated trades DataFrame is empty or missing 'pnl' column.")
            zero_summary = {'total_pnl': 0, 'total_trades': 0, 'win_rate': 0, 'sharpe_ratio': 0, 'max_drawdown': 0, 'avg_return': 0}
            self.db_manager.update_backtest_run_summary(self.run_id, zero_summary)
            return

        total_pnl = all_trades['pnl'].sum()
        total_trades = len(all_trades)
        win_rate = (all_trades['pnl'] > 0).sum() / total_trades if total_trades > 0 else 0
        
        # Overall Sharpe (simplified - should ideally use portfolio returns)
        daily_returns = all_trades.set_index('exit_date')['pnl'].resample('D').sum() * LEVERAGE
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(365) if daily_returns.std() > 0 else 0

        # Overall Max Drawdown
        # This is a simplified calculation. A true portfolio drawdown would track equity curve over time.
        portfolio_curve = (1 + daily_returns).cumprod()
        cumulative_max = portfolio_curve.cummax()
        drawdown = (portfolio_curve - cumulative_max) / cumulative_max
        max_drawdown = drawdown.min() if not np.isnan(drawdown.min()) else 0
        
        summary = {
            'total_pnl': total_pnl, # Note: This is a simple sum, not compounded return
            'total_trades': total_trades,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'avg_return': all_trades['pnl'].mean() * LEVERAGE
        }
        
        # Convert numpy types to native python types
        for key, value in summary.items():
            if hasattr(value, 'item'):
                summary[key] = value.item()
        
        if self.run_id:
            logger.info(f"📈 Attempting to save overall run summary for run_id {self.run_id}...")
            logger.info(f"Summary data to be saved: {summary}")
            self.db_manager.update_backtest_run_summary(self.run_id, summary)
            logger.info(f"✅ Successfully saved summary for run_id {self.run_id}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robust MA backtester (CLI)")
    parser.add_argument('--symbols', type=str, default=','.join(SYMBOLS), help='Comma-separated symbols, e.g. BTCUSDT,ETHUSDT')
    parser.add_argument('--days', type=int, default=30, help='Days of history to fetch')
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--timeframe', type=str, default=str(TIMEFRAME), help='Bybit kline interval (e.g., 60)')
    parser.add_argument('--ma', type=int, default=int(MA_PERIOD), help='MA period')
    parser.add_argument('--fee-bps', type=float, default=5.0, help='Round-trip fee in basis points (per side ~2.5bps)')
    parser.add_argument('--slippage-bps', type=float, default=2.0, help='Adverse slippage in basis points per trade leg')
    parser.add_argument('--cache-ttl', type=int, default=3600, help='Cache TTL seconds')
    parser.add_argument('--max-workers', type=int, default=3, help='Parallel fetch workers')
    parser.add_argument('--no-db', action='store_true', help='Disable DB writes for quick ad-hoc runs')
    parser.add_argument('--no-plot', action='store_true', help='Skip matplotlib plots')
    parser.add_argument('--atr-mult', type=float, default=2.0, help='ATR multiplier for stop distance')
    parser.add_argument('--min-stop-pct', type=float, default=0.01, help='Minimum stop distance as fraction of price')
    parser.add_argument('--atr-gate', action='store_true', help='Enable ATR distance gate from MA to avoid chop')
    parser.add_argument('--atr-gate-mult', type=float, default=0.5, help='ATR gate multiple for MA distance')
    parser.add_argument('--cooldown-bars', type=int, default=0, help='Bars to skip after a losing trade (per symbol)')

    args = parser.parse_args()

    if args.start and args.end:
        BACKTEST_START_DATE = datetime.strptime(args.start, '%Y-%m-%d')
        BACKTEST_END_DATE = datetime.strptime(args.end, '%Y-%m-%d')
    else:
        BACKTEST_START_DATE = datetime.now() - timedelta(days=args.days)
    BACKTEST_END_DATE = datetime.now()

    symbols_list = [s.strip().upper() for s in args.symbols.split(',') if s.strip()]

    run_name = f"CLI Backtest - {', '.join(symbols_list)} - {BACKTEST_START_DATE.strftime('%Y-%m-%d')} to {BACKTEST_END_DATE.strftime('%Y-%m-%d')}"
    run_description = f"Backtest: {len(symbols_list)} symbols, {(BACKTEST_END_DATE - BACKTEST_START_DATE).days} days, TF={args.timeframe}, MA={args.ma}"

    backtester = Backtester(
        symbols=symbols_list,
        start_date=BACKTEST_START_DATE,
        end_date=BACKTEST_END_DATE,
        run_name=run_name,
        run_description=run_description,
        timeframe=args.timeframe,
        ma_period=args.ma,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        cache_ttl_sec=args.cache_ttl,
        max_workers=args.max_workers,
        no_db=args.no_db
        , atr_mult=args.atr_mult
        , min_stop_pct=args.min_stop_pct
        , enable_atr_gate=args.atr_gate
        , atr_gate_mult=args.atr_gate_mult
        , cooldown_bars=args.cooldown_bars
    )
    
    backtester.run()

    if not args.no_plot:
        print("\n🎨 Generating visualizations...")
        backtester.visualize_results()
        print("✅ Backtest complete! Check results/ directory for charts.")
