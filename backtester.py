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

class MovingAverageBacktester:
    """Backtest the moving average strategy on historical data"""

    def __init__(self, symbol='XRPUSDT', ma_period=60, initial_balance=1000):
        self.symbol = symbol
        self.ma_period = ma_period
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position = 0  # 0 = no position, >0 = long position size, <0 = short position size
        self.position_value = 0
        self.trades = []
        self.price_data = []

        # Strategy parameters (matching config)
        self.max_position_usdt = 0.50  # Conservative for backtesting
        self.leverage = 10

        # Load API credentials for historical data
        load_dotenv()

    def get_historical_data(self, days=30):
        """Get historical price data for backtesting"""
        logger.info(f"📊 Starting data download: {days} days of {self.symbol} data")

        try:
            session = HTTP(
                testnet=False,
                api_key=os.getenv('BYBIT_API_KEY'),
                api_secret=os.getenv('BYBIT_API_SECRET')
            )

            # Get data in 1-hour chunks for better granularity
            end_time = int(datetime.now().timestamp() * 1000)
            start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

            all_klines = []

            # Get data in chunks to avoid rate limits
            chunk_size = 200  # Bybit limit per request

            for start in range(start_time, end_time, chunk_size * 60 * 60 * 1000):  # Hourly chunks
                end = min(start + chunk_size * 60 * 60 * 1000, end_time)

                response = session.get_kline(
                    category="linear",
                    symbol=self.symbol,
                    interval="1",  # 1-minute data
                    start=start,
                    end=end,
                    limit=200
                )

                if response['retCode'] == 0:
                    klines = response['result']['list']
                    if klines:
                        all_klines.extend(klines)

                # Small delay to respect rate limits
                import time
                time.sleep(0.1)

            logger.info(f"✅ Downloaded {len(all_klines)} data points")

            # Convert to DataFrame
            df = pd.DataFrame(all_klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
            df = df.sort_values('timestamp')
            df['close'] = df['close'].astype(float)

            logger.info(f"📈 Data processed: {len(df)} rows from {df.iloc[0]['timestamp']} to {df.iloc[-1]['timestamp']}")
            logger.info(f"📊 Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")

            return df

        except Exception as e:
            logger.error(f"❌ Error getting historical data: {str(e)}")
            return None

    def calculate_ma_signal(self, prices, index):
        """Calculate moving average signal"""
        if index < self.ma_period:
            return 0  # Not enough data

        ma = prices.iloc[index-self.ma_period:index]['close'].mean()
        current_price = prices.iloc[index]['close']

        # Simple signal: 1 for buy (price > MA), -1 for sell (price < MA)
        if current_price > ma * 1.001:  # 0.1% threshold
            logger.debug(f"📈 BUY SIGNAL: Price ${current_price:.2f} > MA ${ma:.2f} (threshold: 0.1%)")
            return 1
        elif current_price < ma * 0.999:  # 0.1% threshold
            logger.debug(f"📉 SELL SIGNAL: Price ${current_price:.2f} < MA ${ma:.2f} (threshold: 0.1%)")
            return -1
        else:
            logger.debug(f"⏸️ HOLD SIGNAL: Price ${current_price:.2f} ≈ MA ${ma:.2f}")
            return 0

    def execute_trade(self, signal, price, timestamp, index):
        """Execute a trade based on signal"""
        if signal == 1 and self.position <= 0:  # Go long
            # Close short position first if exists
            if self.position < 0:
                pnl = self.close_position(price, 'short_close')
                self.trades.append({
                    'timestamp': timestamp,
                    'action': 'close_short',
                    'price': price,
                    'pnl': pnl,
                    'balance': self.balance
                })

            # Open long position
            position_size = min(self.max_position_usdt * self.leverage / price, self.balance / price)
            self.position = position_size
            self.position_value = position_size * price
            cost = self.position_value / self.leverage  # Actual cash used

            self.balance -= cost

            logger.info(f"📈 LONG POSITION OPENED: {position_size:.6f} @ ${price:.2f} | Cost: ${cost:.2f} | Balance: ${self.balance:.2f}")

            self.trades.append({
                'timestamp': timestamp,
                'action': 'buy',
                'price': price,
                'quantity': position_size,
                'cost': cost,
                'balance': self.balance
            })

        elif signal == -1 and self.position >= 0:  # Go short
            # Close long position first if exists
            if self.position > 0:
                pnl = self.close_position(price, 'long_close')
                self.trades.append({
                    'timestamp': timestamp,
                    'action': 'close_long',
                    'price': price,
                    'pnl': pnl,
                    'balance': self.balance
                })

            # Open short position
            position_size = min(self.max_position_usdt * self.leverage / price, self.balance / price)
            self.position = -position_size  # Negative for short
            self.position_value = position_size * price
            cost = self.position_value / self.leverage

            self.balance -= cost

            logger.info(f"📉 SHORT POSITION OPENED: {position_size:.6f} @ ${price:.2f} | Cost: ${cost:.2f} | Balance: ${self.balance:.2f}")

            self.trades.append({
                'timestamp': timestamp,
                'action': 'sell',
                'price': price,
                'quantity': position_size,
                'cost': cost,
                'balance': self.balance
            })

        elif signal == 0 and self.position != 0:  # Close position
            pnl = self.close_position(price, 'signal_close')
            logger.info(f"🔄 POSITION CLOSED: Signal neutral @ ${price:.2f} | P&L: ${pnl:.4f} | Balance: ${self.balance:.2f}")
            self.trades.append({
                'timestamp': timestamp,
                'action': 'close_position',
                'price': price,
                'pnl': pnl,
                'balance': self.balance
            })

    def close_position(self, price, close_type='manual'):
        """Close current position and return P&L"""
        if self.position == 0:
            return 0

        if self.position > 0:  # Long position
            pnl = (price - self.position_value / abs(self.position)) * abs(self.position)
        else:  # Short position
            pnl = (self.position_value / abs(self.position) - price) * abs(self.position)

        # Add back the margin used
        margin_used = abs(self.position_value) / self.leverage
        self.balance += margin_used

        # Reset position
        old_position = self.position
        self.position = 0
        self.position_value = 0

        return pnl

    def run_backtest(self, days=30):
        """Run the backtest"""
        logger.info(f"🧪 Starting {days}-day backtest for {self.symbol}")
        logger.info(f"📊 Initial Balance: ${self.initial_balance}")
        logger.info(f"📊 Position Size: ${self.max_position_usdt} per trade")
        logger.info(f"📊 Leverage: {self.leverage}x")

        # Get historical data
        df = self.get_historical_data(days)
        if df is None or len(df) < self.ma_period:
            logger.error("❌ Not enough data for backtesting")
            return

        logger.info(f"📈 Backtesting on {len(df)} data points from {df.iloc[0]['timestamp']} to {df.iloc[-1]['timestamp']}")

        # Run simulation
        for i in range(self.ma_period, len(df)):
            timestamp = df.iloc[i]['timestamp']
            price = df.iloc[i]['close']

            # Get signal
            signal = self.calculate_ma_signal(df, i)

            # Execute trade if signal changes or position exists
            if signal != 0 or self.position != 0:
                self.execute_trade(signal, price, timestamp, i)

        # Close any remaining position at the end
        if self.position != 0:
            final_price = df.iloc[-1]['close']
            final_pnl = self.close_position(final_price, 'end_of_backtest')
            logger.info(f"🏁 FINAL POSITION CLOSED: @ ${final_price:.2f} | P&L: ${final_pnl:.4f}")
            self.trades.append({
                'timestamp': df.iloc[-1]['timestamp'],
                'action': 'final_close',
                'price': final_price,
                'pnl': final_pnl,
                'balance': self.balance
            })

        self.calculate_results(df)

    def calculate_results(self, df):
        """Calculate and display backtest results"""
        logger.info("=" * 60)
        logger.info("📊 BACKTEST RESULTS")
        logger.info("=" * 60)

        # Calculate metrics
        total_return = (self.balance - self.initial_balance) / self.initial_balance * 100
        total_trades = len([t for t in self.trades if t['action'] in ['buy', 'sell']])

        winning_trades = [t for t in self.trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in self.trades if t.get('pnl', 0) < 0]

        win_rate = len(winning_trades) / max(total_trades, 1) * 100

        # Calculate Sharpe ratio (simplified)
        pnl_values = [t.get('pnl', 0) for t in self.trades]
        if pnl_values:
            avg_return = np.mean(pnl_values)
            std_return = np.std(pnl_values)
            sharpe_ratio = avg_return / max(std_return, 0.01) * np.sqrt(252)  # Annualized
        else:
            sharpe_ratio = 0

        # Maximum drawdown
        balance_history = [self.initial_balance]
        for trade in self.trades:
            balance_history.append(trade['balance'])

        peak = balance_history[0]
        max_dd = 0
        for balance in balance_history:
            if balance > peak:
                peak = balance
            dd = (peak - balance) / peak * 100
            max_dd = max(max_dd, dd)

        logger.info(f"📈 Final Balance: ${self.balance:.2f}")
        logger.info(f"📈 Total Return: {total_return:.2f}%")
        logger.info(f"📈 Total Trades: {total_trades}")
        logger.info(f"📈 Win Rate: {win_rate:.1f}%")
        logger.info(f"📈 Sharpe Ratio: {sharpe_ratio:.2f}")
        logger.info(f"📈 Max Drawdown: {max_dd:.2f}%")

        # Show trade summary
        logger.info("\n🏆 TRADE SUMMARY:")
        logger.info(f"   Winning Trades: {len(winning_trades)}")
        logger.info(f"   Losing Trades: {len(losing_trades)}")

        if winning_trades:
            avg_win = np.mean([t['pnl'] for t in winning_trades])
            logger.info(f"   Average Win: ${avg_win:.4f}")

        if losing_trades:
            avg_loss = np.mean([t['pnl'] for t in losing_trades])
            logger.info(f"   Average Loss: ${avg_loss:.4f}")

        # Plot results
        self.plot_results(balance_history)

        logger.info("\n🎯 Backtest Complete!")

    def plot_results(self, balance_history):
        """Plot balance over time"""
        try:
            plt.figure(figsize=(12, 6))

            # Plot balance
            plt.subplot(1, 2, 1)
            plt.plot(balance_history)
            plt.title('Account Balance Over Time')
            plt.xlabel('Trade Number')
            plt.ylabel('Balance ($)')
            plt.grid(True, alpha=0.3)

            # Plot returns
            if len(balance_history) > 1:
                returns = [(balance_history[i] - balance_history[i-1]) / balance_history[i-1] * 100
                          for i in range(1, len(balance_history))]
                plt.subplot(1, 2, 2)
                plt.plot(returns)
                plt.title('Trade Returns')
                plt.xlabel('Trade Number')
                plt.ylabel('Return (%)')
                plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig('backtest_results.png', dpi=150, bbox_inches='tight')
            logger.info("📊 Results chart saved as 'backtest_results.png'")

        except Exception as e:
            logger.warning(f"⚠️ Could not create chart: {str(e)}")

def main():
    """Run the backtester"""
    logger.info("🚀 TGT QR Moving Average Strategy Backtester")
    logger.info("=" * 60)

    # Create backtester
    backtester = MovingAverageBacktester(
        symbol='XRPUSDT',
        ma_period=60,  # 60-minute MA
        initial_balance=1000  # $1000 starting balance
    )

    logger.info(f"🔧 Backtester initialized: {backtester.symbol} MA{backtester.ma_period} ${backtester.initial_balance} balance")

    # Run backtest
    backtester.run_backtest(days=7)  # Test on last 7 days

if __name__ == "__main__":
    main()
