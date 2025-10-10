#!/usr/bin/env python3
"""
Comprehensive Strategy Testing Framework
Test multiple strategies and parameters to find optimal combinations
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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - TESTER - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/strategy_testing.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('TESTER')

class ComprehensiveStrategyTester:
    """Test multiple strategies and parameters systematically"""

    def __init__(self):
        load_dotenv()

    def get_historical_data(self, symbol, days=30):
        """Get historical price data"""
        try:
            session = HTTP(
                testnet=False,
                api_key=os.getenv('BYBIT_API_KEY'),
                api_secret=os.getenv('BYBIT_API_SECRET')
            )

            end_time = int(datetime.now().timestamp() * 1000)
            start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

            all_klines = []
            chunk_size = 200

            for start in range(start_time, end_time, chunk_size * 60 * 60 * 1000):
                end = min(start + chunk_size * 60 * 60 * 1000, end_time)

                response = session.get_kline(
                    category="linear",
                    symbol=symbol,
                    interval="1",
                    start=start,
                    end=end,
                    limit=200
                )

                if response['retCode'] == 0:
                    klines = response['result']['list']
                    if klines:
                        all_klines.extend(klines)

                import time
                time.sleep(0.1)

            df = pd.DataFrame(all_klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
            df = df.sort_values('timestamp')
            df['close'] = df['close'].astype(float)

            logger.info(f"Downloaded {len(df)} data points for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Error getting data for {symbol}: {str(e)}")
            return None

    def calculate_ma_signal(self, prices, index, ma_period, threshold_pct=0.1):
        """Calculate moving average signal"""
        if index < ma_period:
            return 0

        ma = prices.iloc[index-ma_period:index]['close'].mean()
        current_price = prices.iloc[index]['close']
        threshold = ma * (threshold_pct / 100)

        if current_price > ma + threshold:
            return 1
        elif current_price < ma - threshold:
            return -1
        else:
            return 0

    def calculate_rsi_signal(self, prices, index, rsi_period=14, overbought=70, oversold=30):
        """Calculate RSI signal"""
        if index < rsi_period:
            return 0

        # Calculate RSI
        deltas = prices['close'].diff()
        gains = (deltas.where(deltas > 0, 0)).rolling(window=rsi_period).mean()
        losses = (-deltas.where(deltas < 0, 0)).rolling(window=rsi_period).mean()

        rs = gains / losses
        rsi = 100 - (100 / (1 + rs))

        current_rsi = rsi.iloc[index]

        if current_rsi < oversold:
            return 1  # Buy signal
        elif current_rsi > overbought:
            return -1  # Sell signal
        else:
            return 0

    def run_strategy_test(self, symbol, strategy_type='ma', ma_period=60, threshold_pct=0.1,
                         rsi_period=14, initial_balance=1000, position_size_pct=0.5):
        """Run a strategy test"""
        df = self.get_historical_data(symbol, days=30)
        if df is None or len(df) < max(ma_period, rsi_period):
            return None

        balance = initial_balance
        position = 0
        position_value = 0
        trades = []

        max_position_usdt = initial_balance * position_size_pct
        leverage = 10

        for i in range(max(ma_period, rsi_period), len(df)):
            timestamp = df.iloc[i]['timestamp']
            price = df.iloc[i]['close']

            # Get signal based on strategy type
            if strategy_type == 'ma':
                signal = self.calculate_ma_signal(df, i, ma_period, threshold_pct)
            elif strategy_type == 'rsi':
                signal = self.calculate_rsi_signal(df, i, rsi_period)
            else:
                signal = 0

            # Execute trades
            if signal == 1 and position <= 0:
                if position < 0:
                    pnl = self.close_position(price, position, position_value, leverage)
                    balance += pnl
                    trades.append({'timestamp': timestamp, 'action': 'close_short', 'pnl': pnl})

                position_size = min(max_position_usdt * leverage / price, balance / price)
                position = position_size
                position_value = position_size * price
                cost = position_value / leverage
                balance -= cost

                trades.append({
                    'timestamp': timestamp,
                    'action': 'buy',
                    'price': price,
                    'quantity': position_size,
                    'cost': cost
                })

            elif signal == -1 and position >= 0:
                if position > 0:
                    pnl = self.close_position(price, position, position_value, leverage)
                    balance += pnl
                    trades.append({'timestamp': timestamp, 'action': 'close_long', 'pnl': pnl})

                position_size = min(max_position_usdt * leverage / price, balance / price)
                position = -position_size
                position_value = position_size * price
                cost = position_value / leverage
                balance -= cost

                trades.append({
                    'timestamp': timestamp,
                    'action': 'sell',
                    'price': price,
                    'quantity': position_size,
                    'cost': cost
                })

            elif signal == 0 and position != 0:
                pnl = self.close_position(price, position, position_value, leverage)
                balance += pnl
                trades.append({'timestamp': timestamp, 'action': 'close_position', 'pnl': pnl})
                position = 0
                position_value = 0

        # Close final position
        if position != 0:
            final_price = df.iloc[-1]['close']
            pnl = self.close_position(final_price, position, position_value, leverage)
            balance += pnl

        return {
            'symbol': symbol,
            'strategy': strategy_type,
            'ma_period': ma_period,
            'threshold_pct': threshold_pct,
            'rsi_period': rsi_period,
            'final_balance': balance,
            'total_return': (balance - initial_balance) / initial_balance * 100,
            'total_trades': len([t for t in trades if t['action'] in ['buy', 'sell']]),
            'winning_trades': len([t for t in trades if t.get('pnl', 0) > 0]),
            'losing_trades': len([t for t in trades if t.get('pnl', 0) < 0]),
            'max_drawdown': self.calculate_max_drawdown(trades, initial_balance)
        }

    def close_position(self, price, position, position_value, leverage):
        """Close position and return P&L"""
        if position == 0:
            return 0

        if position > 0:
            pnl = (price - position_value / abs(position)) * abs(position)
        else:
            pnl = (position_value / abs(position) - price) * abs(position)

        return pnl

    def calculate_max_drawdown(self, trades, initial_balance):
        """Calculate maximum drawdown"""
        balance_history = [initial_balance]
        for trade in trades:
            if 'pnl' in trade:
                balance_history.append(balance_history[-1] + trade['pnl'])

        peak = balance_history[0]
        max_dd = 0

        for balance in balance_history:
            if balance > peak:
                peak = balance
            dd = (peak - balance) / peak * 100 if peak > 0 else 0
            max_dd = max(max_dd, dd)

        return max_dd

    def run_comprehensive_test(self):
        """Run comprehensive strategy testing"""
        symbols = ['XRPUSDT', 'BTCUSDT', 'ETHUSDT']
        strategies = ['ma', 'rsi']
        ma_periods = [20, 40, 60, 80]
        thresholds = [0.05, 0.1, 0.2, 0.5]
        rsi_periods = [14, 21]
        overbought_levels = [70, 75]
        oversold_levels = [30, 25]

        results = []

        logger.info("Starting comprehensive strategy testing...")
        logger.info("=" * 80)

        # Test Moving Average strategies
        for symbol in symbols:
            logger.info(f"\nTesting {symbol} with Moving Average strategies...")

            for ma_period in ma_periods:
                for threshold in thresholds:
                    result = self.run_strategy_test(symbol, 'ma', ma_period, threshold)

                    if result:
                        results.append(result)
                        logger.info(f"   MA{ma_period:2d} Thr{threshold:4.1f}: "
                                  f"Return {result['total_return']:6.2f}% | "
                                  f"Trades {result['total_trades']:2d} | "
                                  f"Win {result['winning_trades']:2d}/{result['losing_trades']:2d}")

        # Test RSI strategies
        for symbol in symbols:
            logger.info(f"\nTesting {symbol} with RSI strategies...")

            for rsi_period in rsi_periods:
                for overbought in overbought_levels:
                    for oversold in oversold_levels:
                        result = self.run_strategy_test(symbol, 'rsi', rsi_period=rsi_period,
                                                      initial_balance=1000)

                        if result:
                            results.append(result)
                            logger.info(f"   RSI{rsi_period:2d} OB{overbought:2d} OS{oversold:2d}: "
                                      f"Return {result['total_return']:6.2f}% | "
                                      f"Trades {result['total_trades']:2d} | "
                                      f"Win {result['winning_trades']:2d}/{result['losing_trades']:2d}")

        # Sort by total return
        results.sort(key=lambda x: x['total_return'], reverse=True)

        logger.info("\n" + "=" * 80)
        logger.info("TOP PERFORMING STRATEGIES")
        logger.info("=" * 80)

        for i, result in enumerate(results[:15], 1):
            strategy_name = f"{result['strategy'].upper()}{result.get('ma_period', result.get('rsi_period', ''))}"
            logger.info(f"{i:2d}. {result['symbol']:8s} {strategy_name:6s}: "
                       f"Return {result['total_return']:6.2f}% | "
                       f"Trades {result['total_trades']:2d} | "
                       f"Win Rate {result['winning_trades']/max(result['total_trades'],1)*100:5.1f}%")

        # Find best parameters
        best = results[0]
        logger.info("
🎯 BEST STRATEGY:"        logger.info(f"   Symbol: {best['symbol']}")
        logger.info(f"   Strategy: {best['strategy'].upper()}")
        logger.info(f"   Return: {best['total_return']:.2f}%")
        logger.info(f"   Trades: {best['total_trades']}")
        logger.info(f"   Win Rate: {best['winning_trades']/max(best['total_trades'],1)*100:.1f}%")
        logger.info(f"   Max Drawdown: {best['max_drawdown']:.2f}%")

        # Save results
        df = pd.DataFrame(results)
        df.to_csv('comprehensive_strategy_results.csv', index=False)
        logger.info("Results saved to 'comprehensive_strategy_results.csv'")

        return results

def main():
    """Run comprehensive strategy testing"""
    tester = ComprehensiveStrategyTester()
    results = tester.run_comprehensive_test()

    logger.info("Comprehensive strategy testing complete!")

if __name__ == "__main__":
    main()

