#!/usr/bin/env python3
"""
Strategy Optimizer - Test different parameters and symbols for profitability
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sys
import os
from dotenv import load_dotenv
from pybit.unified_trading import HTTP

# Add src to path for imports
sys.path.append('src')

class StrategyOptimizer:
    """Test different strategy parameters and symbols for optimal performance"""

    def __init__(self):
        load_dotenv()

    def get_historical_data(self, symbol, days=7):
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

            return df

        except Exception as e:
            print(f"❌ Error getting data for {symbol}: {str(e)}")
            return None

    def calculate_ma_signal(self, prices, index, ma_period, threshold_pct=0.1):
        """Calculate moving average signal with configurable threshold"""
        if index < ma_period:
            return 0

        ma = prices.iloc[index-ma_period:index]['close'].mean()
        current_price = prices.iloc[index]['close']

        threshold = ma * (threshold_pct / 100)

        if current_price > ma + threshold:
            return 1  # Buy signal
        elif current_price < ma - threshold:
            return -1  # Sell signal
        else:
            return 0  # Hold signal

    def run_single_backtest(self, symbol, ma_period, threshold_pct, initial_balance=1000, position_size_pct=0.5):
        """Run a single backtest with given parameters"""
        df = self.get_historical_data(symbol, days=7)
        if df is None or len(df) < ma_period:
            return None

        balance = initial_balance
        position = 0
        position_value = 0
        trades = []

        # Risk management
        max_position_usdt = initial_balance * position_size_pct
        leverage = 10

        for i in range(ma_period, len(df)):
            timestamp = df.iloc[i]['timestamp']
            price = df.iloc[i]['close']

            signal = self.calculate_ma_signal(df, i, ma_period, threshold_pct)

            if signal == 1 and position <= 0:
                if position < 0:  # Close short first
                    pnl = self.close_position(price, position, position_value, leverage)
                    balance += pnl
                    trades.append({'timestamp': timestamp, 'action': 'close_short', 'pnl': pnl})

                # Open long
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
                if position > 0:  # Close long first
                    pnl = self.close_position(price, position, position_value, leverage)
                    balance += pnl
                    trades.append({'timestamp': timestamp, 'action': 'close_long', 'pnl': pnl})

                # Open short
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
            'ma_period': ma_period,
            'threshold_pct': threshold_pct,
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

        if position > 0:  # Long position
            pnl = (price - position_value / abs(position)) * abs(position)
        else:  # Short position
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

    def optimize_strategy(self):
        """Test different parameters and symbols"""
        symbols = ['XRPUSDT', 'BTCUSDT', 'ETHUSDT']
        ma_periods = [20, 40, 60, 80]
        thresholds = [0.05, 0.1, 0.2, 0.5]

        results = []

        print("🚀 Testing strategy combinations...")
        print("=" * 80)

        for symbol in symbols:
            print(f"\n📈 Testing {symbol}...")

            for ma_period in ma_periods:
                for threshold in thresholds:
                    result = self.run_single_backtest(symbol, ma_period, threshold)

                    if result:
                        results.append(result)

                        print(f"   MA{ma_period:2d} Thr{threshold:4.1f}: "
                              f"Return {result['total_return']:6.2f}% | "
                              f"Trades {result['total_trades']:2d} | "
                              f"Win {result['winning_trades']:2d}/{result['losing_trades']:2d} | "
                              f"DD {result['max_drawdown']:5.2f}%")

        # Sort by total return
        results.sort(key=lambda x: x['total_return'], reverse=True)

        print("\n" + "=" * 80)
        print("🏆 TOP PERFORMING STRATEGIES")
        print("=" * 80)

        for i, result in enumerate(results[:10], 1):
            print(f"{i:2d}. {result['symbol']:8s} MA{result['ma_period']:2d} "
                  f"Thr{result['threshold_pct']:4.1f}: "
                  f"Return {result['total_return']:6.2f}% | "
                  f"Trades {result['total_trades']:2d} | "
                  f"Win Rate {result['winning_trades']/max(result['total_trades'],1)*100:5.1f}%")

        # Find best parameters
        best = results[0]
        print("\n🎯 BEST STRATEGY:")
        print(f"   Symbol: {best['symbol']}")
        print(f"   MA Period: {best['ma_period']}")
        print(f"   Threshold: {best['threshold_pct']}%")
        print(f"   Return: {best['total_return']:.2f}% | Trades: {best['total_trades']}")
        print(f"   Win Rate: {best['winning_trades']/max(best['total_trades'],1)*100:.1f}%")
        print(f"   Max Drawdown: {best['max_drawdown']:.2f}%")

        return results

def main():
    """Run strategy optimization"""
    optimizer = StrategyOptimizer()
    results = optimizer.optimize_strategy()

    # Save results to CSV for analysis
    df = pd.DataFrame(results)
    df.to_csv('strategy_optimization_results.csv', index=False)
    print("📊 Results saved to 'strategy_optimization_results.csv'")

if __name__ == "__main__":
    main()
