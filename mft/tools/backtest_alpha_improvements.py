#!/usr/bin/env python3
"""
Backtest Alpha Improvements with Out-of-Sample Testing
Tests Phase 1 improvements and parameter combinations to find optimal settings
for achieving 1+ trades per day while avoiding overfitting.
"""

import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from tools.backtester import Backtester
from config.config import SYMBOLS, TIMEFRAME, LEVERAGE

class AlphaBacktester:
    """
    Systematic backtesting with train/test splits to avoid overfitting.
    Tests different parameter combinations to find optimal settings.
    """
    
    def __init__(self):
        self.results = []
        
    def run_parameter_sweep(self, symbols=None, days_train=60, days_test=30):
        """
        Test multiple parameter combinations with proper train/test split.
        
        Args:
            symbols: List of symbols to test (default: ETHUSDT, SOLUSDT)
            days_train: Days for training/optimization period
            days_test: Days for out-of-sample testing period
        """
        symbols = symbols or SYMBOLS
        
        # Define parameter combinations to test
        parameter_sets = [
            # Baseline (old parameters)
            {
                'name': 'Baseline (Old)',
                'min_trend_strength': 0.0005,
                'threshold_high': 0.003,
                'threshold_normal': 0.001,
                'threshold_low': 0.0005,
                'use_ma_slope': True,
                'ma_period': 20
            },
            # Phase 1: Remove MA slope + reduce thresholds
            {
                'name': 'Phase 1: No Slope + Reduced Thresholds',
                'min_trend_strength': 0.0002,
                'threshold_high': 0.002,
                'threshold_normal': 0.0005,
                'threshold_low': 0.0003,
                'use_ma_slope': False,
                'ma_period': 20
            },
            # Phase 1 + Shorter MA
            {
                'name': 'Phase 1 + MA15',
                'min_trend_strength': 0.0002,
                'threshold_high': 0.002,
                'threshold_normal': 0.0005,
                'threshold_low': 0.0003,
                'use_ma_slope': False,
                'ma_period': 15
            },
            # Phase 1 + Even Lower Trend Strength
            {
                'name': 'Phase 1 + Lower Trend',
                'min_trend_strength': 0.0001,
                'threshold_high': 0.002,
                'threshold_normal': 0.0005,
                'threshold_low': 0.0003,
                'use_ma_slope': False,
                'ma_period': 20
            },
            # Phase 1 + More Aggressive Thresholds
            {
                'name': 'Phase 1 + Aggressive',
                'min_trend_strength': 0.0002,
                'threshold_high': 0.0015,
                'threshold_normal': 0.0003,
                'threshold_low': 0.0002,
                'use_ma_slope': False,
                'ma_period': 20
            },
        ]
        
        print("=" * 80)
        print("🧪 ALPHA IMPROVEMENTS BACKTEST")
        print("=" * 80)
        print(f"Symbols: {', '.join(symbols)}")
        print(f"Training Period: {days_train} days")
        print(f"Test Period: {days_test} days")
        print(f"Total Parameter Sets: {len(parameter_sets)}")
        print("=" * 80)
        
        # Calculate date ranges
        test_end = datetime.now()
        test_start = test_end - timedelta(days=days_test)
        train_end = test_start - timedelta(days=1)
        train_start = train_end - timedelta(days=days_train)
        
        print(f"\n📅 Date Ranges:")
        print(f"  Training: {train_start.strftime('%Y-%m-%d')} to {train_end.strftime('%Y-%m-%d')}")
        print(f"  Test (OOS): {test_start.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')}")
        print()
        
        all_results = []
        
        for i, params in enumerate(parameter_sets, 1):
            print(f"\n{'='*80}")
            print(f"TEST {i}/{len(parameter_sets)}: {params['name']}")
            print(f"{'='*80}")
            
            # Test on TRAINING data first (for optimization)
            train_result = self._run_single_test(
                symbols=symbols,
                start_date=train_start,
                end_date=train_end,
                params=params,
                set_name=f"{params['name']} (Train)"
            )
            
            # Test on OUT-OF-SAMPLE data (true performance)
            test_result = self._run_single_test(
                symbols=symbols,
                start_date=test_start,
                end_date=test_end,
                params=params,
                set_name=f"{params['name']} (OOS)"
            )
            
            # Combine results
            combined = {
                'name': params['name'],
                'train': train_result,
                'test': test_result,
                'params': params
            }
            all_results.append(combined)
            
            # Print summary
            self._print_summary(combined)
        
        # Find best performing (by trades/day on OOS data)
        print("\n" + "=" * 80)
        print("📊 FINAL RANKINGS (by Trades/Day on Out-of-Sample Data)")
        print("=" * 80)
        
        sorted_results = sorted(
            all_results,
            key=lambda x: x['test'].get('trades_per_day', 0),
            reverse=True
        )
        
        for i, result in enumerate(sorted_results, 1):
            test = result['test']
            train = result['train']
            print(f"\n{i}. {result['name']}")
            print(f"   OOS Trades/Day: {test.get('trades_per_day', 0):.2f}")
            print(f"   OOS Total Trades: {test.get('total_trades', 0)}")
            print(f"   OOS Win Rate: {test.get('win_rate', 0):.1f}%")
            print(f"   OOS Return: {test.get('total_return', 0):.2f}%")
            print(f"   Train Trades/Day: {train.get('trades_per_day', 0):.2f}")
            print(f"   Overfitting Check: {train.get('trades_per_day', 0) - test.get('trades_per_day', 0):.2f} trades/day diff")
        
        # Recommend best parameter set
        best = sorted_results[0]
        print("\n" + "=" * 80)
        print("✅ RECOMMENDED CONFIGURATION")
        print("=" * 80)
        print(f"Parameter Set: {best['name']}")
        print(f"OOS Trades/Day: {best['test'].get('trades_per_day', 0):.2f}")
        print(f"\nParameters:")
        for key, value in best['params'].items():
            if key != 'name':
                print(f"  {key}: {value}")
        
        return sorted_results
    
    def _run_single_test(self, symbols, start_date, end_date, params, set_name):
        """Run a single backtest with given parameters"""
        
        # Create a custom backtester with modified signal logic
        # Use very long cache TTL to reuse cached data and avoid rate limits
        backtester = Backtester(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            run_name=f"{set_name} - {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
            run_description=f"Alpha improvement test: {set_name}",
            timeframe=TIMEFRAME,
            ma_period=params['ma_period'],
            no_db=True,  # Don't save to DB for parameter sweep
            cache_ttl_sec=86400 * 365  # 1 year cache TTL to reuse existing data
        )
        
        # Patch the signal generation method with our parameters
        original_get_signal = backtester._get_signal
        
        def patched_get_signal(historical_prices):
            """Modified signal generation with custom parameters"""
            closes = historical_prices['close']
            if len(historical_prices) < max(params['ma_period'], 20) + 10:
                return 0
            
            current_price = closes.iloc[-1]
            ma = closes.rolling(window=params['ma_period']).mean().iloc[-1]
            
            # Calculate trend strength
            ma_series = closes.rolling(window=params['ma_period']).mean()
            ma_slope = (ma_series.iloc[-1] - ma_series.iloc[-5]) / max(params['ma_period'], 1)
            trend_strength = abs(ma_slope) / max(current_price, 1e-9)
            
            # Check trend strength filter
            if trend_strength < params['min_trend_strength']:
                return 0
            
            # Calculate volatility
            recent_volatility = closes.pct_change().rolling(window=10).std().iloc[-1]
            
            # Select threshold based on volatility
            if recent_volatility > 0.025:  # High volatility
                threshold = params['threshold_high']
            elif recent_volatility > 0.015:  # Normal volatility
                threshold = params['threshold_normal']
            else:  # Low volatility
                threshold = params['threshold_low']
            
            # Generate signal (with or without MA slope requirement)
            if params['use_ma_slope']:
                if current_price > ma * (1 + threshold) and ma_slope > 0:
                    return 1
                if current_price < ma * (1 - threshold) and ma_slope < 0:
                    return -1
            else:
                if current_price > ma * (1 + threshold):
                    return 1
                if current_price < ma * (1 - threshold):
                    return -1
            
            return 0
        
        # Replace the signal method
        backtester._get_signal = patched_get_signal
        
        # Run the backtest
        try:
            backtester.fetch_historical_data()
            
            # Run simulation manually to get metrics
            all_results = {}
            for symbol in symbols:
                if symbol not in backtester.historical_data or backtester.historical_data[symbol].empty:
                    continue
                
                df = backtester.historical_data[symbol]
                trades = []
                position = 0
                entry_price = 0
                stop_loss = 0
                take_profit = 0
                
                required_lookback = max(params['ma_period'], 20) + 10
                for i in range(required_lookback, len(df)):
                    current_slice = df.iloc[:i]
                    signal = patched_get_signal(current_slice)
                    current_price = current_slice['close'].iloc[-1]
                    fee_pct = 5.0 / 10000.0
                    slip_pct = 2.0 / 10000.0
                    
                    # Check exit conditions
                    if position != 0:
                        should_exit = False
                        if position == 1:  # Long
                            if current_price <= stop_loss or current_price >= take_profit:
                                should_exit = True
                                exit_exec = current_price * (1 - slip_pct)
                                pnl = (exit_exec - entry_price) / entry_price - (2 * fee_pct)
                        else:  # Short
                            if current_price >= stop_loss or current_price <= take_profit:
                                should_exit = True
                                exit_exec = current_price * (1 + slip_pct)
                                pnl = (entry_price - exit_exec) / entry_price - (2 * fee_pct)
                        
                        if should_exit:
                            trades[-1].update({'exit_date': df.index[i], 'exit_price': current_price, 'pnl': pnl})
                            position = 0
                            continue
                    
                    # Open new position
                    if position == 0 and signal != 0:
                        position = signal
                        entry_price = current_price * (1 + slip_pct) if signal == 1 else current_price * (1 - slip_pct)
                        
                        # Calculate stop/take profit
                        atr_prices = current_slice.iloc[-20:] if len(current_slice) >= 20 else current_slice
                        atr = backtester._calculate_atr(atr_prices)
                        
                        if atr > 0:
                            stop_distance = max(atr * 2.0, entry_price * 0.01)
                        else:
                            stop_distance = entry_price * 0.02
                        
                        take_profit_distance = stop_distance * 2
                        
                        if signal == 1:  # Long
                            stop_loss = entry_price - stop_distance
                            take_profit = entry_price + take_profit_distance
                        else:  # Short
                            stop_loss = entry_price + stop_distance
                            take_profit = entry_price - take_profit_distance
                        
                        trades.append({
                            'entry_date': df.index[i],
                            'entry_price': entry_price,
                            'type': 'long' if signal == 1 else 'short'
                        })
                
                # Filter completed trades
                completed_trades = [t for t in trades if 'exit_date' in t and 'pnl' in t]
                all_results[symbol] = pd.DataFrame(completed_trades) if completed_trades else pd.DataFrame()
            
            # Calculate metrics
            metrics = self._calculate_metrics(all_results, start_date, end_date)
            return metrics
            
        except Exception as e:
            print(f"❌ Error in backtest: {e}")
            import traceback
            traceback.print_exc()
            return {
                'total_trades': 0,
                'trades_per_day': 0,
                'win_rate': 0,
                'total_return': 0
            }
    
    def _calculate_metrics(self, all_results, start_date, end_date):
        """Calculate key metrics from backtest results"""
        all_trades = []
        for symbol, trades_df in all_results.items():
            if not trades_df.empty and 'pnl' in trades_df.columns:
                all_trades.extend(trades_df.to_dict('records'))
        
        if not all_trades:
            return {
                'total_trades': 0,
                'trades_per_day': 0,
                'win_rate': 0,
                'total_return': 0,
                'avg_return': 0,
                'sharpe_ratio': 0
            }
        
        trades_df = pd.DataFrame(all_trades)
        total_trades = len(trades_df)
        
        # Calculate days
        days = (end_date - start_date).days
        trades_per_day = total_trades / days if days > 0 else 0
        
        # Win rate
        wins = trades_df[trades_df['pnl'] > 0]
        win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0
        
        # Returns (with leverage)
        pnl_with_leverage = (trades_df['pnl'] * LEVERAGE) + 1
        cumulative_return = pnl_with_leverage.cumprod()
        total_return = (cumulative_return.iloc[-1] - 1) * 100 if len(cumulative_return) > 0 else 0
        avg_return = trades_df['pnl'].mean() * LEVERAGE * 100
        
        # Sharpe ratio (simplified)
        daily_returns = trades_df.set_index('exit_date')['pnl'].resample('D').sum() * LEVERAGE if 'exit_date' in trades_df.columns else pd.Series()
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(365)
        else:
            sharpe_ratio = 0
        
        return {
            'total_trades': total_trades,
            'trades_per_day': trades_per_day,
            'win_rate': win_rate,
            'total_return': total_return,
            'avg_return': avg_return,
            'sharpe_ratio': sharpe_ratio
        }
    
    def _print_summary(self, result):
        """Print summary for a single test result"""
        test = result['test']
        train = result['train']
        
        print(f"\n📊 Results for: {result['name']}")
        print(f"  Training Period:")
        print(f"    Trades/Day: {train.get('trades_per_day', 0):.2f}")
        print(f"    Total Trades: {train.get('total_trades', 0)}")
        print(f"    Win Rate: {train.get('win_rate', 0):.1f}%")
        print(f"    Return: {train.get('total_return', 0):.2f}%")
        print(f"  Out-of-Sample Period:")
        print(f"    Trades/Day: {test.get('trades_per_day', 0):.2f} ⭐")
        print(f"    Total Trades: {test.get('total_trades', 0)}")
        print(f"    Win Rate: {test.get('win_rate', 0):.1f}%")
        print(f"    Return: {test.get('total_return', 0):.2f}%")
        print(f"  Overfitting Check: {train.get('trades_per_day', 0) - test.get('trades_per_day', 0):.2f} trades/day difference")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Backtest alpha improvements with OOS testing")
    parser.add_argument('--symbols', type=str, default=','.join(SYMBOLS), help='Comma-separated symbols')
    parser.add_argument('--train-days', type=int, default=60, help='Training period days')
    parser.add_argument('--test-days', type=int, default=30, help='Out-of-sample test period days')
    
    args = parser.parse_args()
    
    symbols_list = [s.strip().upper() for s in args.symbols.split(',') if s.strip()]
    
    tester = AlphaBacktester()
    results = tester.run_parameter_sweep(
        symbols=symbols_list,
        days_train=args.train_days,
        days_test=args.test_days
    )
    
    print("\n✅ Backtest complete!")
    print(f"Best configuration achieves {results[0]['test'].get('trades_per_day', 0):.2f} trades/day on OOS data")

