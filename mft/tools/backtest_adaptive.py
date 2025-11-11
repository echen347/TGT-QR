#!/usr/bin/env python3
"""
Backtest the adaptive parameter system (Phase 2B).
Simulates sliding window adaptation to test if adaptive parameters improve performance.
"""
import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from tools.backtester import Backtester
from config.config import SYMBOLS, TRADING_FEE_BPS, SLIPPAGE_BPS, MIN_TREND_STRENGTH
from signal_calculator import calculate_signal, THRESHOLD_HIGH_VOL, THRESHOLD_NORMAL_VOL, THRESHOLD_LOW_VOL


class AdaptiveBacktester:
    """Backtest with adaptive parameter adjustment simulation"""
    
    def __init__(self, symbols=None, days=60, adaptive_window_days=7, adaptive_update_hours=24):
        self.symbols = symbols or SYMBOLS
        self.days = days
        self.adaptive_window_days = adaptive_window_days
        self.adaptive_update_hours = adaptive_update_hours
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        self.backtester = Backtester(
            symbols=self.symbols,
            start_date=start_date,
            end_date=end_date,
            run_name=f"Adaptive Strategy Test - {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
            run_description=f"Testing adaptive parameters (window={adaptive_window_days}d, update={adaptive_update_hours}h)",
            timeframe='1',
            no_db=True,  # Don't save to DB for quick testing
            cache_ttl_sec=86400 * 365  # Use cached data
        )
    
    def calculate_adaptive_params(self, trades, current_time):
        """Calculate adaptive parameters based on recent trades (simulating PerformanceTracker)"""
        # Filter trades within adaptive window
        cutoff_time = current_time - timedelta(days=self.adaptive_window_days)
        recent_trades = [t for t in trades if t.get('exit_date', current_time) >= cutoff_time]
        
        if len(recent_trades) < 5:
            return {
                'adjust_trend_strength': 1.0,
                'adjust_thresholds': 1.0,
                'adjust_position_size': 1.0,
                'reason': 'insufficient_data'
            }
        
        # Calculate metrics
        winning_trades = [t for t in recent_trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in recent_trades if t.get('pnl', 0) < 0]
        
        win_rate = len(winning_trades) / len(recent_trades) if recent_trades else 0.0
        total_wins = sum(t.get('pnl', 0) for t in winning_trades)
        total_losses = abs(sum(t.get('pnl', 0) for t in losing_trades))
        profit_factor = total_wins / total_losses if total_losses > 0 else 0.0
        
        # Calculate trades per day
        if recent_trades:
            first_trade_time = min(t.get('exit_date', current_time) for t in recent_trades)
            days_span = (current_time - first_trade_time).days or 1
            trades_per_day = len(recent_trades) / days_span
        else:
            trades_per_day = 0
        
        recommendations = {
            'adjust_trend_strength': 1.0,
            'adjust_thresholds': 1.0,
            'adjust_position_size': 1.0,
            'reason': 'default'
        }
        
        # Underperforming: Tighten filters
        if win_rate < 0.45 or profit_factor < 1.0:
            recommendations['adjust_trend_strength'] = 1.5
            recommendations['adjust_thresholds'] = 1.2
            recommendations['adjust_position_size'] = 0.8
            recommendations['reason'] = f'underperforming (wr={win_rate:.2%}, pf={profit_factor:.2f})'
        
        # Performing well: Loosen filters
        elif win_rate > 0.55 and profit_factor > 1.5:
            recommendations['adjust_trend_strength'] = 0.9
            recommendations['adjust_thresholds'] = 0.95
            recommendations['adjust_position_size'] = 1.0
            recommendations['reason'] = f'performing_well (wr={win_rate:.2%}, pf={profit_factor:.2f})'
        
        # Low frequency: Loosen filters
        elif trades_per_day < 0.5:
            recommendations['adjust_trend_strength'] = 0.85
            recommendations['adjust_thresholds'] = 0.9
            recommendations['adjust_position_size'] = 1.0
            recommendations['reason'] = f'low_frequency ({trades_per_day:.2f} trades/day)'
        
        # High frequency but poor: Tighten filters
        elif trades_per_day > 2.0 and win_rate < 0.50:
            recommendations['adjust_trend_strength'] = 1.3
            recommendations['adjust_thresholds'] = 1.15
            recommendations['adjust_position_size'] = 0.9
            recommendations['reason'] = f'high_freq_poor ({trades_per_day:.2f}/day, {win_rate:.2%})'
        
        else:
            recommendations['reason'] = f'neutral (wr={win_rate:.2%}, {trades_per_day:.2f}/day)'
        
        return recommendations
    
    def run_adaptive_backtest(self):
        """Run backtest with adaptive parameters"""
        print("=" * 80)
        print("🧪 ADAPTIVE STRATEGY BACKTEST")
        print("=" * 80)
        print(f"Symbols: {', '.join(self.symbols)}")
        print(f"Period: {self.days} days")
        print(f"Adaptive Window: {self.adaptive_window_days} days")
        print(f"Update Interval: {self.adaptive_update_hours} hours")
        print("=" * 80)
        
        # Fetch data
        self.backtester.fetch_historical_data()
        
        all_results = {}
        adaptive_updates = []
        
        for symbol in self.symbols:
            if symbol not in self.backtester.historical_data or self.backtester.historical_data[symbol].empty:
                continue
            
            print(f"\n📊 Testing {symbol}...")
            df = self.backtester.historical_data[symbol]
            trades = []
            position = 0
            entry_price = 0
            stop_loss = 0
            take_profit = 0
            
            # Adaptive parameters (updated periodically)
            adaptive_params = {
                'adjust_trend_strength': 1.0,
                'adjust_thresholds': 1.0,
                'adjust_position_size': 1.0
            }
            last_adaptive_update = None
            
            required_lookback = max(self.backtester.ma_period, 20) + 10
            fee_pct = TRADING_FEE_BPS / 10000.0
            slip_pct = SLIPPAGE_BPS / 10000.0
            
            for i in range(required_lookback, len(df)):
                current_slice = df.iloc[:i]
                current_price = current_slice['close'].iloc[-1]
                current_time = current_slice.index[-1]
                
                # Update adaptive parameters periodically
                if (last_adaptive_update is None or 
                    (current_time - last_adaptive_update).total_seconds() >= self.adaptive_update_hours * 3600):
                    new_params = self.calculate_adaptive_params(trades, current_time)
                    if new_params['reason'] != 'insufficient_data':
                        adaptive_params = new_params
                        adaptive_updates.append({
                            'symbol': symbol,
                            'time': current_time,
                            'params': adaptive_params.copy()
                        })
                    last_adaptive_update = current_time
                
                # Calculate signal with adaptive parameters
                adaptive_min_trend = MIN_TREND_STRENGTH * adaptive_params['adjust_trend_strength']
                adaptive_threshold_high = THRESHOLD_HIGH_VOL * adaptive_params['adjust_thresholds']
                adaptive_threshold_normal = THRESHOLD_NORMAL_VOL * adaptive_params['adjust_thresholds']
                adaptive_threshold_low = THRESHOLD_LOW_VOL * adaptive_params['adjust_thresholds']
                
                signal_value, signal_name, metadata = calculate_signal(
                    current_slice,
                    min_trend_strength=adaptive_min_trend,
                    threshold_high=adaptive_threshold_high,
                    threshold_normal=adaptive_threshold_normal,
                    threshold_low=adaptive_threshold_low
                )
                
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
                        trades[-1].update({
                            'exit_date': current_time,
                            'exit_price': current_price,
                            'pnl': pnl
                        })
                        position = 0
                        entry_price = 0
                        continue
                
                # Open new position
                if position == 0 and signal_value != 0:
                    position = signal_value
                    entry_price = current_price * (1 + slip_pct) if signal_value == 1 else current_price * (1 - slip_pct)
                    
                    # Calculate stop/take profit
                    atr_prices = current_slice.iloc[-20:] if len(current_slice) >= 20 else current_slice
                    atr = self.backtester._calculate_atr(atr_prices)
                    
                    if atr > 0:
                        stop_distance = max(atr * 2.0, entry_price * 0.01)
                    else:
                        stop_distance = entry_price * 0.02
                    
                    take_profit_distance = stop_distance * 2
                    
                    if signal_value == 1:  # Long
                        stop_loss = entry_price - stop_distance
                        take_profit = entry_price + take_profit_distance
                    else:  # Short
                        stop_loss = entry_price + stop_distance
                        take_profit = entry_price - take_profit_distance
                    
                    trades.append({
                        'entry_date': current_time,
                        'entry_price': entry_price,
                        'type': 'long' if signal_value == 1 else 'short'
                    })
            
            # Filter completed trades
            completed_trades = [t for t in trades if 'exit_date' in t and 'pnl' in t]
            all_results[symbol] = pd.DataFrame(completed_trades) if completed_trades else pd.DataFrame()
        
        # Calculate and print results
        print("\n" + "=" * 80)
        print("📊 ADAPTIVE STRATEGY RESULTS")
        print("=" * 80)
        
        total_trades = 0
        total_pnl = 0
        all_trades_list = []
        
        for symbol, trades_df in all_results.items():
            if trades_df.empty:
                print(f"\n{symbol}: No trades")
                continue
            
            trades_list = trades_df.to_dict('records')
            all_trades_list.extend(trades_list)
            total_trades += len(trades_list)
            total_pnl += sum(t.get('pnl', 0) for t in trades_list)
            
            winning = [t for t in trades_list if t.get('pnl', 0) > 0]
            win_rate = len(winning) / len(trades_list) if trades_list else 0
            
            print(f"\n{symbol}:")
            print(f"  Trades: {len(trades_list)}")
            print(f"  Win Rate: {win_rate:.2%}")
            print(f"  Total Return: {sum(t.get('pnl', 0) for t in trades_list):.2%}")
        
        if all_trades_list:
            days = self.days
            trades_per_day = total_trades / days if days > 0 else 0
            overall_win_rate = len([t for t in all_trades_list if t.get('pnl', 0) > 0]) / len(all_trades_list)
            
            print("\n" + "=" * 80)
            print("📈 OVERALL RESULTS")
            print("=" * 80)
            print(f"Total Trades: {total_trades}")
            print(f"Trades/Day: {trades_per_day:.2f}")
            print(f"Win Rate: {overall_win_rate:.2%}")
            print(f"Total Return: {total_pnl:.2%}")
            print(f"Adaptive Updates: {len(adaptive_updates)}")
            print("=" * 80)
            
            # Show sample adaptive updates
            if adaptive_updates:
                print("\n📊 Sample Adaptive Parameter Updates:")
                for update in adaptive_updates[:5]:  # Show first 5
                    print(f"  {update['symbol']} @ {update['time']}: {update['params']['reason']}")
        
        return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Backtest adaptive parameter system')
    parser.add_argument('--symbols', type=str, default=','.join(SYMBOLS), help='Comma-separated symbols')
    parser.add_argument('--days', type=int, default=60, help='Days of history')
    parser.add_argument('--window', type=int, default=7, help='Adaptive window days')
    parser.add_argument('--update-hours', type=int, default=24, help='Adaptive update interval hours')
    
    args = parser.parse_args()
    symbols_list = [s.strip().upper() for s in args.symbols.split(',') if s.strip()]
    
    backtester = AdaptiveBacktester(
        symbols=symbols_list,
        days=args.days,
        adaptive_window_days=args.window,
        adaptive_update_hours=args.update_hours
    )
    
    backtester.run_adaptive_backtest()

