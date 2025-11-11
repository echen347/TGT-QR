#!/usr/bin/env python3
"""
Performance Monitoring Tool
Analyzes live trading performance vs backtest predictions
"""

import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import sqlite3

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.database import DatabaseManager
from config.config import SYMBOLS

class PerformanceMonitor:
    """Monitor live trading performance"""
    
    def __init__(self):
        self.db = DatabaseManager()
    
    def get_recent_trades(self, days=7):
        """Get recent trades from database"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            trades = self.db.session.query(self.db.TradeRecord)\
                .filter(self.db.TradeRecord.timestamp >= cutoff_date)\
                .order_by(self.db.TradeRecord.timestamp.desc())\
                .all()
            
            trade_data = []
            for trade in trades:
                trade_data.append({
                    'timestamp': trade.timestamp,
                    'symbol': trade.symbol,
                    'side': trade.side,
                    'price': trade.price,
                    'quantity': trade.quantity,
                    'value_usdt': trade.value_usdt,
                    'pnl': trade.pnl,
                    'fees': trade.fees
                })
            
            return pd.DataFrame(trade_data)
        except Exception as e:
            print(f"Error getting trades: {e}")
            return pd.DataFrame()
    
    def calculate_metrics(self, trades_df):
        """Calculate performance metrics"""
        if trades_df.empty:
            return {
                'total_trades': 0,
                'trades_per_day': 0.0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'total_return_pct': 0.0,
                'avg_trade_pnl': 0.0
            }
        
        total_trades = len(trades_df)
        days = (trades_df['timestamp'].max() - trades_df['timestamp'].min()).days + 1
        trades_per_day = total_trades / max(days, 1)
        
        winning_trades = trades_df[trades_df['pnl'] > 0]
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
        
        total_pnl = trades_df['pnl'].sum()
        total_fees = trades_df['fees'].sum()
        net_pnl = total_pnl - total_fees
        
        # Estimate return based on position sizes
        total_value = trades_df['value_usdt'].sum()
        return_pct = (net_pnl / total_value * 100) if total_value > 0 else 0
        
        avg_trade_pnl = trades_df['pnl'].mean()
        
        return {
            'total_trades': total_trades,
            'trades_per_day': trades_per_day,
            'win_rate': win_rate,
            'total_pnl': net_pnl,
            'total_return_pct': return_pct,
            'avg_trade_pnl': avg_trade_pnl,
            'winning_trades': len(winning_trades),
            'losing_trades': total_trades - len(winning_trades)
        }
    
    def compare_to_backtest(self, live_metrics):
        """Compare live metrics to backtest predictions"""
        # Phase 1 backtest results
        backtest_metrics = {
            'trades_per_day': 7.67,
            'win_rate': 52.2,
            'return_pct': 112.93
        }
        
        comparison = {
            'trades_per_day': {
                'backtest': backtest_metrics['trades_per_day'],
                'live': live_metrics['trades_per_day'],
                'diff': live_metrics['trades_per_day'] - backtest_metrics['trades_per_day'],
                'diff_pct': ((live_metrics['trades_per_day'] - backtest_metrics['trades_per_day']) / backtest_metrics['trades_per_day'] * 100) if backtest_metrics['trades_per_day'] > 0 else 0
            },
            'win_rate': {
                'backtest': backtest_metrics['win_rate'],
                'live': live_metrics['win_rate'],
                'diff': live_metrics['win_rate'] - backtest_metrics['win_rate']
            }
        }
        
        return comparison
    
    def print_report(self, days=7):
        """Print performance report"""
        print("=" * 60)
        print("📊 LIVE TRADING PERFORMANCE REPORT")
        print("=" * 60)
        print(f"Period: Last {days} days")
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        trades_df = self.get_recent_trades(days)
        
        if trades_df.empty:
            print("⚠️  No trades found in the specified period")
            return
        
        metrics = self.calculate_metrics(trades_df)
        comparison = self.compare_to_backtest(metrics)
        
        print("\n📈 LIVE METRICS:")
        print(f"  Total Trades: {metrics['total_trades']}")
        print(f"  Trades/Day: {metrics['trades_per_day']:.2f}")
        print(f"  Win Rate: {metrics['win_rate']:.1f}%")
        print(f"  Total PnL: ${metrics['total_pnl']:.2f}")
        print(f"  Return: {metrics['total_return_pct']:.2f}%")
        print(f"  Avg Trade PnL: ${metrics['avg_trade_pnl']:.2f}")
        
        print("\n🔬 COMPARISON TO BACKTEST:")
        print(f"  Trades/Day:")
        print(f"    Backtest: {comparison['trades_per_day']['backtest']:.2f}")
        print(f"    Live: {comparison['trades_per_day']['live']:.2f}")
        print(f"    Difference: {comparison['trades_per_day']['diff']:+.2f} ({comparison['trades_per_day']['diff_pct']:+.1f}%)")
        
        print(f"  Win Rate:")
        print(f"    Backtest: {comparison['win_rate']['backtest']:.1f}%")
        print(f"    Live: {comparison['win_rate']['live']:.1f}%")
        print(f"    Difference: {comparison['win_rate']['diff']:+.1f}%")
        
        print("\n✅ SUCCESS CRITERIA:")
        trades_per_day_ok = metrics['trades_per_day'] >= 1.0
        win_rate_ok = metrics['win_rate'] >= 50.0
        print(f"  Trades/Day ≥ 1.0: {'✅' if trades_per_day_ok else '❌'} ({metrics['trades_per_day']:.2f})")
        print(f"  Win Rate ≥ 50%: {'✅' if win_rate_ok else '❌'} ({metrics['win_rate']:.1f}%)")
        
        if not trades_per_day_ok or not win_rate_ok:
            print("\n⚠️  WARNING: Performance below targets!")
            if not trades_per_day_ok:
                print("   - Consider Phase 2 improvements if this persists")
            if not win_rate_ok:
                print("   - Consider Phase 1 + MA15 (53.3% win rate)")
        
        print("\n" + "=" * 60)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Monitor live trading performance')
    parser.add_argument('--days', type=int, default=7, help='Number of days to analyze')
    args = parser.parse_args()
    
    monitor = PerformanceMonitor()
    monitor.print_report(args.days)

if __name__ == '__main__':
    main()

