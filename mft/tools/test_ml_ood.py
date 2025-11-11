#!/usr/bin/env python3
"""
OOD Validation for ML Strategies
Tests ML strategies with train/test split to check for overfitting
"""

import sys
import os
from datetime import datetime, timedelta

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from tools.backtester import Backtester

def test_ml_ood(symbol='ETHUSDT', strategy='ml_rf', train_days=36, test_days=24):
    """Test ML strategy with train/test split"""
    print("=" * 70)
    print(f"🔬 ML Strategy OOD Validation: {strategy.upper()} on {symbol}")
    print("=" * 70)
    
    end_date = datetime.now()
    test_start = end_date - timedelta(days=test_days)
    train_start = test_start - timedelta(days=train_days)
    
    print(f"\n📅 Train: {train_start.strftime('%Y-%m-%d')} to {test_start.strftime('%Y-%m-%d')} ({train_days} days)")
    print(f"📅 Test:  {test_start.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} ({test_days} days)")
    
    # Train period
    print(f"\n{'='*70}")
    print("📊 TRAINING PERIOD")
    print(f"{'='*70}")
    train_bt = Backtester(
        symbols=[symbol],
        start_date=train_start,
        end_date=test_start,
        run_name=f"{symbol} {strategy} Train",
        strategy=strategy,
        no_db=True
    )
    train_bt.run()
    train_metrics = train_bt.last_metrics.get(symbol, {})
    
    # Test period
    print(f"\n{'='*70}")
    print("📊 TEST PERIOD (OOD)")
    print(f"{'='*70}")
    test_bt = Backtester(
        symbols=[symbol],
        start_date=test_start,
        end_date=end_date,
        run_name=f"{symbol} {strategy} Test",
        strategy=strategy,
        no_db=True
    )
    test_bt.run()
    test_metrics = test_bt.last_metrics.get(symbol, {})
    
    # Summary
    print(f"\n{'='*70}")
    print("📈 RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"\nTrain: Return={train_metrics.get('total_return_pct', 0):.2f}%, Win Rate={train_metrics.get('win_rate_pct', 0):.2f}%, Trades={train_metrics.get('total_trades', 0)}")
    print(f"Test:  Return={test_metrics.get('total_return_pct', 0):.2f}%, Win Rate={test_metrics.get('win_rate_pct', 0):.2f}%, Trades={test_metrics.get('total_trades', 0)}")
    
    test_return = test_metrics.get('total_return_pct', 0)
    test_wr = test_metrics.get('win_rate_pct', 0)
    
    if test_return > 0 and test_wr >= 40:
        print(f"\n✅ PROFITABLE ON TEST SET! Return: {test_return:.2f}%, Win Rate: {test_wr:.2f}%")
        return True
    else:
        print(f"\n❌ Not profitable on test set")
        return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', default='ETHUSDT')
    parser.add_argument('--strategy', default='ml_rf', choices=['ml_rf', 'ml_lr'])
    parser.add_argument('--train-days', type=int, default=36)
    parser.add_argument('--test-days', type=int, default=24)
    args = parser.parse_args()
    
    test_ml_ood(args.symbol, args.strategy, args.train_days, args.test_days)

