#!/usr/bin/env python3
"""
OOD Validation for ML Strategies
Tests ML strategies with train/test split to check for overfitting

IMPORTANT: This script clears the ML model cache between train and test runs
to ensure proper out-of-distribution validation. Without cache clearing,
train and test might use the same model leading to identical results.
"""

import sys
import os
from datetime import datetime, timedelta

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from tools.backtester import Backtester

# Import cache clearing function
try:
    from tools.ml_strategy import clear_ml_cache
    HAS_ML_CACHE = True
except ImportError:
    HAS_ML_CACHE = False
    print("⚠️ Could not import clear_ml_cache - ML cache will not be cleared")

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
    
    # Clear ML cache before train run
    if HAS_ML_CACHE:
        clear_ml_cache()
        print("🧹 ML model cache cleared before TRAIN run")

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
    
    # Clear ML cache before test run (CRITICAL - ensures fresh model for OOD)
    if HAS_ML_CACHE:
        clear_ml_cache()
        print("🧹 ML model cache cleared before TEST run (ensures true OOD)")

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

    train_return = train_metrics.get('total_return_pct', 0)
    train_wr = train_metrics.get('win_rate_pct', 0)
    train_trades = train_metrics.get('total_trades', 0)

    test_return = test_metrics.get('total_return_pct', 0)
    test_wr = test_metrics.get('win_rate_pct', 0)
    test_trades = test_metrics.get('total_trades', 0)

    print(f"\nTrain: Return={train_return:.2f}%, Win Rate={train_wr:.2f}%, Trades={train_trades}")
    print(f"Test:  Return={test_return:.2f}%, Win Rate={test_wr:.2f}%, Trades={test_trades}")

    # Check for suspicious identical results (indicates bug)
    if train_return == test_return and train_wr == test_wr and train_trades == test_trades:
        print(f"\n⚠️ WARNING: Train and test results are IDENTICAL!")
        print(f"   This suggests a bug in data filtering or model caching.")
        print(f"   Results should NOT be trusted until this is investigated.")

    # Check for overfitting (test << train)
    if train_return > 0 and test_return > 0:
        perf_ratio = test_return / train_return if train_return != 0 else 0
        if perf_ratio < 0.8:
            print(f"\n⚠️ OVERFITTING WARNING: Test performance is {perf_ratio:.1%} of train")
            print(f"   (Test < 80% of train suggests strategy may be overfit)")

    # Check for statistical significance
    if test_trades < 20:
        print(f"\n⚠️ WARNING: Only {test_trades} test trades (need ≥20 for statistical significance)")

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


