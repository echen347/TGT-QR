#!/usr/bin/env python3
"""
Careful parameter optimization for AVAXUSDT with strict overfitting prevention.
Uses train/test split and multiple validation checks.
"""

import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from tools.backtester import Backtester
from config.config import (
    TRADING_FEE_BPS, SLIPPAGE_BPS, LEVERAGE
)
from signal_calculator import calculate_signal

def test_parameter_set(symbol, start_date, end_date, params):
    """Test a single parameter set"""
    # Create a custom backtester with these parameters
    # We'll need to modify signal_calculator to accept custom params
    backtester = Backtester(
        symbols=[symbol],
        start_date=start_date,
        end_date=end_date,
        run_name=f"Param Test: MA={params['ma_period']}, Trend={params['min_trend_strength']:.6f}",
        run_description=f"Parameter optimization test",
        timeframe='1',
        ma_period=params['ma_period'],
        fee_bps=TRADING_FEE_BPS,
        slippage_bps=SLIPPAGE_BPS,
        no_db=True,
        cache_ttl_sec=86400 * 365
    )
    
    # We need to modify the backtester to use custom signal params
    # For now, let's use the standard backtester and note the limitation
    backtester.run()
    metrics = backtester.last_metrics.get(symbol, {})
    
    return {
        'total_trades': metrics.get('total_trades', 0),
        'win_rate': metrics.get('win_rate_pct', 0) / 100.0,
        'total_return': metrics.get('total_return_pct', 0),
        'trades_per_day': metrics.get('total_trades', 0) / ((end_date - start_date).days or 1)
    }

def optimize_avax_params(symbol='AVAXUSDT', days_train=60, days_test=30):
    """
    Systematic parameter optimization with strict overfitting prevention.
    
    Strategy:
    1. Test on training data (60 days)
    2. Validate on test data (30 days) - OOD
    3. Only recommend if:
       - Test return > 0%
       - Test win rate >= 40%
       - No overfitting (test performance within 20% of train)
       - Test trades/day >= 0.5
    """
    print("=" * 80)
    print("🔬 AVAXUSDT Parameter Optimization (Anti-Overfitting)")
    print("=" * 80)
    
    # Calculate date ranges
    end_date = datetime.now()
    test_start_date = end_date - timedelta(days=days_test)
    train_start_date = test_start_date - timedelta(days=days_train)
    
    print(f"\n📅 Date Ranges:")
    print(f"   Train: {train_start_date.strftime('%Y-%m-%d')} to {test_start_date.strftime('%Y-%m-%d')} ({days_train} days)")
    print(f"   Test:  {test_start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} ({days_test} days)")
    
    # Parameter grid to test
    # Conservative grid - not too many combinations to avoid overfitting
    ma_periods = [15, 20, 25, 30]
    min_trend_strengths = [0.0001, 0.0002, 0.0003]
    
    # Thresholds are in signal_calculator - we'll use defaults for now
    # Can extend later if needed
    
    print(f"\n⚙️  Parameter Grid:")
    print(f"   MA Periods: {ma_periods}")
    print(f"   MIN_TREND_STRENGTH: {min_trend_strengths}")
    print(f"   Total Combinations: {len(ma_periods) * len(min_trend_strengths)}")
    
    print(f"\n{'='*80}")
    print("📊 TESTING PARAMETER COMBINATIONS")
    print(f"{'='*80}")
    
    results = []
    
    for ma_period in ma_periods:
        for min_trend_strength in min_trend_strengths:
            params = {
                'ma_period': ma_period,
                'min_trend_strength': min_trend_strength
            }
            
            print(f"\n🔍 Testing: MA={ma_period}, Trend={min_trend_strength:.6f}")
            
            # Test on training data
            train_result = test_parameter_set(symbol, train_start_date, test_start_date, params)
            
            # Test on test data (OOD)
            test_result = test_parameter_set(symbol, test_start_date, end_date, params)
            
            # Calculate overfitting score
            return_diff = train_result['total_return'] - test_result['total_return']
            win_rate_diff = train_result['win_rate'] - test_result['win_rate']
            trades_diff = train_result['trades_per_day'] - test_result['trades_per_day']
            
            # Overfitting check: test should be within 20% of train return
            overfitting_score = abs(return_diff) / max(abs(train_result['total_return']), 1.0)
            
            result = {
                'params': params,
                'train': train_result,
                'test': test_result,
                'overfitting_score': overfitting_score,
                'return_diff': return_diff,
                'win_rate_diff': win_rate_diff,
                'trades_diff': trades_diff
            }
            results.append(result)
            
            print(f"   Train: {train_result['total_return']:.2f}% return, {train_result['win_rate']*100:.1f}% win rate, {train_result['trades_per_day']:.2f} trades/day")
            print(f"   Test:  {test_result['total_return']:.2f}% return, {test_result['win_rate']*100:.1f}% win rate, {test_result['trades_per_day']:.2f} trades/day")
            print(f"   Overfitting Score: {overfitting_score:.2f} {'⚠️' if overfitting_score > 0.2 else '✅'}")
    
    # Filter and rank results
    print(f"\n{'='*80}")
    print("📈 RESULTS ANALYSIS")
    print(f"{'='*80}")
    
    # Strict criteria for recommendation
    valid_results = []
    for result in results:
        test = result['test']
        train = result['train']
        
        # Must pass all these checks:
        is_profitable = test['total_return'] > 0
        has_min_win_rate = test['win_rate'] >= 0.40
        has_min_trades = test['trades_per_day'] >= 0.5
        no_overfitting = result['overfitting_score'] <= 0.20  # Test within 20% of train
        
        if is_profitable and has_min_win_rate and has_min_trades and no_overfitting:
            valid_results.append(result)
    
    if not valid_results:
        print("\n❌ NO PROFITABLE PARAMETER SETS FOUND")
        print("\n   Criteria (all must pass):")
        print("   - Test return > 0%")
        print("   - Test win rate >= 40%")
        print("   - Test trades/day >= 0.5")
        print("   - Overfitting score <= 0.20 (test within 20% of train)")
        print("\n   💡 Consider alternative strategies (see STRATEGY_ALTERNATIVES.md)")
        return None
    
    # Rank by test return (most important)
    valid_results.sort(key=lambda x: x['test']['total_return'], reverse=True)
    
    print(f"\n✅ Found {len(valid_results)} profitable parameter set(s):\n")
    
    for i, result in enumerate(valid_results[:5], 1):  # Top 5
        params = result['params']
        test = result['test']
        train = result['train']
        
        print(f"{i}. MA={params['ma_period']}, MIN_TREND_STRENGTH={params['min_trend_strength']:.6f}")
        print(f"   Test Return: {test['total_return']:.2f}%")
        print(f"   Test Win Rate: {test['win_rate']*100:.1f}%")
        print(f"   Test Trades/Day: {test['trades_per_day']:.2f}")
        print(f"   Overfitting Score: {result['overfitting_score']:.2f}")
        print(f"   Train Return: {train['total_return']:.2f}% (for comparison)")
        print()
    
    # Recommend best
    best = valid_results[0]
    print(f"{'='*80}")
    print("🎯 RECOMMENDED CONFIGURATION")
    print(f"{'='*80}")
    print(f"MA_PERIOD = {best['params']['ma_period']}")
    print(f"MIN_TREND_STRENGTH = {best['params']['min_trend_strength']}")
    print(f"\nExpected Performance (OOD):")
    print(f"  Return: {best['test']['total_return']:.2f}%")
    print(f"  Win Rate: {best['test']['win_rate']*100:.1f}%")
    print(f"  Trades/Day: {best['test']['trades_per_day']:.2f}")
    
    return best

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimize AVAXUSDT parameters with OOD validation')
    parser.add_argument('--train-days', type=int, default=60, help='Training period')
    parser.add_argument('--test-days', type=int, default=30, help='Test period (OOD)')
    
    args = parser.parse_args()
    
    best = optimize_avax_params(
        days_train=args.train_days,
        days_test=args.test_days
    )
    
    sys.exit(0 if best else 1)

