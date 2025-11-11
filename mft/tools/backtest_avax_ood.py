#!/usr/bin/env python3
"""
Backtest AVAXUSDT strategy with train/test split to check for overfitting.
This script verifies profitability and OOD (out-of-distribution) performance.
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
from config.config import (
    SYMBOLS, MA_PERIOD, LEVERAGE, TRADING_FEE_BPS, SLIPPAGE_BPS,
    MIN_TREND_STRENGTH, VOLATILITY_THRESHOLD_HIGH, VOLATILITY_THRESHOLD_LOW
)
from signal_calculator import THRESHOLD_HIGH_VOL, THRESHOLD_NORMAL_VOL, THRESHOLD_LOW_VOL

def run_ood_backtest(symbol='AVAXUSDT', days_train=60, days_test=30):
    """
    Run backtest with train/test split to check for overfitting.
    
    Args:
        symbol: Symbol to test (default: AVAXUSDT)
        days_train: Training period (default: 60 days)
        days_test: Test period (default: 30 days)
    """
    print("=" * 70)
    print("🔬 AVAXUSDT OOD Backtest - Train/Test Split")
    print("=" * 70)
    
    # Calculate date ranges
    end_date = datetime.now()
    test_start_date = end_date - timedelta(days=days_test)
    train_start_date = test_start_date - timedelta(days=days_train)
    
    print(f"\n📅 Date Ranges:")
    print(f"   Train: {train_start_date.strftime('%Y-%m-%d')} to {test_start_date.strftime('%Y-%m-%d')} ({days_train} days)")
    print(f"   Test:  {test_start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} ({days_test} days)")
    
    print(f"\n⚙️  Strategy Parameters:")
    print(f"   MA Period: {MA_PERIOD}")
    print(f"   MIN_TREND_STRENGTH: {MIN_TREND_STRENGTH}")
    print(f"   Thresholds: High={THRESHOLD_HIGH_VOL*100:.3f}%, Normal={THRESHOLD_NORMAL_VOL*100:.3f}%, Low={THRESHOLD_LOW_VOL*100:.3f}%")
    print(f"   Leverage: {LEVERAGE}x")
    print(f"   Fees: {TRADING_FEE_BPS} bps (round-trip)")
    print(f"   Slippage: {SLIPPAGE_BPS} bps")
    
    # Run train backtest
    print(f"\n{'='*70}")
    print("📊 TRAINING PERIOD BACKTEST")
    print(f"{'='*70}")
    
    train_backtester = Backtester(
        symbols=[symbol],
        start_date=train_start_date,
        end_date=test_start_date,
        run_name=f"AVAXUSDT Train - {train_start_date.strftime('%Y-%m-%d')} to {test_start_date.strftime('%Y-%m-%d')}",
        run_description=f"Training period backtest for OOD validation",
        timeframe='1',
        ma_period=MA_PERIOD,
        fee_bps=TRADING_FEE_BPS,
        slippage_bps=SLIPPAGE_BPS,
        no_db=True,  # Don't save to DB
        cache_ttl_sec=86400 * 365  # Use cached data
    )
    
    train_backtester.run()
    train_metrics = train_backtester.last_metrics.get(symbol, {})
    
    # Run test backtest
    print(f"\n{'='*70}")
    print("📊 TEST PERIOD BACKTEST (OOD)")
    print(f"{'='*70}")
    
    test_backtester = Backtester(
        symbols=[symbol],
        start_date=test_start_date,
        end_date=end_date,
        run_name=f"AVAXUSDT Test - {test_start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
        run_description=f"Out-of-sample test period backtest",
        timeframe='1',
        ma_period=MA_PERIOD,
        fee_bps=TRADING_FEE_BPS,
        slippage_bps=SLIPPAGE_BPS,
        no_db=True,  # Don't save to DB
        cache_ttl_sec=86400 * 365  # Use cached data
    )
    
    test_backtester.run()
    test_metrics = test_backtester.last_metrics.get(symbol, {})
    
    # Analyze results
    print(f"\n{'='*70}")
    print("📈 RESULTS SUMMARY")
    print(f"{'='*70}")
    
    train_total_trades = train_metrics.get('total_trades', 0)
    train_win_rate_pct = train_metrics.get('win_rate_pct', 0)
    train_win_rate = train_win_rate_pct / 100.0  # Convert from percentage to decimal
    train_total_return = train_metrics.get('total_return_pct', 0)
    train_trades_per_day = train_total_trades / days_train if days_train > 0 else 0
    
    test_total_trades = test_metrics.get('total_trades', 0)
    test_win_rate_pct = test_metrics.get('win_rate_pct', 0)
    test_win_rate = test_win_rate_pct / 100.0  # Convert from percentage to decimal
    test_total_return = test_metrics.get('total_return_pct', 0)
    test_trades_per_day = test_total_trades / days_test if days_test > 0 else 0
    
    print(f"\n📊 Training Period ({days_train} days):")
    print(f"   Total Trades: {train_total_trades}")
    print(f"   Trades/Day: {train_trades_per_day:.2f}")
    print(f"   Win Rate: {train_win_rate*100:.2f}%")
    print(f"   Total Return: {train_total_return:.2f}%")
    
    print(f"\n📊 Test Period ({days_test} days) - OOD:")
    print(f"   Total Trades: {test_total_trades}")
    print(f"   Trades/Day: {test_trades_per_day:.2f}")
    print(f"   Win Rate: {test_win_rate*100:.2f}%")
    print(f"   Total Return: {test_total_return:.2f}%")
    
    # Overfitting check
    print(f"\n{'='*70}")
    print("🔍 OVERFITTING ANALYSIS")
    print(f"{'='*70}")
    
    trades_per_day_diff = train_trades_per_day - test_trades_per_day
    win_rate_diff = train_win_rate - test_win_rate
    return_diff = train_total_return - test_total_return
    
    print(f"   Trades/Day Difference: {trades_per_day_diff:+.2f} (train - test)")
    print(f"   Win Rate Difference: {win_rate_diff*100:+.2f}% (train - test)")
    print(f"   Return Difference: {return_diff:+.2f}% (train - test)")
    
    # Overfitting indicators
    overfitting_score = 0
    overfitting_issues = []
    
    if trades_per_day_diff > 0.5:
        overfitting_score += 1
        overfitting_issues.append("⚠️  Train trades/day >> Test trades/day (possible overfitting)")
    
    if win_rate_diff > 0.15:
        overfitting_score += 1
        overfitting_issues.append("⚠️  Train win rate >> Test win rate (possible overfitting)")
    
    if return_diff > 20:
        overfitting_score += 1
        overfitting_issues.append("⚠️  Train return >> Test return (possible overfitting)")
    
    if overfitting_score == 0:
        print("\n✅ No overfitting detected - test performance is consistent with train")
    else:
        print(f"\n⚠️  {overfitting_score} overfitting indicator(s) detected:")
        for issue in overfitting_issues:
            print(f"   {issue}")
    
    # Profitability check
    print(f"\n{'='*70}")
    print("💰 PROFITABILITY CHECK")
    print(f"{'='*70}")
    
    is_profitable_train = train_total_return > 0 and train_win_rate >= 0.40
    is_profitable_test = test_total_return > 0 and test_win_rate >= 0.40
    
    print(f"   Train Period: {'✅ Profitable' if is_profitable_train else '❌ Not Profitable'}")
    print(f"   Test Period:  {'✅ Profitable' if is_profitable_test else '❌ Not Profitable'}")
    
    # Final recommendation
    print(f"\n{'='*70}")
    print("🎯 DEPLOYMENT RECOMMENDATION")
    print(f"{'='*70}")
    
    if is_profitable_test and overfitting_score == 0:
        print("\n✅ ✅ ✅ RECOMMENDED FOR PRODUCTION ✅ ✅ ✅")
        print("\n   Reasons:")
        print(f"   - Test period is profitable ({test_total_return:.2f}% return, {test_win_rate*100:.2f}% win rate)")
        print(f"   - No overfitting detected (test performance matches train)")
        print(f"   - Trades/Day: {test_trades_per_day:.2f} (target: ≥1.0)")
        return True
    elif is_profitable_test and overfitting_score <= 1:
        print("\n⚠️  CAUTIOUSLY RECOMMENDED")
        print("\n   Reasons:")
        print(f"   - Test period is profitable ({test_total_return:.2f}% return)")
        print(f"   - Minor overfitting concerns (monitor closely)")
        return True
    else:
        print("\n❌ NOT RECOMMENDED FOR PRODUCTION")
        print("\n   Reasons:")
        if not is_profitable_test:
            print(f"   - Test period not profitable ({test_total_return:.2f}% return, {test_win_rate*100:.2f}% win rate)")
        if overfitting_score > 1:
            print(f"   - Significant overfitting detected ({overfitting_score} indicators)")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Backtest AVAXUSDT with OOD validation')
    parser.add_argument('--symbol', type=str, default='AVAXUSDT', help='Symbol to test')
    parser.add_argument('--train-days', type=int, default=60, help='Training period in days')
    parser.add_argument('--test-days', type=int, default=30, help='Test period in days')
    
    args = parser.parse_args()
    
    recommended = run_ood_backtest(
        symbol=args.symbol,
        days_train=args.train_days,
        days_test=args.test_days
    )
    
    sys.exit(0 if recommended else 1)

