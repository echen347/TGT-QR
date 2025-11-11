#!/usr/bin/env python3
"""
Batch Strategy Tester - Test multiple strategies on multiple tickers
Follows scientific method: quick test, then OOD validation if promising
"""
import subprocess
import sys
import os
from datetime import datetime

# Strategies to test
STRATEGIES = [
    'volatility_breakout',
    'momentum_mr_hybrid',
    'mean_reversion',
    'vwma',
    'atr_dynamic',
    'rsi_mr',
]

# Top liquid tickers to test (from discover_tickers.py)
TICKERS = [
    'DOGEUSDT',  # High volume, established
    'GALAUSDT',  # High volume
    'COTIUSDT',  # High volume
    'JASMYUSDT', # High volume
    'AVAXUSDT',  # Our focus ticker
    'ETHUSDT',   # Major pair
    'SOLUSDT',   # Major pair
]

def run_backtest(symbol, strategy, days=60):
    """Run a single backtest"""
    cmd = [
        'python3', 'tools/backtester.py',
        '--symbols', symbol,
        '--days', str(days),
        '--strategy', strategy,
        '--no-plot',
        '--no-db'  # Skip DB writes for batch testing
    ]
    
    try:
        # Get the project root (parent of tools directory)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, '', str(e)

def parse_results(output):
    """Parse backtest results from output"""
    lines = output.split('\n')
    results = {}
    
    for line in lines:
        if 'Total Return:' in line:
            try:
                results['return'] = float(line.split(':')[1].strip().replace('%', ''))
            except:
                pass
        elif 'Win Rate:' in line:
            try:
                results['win_rate'] = float(line.split(':')[1].strip().replace('%', ''))
            except:
                pass
        elif 'Total Trades:' in line:
            try:
                results['trades'] = int(line.split(':')[1].strip())
            except:
                pass
    
    return results

def main():
    print("=" * 80)
    print("🧪 BATCH STRATEGY TESTER - Scientific Method Approach")
    print("=" * 80)
    print(f"Testing {len(STRATEGIES)} strategies on {len(TICKERS)} tickers")
    print(f"Period: 60 days (recent data)")
    print("=" * 80)
    print()
    
    all_results = []
    
    for strategy in STRATEGIES:
        print(f"\n📊 Testing Strategy: {strategy.upper()}")
        print("-" * 80)
        
        for ticker in TICKERS:
            print(f"  Testing {ticker}...", end=' ', flush=True)
            success, stdout, stderr = run_backtest(ticker, strategy)
            
            if success:
                results = parse_results(stdout)
                if results:
                    results['strategy'] = strategy
                    results['ticker'] = ticker
                    all_results.append(results)
                    
                    return_pct = results.get('return', 0)
                    win_rate = results.get('win_rate', 0)
                    trades = results.get('trades', 0)
                    
                    status = "✅" if return_pct > 0 and win_rate >= 40 else "❌"
                    print(f"{status} Return: {return_pct:.2f}%, Win Rate: {win_rate:.2f}%, Trades: {trades}")
                else:
                    print("⚠️ Could not parse results")
            else:
                print(f"❌ Failed: {stderr[:100]}")
    
    # Summary
    print("\n" + "=" * 80)
    print("📈 SUMMARY - Promising Strategies (Return > 0%, Win Rate ≥ 40%)")
    print("=" * 80)
    
    promising = [r for r in all_results if r.get('return', 0) > 0 and r.get('win_rate', 0) >= 40]
    
    if promising:
        print(f"\n✅ Found {len(promising)} promising results:\n")
        for r in sorted(promising, key=lambda x: x.get('return', 0), reverse=True):
            print(f"  {r['strategy']:20s} | {r['ticker']:12s} | Return: {r.get('return', 0):7.2f}% | Win Rate: {r.get('win_rate', 0):5.2f}% | Trades: {r.get('trades', 0):4d}")
    else:
        print("\n❌ No promising strategies found. Continue research...")
    
    print("\n" + "=" * 80)
    print("💡 Next Steps:")
    print("  1. For promising strategies: Run OOD validation (train/test split)")
    print("  2. For unprofitable strategies: Document learnings and try new ideas")
    print("  3. Consider: Different timeframes, parameter optimization, ensemble methods")
    print("=" * 80)

if __name__ == "__main__":
    main()

