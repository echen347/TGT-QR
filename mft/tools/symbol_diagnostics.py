#!/usr/bin/env python3
"""
Symbol Diagnostics Tool
Analyzes and compares data characteristics across different trading symbols.

Purpose: Understand why strategies work on ETHUSDT but fail on other symbols.

Hypotheses to test:
1. Data Quality: Is ETHUSDT data denser or higher quality?
2. Liquidity: Does higher liquidity make patterns more predictable?
3. Volatility: Does ETHUSDT have different volatility characteristics?
4. Time Period: Was there something special about the test period?
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Try to import visualization libraries
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def load_symbol_cache(symbol, data_dir=None):
    """Load cached data for a symbol"""
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')

    cache_path = os.path.join(data_dir, f"{symbol}_cache.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    return None


def analyze_data_quality(df, symbol):
    """Analyze data quality for a symbol"""
    print(f"\n{'='*60}")
    print(f"DATA QUALITY ANALYSIS: {symbol}")
    print(f"{'='*60}")

    # Basic stats
    print(f"\nBasic Statistics:")
    print(f"  Total Rows: {len(df):,}")
    print(f"  Date Range: {df.index.min()} to {df.index.max()}")
    print(f"  Days Covered: {(df.index.max() - df.index.min()).days}")

    # Calculate data density (rows per day)
    days = max((df.index.max() - df.index.min()).days, 1)
    density = len(df) / days
    print(f"  Data Density: {density:.1f} rows/day")

    # Check for gaps
    time_diffs = df.index.to_series().diff()
    expected_interval = time_diffs.mode().iloc[0] if len(time_diffs.mode()) > 0 else pd.Timedelta(minutes=1)
    gaps = time_diffs[time_diffs > expected_interval * 2]
    print(f"  Expected Interval: {expected_interval}")
    print(f"  Data Gaps (>2x interval): {len(gaps)}")

    if len(gaps) > 0:
        print(f"  Largest Gap: {gaps.max()}")

    # Missing values
    missing = df.isnull().sum().sum()
    print(f"  Missing Values: {missing}")

    return {
        'rows': len(df),
        'days': days,
        'density': density,
        'gaps': len(gaps),
        'missing': missing
    }


def analyze_price_characteristics(df, symbol):
    """Analyze price and volatility characteristics"""
    print(f"\n{'='*60}")
    print(f"PRICE CHARACTERISTICS: {symbol}")
    print(f"{'='*60}")

    # Calculate returns
    df = df.copy()
    df['returns'] = df['close'].pct_change()

    print(f"\nPrice Statistics:")
    print(f"  Mean Price: ${df['close'].mean():.2f}")
    print(f"  Price Range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    print(f"  Price Std Dev: ${df['close'].std():.2f}")

    print(f"\nVolatility Statistics:")
    print(f"  Mean Daily Return: {df['returns'].mean() * 100:.4f}%")
    print(f"  Return Std Dev: {df['returns'].std() * 100:.4f}%")
    print(f"  Annualized Vol: {df['returns'].std() * np.sqrt(365*24*60) * 100:.2f}%")

    # Skewness and kurtosis
    skew = df['returns'].skew()
    kurt = df['returns'].kurtosis()
    print(f"  Skewness: {skew:.3f} ({'positive' if skew > 0 else 'negative'})")
    print(f"  Kurtosis: {kurt:.3f} ({'fat tails' if kurt > 0 else 'thin tails'})")

    # Volatility clustering (GARCH-like behavior)
    returns_sq = df['returns'] ** 2
    autocorr = returns_sq.autocorr(lag=1)
    print(f"  Volatility Clustering (lag-1 autocorr of returns^2): {autocorr:.3f}")

    return {
        'mean_price': df['close'].mean(),
        'volatility': df['returns'].std(),
        'skewness': skew,
        'kurtosis': kurt,
        'vol_clustering': autocorr
    }


def analyze_volume(df, symbol):
    """Analyze volume/liquidity characteristics"""
    print(f"\n{'='*60}")
    print(f"VOLUME/LIQUIDITY: {symbol}")
    print(f"{'='*60}")

    print(f"\nVolume Statistics:")
    print(f"  Mean Volume: {df['volume'].mean():,.0f}")
    print(f"  Median Volume: {df['volume'].median():,.0f}")
    print(f"  Volume Std Dev: {df['volume'].std():,.0f}")
    print(f"  Volume CV (Std/Mean): {df['volume'].std() / df['volume'].mean():.2f}")

    # Volume spikes
    vol_mean = df['volume'].mean()
    vol_std = df['volume'].std()
    spikes = len(df[df['volume'] > vol_mean + 2 * vol_std])
    print(f"  Volume Spikes (>2 std): {spikes} ({spikes/len(df)*100:.2f}%)")

    # Dollar volume (proxy for liquidity)
    df = df.copy()
    df['dollar_volume'] = df['close'] * df['volume']
    print(f"  Mean Dollar Volume: ${df['dollar_volume'].mean():,.0f}")

    return {
        'mean_volume': df['volume'].mean(),
        'volume_cv': df['volume'].std() / df['volume'].mean(),
        'dollar_volume': df['dollar_volume'].mean()
    }


def analyze_patterns(df, symbol):
    """Analyze predictability patterns"""
    print(f"\n{'='*60}")
    print(f"PATTERN ANALYSIS: {symbol}")
    print(f"{'='*60}")

    df = df.copy()
    df['returns'] = df['close'].pct_change()

    # Serial correlation (momentum/mean reversion signal)
    autocorrs = []
    for lag in [1, 5, 10, 20, 60]:
        if len(df) > lag:
            ac = df['returns'].autocorr(lag=lag)
            autocorrs.append((lag, ac))
            print(f"  Return Autocorrelation (lag {lag}): {ac:.4f}")

    # Trend strength
    df['ma20'] = df['close'].rolling(20).mean()
    df['above_ma'] = (df['close'] > df['ma20']).astype(int)
    trend_persistence = df['above_ma'].diff().abs().mean()
    print(f"  Trend Persistence (lower = stronger trends): {trend_persistence:.3f}")

    # Directional accuracy potential (how often does direction persist)
    df['direction'] = np.sign(df['returns'])
    direction_persist = (df['direction'] == df['direction'].shift(1)).mean()
    print(f"  Direction Persistence: {direction_persist:.2%}")

    return {
        'autocorr_1': df['returns'].autocorr(lag=1) if len(df) > 1 else 0,
        'trend_persistence': trend_persistence,
        'direction_persist': direction_persist
    }


def compare_symbols(symbols, data_dir=None):
    """Compare multiple symbols"""
    print("\n" + "="*80)
    print("SYMBOL COMPARISON REPORT")
    print("="*80)

    results = {}

    for symbol in symbols:
        print(f"\nLoading {symbol}...")
        data = load_symbol_cache(symbol, data_dir)

        if data is None:
            print(f"  No cache found for {symbol}")
            continue

        # Convert to DataFrame if needed
        if isinstance(data, dict):
            # Cache format: {'data': DataFrame, 'timestamp': ...}
            if 'data' in data and isinstance(data['data'], pd.DataFrame):
                df = data['data'].copy()
            else:
                # Try to create DataFrame from dict
                try:
                    df = pd.DataFrame(data)
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df.set_index('timestamp', inplace=True)
                except ValueError:
                    print(f"  Could not parse data format for {symbol}")
                    continue
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            print(f"  Unexpected data format for {symbol}")
            continue

        quality = analyze_data_quality(df, symbol)
        price = analyze_price_characteristics(df, symbol)
        volume = analyze_volume(df, symbol)
        patterns = analyze_patterns(df, symbol)

        results[symbol] = {
            **quality,
            **price,
            **volume,
            **patterns
        }

    # Summary comparison
    if len(results) > 1:
        print("\n" + "="*80)
        print("COMPARISON SUMMARY")
        print("="*80)

        print(f"\n{'Symbol':<12} {'Rows':>10} {'Volatility':>12} {'Volume':>15} {'Autocorr':>10} {'Trend':>8}")
        print("-" * 70)

        for symbol, r in results.items():
            print(f"{symbol:<12} {r['rows']:>10,} {r['volatility']*100:>11.2f}% {r['mean_volume']:>15,.0f} {r['autocorr_1']:>10.4f} {r['trend_persistence']:>8.3f}")

        # Key insights
        print("\n" + "="*80)
        print("KEY INSIGHTS")
        print("="*80)

        if 'ETHUSDT' in results:
            eth = results['ETHUSDT']
            for symbol, r in results.items():
                if symbol == 'ETHUSDT':
                    continue

                print(f"\n{symbol} vs ETHUSDT:")
                if r['rows'] != eth['rows']:
                    ratio = r['rows'] / eth['rows']
                    print(f"  Data: {ratio:.1f}x rows ({r['rows']:,} vs {eth['rows']:,})")
                if r['volatility'] != eth['volatility']:
                    ratio = r['volatility'] / eth['volatility']
                    print(f"  Volatility: {ratio:.2f}x")
                if r['mean_volume'] != eth['mean_volume']:
                    ratio = r['mean_volume'] / eth['mean_volume']
                    print(f"  Volume: {ratio:.2f}x")
                if r['autocorr_1'] != eth['autocorr_1']:
                    diff = r['autocorr_1'] - eth['autocorr_1']
                    print(f"  Autocorr diff: {diff:+.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Symbol Diagnostics Tool')
    parser.add_argument('--symbols', type=str, default='ETHUSDT,AVAXUSDT,BTCUSDT,SOLUSDT',
                        help='Comma-separated list of symbols to analyze')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Directory containing cache files')
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(',')]
    compare_symbols(symbols, args.data_dir)


if __name__ == '__main__':
    main()
