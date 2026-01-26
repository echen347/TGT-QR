#!/usr/bin/env python3
"""
Experiment: Test Market Regime Hypothesis (H1)

Hypothesis: ETHUSDT exhibits momentum behavior while other symbols exhibit mean reversion.
Strategies optimized for momentum should work on ETHUSDT but fail on mean-reverting assets.

Test: Run momentum and mean-reversion strategies across all symbols using cached data only.
"""

import sys
import os
import pickle
import pandas as pd
import numpy as np

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))


def load_cached_data(symbol):
    """Load cached data directly from pickle files"""
    cache_path = os.path.join(os.path.dirname(__file__), '..', 'data', f'{symbol}_cache.pkl')
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        return data['data']
    return None


def calculate_ema(series, period):
    """Calculate EMA"""
    return series.ewm(span=period, adjust=False).mean()


def calculate_rsi(prices, period=14):
    """Calculate RSI"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def momentum_strategy_macd(df, fast=12, slow=26, signal=9):
    """
    MACD momentum strategy - trend following.
    Buy on MACD crossover up, sell on crossover down.
    """
    closes = df['close']
    macd_line = calculate_ema(closes, fast) - calculate_ema(closes, slow)
    signal_line = calculate_ema(macd_line, signal)

    signals = pd.Series(0, index=df.index)

    # Generate signals on crossovers
    for i in range(1, len(df)):
        if pd.isna(macd_line.iloc[i]) or pd.isna(signal_line.iloc[i]):
            continue
        if pd.isna(macd_line.iloc[i-1]) or pd.isna(signal_line.iloc[i-1]):
            continue

        # Bullish crossover
        if macd_line.iloc[i] > signal_line.iloc[i] and macd_line.iloc[i-1] <= signal_line.iloc[i-1]:
            signals.iloc[i] = 1
        # Bearish crossover
        elif macd_line.iloc[i] < signal_line.iloc[i] and macd_line.iloc[i-1] >= signal_line.iloc[i-1]:
            signals.iloc[i] = -1

    return signals


def momentum_strategy_donchian(df, period=20):
    """
    Donchian channel breakout - trend following momentum.
    Buy on breakout above upper band, sell on breakdown below lower band.
    """
    highs = df['high'].rolling(window=period).max()
    lows = df['low'].rolling(window=period).min()
    closes = df['close']

    signals = pd.Series(0, index=df.index)

    for i in range(period + 1, len(df)):
        prev_close = closes.iloc[i - 1]
        upper = highs.iloc[i - 2]  # Use previous period's band
        lower = lows.iloc[i - 2]

        if pd.isna(upper) or pd.isna(lower):
            continue

        if prev_close > upper:
            signals.iloc[i] = 1
        elif prev_close < lower:
            signals.iloc[i] = -1

    return signals


def mean_reversion_strategy_rsi(df, period=14, oversold=30, overbought=70):
    """
    RSI mean reversion strategy.
    Buy when oversold (RSI < 30), sell when overbought (RSI > 70).
    """
    rsi = calculate_rsi(df['close'], period)

    signals = pd.Series(0, index=df.index)

    for i in range(period + 1, len(df)):
        if pd.isna(rsi.iloc[i]):
            continue

        if rsi.iloc[i] < oversold:
            signals.iloc[i] = 1
        elif rsi.iloc[i] > overbought:
            signals.iloc[i] = -1

    return signals


def backtest_strategy(df, signals, fee_bps=11):
    """
    Simple backtest: enter on signal, hold until opposite signal.
    Returns total return percentage.
    """
    position = 0  # 1 = long, -1 = short, 0 = flat
    entry_price = 0
    total_pnl = 0
    trades = 0

    closes = df['close'].values
    sigs = signals.values

    for i in range(len(df)):
        sig = sigs[i]
        price = closes[i]

        if sig == 1 and position != 1:
            # Close short if any
            if position == -1:
                pnl = (entry_price - price) / entry_price * 100
                pnl -= fee_bps / 100 * 2  # Entry + exit fees
                total_pnl += pnl
                trades += 1

            # Go long
            position = 1
            entry_price = price

        elif sig == -1 and position != -1:
            # Close long if any
            if position == 1:
                pnl = (price - entry_price) / entry_price * 100
                pnl -= fee_bps / 100 * 2
                total_pnl += pnl
                trades += 1

            # Go short
            position = -1
            entry_price = price

    # Close any open position at end
    if position == 1:
        pnl = (closes[-1] - entry_price) / entry_price * 100
        pnl -= fee_bps / 100 * 2
        total_pnl += pnl
        trades += 1
    elif position == -1:
        pnl = (entry_price - closes[-1]) / entry_price * 100
        pnl -= fee_bps / 100 * 2
        total_pnl += pnl
        trades += 1

    return total_pnl, trades


def main():
    print("=" * 80)
    print("EXPERIMENT: Market Regime Hypothesis (H1)")
    print("=" * 80)
    print()
    print("Hypothesis: ETHUSDT shows momentum, others show mean-reversion.")
    print("Expected: Momentum strategies work on ETHUSDT, mean-reversion works on others.")
    print()

    symbols = ['ETHUSDT', 'AVAXUSDT', 'SOLUSDT']

    strategies = {
        'MACD (Momentum)': momentum_strategy_macd,
        'Donchian (Momentum)': momentum_strategy_donchian,
        'RSI (Mean-Rev)': mean_reversion_strategy_rsi,
    }

    results = {}

    print("Loading cached data and running backtests...")
    print("-" * 80)

    for symbol in symbols:
        df = load_cached_data(symbol)
        if df is None:
            print(f"  {symbol}: No cached data found")
            continue

        print(f"\n{symbol}: {len(df)} candles from {df.index.min().date()} to {df.index.max().date()}")
        results[symbol] = {}

        for name, strategy_fn in strategies.items():
            signals = strategy_fn(df)
            ret, trades = backtest_strategy(df, signals)
            results[symbol][name] = {'return': ret, 'trades': trades}
            sign = '+' if ret >= 0 else ''
            print(f"  {name:<20} {sign}{ret:.1f}% ({trades} trades)")

    # Print results matrix
    print()
    print("=" * 80)
    print("RESULTS MATRIX: Total Return (%)")
    print("=" * 80)
    print()

    # Header
    header = f"{'Strategy':<22}"
    for symbol in symbols:
        header += f"{symbol:>12}"
    print(header)
    print("-" * (22 + 12 * len(symbols)))

    for strat_name in strategies.keys():
        row = f"{strat_name:<22}"
        for symbol in symbols:
            res = results.get(symbol, {}).get(strat_name, {})
            ret = res.get('return')
            if ret is not None:
                sign = '+' if ret >= 0 else ''
                row += f"{sign}{ret:>10.1f}%"
            else:
                row += f"{'N/A':>11}"
        print(row)

    # Analysis
    print()
    print("=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    # Separate momentum and mean-reversion results
    momentum_strats = ['MACD (Momentum)', 'Donchian (Momentum)']
    meanrev_strats = ['RSI (Mean-Rev)']

    eth_momentum = []
    eth_meanrev = []
    other_momentum = []
    other_meanrev = []

    for symbol in symbols:
        for strat in momentum_strats:
            ret = results.get(symbol, {}).get(strat, {}).get('return')
            if ret is not None:
                if symbol == 'ETHUSDT':
                    eth_momentum.append(ret)
                else:
                    other_momentum.append(ret)

        for strat in meanrev_strats:
            ret = results.get(symbol, {}).get(strat, {}).get('return')
            if ret is not None:
                if symbol == 'ETHUSDT':
                    eth_meanrev.append(ret)
                else:
                    other_meanrev.append(ret)

    print()
    print("Average Returns by Category:")
    if eth_momentum:
        avg = sum(eth_momentum) / len(eth_momentum)
        print(f"  ETHUSDT + Momentum:      {avg:+.1f}%")
    if eth_meanrev:
        avg = sum(eth_meanrev) / len(eth_meanrev)
        print(f"  ETHUSDT + Mean-Rev:      {avg:+.1f}%")
    if other_momentum:
        avg = sum(other_momentum) / len(other_momentum)
        print(f"  Others + Momentum:       {avg:+.1f}%")
    if other_meanrev:
        avg = sum(other_meanrev) / len(other_meanrev)
        print(f"  Others + Mean-Rev:       {avg:+.1f}%")

    # Hypothesis evaluation
    print()
    print("-" * 80)
    print("HYPOTHESIS EVALUATION (H1)")
    print("-" * 80)

    eth_mom = sum(eth_momentum) / len(eth_momentum) if eth_momentum else 0
    eth_mr = sum(eth_meanrev) / len(eth_meanrev) if eth_meanrev else 0
    oth_mom = sum(other_momentum) / len(other_momentum) if other_momentum else 0
    oth_mr = sum(other_meanrev) / len(other_meanrev) if other_meanrev else 0

    h1_support = 0

    # Check 1: Momentum works better on ETHUSDT than mean-reversion
    if eth_mom > eth_mr:
        print(f"✓ ETHUSDT: Momentum ({eth_mom:+.1f}%) > Mean-Rev ({eth_mr:+.1f}%)")
        h1_support += 1
    else:
        print(f"✗ ETHUSDT: Momentum ({eth_mom:+.1f}%) <= Mean-Rev ({eth_mr:+.1f}%)")

    # Check 2: Mean-reversion works better on others than momentum
    if oth_mr > oth_mom:
        print(f"✓ Others:  Mean-Rev ({oth_mr:+.1f}%) > Momentum ({oth_mom:+.1f}%)")
        h1_support += 1
    else:
        print(f"✗ Others:  Mean-Rev ({oth_mr:+.1f}%) <= Momentum ({oth_mom:+.1f}%)")

    # Check 3: ETHUSDT momentum > Others momentum
    if eth_mom > oth_mom:
        print(f"✓ Momentum: ETHUSDT ({eth_mom:+.1f}%) > Others ({oth_mom:+.1f}%)")
        h1_support += 1
    else:
        print(f"✗ Momentum: ETHUSDT ({eth_mom:+.1f}%) <= Others ({oth_mom:+.1f}%)")

    print()
    if h1_support >= 2:
        print(f"CONCLUSION: H1 SUPPORTED ({h1_support}/3 criteria met)")
        print("Market regime differences likely explain ETHUSDT's performance.")
    else:
        print(f"CONCLUSION: H1 NOT SUPPORTED ({h1_support}/3 criteria met)")
        print("Market regime hypothesis does not explain the difference.")
        print("Consider testing H2 (data quality artifact).")


if __name__ == '__main__':
    main()
