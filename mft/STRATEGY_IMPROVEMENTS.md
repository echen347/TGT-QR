# Strategy Improvements & Re-evaluation Plan

## Current Issues

### Performance Analysis
- **ETHUSDT**: 0 trades (no signals generated) - strategy too conservative or wrong parameters
- **SOLUSDT**: 13 trades, -39.54% return, 38.46% win rate - strategy underperforming
- **Previous success**: BTCUSDT + ETHUSDT showed 112% return, 7.67 trades/day
- **Key insight**: Strategy works better on BTCUSDT than SOLUSDT

### Root Causes
1. **Fixed parameters** - Same MA period/thresholds for all symbols regardless of volatility
2. **Limited symbols** - Only 2 symbols (ETHUSDT, SOLUSDT) vs successful test (BTCUSDT, ETHUSDT)
3. **No adaptation** - Strategy doesn't adapt to changing market conditions
4. **Time period dependency** - Recent 60 days may be unfavorable market regime

## Proposed Improvements

### Phase 2A: Add More Symbols ⭐ HIGH PRIORITY
**Why**: Previous successful backtest used BTCUSDT. More symbols = more opportunities.

**Implementation**:
```python
SYMBOLS = [
    'ETHUSDT', 'SOLUSDT', 'BTCUSDT', 'AVAXUSDT', 'MATICUSDT'
]
```

**Expected Impact**: 
- 2.5x more trading opportunities (2 → 5 symbols)
- Diversification across different market behaviors
- Some symbols may perform better than others

**Risk**: Low - Same strategy, just more symbols

---

### Phase 2B: Sliding Window Adaptive Strategy ⭐⭐ HIGH PRIORITY
**Why**: Markets change. Strategy should adapt to recent performance.

**Concept**: 
- Track performance over rolling windows (e.g., last 7 days, 14 days, 30 days)
- Adjust parameters based on recent win rate and returns
- If performance drops, tighten filters or reduce position sizes

**Implementation**:
```python
# Track recent performance
recent_trades = get_trades_last_n_days(7)
recent_win_rate = calculate_win_rate(recent_trades)
recent_return = calculate_return(recent_trades)

# Adaptive parameters
if recent_win_rate < 0.45:  # Underperforming
    # Tighten filters - reduce signal frequency
    MIN_TREND_STRENGTH *= 1.5  # Require stronger trends
    THRESHOLD_HIGH_VOL *= 1.2  # Require larger deviations
    MAX_POSITION_USDT *= 0.8  # Reduce position size
elif recent_win_rate > 0.55:  # Performing well
    # Loosen filters - increase signal frequency
    MIN_TREND_STRENGTH *= 0.9
    THRESHOLD_HIGH_VOL *= 0.95
```

**Expected Impact**:
- Better adaptation to market regimes
- Reduced drawdowns during bad periods
- Increased activity during good periods

**Risk**: Medium - Need to avoid over-optimization

---

### Phase 2C: Symbol-Specific Parameters ⭐ MEDIUM PRIORITY
**Why**: Different symbols have different volatility and behavior patterns.

**Implementation**:
```python
SYMBOL_PARAMS = {
    'BTCUSDT': {
        'ma_period': 20,
        'threshold_high': 0.002,
        'threshold_normal': 0.0005,
        'threshold_low': 0.0003
    },
    'ETHUSDT': {
        'ma_period': 18,  # Slightly shorter for ETH
        'threshold_high': 0.0025,  # ETH more volatile
        'threshold_normal': 0.0006,
        'threshold_low': 0.0004
    },
    'SOLUSDT': {
        'ma_period': 15,  # SOL more volatile, shorter MA
        'threshold_high': 0.003,  # Larger thresholds for SOL
        'threshold_normal': 0.0008,
        'threshold_low': 0.0005
    }
}
```

**Expected Impact**:
- Better signal quality per symbol
- Reduced false signals on volatile symbols
- More signals on stable symbols

**Risk**: Medium - Need backtesting per symbol

---

### Phase 2D: Multi-Timeframe Confirmation ⭐⭐ MEDIUM PRIORITY
**Why**: Confirm signals across multiple timeframes to reduce false signals.

**Implementation**:
```python
# Check signals on multiple timeframes
signal_1m = calculate_signal(prices_1m)  # Current
signal_5m = calculate_signal(prices_5m)  # 5-minute candles
signal_15m = calculate_signal(prices_15m)  # 15-minute candles

# Require alignment across timeframes
if signal_1m == signal_5m == signal_15m:
    final_signal = signal_1m  # Strong signal
elif signal_1m == signal_5m:
    final_signal = signal_1m  # Medium signal
else:
    final_signal = 0  # No signal - conflicting timeframes
```

**Expected Impact**:
- Higher win rate (fewer false signals)
- Lower trade frequency (more selective)
- Better entry timing

**Risk**: Low - Conservative approach

---

### Phase 2E: Dynamic MA Period Based on Volatility ⭐ LOW PRIORITY
**Why**: Volatile markets need shorter MA, stable markets can use longer MA.

**Implementation**:
```python
# Calculate recent volatility
volatility = prices_df['close'].pct_change().rolling(window=20).std().iloc[-1]

# Adjust MA period based on volatility
if volatility > 0.03:  # High volatility
    ma_period = 15  # Shorter MA, more responsive
elif volatility < 0.01:  # Low volatility
    ma_period = 25  # Longer MA, smoother
else:
    ma_period = 20  # Default
```

**Expected Impact**:
- Better adaptation to market conditions
- More signals in volatile markets
- Fewer false signals in stable markets

**Risk**: Medium - Need careful calibration

---

## Implementation Priority

### Immediate (This Week)
1. ✅ **Add BTCUSDT** - Previous backtest showed it works well
2. ✅ **Add AVAXUSDT** - High volume, good liquidity
3. ✅ **Test on new symbols** - Backtest with expanded symbol list

### Short-term (Next Week)
1. **Sliding Window Adaptation** - Track 7-day performance, adjust parameters
2. **Symbol-Specific Parameters** - Tune for BTCUSDT, ETHUSDT, SOLUSDT separately

### Medium-term (Next Month)
1. **Multi-Timeframe Confirmation** - Add 5m and 15m confirmation
2. **Dynamic MA Period** - Adjust based on volatility

## Testing Plan

### Step 1: Expand Symbols
```bash
# Test with expanded symbol list
python3 tools/backtester.py --symbols ETHUSDT,SOLUSDT,BTCUSDT,AVAXUSDT --days 60
```

### Step 2: Test Sliding Window
- Implement performance tracking
- Test adaptive parameters on historical data
- Compare to fixed parameters

### Step 3: Test Symbol-Specific Params
- Backtest each symbol with optimized parameters
- Compare to unified parameters

## Success Metrics

- **Trades/Day**: ≥1.0 (current: 0.2-0.4 on SOLUSDT)
- **Win Rate**: ≥50% (current: 38.46% on SOLUSDT)
- **Total Return**: Positive (current: -39.54% on SOLUSDT)
- **Sharpe Ratio**: >1.0 (current: -7.82 on SOLUSDT)

## Risk Management

- Keep exit-on-profit mode enabled during testing
- Monitor live performance closely
- Reduce position sizes if performance degrades
- Consider pausing if win rate drops below 40%

