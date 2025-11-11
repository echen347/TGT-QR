# Strategy Improvements & Re-evaluation Plan

## Current Status (Updated)

### Recent Backtest Results (60 days)
- **AVAXUSDT**: +26.90% return, 42.11% win rate, 38 trades ✅ **ONLY PROFITABLE SYMBOL**
- **ETHUSDT**: 0 trades (no signals generated)
- **SOLUSDT**: -4.68% return, 20% win rate, 5 trades
- **BTCUSDT**: -142.64% return, 0% win rate, 1 trade
- **Adaptive System**: -120.42% return, 38.64% win rate ❌ **DISABLED**

### Current Strategy Focus
- **Symbol**: AVAXUSDT only (only profitable symbol)
- **Adaptive System**: Disabled (not improving performance)
- **Max Positions**: 1 (single symbol focus)
- **Parameters**: MA=20, thresholds 0.2%/0.05%/0.03%

### Key Insights
1. **Symbol-specific performance** - AVAXUSDT works, others don't
2. **Adaptive system failed** - Made performance worse (-120% return)
3. **Simple is better** - Rule-based strategy outperforms adaptive
4. **Focus needed** - One symbol is easier to optimize than multiple

## Proposed Improvements

### Phase 1: AVAXUSDT Parameter Optimization ⭐⭐⭐ HIGHEST PRIORITY
**Status**: 🔄 **IN PROGRESS** - Current MA strategy not profitable (-43.77% return)

**Why**: Current MA strategy shows -43.77% return, 38.01% win rate on recent data. Need to find profitable parameters or alternative strategy.

**Approach**: 
1. **Parameter Sweep** (with OOD validation):
   ```bash
   # Test different MA periods with train/test split
   python3 tools/optimize_avax_params.py --train-days 60 --test-days 30
   ```

2. **Alternative Strategies** (if parameter optimization fails):
   - See `STRATEGY_ALTERNATIVES.md` for ideas (RSI Mean Reversion, Breakout, ATR-Based, etc.)
   - Test each on recent 60 days of data
   - Only deploy if profitable on OOD test data

**Expected Impact**: 
- Find profitable parameters OR alternative strategy
- Achieve >0% return, ≥40% win rate on OOD test data
- Avoid overfitting (test performance within 20% of train)

**Risk**: Medium - Must be careful about overfitting, may need to try multiple strategies

---

### Phase 2: Entry/Exit Improvements ⭐⭐ MEDIUM PRIORITY
**Status**: 🔄 **NEXT** - After Phase 1 optimization

**Why**: Current win rate (42%) is below target (≥50%). Better entry/exit logic can help.

**Options**:
1. **Multi-timeframe confirmation** - Require 1m + 5m alignment
2. **Volume confirmation** - Require volume spike for entry
3. **ATR-based thresholds** - Use ATR instead of fixed percentages
4. **Trailing stops** - Protect profits better

**Expected Impact**:
- Higher win rate (42% → ≥50%)
- Better risk/reward ratio
- Fewer false signals

**Risk**: Medium - Need careful backtesting

---

### Phase 3: Add More Symbols (Only if AVAXUSDT works) ⭐ LOW PRIORITY
**Status**: ⏸️ **PAUSED** - Focus on AVAXUSDT first

**Why**: Once AVAXUSDT is profitable, test other symbols individually.

**Implementation**:
```python
# Only add symbols that show >50% win rate in backtest
SYMBOLS = [
    'AVAXUSDT',  # Proven profitable
    # Add others only after individual backtests show >50% win rate
]
```

**Expected Impact**: 
- More trading opportunities
- Diversification

**Risk**: Medium - Need to avoid adding unprofitable symbols

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

### Immediate (This Week) ✅ COMPLETED
1. ✅ **Disable Adaptive System** - Not improving performance
2. ✅ **Focus on AVAXUSDT** - Only profitable symbol
3. ✅ **Set MAX_POSITIONS = 1** - Single symbol focus

### Short-term (Next Week)
1. **Parameter Optimization** - MA period sweep (15, 20, 25, 30)
2. **Threshold Optimization** - Test tighter/looser thresholds
3. **Trend Strength Tuning** - Test different MIN_TREND_STRENGTH values

### Medium-term (Next Month)
1. **Entry/Exit Improvements** - Multi-timeframe, volume confirmation
2. **Add More Symbols** - Only if AVAXUSDT achieves ≥50% win rate

## Testing Plan

### Step 1: AVAXUSDT Parameter Optimization ✅ IN PROGRESS
```bash
# Test different MA periods
python3 tools/backtester.py --symbols AVAXUSDT --days 60 --ma-period 15
python3 tools/backtester.py --symbols AVAXUSDT --days 60 --ma-period 20
python3 tools/backtester.py --symbols AVAXUSDT --days 60 --ma-period 25
python3 tools/backtester.py --symbols AVAXUSDT --days 60 --ma-period 30

# Compare results and select best MA period
```

### Step 2: Threshold Optimization
- Test different volatility thresholds
- Test different MIN_TREND_STRENGTH values
- Find optimal combination for AVAXUSDT

### Step 3: Entry/Exit Improvements
- Test multi-timeframe confirmation
- Test volume confirmation
- Compare to baseline

## Success Metrics

- **Trades/Day**: ≥1.0 (current: 0.63 on AVAXUSDT)
- **Win Rate**: ≥50% (current: 42.11% on AVAXUSDT - needs improvement)
- **Total Return**: Positive (current: +26.90% on AVAXUSDT ✅)
- **Sharpe Ratio**: >1.0 (to be calculated)

## Risk Management

- Keep exit-on-profit mode enabled during testing
- Monitor live performance closely
- Reduce position sizes if performance degrades
- Consider pausing if win rate drops below 40%

