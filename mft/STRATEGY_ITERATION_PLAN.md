# Strategy Iteration Plan - Finding a Profitable Strategy

## Current Status

### Adaptive System Results (60 days)
- **Total Return**: -120.42% ❌
- **Win Rate**: 38.64% ❌ (target: ≥50%)
- **Trades/Day**: 0.73 ⚠️ (target: ≥1.0)
- **Best Performer**: AVAXUSDT (+26.90%, 42.11% win rate)

### Key Issues
1. **Overall negative returns** - Strategy is losing money
2. **Low win rate** - Below 50% threshold
3. **Adaptive system tightening filters** - But not improving performance enough
4. **Symbol-specific performance** - AVAXUSDT works better than others

## Iteration Strategy

### Phase 1: Symbol Selection & Filtering
**Hypothesis**: Some symbols work better than others. Focus on profitable symbols.

**Tests**:
1. ✅ **Test individual symbols** - Identify which symbols are profitable
2. **Remove unprofitable symbols** - Focus only on symbols with >50% win rate
3. **Test symbol combinations** - Find optimal symbol mix

**Expected**: Improve overall win rate by removing bad symbols

---

### Phase 2: Parameter Optimization
**Hypothesis**: Current parameters (MA=20, thresholds) may not be optimal for all symbols.

**Tests**:
1. **MA Period Sweep**: Test MA periods 10, 15, 20, 25, 30
2. **Threshold Sweep**: Test different volatility thresholds
3. **Trend Strength Sweep**: Test MIN_TREND_STRENGTH values
4. **Symbol-specific parameters**: Optimize per symbol

**Expected**: Find parameters that achieve >50% win rate

---

### Phase 3: Entry/Exit Improvements
**Hypothesis**: Better entry/exit logic can improve win rate.

**Tests**:
1. **Multi-timeframe confirmation**: Require signal alignment on 1m, 5m, 15m
2. **Volume confirmation**: Require volume spike for entry
3. **ATR-based thresholds**: Use ATR instead of fixed percentages
4. **Trailing stops**: Use trailing stop-loss instead of fixed

**Expected**: Higher win rate, better risk/reward

---

### Phase 4: Risk Management Improvements
**Hypothesis**: Better risk management can improve overall returns even with same win rate.

**Tests**:
1. **Dynamic position sizing**: Size based on volatility/confidence
2. **Portfolio-level stops**: Stop trading if overall portfolio underperforms
3. **Correlation filtering**: Avoid correlated positions (e.g., BTC + ETH together)
4. **Time-based filters**: Avoid trading during low-liquidity hours

**Expected**: Better risk-adjusted returns

---

## Testing Protocol

### For Each Iteration:
1. **Backtest on 60+ days** of historical data
2. **Train/Test Split**: Use 70% for optimization, 30% for validation
3. **Check for Overfitting**: Test performance should match train performance
4. **Success Criteria**:
   - Win Rate ≥ 50%
   - Total Return > 0%
   - Trades/Day ≥ 1.0
   - Sharpe Ratio > 1.0

### Avoid Overfitting:
- ✅ Use out-of-sample test data
- ✅ Test on multiple symbols
- ✅ Test on different time periods
- ✅ Don't optimize on test data
- ✅ Keep parameter changes simple and explainable

---

## Immediate Next Steps

### Step 1: Symbol Analysis
```bash
# Test each symbol individually
python3 tools/backtester.py --symbols ETHUSDT --days 60 --no-plot
python3 tools/backtester.py --symbols SOLUSDT --days 60 --no-plot
python3 tools/backtester.py --symbols BTCUSDT --days 60 --no-plot
python3 tools/backtester.py --symbols AVAXUSDT --days 60 --no-plot
```

**Action**: Remove symbols with <45% win rate or negative returns

### Step 2: Parameter Sweep
```bash
# Test different MA periods
for ma in 10 15 20 25 30; do
    python3 tools/backtester.py --ma $ma --days 60 --no-plot
done
```

**Action**: Find optimal MA period for remaining symbols

### Step 3: Test Improvements
- Test multi-timeframe confirmation
- Test ATR-based thresholds
- Test volume confirmation

---

## Success Metrics

A strategy is considered "good" if it meets ALL of these:
- ✅ **Win Rate**: ≥50% (profitable)
- ✅ **Total Return**: >0% (makes money)
- ✅ **Trades/Day**: ≥1.0 (meets frequency target)
- ✅ **Sharpe Ratio**: >1.0 (good risk-adjusted returns)
- ✅ **No Overfitting**: Test performance matches train

---

## Notes

- **Current adaptive system**: Tightening filters when underperforming, but not enough improvement
- **AVAXUSDT shows promise**: +26.90% return suggests strategy can work
- **Need systematic iteration**: Test one change at a time, measure impact
- **Avoid overfitting**: Always validate on out-of-sample data

