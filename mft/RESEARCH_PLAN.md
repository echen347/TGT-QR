# Research Plan - Finding Profitable Strategy for AVAXUSDT

**Status**: ⏸️ **TRADING PAUSED** - Research Mode  
**Date**: 2025-11-11  
**Goal**: Find profitable trading strategy for AVAXUSDT

---

## Current Situation

- **All tested strategies are unprofitable** on recent 60 days of data
- **Best result**: MA=20 with -43.77% return, 38.01% win rate
- **System paused**: No new positions will be opened
- **Old positions**: Will exit at 1% profit (defensive mode)

---

## Research Areas

### 1. Parameter Optimization ⭐⭐⭐
**Status**: In Progress

**Approach**:
- Test different MA periods (15, 20, 25, 30) ✅ Done - all unprofitable
- Test different trend strength thresholds
- Test different volatility thresholds
- Test different stop loss/take profit ratios

**Tools**:
```bash
python3 tools/optimize_avax_params.py --train-days 60 --test-days 30
```

**Next Steps**:
- Test threshold combinations
- Test stop loss/take profit ratios (1:1, 1:2, 1:3)
- Test position sizing strategies

---

### 2. Alternative Strategies ⭐⭐⭐⭐
**Status**: Partially Tested

**Tested**:
- ✅ RSI Mean Reversion: -99.65% return ❌
- ✅ MACD: -99.83% return ❌
- ✅ HTF Pullback: -87.16% return ❌
- ❌ Donchian: Error

**To Test**:
- **Volume-Weighted MA**: Use VWMA instead of simple MA
- **ATR-Based Dynamic**: Use ATR for dynamic thresholds
- **Multi-Timeframe Confirmation**: Require 1m + 5m + 15m alignment
- **Order Flow**: Track buy/sell volume imbalance
- **Breakout Strategy**: Trade breakouts from consolidation

**Tools**:
```bash
# Test available strategies
python3 tools/backtester.py --symbols AVAXUSDT --days 60 --strategy <strategy_name>
```

---

### 3. Entry/Exit Improvements ⭐⭐⭐
**Status**: Not Tested

**Ideas**:
- **Volume Confirmation**: Only enter if volume > 1.5x average
- **Time-of-Day Filter**: Avoid trading during low liquidity hours
- **Volatility Filter**: Skip trades during extreme volatility
- **Trailing Stops**: Use trailing stop-loss instead of fixed
- **Partial Exits**: Exit 50% at 1% profit, let rest run

**Implementation**: Modify `strategy.py` and backtest

---

### 4. Market Regime Analysis ⭐⭐
**Status**: Not Started

**Questions**:
- What market conditions favor trend-following?
- What conditions favor mean reversion?
- Can we detect regime changes and switch strategies?

**Approach**:
- Analyze historical performance by market regime
- Identify regime indicators (volatility, trend strength, volume)
- Build regime detection system

---

### 5. Risk Management Improvements ⭐⭐⭐
**Status**: Partially Implemented

**Current**:
- ✅ Stop loss: 1%
- ✅ Take profit: 2%
- ✅ Max position hold: 24 hours
- ✅ Exit old positions at 1% profit

**To Test**:
- Dynamic position sizing based on volatility
- Portfolio-level risk limits
- Correlation filtering (avoid correlated positions)

---

## Testing Protocol

### Step 1: Quick Backtest (60 days)
```bash
python3 tools/backtester.py --symbols AVAXUSDT --days 60 --strategy <strategy> --no-plot
```

**Criteria**: Return > 0%, Win Rate >= 40%

### Step 2: OOD Validation (Train/Test Split)
```bash
python3 tools/backtest_avax_ood.py --train-days 60 --test-days 30
```

**Criteria**:
- Test return > 0%
- Test win rate >= 40%
- No overfitting (test within 20% of train)
- Test trades/day >= 0.5

### Step 3: Deploy
- Only deploy if passes OOD validation
- Monitor live performance closely
- Be ready to pause if performance degrades

---

## Priority Order

1. **Test Volume-Weighted MA** (similar to current but with volume filter)
2. **Test ATR-Based Dynamic** (adaptive to volatility)
3. **Test Multi-Timeframe Confirmation** (reduce false signals)
4. **Parameter optimization** (if any strategy shows promise)
5. **Entry/Exit improvements** (refine best strategy)

---

## Success Criteria

**Must Have**:
- ✅ Test return > 0% (on OOD data)
- ✅ Test win rate >= 40%
- ✅ No overfitting (test within 20% of train)
- ✅ Test trades/day >= 0.5

**Nice to Have**:
- Win rate >= 50%
- Sharpe ratio > 1.0
- Max drawdown < 20%

---

## Timeline

- **Week 1**: Test alternative strategies (Volume-Weighted MA, ATR-Based, Multi-Timeframe)
- **Week 2**: Parameter optimization for promising strategies
- **Week 3**: Entry/Exit improvements
- **Week 4**: OOD validation and deployment

---

## Notes

- **Stick to recent data** - Don't cherry-pick favorable time periods
- **Avoid overfitting** - Always use train/test split
- **Be patient** - Finding profitable strategies takes time
- **Monitor live** - Even paused, system monitors market conditions

---

## Resources

- `BACKTEST_RESULTS_SUMMARY.md` - All backtest results
- `STRATEGY_ALTERNATIVES.md` - Alternative strategy ideas
- `tools/backtester.py` - Main backtesting tool
- `tools/backtest_avax_ood.py` - OOD validation tool
- `tools/optimize_avax_params.py` - Parameter optimization tool

