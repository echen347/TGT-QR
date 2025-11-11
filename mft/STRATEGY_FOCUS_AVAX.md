# Strategy Focus: AVAXUSDT Optimization

## Decision: Focus on Profitable Symbol

**Backtest Results (60 days):**
- AVAXUSDT: +26.90% return, 42.11% win rate, 38 trades
- ETHUSDT: No trades
- SOLUSDT: -4.68% return, 20% win rate, 5 trades
- BTCUSDT: -142.64% return, 0% win rate, 1 trade

**Conclusion**: AVAXUSDT is the only profitable symbol. Focus optimization here.

---

## Current Configuration

**Symbols**: `['AVAXUSDT']` (single symbol focus)
**Adaptive System**: Disabled (not improving performance)
**Max Positions**: 1 (focus on AVAXUSDT only)

**Current Parameters**:
- MA Period: 20
- MIN_TREND_STRENGTH: 0.0002
- Thresholds: 0.2%/0.05%/0.03% (High/Normal/Low vol)
- Leverage: 5x
- Position Size: $10 USDT margin

---

## Optimization Plan

### Phase 1: Parameter Optimization for AVAXUSDT

**Test different parameter combinations:**

1. **MA Period Sweep**: 15, 20, 25, 30
2. **Threshold Optimization**: Test tighter/looser thresholds
3. **Trend Strength**: Test different MIN_TREND_STRENGTH values

**Goal**: Find parameters that achieve:
- Win Rate ≥ 50%
- Total Return > 0%
- Trades/Day ≥ 1.0

### Phase 2: Entry/Exit Improvements

**If Phase 1 doesn't achieve goals:**

1. **Multi-timeframe confirmation**: Require 1m + 5m alignment
2. **Volume confirmation**: Require volume spike for entry
3. **ATR-based thresholds**: Use ATR instead of fixed percentages

### Phase 3: Add More Symbols (Only if AVAXUSDT works)

**Once AVAXUSDT is profitable:**
- Test other symbols individually
- Only add symbols that show >50% win rate in backtest

---

## Testing Protocol

```bash
# Test AVAXUSDT with different parameters
python3 tools/backtester.py --symbols AVAXUSDT --days 60 --ma-period 15
python3 tools/backtester.py --symbols AVAXUSDT --days 60 --ma-period 20
python3 tools/backtester.py --symbols AVAXUSDT --days 60 --ma-period 25
python3 tools/backtester.py --symbols AVAXUSDT --days 60 --ma-period 30
```

**Success Criteria:**
- Win Rate ≥ 50%
- Total Return > 0%
- Trades/Day ≥ 1.0
- No overfitting (test performance matches train)

---

## Next Steps

1. ✅ Disable adaptive system
2. ✅ Focus on AVAXUSDT only
3. 🔄 Run parameter optimization backtests
4. 📊 Analyze results and select best parameters
5. 🚀 Deploy optimized strategy
6. 📈 Monitor live performance

---

## Why This Approach?

1. **Focus**: One symbol is easier to optimize than four
2. **Proven**: AVAXUSDT already shows profitability
3. **Simple**: Rule-based, no ML complexity
4. **Low Risk**: Can't overfit on one symbol if we test properly
5. **Iterative**: Once AVAXUSDT works, expand to other symbols

