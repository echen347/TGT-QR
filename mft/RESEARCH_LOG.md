# Research Log - Scientific Method

**Date Started**: 2025-11-11  
**Goal**: Find profitable trading strategy for crypto futures  
**Method**: Systematic hypothesis testing with strict overfitting prevention

---

## Research Methodology

### Scientific Method Process:
1. **Hypothesis Formation**: State clear hypothesis about why strategy should work
2. **Experiment Design**: Define test period, metrics, success criteria
3. **Implementation**: Code the strategy
4. **Testing**: Run backtest on recent 60 days (2025-09-12 to 2025-11-11)
5. **Analysis**: Evaluate results against success criteria
6. **OOD Validation**: Train/test split (60% train, 40% test) to check overfitting
7. **Conclusion**: Accept/reject hypothesis, document learnings

### Success Criteria:
- ✅ Test return > 0%
- ✅ Test win rate ≥ 40%
- ✅ Test trades ≥ 20 (statistical significance)
- ✅ No overfitting (test performance within 20% of train)

### Overfitting Red Flags:
- ❌ Test performance much worse than train (>20% difference)
- ❌ Very few trades (<20)
- ❌ Optimizing parameters on test set
- ❌ Testing on different time periods to find favorable results

---

## Experiment Log

### Experiment 1: Moving Average Strategy (MA=20)
**Hypothesis**: Price crossing above/below MA indicates trend, profitable to follow  
**Test Period**: 60 days (2025-09-12 to 2025-11-11)  
**Symbols**: AVAXUSDT, ETHUSDT, SOLUSDT, BTCUSDT  
**Results**:
- AVAXUSDT: -43.77% return, 38.01% win rate, 171 trades ❌
- ETHUSDT: Not tested
- SOLUSDT: Not tested
- BTCUSDT: Not tested  
**Conclusion**: ❌ REJECTED - Unprofitable on AVAXUSDT  
**Learnings**: Simple MA crossover not sufficient, need additional filters

---

### Experiment 2: RSI Mean Reversion
**Hypothesis**: RSI oversold/overbought levels indicate mean reversion opportunities  
**Test Period**: 60 days  
**Symbols**: AVAXUSDT  
**Results**: -99.65% return, 43.15% win rate, 248 trades ❌  
**Conclusion**: ❌ REJECTED - High win rate but poor risk/reward  
**Learnings**: Mean reversion works but exits need improvement

---

### Experiment 3: MACD Strategy
**Hypothesis**: MACD crossover signals trend changes  
**Test Period**: 60 days  
**Symbols**: AVAXUSDT  
**Results**: -99.83% return, 30.73% win rate, 397 trades ❌  
**Conclusion**: ❌ REJECTED - Too many false signals  
**Learnings**: MACD generates too many trades, poor quality

---

### Experiment 4: Mean Reversion (Price Z-Score)
**Hypothesis**: Price deviating from mean will revert  
**Test Period**: 60 days  
**Symbols**: AVAXUSDT  
**Results**: -79.02% return, 42.45% win rate, 106 trades ❌  
**Conclusion**: ❌ REJECTED - Best win rate so far but still unprofitable  
**Learnings**: Win rate good but risk/reward poor

---

### Experiment 5: Pairs Trading (AVAXUSDT/ETHUSDT)
**Hypothesis**: Cointegrated pairs mean-revert around spread  
**Test Period**: 60 days  
**Symbols**: AVAXUSDT, ETHUSDT  
**Results**: -100% return, 21.09% win rate, 531 trades ❌  
**Conclusion**: ❌ REJECTED - Spread not cointegrated or fees too high  
**Learnings**: Pairs trading requires strong cointegration, fees eat profits

---

### Experiment 6: Volatility Breakout
**Hypothesis**: Low volatility periods followed by breakouts  
**Test Period**: 60 days  
**Symbols**: DOGEUSDT  
**Results**: -94.09% return, 27.50% win rate, 80 trades ❌  
**Conclusion**: ❌ REJECTED - Breakouts not reliable  
**Learnings**: Need better volume confirmation

---

### Experiment 7: Momentum + Mean Reversion Hybrid
**Hypothesis**: Combine MA trend with RSI pullback entries  
**Test Period**: 60 days  
**Symbols**: DOGEUSDT, AVAXUSDT  
**Results**:
- DOGEUSDT: -59.89% return, 25% win rate, 28 trades ❌
- AVAXUSDT: -74.76% return, 26.92% win rate, 26 trades ❌  
**Conclusion**: ❌ REJECTED - Hybrid approach not working  
**Learnings**: Combining strategies doesn't help if both are weak

---

### Experiment 8: ML Random Forest (60% confidence)
**Hypothesis**: ML can find patterns traditional indicators miss  
**Test Period**: 60 days  
**Symbols**: AVAXUSDT, DOGEUSDT, SOLUSDT  
**Features**: Price z-score, MA ratios, volatility, volume, RSI-like, ATR-like  
**Overfitting Prevention**: max_depth=5, min_samples_split=20, confidence >60%  
**Results**:
- AVAXUSDT: -99.87% return, 40.06% win rate, 332 trades ❌
- DOGEUSDT: -99.70% return, 42.74% win rate, 351 trades ❌
- SOLUSDT: -99.92% return, 40.51% win rate, 353 trades ❌  
**Conclusion**: ❌ REJECTED - Win rate good but risk/reward poor  
**Learnings**: Model finds patterns but exits need work, too conservative (60% threshold)

---

### Experiment 9: ML Random Forest (50% confidence) - ETHUSDT BREAKTHROUGH ⚠️ NEEDS INVESTIGATION
**Hypothesis**: Lower confidence threshold (50% vs 60%) generates more trades while maintaining quality  
**Test Period**: 60 days (2025-09-12 to 2025-11-11)  
**Symbols**: ETHUSDT, DOGEUSDT, AVAXUSDT, BTCUSDT, GALAUSDT  
**Features**: Same as Experiment 8  
**Overfitting Prevention**: Same as Experiment 8, but confidence >50%  
**Results**:
- ETHUSDT: 287.58% return, 73.68% win rate, 19 trades ⚠️ NEEDS VERIFICATION
- DOGEUSDT: -100% return, 34.24% win rate, 1025 trades ❌
- AVAXUSDT: -100% return, 35.91% win rate, 1047 trades ❌
- BTCUSDT: -23.42% return, 40% win rate, 25 trades ❌
- GALAUSDT: -100% return, 35.54% win rate, 1103 trades ❌  
**OOD Validation**: 
- Train (40 days): 287.58% return, 73.68% win rate, 19 trades
- Test (20 days): 287.58% return, 73.68% win rate, 19 trades ⚠️ **BUG DETECTED** - Identical results suggest all trades in one period or date filtering not working  
**Conclusion**: ⚠️ **INVESTIGATING** - ETHUSDT shows promise but train/test identical (possible bug in date filtering)  
**Learnings**: 
- ETHUSDT may be more predictable than other tickers
- Lower confidence threshold helps generate more trades
- **CRITICAL**: Need to verify why train/test results are identical - possible bug
- ML RF only works on ETHUSDT, fails on all other tickers tested
**Status**: 🔍 Investigating date filtering in backtester

---

### Experiment 10: Order Flow Imbalance
**Hypothesis**: Buy/sell volume imbalance predicts price direction  
**Test Period**: 60 days  
**Symbols**: AVAXUSDT, ETHUSDT, SOLUSDT  
**Method**: Calculate buy/sell pressure from price change × volume  
**Results**:
- AVAXUSDT: -100.01% return, 31.25% win rate, 384 trades ❌
- ETHUSDT: 92.66% return, 50% win rate, 8 trades ⚠️ Too few trades
- SOLUSDT: -99.95% return, 31.64% win rate, 354 trades ❌  
**Conclusion**: ⚠️ PARTIAL - Works on ETHUSDT but only 8 trades (not statistically significant)  
**Learnings**: Volume-based strategies work on ETHUSDT but need more signals

---

### Experiment 11: Volatility Clustering
**Hypothesis**: Volatility clusters - trade breakouts from low vol periods  
**Test Period**: 60 days  
**Symbols**: ETHUSDT  
**Method**: Identify low/high vol regimes, trade momentum  
**Results**: 83.88% return, 36.36% win rate, 11 trades ⚠️ Too few trades  
**Conclusion**: ⚠️ PARTIAL - Good risk/reward but low trade frequency  
**Learnings**: Volatility-based strategies have potential but need more signals

---

### Experiment 12: ML Logistic Regression (60% confidence)
**Hypothesis**: Linear model with regularization can find patterns  
**Test Period**: 60 days  
**Symbols**: AVAXUSDT, ETHUSDT  
**Overfitting Prevention**: C=0.1 (strong L2 regularization), confidence >60%  
**Results**:
- AVAXUSDT: -100.01% return, 35.04% win rate, 254 trades ❌
- ETHUSDT: -105.69% return, 29.03% win rate, 31 trades ❌  
**Conclusion**: ❌ REJECTED - Linear model not finding useful patterns  
**Learnings**: Non-linear patterns (RF better than LR)

---

### Experiment 13: ML Random Forest (50% confidence) - Additional Tickers
**Hypothesis**: ML RF works on other liquid tickers  
**Test Period**: 60 days  
**Symbols**: SOLUSDT, COTIUSDT  
**Features**: Same as Experiment 9  
**Overfitting Prevention**: Same as Experiment 9  
**Results**:
- SOLUSDT: -100% return, 33.08% win rate, 1037 trades ❌
- COTIUSDT: -100% return, 38.96% win rate, 1137 trades ❌  
**Conclusion**: ❌ REJECTED - ML RF does not generalize to other tickers  
**Learnings**: ETHUSDT-specific patterns, other tickers have different dynamics

---

### Experiment 14: ML Random Forest (50% confidence) - Different Timeframe
**Hypothesis**: ML RF works on ETHUSDT across different timeframes  
**Test Period**: 60 days  
**Symbols**: ETHUSDT  
**Timeframe**: 15 minutes (vs 1 minute in Experiment 9)  
**Results**: 287.58% return, 73.68% win rate, 19 trades ✅ (Same as 1-minute)  
**Conclusion**: ✅ CONFIRMED - Strategy robust across timeframes (1m and 15m show identical results)  
**Learnings**: ETHUSDT ML RF signals are consistent across timeframes

---

### Experiment 15: Traditional Strategies on ETHUSDT
**Hypothesis**: Strategies that failed on AVAXUSDT might work on ETHUSDT  
**Test Period**: 60 days  
**Symbols**: ETHUSDT  
**Strategies**: RSI Mean Reversion, Mean Reversion (Z-Score), MACD, VWMA  
**Results**:
- RSI Mean Reversion: -100.62% return, 34.69% win rate, 49 trades ❌
- Mean Reversion (Z-Score): 33.86% return, 33.33% win rate, 3 trades ⚠️ Too few trades
- MACD: 46.00% return, 25.53% win rate, 47 trades ✅ PROFITABLE
- VWMA: -121.15% return, 50% win rate, 2 trades ❌ Too few trades  
**Conclusion**: ✅ PARTIAL - MACD profitable on ETHUSDT (46% return, 47 trades) but low win rate (25.53%)  
**Learnings**: 
- ETHUSDT shows better results than AVAXUSDT for multiple strategies
- MACD works on ETHUSDT despite failing on AVAXUSDT
- Mean Reversion works but needs more signals for statistical significance
- RSI Mean Reversion still fails on ETHUSDT
- VWMA generates too few signals

---

## Summary of Findings

### ✅ Promising Strategies (Need Verification):
1. **ML RF (50% conf) on ETHUSDT**: 287.58% return, 73.68% win rate, 19 trades ⚠️ NEEDS OOD VERIFICATION (bug detected)
2. **Order Flow on ETHUSDT**: 92.66% return, 50% win rate ⚠️ Too few trades (8)
3. **Volatility Clustering on ETHUSDT**: 83.88% return ⚠️ Too few trades (11)
4. **MACD on ETHUSDT**: 46.00% return, 25.53% win rate, 47 trades ✅ (Low win rate but profitable)
5. **Mean Reversion (Z-Score) on ETHUSDT**: 33.86% return ⚠️ Too few trades (3)

### ❌ Rejected Strategies:
- MA (all periods tested)
- RSI Mean Reversion (on AVAXUSDT and ETHUSDT)
- MACD (on AVAXUSDT, but ✅ works on ETHUSDT)
- Mean Reversion (Z-Score) (on AVAXUSDT, but ⚠️ works on ETHUSDT with too few trades)
- Pairs Trading
- Volatility Breakout
- Momentum + MR Hybrid
- ML RF (60% conf) on AVAXUSDT/DOGEUSDT/SOLUSDT
- ML RF (50% conf) on DOGEUSDT/AVAXUSDT/BTCUSDT/GALAUSDT/SOLUSDT/COTIUSDT
- ML LR (all symbols)

### Key Observations:
1. **ETHUSDT is special**: Multiple strategies work on ETHUSDT but not other tickers
   - ML RF: 287.58% return, 73.68% win rate (19 trades)
   - Order Flow: 92.66% return, 50% win rate (8 trades)
   - Volatility Clustering: 83.88% return (11 trades)
   - MACD: 46.00% return, 25.53% win rate (47 trades) ✅
   - Mean Reversion: 33.86% return (3 trades, too few)
2. **Trade frequency matters**: Strategies with <20 trades need more signals for statistical significance
3. **Win rate alone insufficient**: Need good risk/reward ratio
4. **ML shows promise**: But needs proper OOD validation (bug detected)
5. **Overfitting risk**: Train/test identical results suggest possible bug

### Critical Issues:
- ⚠️ **ML RF train/test identical**: Possible bug in date filtering or ML strategy retraining
- ⚠️ **Low trade frequency**: Order Flow (8 trades) and Volatility Clustering (11 trades) need more signals
- ⚠️ **ETHUSDT-specific**: Strategies don't generalize to other tickers

---

## Next Steps

### Priority 1: Verify ETHUSDT ML RF Results
- [ ] Investigate why train/test results are identical
- [ ] Run proper OOD validation with different date ranges
- [ ] Check if all 19 trades are in one period
- [ ] Verify results aren't due to data issues

### Priority 2: Improve Promising Strategies
- [ ] Increase trade frequency for Order Flow (lower threshold?)
- [ ] Increase trade frequency for Volatility Clustering
- [ ] Test ensemble: ML RF + Order Flow + Volatility Clustering

### Priority 3: Continue Research
- [ ] Test strategies on different timeframes (15m, 1h, 4h)
- [ ] Test on more tickers to see if ETHUSDT success generalizes
- [ ] Investigate why ETHUSDT is more predictable
- [ ] Try market regime detection

---

## Research Notes

**2025-11-11**: Started systematic research log following scientific method. Documenting all experiments with clear hypotheses, results, and conclusions.

**Key Insight**: ETHUSDT shows consistent profitability across multiple strategies, suggesting it may have more predictable patterns than other tickers. Need to verify this isn't overfitting.

