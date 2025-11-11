# Research Workflow - Scientific Method

**Goal**: Find profitable trading strategy for AVAXUSDT  
**Method**: Systematic hypothesis testing with strict overfitting prevention

---

## Scientific Method Process

### 1. Hypothesis Formation
- **Question**: "Will strategy X be profitable on AVAXUSDT?"
- **Theory**: Why should this strategy work? (e.g., mean reversion, trend following, arbitrage)
- **Prediction**: Expected performance (win rate, return, risk)

### 2. Experiment Design
- **Data**: Recent 60 days (2025-09-12 to 2025-11-11)
- **Train/Test Split**: 60% train (36 days), 40% test (24 days)
- **Metrics**: Return, Win Rate, Sharpe Ratio, Max Drawdown
- **Success Criteria**: 
  - Test return > 0%
  - Test win rate ≥ 40%
  - No overfitting (test within 20% of train)

### 3. Implementation
- Implement strategy in backtester
- Use realistic fees (11 bps) and slippage (2 bps)
- Test on AVAXUSDT

### 4. Testing
```bash
# Quick test (full period)
python3 tools/backtester.py --symbols AVAXUSDT --days 60 --strategy <strategy> --no-plot

# Train/test split (if promising)
python3 tools/backtest_avax_ood.py --train-days 36 --test-days 24 --strategy <strategy>
```

### 5. Analysis
- **If Promising** (test return > 0%, win rate ≥ 40%):
  - Parameter tuning with train/test split
  - Test robustness (different time periods)
  - Document results
  
- **If Not Promising**:
  - Document why it failed
  - Move to next strategy
  - Learn from failure

### 6. Iteration
- Keep testing new strategies
- Build on learnings from previous tests
- Consider: pairs trading, statistical arbitrage, volatility strategies, etc.

---

## Overfitting Prevention

### Rules:
1. **Never optimize on test data** - Test data is only for final validation
2. **Train/test split** - Always use separate train/test periods
3. **No cherry-picking** - Use recent 60 days consistently
4. **Parameter bounds** - Limit parameter search space
5. **Minimum trades** - Require ≥20 trades for statistical significance
6. **Out-of-sample validation** - Test on completely unseen data if possible

### Red Flags:
- ❌ Test performance much worse than train (>20% difference)
- ❌ Very few trades (<20)
- ❌ Optimizing parameters on test set
- ❌ Testing on different time periods to find favorable results

---

## Strategy Ideas to Test

### 1. Pairs Trading ⭐⭐⭐⭐⭐
**Theory**: Cointegrated pairs mean-revert around their spread
**Implementation**: 
- Find cointegrated pair (AVAXUSDT vs ETHUSDT/SOLUSDT/BTCUSDT)
- Trade spread when it deviates from mean
- Exit when spread returns to mean

### 2. Statistical Arbitrage ⭐⭐⭐⭐
**Theory**: Price relationships between correlated assets
**Implementation**:
- Calculate z-score of price ratio
- Trade when z-score > 2 (short) or < -2 (long)
- Exit when z-score returns to 0

### 3. Volatility Breakout ⭐⭐⭐
**Theory**: Low volatility periods followed by high volatility moves
**Implementation**:
- Identify low volatility periods (Bollinger Band squeeze)
- Enter on breakout with volume confirmation
- Use ATR-based stops

### 4. Momentum + Mean Reversion Hybrid ⭐⭐⭐⭐
**Theory**: Combine trend following with mean reversion
**Implementation**:
- Use MA for trend direction
- Enter on pullbacks (RSI oversold/overbought)
- Exit on trend reversal or target

### 5. Order Flow Imbalance ⭐⭐⭐
**Theory**: Buy/sell volume imbalance predicts price moves
**Implementation**:
- Track buy vs sell volume
- Enter when imbalance exceeds threshold
- Exit on reversal or target

### 6. Market Regime Detection ⭐⭐⭐⭐⭐
**Theory**: Different strategies work in different market conditions
**Implementation**:
- Detect trending vs ranging markets
- Use trend-following in trending, mean reversion in ranging
- Switch strategies based on regime

---

## Current Research Status

**Strategies Tested**: 9 (all unprofitable)
- MA (15, 20, 25, 30)
- RSI MR, MACD, HTF Pullback, MA Cross
- Mean Reversion, VWMA, ATR Dynamic

**Next Strategy**: Pairs Trading (cointegration-based)

---

## Research Log

### Strategy 1: Pairs Trading (AVAXUSDT/ETHUSDT) ❌ FAILED
- **Hypothesis**: AVAXUSDT and ETHUSDT are cointegrated, spread mean-reverts
- **Test**: 60 days, z-score entry >2, exit at 0
- **Results**: 
  - Return: -100%
  - Win Rate: 21.09%
  - Trades: 531
  - **Conclusion**: Not profitable - spread may not be cointegrated or fees/slippage too high
- **Next**: Test AVAXUSDT/SOLUSDT pair, or move to different strategy

### Strategy 2: TBD

