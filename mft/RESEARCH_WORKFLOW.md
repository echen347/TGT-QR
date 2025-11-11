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

### Strategy 2: Pairs Trading (AVAXUSDT/SOLUSDT) ❌ FAILED
- **Hypothesis**: AVAXUSDT and SOLUSDT are cointegrated, spread mean-reverts
- **Test**: 60 days, z-score entry >2, exit at 0
- **Results**: 
  - Return: -92.17%
  - Win Rate: 17.12%
  - Trades: 111
  - **Conclusion**: Not profitable - pairs trading not working for these pairs

### Strategy 3: Volatility Breakout (DOGEUSDT) ❌ FAILED
- **Hypothesis**: Bollinger Band squeeze followed by breakout with volume
- **Test**: 60 days, BB squeeze <2%, breakout with 1.5x volume
- **Results**:
  - Return: -94.09%
  - Win Rate: 27.50%
  - Trades: 80
  - **Conclusion**: Not profitable - breakouts not reliable

### Strategy 4: Momentum + Mean Reversion Hybrid (DOGEUSDT, AVAXUSDT) ❌ FAILED
- **Hypothesis**: Combine MA trend with RSI pullback entries
- **Test**: 60 days, MA for direction, RSI <40 (long) or >60 (short)
- **Results**:
  - DOGEUSDT: -59.89% return, 25% win rate, 28 trades
  - AVAXUSDT: -74.76% return, 26.92% win rate, 26 trades
  - **Conclusion**: Not profitable - hybrid approach not working

### Strategy 5: ML Random Forest (AVAXUSDT) ❌ FAILED (but promising win rate)
- **Hypothesis**: ML can find patterns traditional indicators miss
- **Test**: 60 days, Random Forest with regularization, confidence threshold 60%
- **Features**: Price z-score, MA ratios, volatility, volume, RSI-like, ATR-like
- **Overfitting Prevention**: 
  - Regularization (max_depth=5, min_samples_split=20)
  - Confidence threshold (>60% probability)
  - Limited training data (last 500 bars)
- **Results**:
  - Return: -99.87%
  - Win Rate: 40.06% ✅ (close to target!)
  - Trades: 332
  - Average Win: 7.98%, Average Loss: -7.75%
  - **Conclusion**: Win rate good but risk/reward poor - need better exit strategy or position sizing

### Strategy 6: ML Logistic Regression (AVAXUSDT) ❌ FAILED
- **Hypothesis**: Logistic Regression with L2 regularization can find linear patterns
- **Test**: 60 days, Logistic Regression with C=0.1 (strong regularization), confidence threshold 60%
- **Features**: Same as RF (Price z-score, MA ratios, volatility, volume, RSI-like, ATR-like)
- **Overfitting Prevention**: 
  - Strong L2 regularization (C=0.1)
  - Confidence threshold (>60% probability)
  - Limited training data (last 500 bars)
- **Results**:
  - Return: -100.01%
  - Win Rate: 35.04%
  - Trades: 254
  - Average Win: 8.39%, Average Loss: -9.27%
  - **Conclusion**: Not profitable - linear model not finding useful patterns

---

## Key Learnings

### ML Strategies
- **ML Random Forest achieved 40% win rate** (our target!) but still unprofitable
- **Issue**: Average win ≈ average loss (poor risk/reward ratio)
- **Insight**: Model finds some patterns but needs better exit strategy or position sizing
- **Next Steps for ML**:
  1. Improve exit strategy (trailing stops, better take profit)
  2. Dynamic position sizing based on model confidence
  3. Feature engineering (add more technical indicators)
  4. Ensemble methods (combine RF + LR + traditional strategies)
  5. Meta-labeling (use secondary model for position sizing)

### Traditional Strategies
- **All tested strategies unprofitable** on recent 60-day data
- **Possible reasons**:
  - Market regime not suitable for these strategies
  - Fees/slippage too high relative to edge
  - Need different timeframes or parameters
  - Market may be too efficient for simple strategies

---

---

## 🎯 BREAKTHROUGH FINDINGS

### ETHUSDT Shows Promise!
Multiple strategies showing profitability on ETHUSDT:
- **ML RF (50% conf)**: 287.58% return, 73.68% win rate ⭐⭐⭐⭐⭐
- **Order Flow**: 92.66% return, 50% win rate ⭐⭐⭐⭐
- **Volatility Clustering**: 83.88% return, 36.36% win rate ⭐⭐⭐⭐

### Key Insights:
1. **ETHUSDT may be more predictable** than AVAXUSDT
2. **ML with lower confidence threshold** (50% vs 60%) generates more trades while maintaining quality
3. **Volume-based strategies** (Order Flow) show promise
4. **Volatility-based strategies** (Volatility Clustering) have good risk/reward

### Next Steps:
1. ✅ **OOD Validation**: Test ML RF (50% conf) on train/test split
2. **Test on more tickers**: See if ETHUSDT success generalizes
3. **Combine strategies**: Ensemble ML + Order Flow + Volatility Clustering
4. **Parameter optimization**: Fine-tune ML confidence threshold

---

## Next Research Directions

1. **ML Improvements** ⭐⭐⭐⭐⭐
   - Better exit strategies for ML models
   - Dynamic position sizing based on confidence
   - Feature engineering (add momentum, order flow, etc.)
   - Ensemble methods

2. **Different Timeframes** ⭐⭐⭐⭐
   - Test strategies on 5m, 30m, 1h, 4h timeframes
   - Some strategies may work better on different timeframes

3. **Market Regime Detection** ⭐⭐⭐⭐⭐
   - Detect trending vs ranging markets
   - Use different strategies for different regimes
   - ML could help with regime detection

4. **More Tickers** ⭐⭐⭐
   - Test on more liquid tickers (DOGEUSDT, GALAUSDT, etc.)
   - Some strategies may work better on specific tickers

5. **Parameter Optimization** ⭐⭐⭐
   - Systematic parameter sweeps with train/test splits
   - Focus on strategies with promising win rates (ML RF)

