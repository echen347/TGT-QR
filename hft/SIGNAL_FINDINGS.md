# Signal Analysis Findings

**Date:** November 10, 2025  
**Analysis:** Signal quality metrics from Lasso model

---

## Key Discovery: IC = 0.1638 ✅

**This is a STRONG signal!**

- IC > 0.05 = Strong signal
- IC 0.02-0.05 = Moderate signal  
- IC < 0.02 = Weak signal

**Our IC: 0.1638** - This is in the strong category, much better than the R² (0.0265) suggested!

---

## Signal Characteristics

### Distribution
- **Mean:** -0.528 (all signals negative)
- **Std:** 0.084
- **Range:** -0.753 to -0.069
- **Skewness:** 0.44 (slightly right-skewed)

### Interpretation
**Important:** All signals are negative, but IC is positive (0.1638). This means:
- Higher (less negative) signals → Higher returns
- Lower (more negative) signals → Lower returns
- **Use long-short strategy**, not long-only!

### Signal-Target Correlation
- **Correlation:** 0.1638
- **IC:** 0.1638
- **Interpretation:** Strong predictive power

---

## Strategy Performance

### Long-Only Strategy (Top 25% signals)
- **Mean Return:** -0.436 (negative - don't use!)
- **Sharpe Ratio:** -0.88 (bad)
- **Win Rate:** 28.60% (below breakeven)
- **Max Drawdown:** -1667 (extreme)

**Conclusion:** Long-only doesn't work with these signals.

### Long-Short Strategy (Top/Bottom 25%) ✅
- **Mean Return:** +0.105 (positive!)
- **Sharpe Ratio:** 0.14 (needs improvement but positive)
- **Win Rate:** 55.42% (above breakeven)
- **Number of Trades:** 7,644

**Conclusion:** Long-short strategy works! This is the way forward.

---

## Signal Decay Analysis

### Autocorrelation
- **Lag 1:** 0.78 (high persistence)
- **Lag 2:** 0.68
- **Lag 5:** 0.47 (still significant)
- **Lag 10:** 0.25 (decaying but present)

**Insight:** Signals persist for multiple periods, good for HFT.

### Predictive Power Decay
- **Lag 0:** IC = 0.1638
- **Lag 1:** IC = 0.1342 (still strong)
- **Lag 2:** IC = 0.1196 (still strong)
- **Lag 3:** IC = 0.0869 (moderate)
- **Lag 4:** IC = 0.0755 (moderate)

**Insight:** Predictive power decays but remains positive for several periods.

---

## Turnover Analysis

### Statistics
- **Turnover Rate:** 8.05% (low!)
- **Signal Changes:** 1,230 out of 15,284 periods
- **Average Holding Period:** 12.4 periods

### Signal Persistence
- **Mean Duration:** 2.5 periods
- **Median Duration:** 1.0 period
- **Max Duration:** 26 periods

**Insight:** 
- Low turnover = manageable transaction costs
- Signals flip frequently (appropriate for HFT)
- Average holding period of 12.4 periods is reasonable

---

## Conditional Returns by Signal Strength

| Bin | Signal Range | Return | Sharpe | N |
|-----|--------------|--------|--------|---|
| 1 (Lowest) | [-0.753, -0.603) | -0.653 | -1.37 | 3,057 |
| 2 | [-0.603, -0.556) | -0.577 | -1.17 | 3,057 |
| 3 | [-0.556, -0.511) | -0.538 | -1.08 | 3,057 |
| 4 | [-0.511, -0.458) | -0.491 | -0.98 | 3,057 |
| 5 (Highest) | [-0.458, -0.069) | -0.425 | -0.86 | 3,056 |

**Key Insight:** 
- Higher signals (less negative) → Better returns (less negative)
- All returns are negative (market going down in test period?)
- **Long-short strategy essential** - go long top quintile, short bottom quintile

---

## Recommendations

### 1. Use Long-Short Strategy ✅
- Long top 25% of signals
- Short bottom 25% of signals
- Win rate: 55.42%
- Sharpe: 0.14 (needs improvement but positive)

### 2. Signal Interpretation
- **Don't use signals as absolute predictions**
- **Use signals as relative rankings**
- Higher signal = relatively better (less negative) return
- Lower signal = relatively worse (more negative) return

### 3. Holding Period
- Average: 12.4 periods
- Signals persist for multiple periods
- Can hold positions for several periods

### 4. Transaction Costs
- Turnover: 8.05% (low)
- Manageable transaction costs
- Need to estimate actual costs to get net Sharpe

### 5. Next Steps
- **Improve Sharpe from 0.14 → 1.0+**
  - Better features
  - Ensemble models
  - Regime-dependent models
- **Validate on more data**
  - Different time periods
  - Different market conditions
- **Transaction cost analysis**
  - Estimate realistic costs
  - Calculate net Sharpe

---

## Why IC is Better Metric Than R²

**R² = 0.0265** suggests weak predictive power  
**IC = 0.1638** suggests strong predictive power

**Why the difference?**
- R² measures explained variance (hard to get high R² in noisy markets)
- IC measures rank correlation (better for trading signals)
- **IC is what matters for trading!**

**Conclusion:** We have a strong signal (IC = 0.16), focus on:
1. Improving Sharpe (better features/models)
2. Validating robustness (more data/periods)
3. Transaction cost analysis (net returns)

---

**Bottom Line:** 
- ✅ Strong signal (IC = 0.16)
- ✅ Long-short strategy works (55% win rate)
- ⚠️ Need to improve Sharpe (0.14 → 1.0+)
- ✅ Low turnover (manageable costs)
- ✅ Signals persist (good for HFT)

