# Adaptive System Explained

## How It Works

### Signal Generation Logic

**Signals ALWAYS generate** if market conditions meet the criteria. The adaptive system doesn't stop signal generation - it adjusts the **thresholds** for what counts as a signal.

### When Performance is Poor (Unprofitable)

**Current Behavior:**
- **TIGHTENS filters** → Fewer signals, but higher quality
- Requires stronger trends (1.3x stricter)
- Requires larger price deviations (1.15x stricter)
- Reduces position size (0.85x)

**Why this makes sense:**
- When losing money, we want to be MORE selective
- Fewer trades = less exposure to losing strategy
- Stricter criteria = only take the best opportunities

**Example:**
- Baseline: Signal when price deviates 0.2% from MA
- When unprofitable: Signal when price deviates 0.23% from MA (1.15x)
- Result: Fewer signals, but they're stronger setups

### When Performance is Good (Profitable)

**Current Behavior:**
- **LOOSENS filters slightly** → More signals, but still selective
- Allows slightly weaker trends (0.92x)
- Catches slightly smaller deviations (0.97x)
- Maintains position size (1.0x)

**Why this makes sense:**
- When making money, strategy is working
- Can afford to take slightly more opportunities
- But still selective (not going crazy)

**Example:**
- Baseline: Signal when price deviates 0.2% from MA
- When profitable: Signal when price deviates 0.194% from MA (0.97x)
- Result: Slightly more signals, still quality setups

## Anti-Overfitting Safeguards

### 1. Minimum Data Requirement
- **Requires 10+ trades** before adapting (was 5)
- Prevents adapting on small sample noise
- Configurable via `ADAPTIVE_MIN_TRADES`

### 2. Bounds on Adjustments
- **Trend strength**: 0.7x to 1.5x (max 30% change)
- **Thresholds**: 0.7x to 1.5x (max 30% change)
- **Position size**: 0.7x to 1.2x (max 20% change)
- Prevents extreme swings that could overfit

### 3. Stricter Performance Thresholds
- **Underperforming**: Win rate < 40% OR profit factor < 0.8 (was 45%/1.0)
- **Performing well**: Win rate > 60% AND profit factor > 1.8 (was 55%/1.5)
- Only adapts when performance is SIGNIFICANTLY different
- Small variations are ignored (noise)

### 4. Longer Windows
- **Window**: 14 days (was 7) - smoother, less reactive
- **Update interval**: 48 hours (was 24) - less churn
- Reduces sensitivity to short-term noise

### 5. Conservative Adjustments
- Maximum adjustment: 30% (not 50%+)
- Gradual changes, not dramatic swings
- Prevents over-optimization to recent patterns

## Testing for Overfitting

### How to Verify

1. **Out-of-Sample Testing**
   ```bash
   python3 tools/backtest_adaptive.py --days 60 --symbols ETHUSDT,SOLUSDT
   ```
   - Test on data NOT used for development
   - Compare train vs test performance
   - If test << train → overfitting

2. **Walk-Forward Analysis**
   - Test on rolling windows
   - Each window is independent
   - If performance degrades → overfitting

3. **Live Performance Monitoring**
   ```bash
   python3 tools/monitor_performance.py --days 7
   ```
   - Compare live vs backtest
   - If live << backtest → overfitting

### Red Flags

- ✅ **Good**: Test performance ≈ Train performance
- ⚠️ **Warning**: Test performance < Train performance by 20%+
- ❌ **Bad**: Test performance < 0% while Train > 50%

## Current Status

**Adaptive System**: ✅ Enabled
- Window: 14 days
- Update: Every 48 hours
- Min trades: 10
- Bounds: ±30% max

**Recommendation**: 
- Monitor live performance for 2-4 weeks
- Compare to backtest predictions
- If overfitting detected, disable adaptive system (`ENABLE_ADAPTIVE_PARAMS = False`)

## Disabling Adaptive System

If you're concerned about overfitting, you can disable it:

```python
# In config/config.py
ENABLE_ADAPTIVE_PARAMS = False
```

This will use baseline parameters (Phase 1) with no adaptation.

