# Backtesting Alpha Improvements Guide

## Overview
This guide explains how to backtest the Phase 1 alpha improvements with proper out-of-sample (OOS) testing to avoid overfitting.

## Quick Start

### Run the Alpha Improvements Backtest
```bash
cd mft
python3 tools/backtest_alpha_improvements.py
```

### Customize Parameters
```bash
# Test with specific symbols
python3 tools/backtest_alpha_improvements.py --symbols ETHUSDT,SOLUSDT,BTCUSDT

# Adjust train/test split (default: 60 days train, 30 days test)
python3 tools/backtest_alpha_improvements.py --train-days 90 --test-days 30
```

## What It Tests

The backtester compares **5 parameter combinations**:

1. **Baseline (Old)** - Original conservative parameters
   - MA slope required
   - Higher thresholds
   - MIN_TREND_STRENGTH = 0.0005

2. **Phase 1: No Slope + Reduced Thresholds** ⭐ (Current Implementation)
   - No MA slope requirement
   - Reduced thresholds (0.2%/0.05%/0.03%)
   - MIN_TREND_STRENGTH = 0.0002

3. **Phase 1 + MA15** - Same as Phase 1 but with shorter MA period
4. **Phase 1 + Lower Trend** - Even more lenient trend strength
5. **Phase 1 + Aggressive** - More aggressive thresholds

## Key Metrics Tracked

- **Trades/Day** - Primary metric (target: ≥1.0)
- **Total Trades** - Total number of completed trades
- **Win Rate** - Percentage of profitable trades
- **Total Return** - Cumulative return percentage
- **Sharpe Ratio** - Risk-adjusted returns

## Overfitting Prevention

The backtester uses a **train/test split**:
- **Training Period**: Used to optimize parameters (default: 60 days)
- **Test Period (OOS)**: Used to validate performance (default: 30 days)

**Key Check**: Compare trades/day between train and test. If train >> test, we're overfitting.

## Interpreting Results

### Good Results ✅
- OOS trades/day ≥ 1.0
- Train/test trades/day difference < 0.5 (not overfitting)
- Positive OOS returns
- Reasonable win rate (>40%)

### Warning Signs ⚠️
- Train trades/day >> Test trades/day (overfitting)
- Negative OOS returns despite high train returns
- Win rate < 30% (too many false signals)

### Example Output
```
📊 FINAL RANKINGS (by Trades/Day on Out-of-Sample Data)
================================================================================

1. Phase 1: No Slope + Reduced Thresholds
   OOS Trades/Day: 1.23 ⭐
   OOS Total Trades: 37
   OOS Win Rate: 45.9%
   OOS Return: 2.34%
   Train Trades/Day: 1.45
   Overfitting Check: 0.22 trades/day diff ✅

✅ RECOMMENDED CONFIGURATION
================================================================================
Parameter Set: Phase 1: No Slope + Reduced Thresholds
OOS Trades/Day: 1.23
```

## Standard Backtest (Single Configuration)

For testing a single configuration on historical data:

```bash
python3 tools/backtester.py \
  --symbols ETHUSDT,SOLUSDT \
  --days 30 \
  --timeframe 15 \
  --ma 20 \
  --strategy ma
```

## Next Steps After Backtesting

1. **If OOS trades/day ≥ 1.0**: ✅ Deploy Phase 1 changes to live
2. **If OOS trades/day < 1.0**: Consider Phase 2 improvements:
   - Add more symbols (BTCUSDT, AVAXUSDT)
   - Reduce MA period further (15 → 12)
   - Add RSI signals

3. **Monitor Live Performance**: After deployment, compare live trades/day to backtest predictions

## Notes

- Backtests use realistic fees (5 bps round-trip) and slippage (2 bps)
- All tests use 5x leverage (matching live config)
- Results are saved to database (unless `--no-db` flag used)
- Visualizations available in `results/` directory

## Troubleshooting

**Issue**: "No data for symbol"
- **Fix**: Check API keys in `.env` file
- **Fix**: Reduce `--days` if hitting rate limits

**Issue**: "No trades executed"
- **Fix**: Check if parameters are too conservative
- **Fix**: Verify date range has sufficient market activity

**Issue**: High overfitting (train >> test)
- **Fix**: Use more conservative parameters
- **Fix**: Increase test period size

