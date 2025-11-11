# Backtest Results Directory

This directory contains backtest results and visualizations.

## Directory Structure

- `backtest_results/` - CSV files with detailed backtest metrics
- `live_results/` - Live trading performance data
- `*_backtest_analysis.png` - Visualization charts for each symbol

## Current Status

**Phase 1 Alpha Improvements Active** (as of latest update):
- MA slope requirement: REMOVED
- MIN_TREND_STRENGTH: 0.0002 (reduced from 0.0005)
- Thresholds: 0.2%/0.05%/0.03% (reduced for more signals)
- Target: ≥1 trade/day

## Running New Backtests

To generate fresh results:

```bash
# Test Phase 1 improvements with OOS validation
python3 tools/backtest_alpha_improvements.py

# Standard backtest
python3 tools/backtester.py --symbols ETHUSDT,SOLUSDT --days 30
```

## Notes

- Old results have been cleared to avoid confusion
- All new backtests use Phase 1 improved parameters
- Results are automatically saved to database (unless `--no-db` flag used)
