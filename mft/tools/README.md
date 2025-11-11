# Trading Analysis Tools

This directory contains backtesting, monitoring, and strategy optimization tools.

## Primary Tools (Active)

### `backtester.py` ⭐
Comprehensive backtesting framework for the moving average strategy.
- Downloads historical data from Bybit API (with database fallback)
- Simulates strategy execution without real trades
- Calculates performance metrics (Sharpe ratio, max drawdown, win rate)
- Generates performance charts
- Supports Phase 1 alpha improvements

**Usage:**
```bash
python3 tools/backtester.py --symbols ETHUSDT,SOLUSDT --days 30
```

### `backtest_alpha_improvements.py` ⭐
Systematic backtesting with out-of-sample validation.
- Tests multiple parameter combinations
- Train/test split to prevent overfitting
- Reports trades/day, win rate, returns
- Compares Phase 1 variants

**Usage:**
```bash
python3 tools/backtest_alpha_improvements.py --symbols ETHUSDT,BTCUSDT --train-days 30 --test-days 14
```

### `monitor_performance.py` ⭐ NEW
Live performance monitoring and analysis.
- Compares live trading to backtest predictions
- Calculates trades/day, win rate, returns
- Tracks success criteria
- Identifies performance issues

**Usage:**
```bash
python3 tools/monitor_performance.py --days 7
```

## Legacy Tools (Deprecated)

### `strategy_optimizer.py`
⚠️ **Deprecated** - Use `backtest_alpha_improvements.py` instead
- Legacy parameter optimization tool
- Superseded by systematic OOS testing

### `strategy_tester.py`
⚠️ **Deprecated** - Use `backtester.py` instead
- Legacy strategy testing framework
- Superseded by comprehensive backtester

## Output

- Backtest results: `results/backtest_results/`
- Live monitoring: Run `monitor_performance.py` for real-time analysis

