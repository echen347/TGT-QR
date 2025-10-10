# Trading Analysis Tools

This directory contains backtesting and strategy optimization tools.

## Available Tools

### `backtester.py`
Comprehensive backtesting framework for the moving average strategy.
- Downloads historical data from Bybit API
- Simulates strategy execution without real trades
- Calculates performance metrics (Sharpe ratio, max drawdown, win rate)
- Generates performance charts

**Usage:**
```bash
python3 tools/backtester.py
```

### `strategy_optimizer.py`
Parameter optimization tool for finding optimal strategy settings.
- Tests multiple MA periods (30, 60, 120, 240)
- Tests different threshold percentages (0.05%, 0.1%, 0.2%, 0.5%)
- Tests across multiple symbols (BTCUSDT, ETHUSDT, XRPUSDT)
- Exports results to CSV

**Usage:**
```bash
python3 tools/strategy_optimizer.py
```

### `strategy_tester.py`
Comprehensive strategy testing framework.
- Supports multiple strategy types (MA, RSI)
- Parameter sensitivity analysis
- Multi-symbol testing
- Detailed performance reporting

**Usage:**
```bash
python3 tools/strategy_tester.py
```

## Output

All tools save results to `results/backtest_results/` directory.

