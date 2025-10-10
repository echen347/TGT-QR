# TGT QR Trading System - Project Structure

## Overview
Clean, organized structure with separated concerns for documentation, tools, results, and core trading system.

## Directory Structure

```
TGT QR/
├── docs/                          # All documentation
│   ├── ALTERNATIVE_DATA_SOURCES.md
│   ├── AWS_DEPLOYMENT_GUIDE.md
│   ├── COMPREHENSIVE_TRADING_SYSTEM.md
│   ├── trading_strategy.pdf
│   ├── trading_strategy.tex
│   └── ... (LaTeX build artifacts)
│
├── tools/                         # Analysis and optimization tools
│   ├── README.md
│   ├── backtester.py             # Backtest moving average strategy
│   ├── strategy_optimizer.py     # Parameter optimization
│   └── strategy_tester.py        # Comprehensive strategy testing
│
├── tests/                         # Test utilities
│   ├── README.md
│   ├── test_system.py            # System integration tests
│   └── symbol_check.py           # Check available trading symbols
│
├── results/                       # Trading results
│   ├── README.md
│   ├── backtest_results/         # Historical analysis (tracked)
│   │   └── strategy_optimization_results.csv
│   └── live_results/             # Live trading data (gitignored)
│
├── logs/                          # Daily rotating logs
│   ├── trading.log               # Current log file
│   ├── trading.log.YYYY-MM-DD    # Archived daily logs
│   └── archive/                  # Old logs
│
├── src/                           # Core trading system
│   ├── dashboard.py              # Web dashboard
│   ├── database.py               # Data persistence
│   ├── risk_manager.py           # Risk management
│   ├── run_trading_system.py    # Main entry point
│   ├── scheduler.py              # Strategy scheduler
│   ├── strategy.py               # Trading strategy
│   └── templates/
│       └── dashboard.html        # Dashboard UI
│
├── config/                        # Configuration
│   └── config.py                 # System configuration
│
├── data/                          # Database files
│   └── trading_data.db           # SQLite database
│
├── .gitignore                     # Git ignore rules
├── .env                           # API credentials (gitignored)
├── README.md                      # Project overview
├── requirements.txt               # Python dependencies
└── deploy_aws.sh                  # AWS deployment script
```

## Key Features

### 1. Documentation (`docs/`)
- Comprehensive trading strategy documentation
- AWS deployment guide
- Alternative data sources research
- LaTeX technical documentation

### 2. Analysis Tools (`tools/`)
- Backtesting framework
- Parameter optimization
- Strategy testing utilities
- All tools save results to `results/backtest_results/`

### 3. Test Utilities (`tests/`)
- System integration tests
- Symbol availability checker
- API connection verification

### 4. Results Management (`results/`)
- **Backtest Results**: Versioned historical analysis (tracked in git)
- **Live Results**: Real trading data (gitignored for privacy)

### 5. Logging System
- **Daily Rotation**: New log file created at midnight
- **Retention**: Keeps 30 days of logs
- **Format**: `trading.log.YYYY-MM-DD`
- **Archive**: Old logs moved to `logs/archive/`

### 6. Security
- `.pem` files ignored (private keys)
- `.env` files ignored (API credentials)
- Live trading results not committed
- Sensitive data protected

## Running the System

### Start Trading System
```bash
python3 src/run_trading_system.py
```

### Access Dashboard
```
http://localhost:5000
```

### Run Backtests
```bash
python3 tools/backtester.py
```

### Optimize Strategy
```bash
python3 tools/strategy_optimizer.py
```

### Test System
```bash
python3 tests/test_system.py
```

## Configuration

All settings in `config/config.py`:
- Trading symbols
- Risk management parameters
- Logging configuration
- Dashboard settings

## Changes Made

1. ✅ Created organized folder structure
2. ✅ Moved documentation to `docs/`
3. ✅ Moved analysis tools to `tools/`
4. ✅ Moved test utilities to `tests/`
5. ✅ Created `results/` with backtest and live subdirectories
6. ✅ Deleted obsolete test/debug scripts
7. ✅ Updated `.gitignore` for private files
8. ✅ Implemented daily rotating logs
9. ✅ Added README files for each directory
10. ✅ Protected sensitive data from git commits

## Notes

- Bot can continue running during reorganization (only file moves)
- Log rotation will take effect on next system restart
- All private data (keys, live results) are gitignored
- Documentation and results are properly versioned

