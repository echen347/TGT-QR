# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Cryptocurrency futures trading system for Bybit USDT-perpetuals with:
- **mft/**: Medium-frequency trading bot with Flask dashboard and backtesting framework
- **hft/**: High-frequency order book analysis (research only)

Current status: **PAUSED** (research mode) - trading disabled via `PAUSE_TRADING=True` and `MAX_POSITIONS=0`.

## Commands

### Setup
```bash
cd mft
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
cp .env.example .env  # Add BYBIT_API_KEY, BYBIT_API_SECRET
```

### Live Trading
```bash
python3 src/run_trading_system.py  # Dashboard at http://localhost:5000
```

### Backtesting
```bash
python3 tools/backtester.py                                    # Default: 30 days, config symbols
python3 tools/backtester.py --symbols ETHUSDT,BTCUSDT --days 60
python3 tools/backtester.py --start 2025-08-01 --end 2025-09-01 --timeframe 60 --ma 20
python3 tools/backtester.py --symbols ETHUSDT --days 14 --no-db --no-plot  # Fast mode
```

### Out-of-Sample Validation
```bash
python3 tools/backtest_alpha_improvements.py  # Tests parameter combinations with train/test split
```

### Testing
```bash
python3 tests/test_system.py     # Integration tests
python3 tests/symbol_check.py    # List available Bybit symbols
```

### AWS Deployment
```bash
# SSH into AWS (from local machine)
ssh -i "tgt-qr-key-oct-9.pem" ubuntu@16.171.196.141

# On AWS: Pull and restart
cd /home/ubuntu/TGT-QR/mft
git pull origin master
sudo systemctl restart tgt-trading.service tgt-dashboard.service

# Check status
sudo systemctl status tgt-dashboard.service
sudo journalctl -u tgt-trading.service -f
```
Dashboard URL: http://16.171.196.141:5000

## Architecture

### Core Components (mft/src/)
- **strategy.py**: `MovingAverageStrategy` singleton - Bybit API integration, position management, signal execution
- **signal_calculator.py**: Signal pipeline (MA → trend strength → volatility → RSI → ADX → price deviation)
- **risk_manager.py**: Position limits, daily loss caps, emergency stops, volume filters
- **dashboard.py**: Flask UI with real-time monitoring (port 5000)
- **database.py**: SQLAlchemy models for SQLite (`data/trading_data.db`)
- **run_trading_system.py**: Entry point - spawns strategy, dashboard, and scheduler threads

### Backtesting (mft/tools/)
- **backtester.py**: Main backtester - deterministic bar-by-bar simulation, no lookahead bias
- **backtest_alpha_improvements.py**: OOS validation with train/test splits
- **ml_strategy.py**: Random Forest and Lasso-based strategies
- **ensemble_strategy.py**: Multi-model approach

### Data Flow
```
Bybit API → Historical Data → signal_calculator.calculate_signal() → Strategy Decision
                ↓                        ↓
         Data Cache (pickle)      Signal: STRONG_BUY/NEUTRAL/STRONG_SELL
```

### Configuration
All parameters in `mft/config/config.py`, loaded from `.env`:
- Risk: $10 max position, 5x leverage, 1% stop loss, 2% take profit
- Timeframe: 15-minute candles, checked every 5 minutes
- Symbols: Currently AVAXUSDT only (profitable in backtests)

## Research Methodology

The project follows scientific method for strategy development (see `RESEARCH_LOG.md`):
1. Hypothesis formation
2. Implementation
3. Backtest on 60-day window
4. OOD validation (60/40 train/test split)
5. Accept/reject based on: return > 0%, win rate ≥ 40%, trades ≥ 20

Key finding: ETHUSDT shows profitability across multiple strategies (MACD: 46% return, 47 trades).

## Key Files
- `mft/README.md`: Detailed setup and CLI reference
- `mft/RESEARCH_LOG.md`: 14+ documented experiments with results
- `mft/BACKTESTING_GUIDE.md`: Backtesting instructions
- `mft/config/config.py`: All tunables (risk, signals, intervals)

## Database Schema (SQLite)
Tables: price_data, trade_records, position_records, signal_records, funding_records, balance_snapshots, backtest_runs
