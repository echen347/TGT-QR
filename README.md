# TGT QR Trading System

## Overview
**Comprehensive algorithmic trading system** designed for learning quantitative trading strategies while maintaining ultra-conservative risk management. Features live trading, backtesting, strategy optimization, and real-time monitoring with a focus on educational value and capital preservation.

## ⚠️ Risk Management Philosophy
- **Maximum Position Size**: $0.50 USDT (50¢ risk per trade)
- **Daily Loss Limit**: $5.00 USDT (aligns with 5 USDT minimum)
- **Total Loss Limit**: $10.00 USDT emergency stop
- **Leverage**: 10x for smaller position sizes
- **Volume Filtering**: Only highly liquid pairs (500K+ USD volume)
- **Emergency Stop**: Automatic shutdown on loss limits

## Architecture

### Core Components
```
├── src/                           # Core trading system
│   ├── strategy.py                # Moving Average strategy with risk management
│   ├── database.py                # Data persistence (SQLite)
│   ├── risk_manager.py            # Ultra-conservative risk controls
│   ├── dashboard.py               # Mobile-friendly web dashboard
│   ├── scheduler.py               # Continuous execution scheduler
│   └── run_trading_system.py      # Main startup script
├── tools/                         # Analysis and optimization tools
│   ├── backtester.py              # Historical backtesting framework
│   ├── strategy_optimizer.py      # Parameter optimization
│   └── strategy_tester.py         # Comprehensive strategy testing
├── tests/                         # Testing utilities
│   ├── test_system.py             # System integration tests
│   └── symbol_check.py            # Trading symbol validation
├── config/                        # Configuration management
│   └── config.py                  # System configuration
├── logs/                          # Comprehensive logging
├── data/                          # Database and cache files
└── results/                       # Backtest and live results
```

### Strategy Details
- **Primary Signal**: 1-hour Moving Average on 1-minute price data
- **Execution**: Every 15 minutes (conservative frequency)
- **Risk Checks**: Pre-trade validation for all positions
- **Persistence**: All data survives system crashes
- **Multi-Symbol**: Support for multiple trading pairs

## Quick Start

### 1. Setup Environment
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip3 install -r requirements.txt
```

### 2. Configure API Keys
```bash
# Copy and edit environment file
cp .env.example .env
# Edit .env with your Bybit API credentials
```

### 3. Run Live Trading System
```bash
# Start the complete trading system
python3 src/run_trading_system.py

# Dashboard available at: http://localhost:5000
```

### 4. AWS Deployment
```bash
# On your AWS server, run:
chmod +x deploy_aws.sh
./deploy_aws.sh
```

## System Features

### Live Trading
- **Real-time Execution**: Automated trading with 15-minute intervals
- **Risk Management**: Ultra-conservative $0.50 position limits
- **Multi-Symbol Support**: Trade BTCUSDT, ETHUSDT, and other pairs
- **Emergency Controls**: Dashboard shutdown and manual intervention

### Analysis Tools
- **Backtesting Framework**: Historical strategy validation
- **Strategy Optimization**: Parameter tuning and performance analysis
- **Comprehensive Testing**: Multiple strategy comparison
- **Performance Metrics**: Sharpe, Sortino, Calmar ratios and more

### Dashboard Features
- **📈 PnL Charts**: 24-hour performance visualization
- **📊 Positions**: Real-time position monitoring
- **📡 Signals**: Recent trading signals with timestamps
- **🛡️ Risk Status**: Live risk management metrics
- **⚙️ Controls**: Emergency shutdown and parameter adjustment
- **📱 Mobile-Friendly**: Optimized for phone access

### Documentation
- **Comprehensive LaTeX Documentation**: `docs/trading_strategy.pdf`
- **Alternative Data Research**: `docs/ALTERNATIVE_DATA_SOURCES.md`
- **AWS Deployment Guide**: `docs/AWS_DEPLOYMENT_GUIDE.md`
- **System Architecture**: `docs/COMPREHENSIVE_TRADING_SYSTEM.md`

## Key Features ✅

### Live Trading System
- ✅ **Bybit API Integration**: Full API setup and testing
- ✅ **Moving Average Strategy**: 1-hour MA with 0.1% threshold
- ✅ **Ultra-Conservative Risk Management**: $0.50 position limits, 10x leverage
- ✅ **Real-time Dashboard**: Mobile-friendly monitoring interface
- ✅ **Emergency Controls**: Dashboard shutdown and manual intervention
- ✅ **Comprehensive Logging**: All trades, signals, and P&L tracked

### Analysis & Research Tools
- ✅ **Backtesting Framework**: Historical strategy validation with multiple symbols
- ✅ **Strategy Optimization**: Parameter tuning and performance analysis
- ✅ **Performance Metrics**: Sharpe, Sortino, Calmar ratios and drawdown analysis
- ✅ **Alternative Data Research**: Order book, funding rates, social sentiment analysis
- ✅ **Multi-Strategy Testing**: Compare different trading approaches

### System Architecture
- ✅ **Modular Design**: Separate strategy, risk, database, and UI components
- ✅ **Data Persistence**: SQLite database with crash recovery
- ✅ **AWS Deployment**: Automated deployment with systemd service
- ✅ **Documentation**: Comprehensive LaTeX documentation and guides

## Usage Examples

### Run Live Trading
```bash
# Start the complete trading system
python3 src/run_trading_system.py
# Dashboard: http://localhost:5000
```

### Run Backtesting
```bash
# Test strategy on historical data
python3 tools/backtester.py
```

### Optimize Strategy Parameters
```bash
# Find optimal MA period and thresholds
python3 tools/strategy_optimizer.py
```

### Test System Components
```bash
# Run integration tests
python3 tests/test_system.py
```

## Configuration

### Risk Management Settings
```bash
# In config/config.py
MAX_POSITION_USDT = 0.50      # 50¢ per trade
MAX_DAILY_LOSS_USDT = 5.00    # $5 daily limit
LEVERAGE = 10                 # 10x leverage
STOP_LOSS_PCT = 0.02         # 2% stop loss
TAKE_PROFIT_PCT = 0.04       # 4% take profit
```

### Strategy Parameters
```bash
# Moving Average settings
MA_PERIOD = 60               # 60-minute moving average
STRATEGY_INTERVAL_MINUTES = 15  # Run every 15 minutes
TIMEFRAME = '60'             # 1-hour candles
```

## Support & Monitoring

### System Status
- **Dashboard**: http://localhost:5000 (when running)
- **Logs**: Check `logs/trading_system.log` for detailed activity
- **Database**: `data/trading_data.db` contains all historical data
- **Results**: `results/` contains backtest and live trading results

### Emergency Procedures
1. **Dashboard Shutdown**: Use the emergency stop button on the dashboard
2. **Manual Stop**: `sudo systemctl stop tgt-trading.service` (on AWS)
3. **Position Closure**: System automatically closes positions on loss limits
4. **Log Review**: Check logs for errors and review recent activity

### Useful Commands
```bash
# System management (on AWS deployment)
sudo systemctl status tgt-trading.service
sudo journalctl -u tgt-trading.service -f

# Local development
python3 src/run_trading_system.py
python3 tools/backtester.py
python3 tests/test_system.py
```
