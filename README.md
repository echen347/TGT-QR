# TGT QR Trading System

## Overview
**Ultra-conservative automated trading system** designed for minimum risk exposure while learning quantitative trading strategies. Built for Week 1 QR assignment with comprehensive risk management and educational focus.

## ⚠️ Risk Management Philosophy
- **Maximum Position Size**: $1.00 USDT
- **Daily Loss Limit**: $2.00 USDT
- **Total Loss Limit**: $5.00 USDT
- **No Leverage**: 1x only for maximum safety
- **Only 2 Assets**: BTCUSDT, ETHUSDT initially
- **Emergency Stop**: Automatic shutdown on loss limits

## Architecture

### Core Components
```
├── src/
│   ├── strategy.py          # Moving Average strategy with risk management
│   ├── database.py          # Data persistence (SQLite)
│   ├── risk_manager.py      # Ultra-conservative risk controls
│   ├── dashboard.py         # Mobile-friendly web dashboard
│   ├── scheduler.py         # Continuous execution scheduler
│   └── run_trading_system.py # Main startup script
├── config/
│   └── config.py            # Configuration with conservative defaults
├── logs/                    # Comprehensive logging
├── data/                    # Persistent database storage
└── templates/               # Dashboard HTML templates
```

### Strategy Details
- **Primary Signal**: 1-hour Moving Average on 1-minute price data
- **Execution**: Every 15 minutes (conservative frequency)
- **Risk Checks**: Before every trade execution
- **Persistence**: All data survives system crashes
- **Logging**: Comprehensive logs for analysis

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

### 3. Run Locally
```bash
# Start the complete system
python3 src/run_trading_system.py

# Dashboard available at: http://localhost:5000
```

### 4. AWS Deployment
```bash
# On your AWS server, run:
chmod +x deploy_aws.sh
./deploy_aws.sh
```

## Dashboard Features
- **📈 PnL Charts**: 24-hour performance visualization
- **📊 Positions**: Real-time position monitoring
- **📡 Signals**: Recent trading signals with timestamps
- **🛡️ Risk Status**: Live risk management metrics
- **⚙️ Controls**: Emergency shutdown and parameter adjustment
- **📱 Mobile-Friendly**: Optimized for phone access

## Files Structure
- **README.md**: Project overview and documentation
- **requirements.txt**: Python dependencies
- **documentation/trading\_strategy.tex**: Comprehensive LaTeX documentation
- **src/**: Core application modules
- **config/**: Configuration files
- **.gitignore**: Excludes sensitive files (.env, logs, etc.)

## Week 1 Deliverables ✅
- ✅ **Funding Rate Understanding**: Complete mechanism documentation
- ✅ **Bybit API Integration**: Full API setup and testing
- ✅ **Simple Strategy**: 1-hour MA strategy implemented
- ✅ **Risk Management**: Ultra-conservative $1 position limits
- ✅ **Data Persistence**: SQLite database with crash recovery
- ✅ **Comprehensive Logging**: Position, PnL, fees, signals tracked
- ✅ **Mobile Dashboard**: Charts, controls, real-time monitoring
- ✅ **Alternative Data Research**: Comprehensive analysis provided

## Next Steps
1. **API Keys**: Add Bybit credentials to `.env`
2. **AWS Server**: Deploy for 24/7 operation
3. **Signal Enhancement**: Implement order book and funding rate signals
4. **Backtesting**: Historical performance analysis
5. **Risk Monitoring**: Continuous system oversight

## Support
- **Dashboard**: http://localhost:5000 (when running)
- **Logs**: Check `logs/trading_system.log` for detailed activity
- **Database**: `data/trading_data.db` contains all historical data
