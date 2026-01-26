# MFT - Medium Frequency Trading Bot

Live trading system for Bybit USDT-perpetuals with Flask dashboard.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
cp .env.example .env  # Add BYBIT_API_KEY, BYBIT_API_SECRET
```

## Usage

### Live Trading
```bash
python3 src/run_trading_system.py
# Dashboard at http://localhost:5000
```

### Backtesting
```bash
# Default: 30 days, config symbols
python3 tools/backtester.py

# Custom
python3 tools/backtester.py --symbols ETHUSDT,BTCUSDT --days 60
python3 tools/backtester.py --start 2025-08-01 --end 2025-09-01 --timeframe 60

# Fast mode (no DB, no plots)
python3 tools/backtester.py --symbols ETHUSDT --days 14 --no-db --no-plot
```

### Out-of-Sample Validation
```bash
python3 tools/backtest_alpha_improvements.py
```

## Project Structure

```
mft/
├── src/
│   ├── strategy.py           # Trading strategy
│   ├── signal_calculator.py  # Signal generation
│   ├── risk_manager.py       # Risk controls
│   ├── dashboard.py          # Flask UI
│   ├── database.py           # SQLite models
│   └── run_trading_system.py # Entry point
├── tools/
│   ├── backtester.py         # CLI backtester
│   └── ml_strategy.py        # ML strategies
├── config/
│   └── config.py             # All parameters
├── data/                     # Cache and DB
└── logs/                     # Log files
```

## Configuration

Edit `config/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| MAX_POSITION_USDT | $10 | Max position size |
| LEVERAGE | 5x | Trading leverage |
| STOP_LOSS_PCT | 1% | Stop loss |
| TAKE_PROFIT_PCT | 2% | Take profit |
| TIMEFRAME | 15 | Candle minutes |

## AWS Deployment

```bash
sudo systemctl restart tgt-trading.service tgt-dashboard.service
sudo journalctl -u tgt-trading.service -f
```

See root README for full deployment guide.
