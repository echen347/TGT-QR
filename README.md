# ethanTGT Trading System

## Overview
Lightweight live trading bot with a clean Flask dashboard and a robust, CLI‑first backtester. Built for safety, reproducibility, and quick iteration on strategies (Bybit USDT‑perps).

## Current risk profile (config/config.py)
- **MAX_POSITION_USDT**: $3.00
- **LEVERAGE**: 5.0x
- **MAX_POSITIONS**: 2 (ETHUSDT, SOLUSDT)
- **STOP_LOSS_PCT / TAKE_PROFIT_PCT**: 1% / 2%
- **MAX_POSITION_HOLD_HOURS**: 24 (auto-close by time)
- **TIMEFRAME**: 15 minutes
- **INTERVAL**: every 5 minutes

## Project layout
```
├── src/
│   ├── strategy.py          # Live strategy + sync with Bybit positions
│   ├── risk_manager.py      # Daily/total loss gates, emergency stop
│   ├── database.py          # SQLite models and helpers
│   ├── dashboard.py         # Main dashboard (no backtest UI)
│   ├── run_trading_system.py# Entry point for live bot
│   └── scheduler.py         # Interval scheduling
├── tools/
│   ├── backtester.py        # CLI backtester (robust, fast, flexible)
│   ├── strategy_optimizer.py
│   └── strategy_tester.py
├── config/config.py         # All tunables
├── logs/                    # trading.log, backtesting.log, etc.
├── data/                    # SQLite DB, caches
└── results/                 # Charts/exports
```

## Setup
```bash
# Create venv and install deps
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt

# Add API keys
cp .env.example .env   # then edit BYBIT_API_KEY / BYBIT_API_SECRET / BYBIT_TESTNET
```

## Run the live bot
```bash
python3 src/run_trading_system.py
# Dashboard at http://localhost:5000
```

### AWS (systemd)
```bash
sudo systemctl restart tgt-trading.service tgt-dashboard.service
sudo systemctl status tgt-trading.service
sudo journalctl -u tgt-trading.service -f
```

## CLI backtesting (robust)
The dashboard backtest UI was removed for simplicity. Use the CLI instead.

Core command (defaults: 30d, DB on, plots on):
```bash
python3 tools/backtester.py
```

Common examples:
```bash
# Specific symbols and days
python3 tools/backtester.py --symbols BTCUSDT,ETHUSDT --days 60

# Exact date range + overrides (1h candles, MA(20), fees/slippage)
python3 tools/backtester.py \
  --symbols BTCUSDT \
  --start 2025-08-01 --end 2025-09-01 \
  --timeframe 60 --ma 20 --fee-bps 5 --slippage-bps 2

# Fast ad‑hoc: skip DB writes and plots
python3 tools/backtester.py --symbols BTCUSDT,ETHUSDT --days 14 --no-db --no-plot
```

Backtester features:
- Deterministic, bar‑by‑bar, no lookahead
- Caching with TTL (default 3600s)
- Parallel data fetch (configurable workers)
- Fees/slippage modeling (bps, per leg)
- ATR‑based stops, 2:1 take‑profit
- Saves metrics and trades to DB (unless `--no-db`)
- Optional matplotlib visuals to `results/`

Flags (partial):
- `--symbols` CSV list (default from `config.py`)
- `--days` or `--start --end`
- `--timeframe` (Bybit interval, e.g., 60)
- `--ma` (MA period)
- `--fee-bps`, `--slippage-bps`
- `--cache-ttl`, `--max-workers`
- `--no-db`, `--no-plot`

## Monitoring & logs
- Dashboard: `http://localhost:5000`
- Live logs: `logs/trading.log` (also `/logs` page)
- Backtests: `logs/backtesting.log`, trades/metrics in SQLite `data/trading_data.db`

## Notes
- Live strategy syncs `market_state` with Bybit each cycle (no stale positions).
- Emergency stop closes all positions and prevents new ones.
- Time‑limit exit (24h) prevents stranded positions.

## Disclaimer
This software is for research/education. Markets are risky; use at your own discretion.
