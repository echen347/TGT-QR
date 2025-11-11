# ethanTGT Trading System

## Overview
Lightweight live trading bot with a clean Flask dashboard and a robust, CLI‑first backtester. Built for safety, reproducibility, and quick iteration on strategies (Bybit USDT‑perps).

## Current risk profile (config/config.py)
- **MAX_POSITION_USDT**: $10.00
- **LEVERAGE**: 5.0x
- **MAX_POSITIONS**: 2 (ETHUSDT, SOLUSDT)
- **STOP_LOSS_PCT / TAKE_PROFIT_PCT**: 1% / 2%
- **MAX_POSITION_HOLD_HOURS**: 24 (auto-close by time)
- **MAX_DAILY_LOSS_USDT**: $6.00 (~10% of bankroll)
- **MAX_TOTAL_LOSS_USDT**: $10.00 (~10% of bankroll)
- **TIMEFRAME**: 15 minutes
- **INTERVAL**: every 5 minutes

## Phase 1 Alpha Improvements (Active)
**Goal**: Achieve ≥1 trade/day while maintaining risk management

**Changes**:
- ✅ **MA slope requirement REMOVED** - Price deviation from MA is sufficient signal
- ✅ **MIN_TREND_STRENGTH**: 0.0005 → 0.0002 (more lenient trend filter)
- ✅ **Reduced thresholds**: 0.3%/0.1%/0.05% → 0.2%/0.05%/0.03% (High/Normal/Low vol)

**Backtesting**: Use `tools/backtest_alpha_improvements.py` for systematic testing with out-of-sample validation.

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
│   ├── backtest_alpha_improvements.py  # OOS testing for alpha improvements
│   ├── monitor_performance.py          # Live performance monitoring
│   ├── strategy_optimizer.py           # Legacy (deprecated)
│   └── strategy_tester.py              # Legacy (deprecated)
├── config/config.py         # All tunables
├── logs/                    # trading.log, backtesting.log, etc.
├── data/                    # SQLite DB, caches
├── results/                 # Charts/exports
├── RESEARCH_NOTES.md        # Research log (scientific method)
├── DEPLOYMENT_CHECKLIST.md  # Deployment steps
└── BACKTESTING_GUIDE.md     # Backtesting instructions
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

### Standard Backtest
Core command (defaults: 30d, DB on, plots on):
```bash
python3 tools/backtester.py
```

### Alpha Improvements Testing (with OOS validation)
Test Phase 1 improvements and parameter combinations:
```bash
python3 tools/backtest_alpha_improvements.py
# Tests 5 parameter combinations with train/test split
# Target: Find config achieving ≥1 trade/day on out-of-sample data
```

### Performance Monitoring
Monitor live trading performance vs backtest predictions:
```bash
python3 tools/monitor_performance.py --days 7
# Compares live metrics (trades/day, win rate) to backtest
# Identifies if performance matches expectations
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
