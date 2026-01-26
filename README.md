# TGT QR - Quantitative Research Infrastructure

Cryptocurrency futures trading system for Bybit USDT-perpetuals with centralized backtesting and data infrastructure.

## Project Structure

```
TGT-QR/
├── qr_data/              # Data ingestion & caching package
├── qr_backtester/        # Realistic backtesting engine
├── mft/                  # Medium-frequency trading bot
│   ├── src/              # Core trading system
│   ├── tools/            # CLI tools & backtesting
│   └── config/           # Configuration
└── hft/                  # HFT research (experimental)
```

## Quick Start

### Installation

```bash
# Clone and setup
git clone https://github.com/echen347/TGT-QR.git
cd TGT-QR

# Install packages (from root)
pip install -e .

# Or install dependencies directly
pip install pandas numpy pyarrow pybit python-dotenv

# For MFT trading bot
cd mft
pip install -r requirements.txt
cp .env.example .env  # Add BYBIT_API_KEY, BYBIT_API_SECRET
```

### Using qr_data (Data Package)

```python
from qr_data import BybitProvider, DataCache
from pathlib import Path
from datetime import datetime, timedelta

# Initialize
provider = BybitProvider()
cache = DataCache(Path("./data_cache"), provider)

# Fetch data (auto-caches as Parquet)
data = cache.get(
    symbol="ETHUSDT",
    timeframe="15m",
    start=datetime.now() - timedelta(days=30),
    end=datetime.now()
)

print(f"Loaded {len(data)} bars")
print(f"Valid: {data.is_valid()}")
```

### Using qr_backtester

```python
from qr_backtester import BacktestEngine, Strategy, Order

class MACrossStrategy(Strategy):
    def __init__(self, fast: int = 10, slow: int = 30):
        self.fast = fast
        self.slow = slow

    @property
    def name(self) -> str:
        return f"MA_{self.fast}_{self.slow}"

    def on_bar(self, bar, portfolio):
        if len(portfolio.history) < self.slow:
            return []

        fast_ma = portfolio.history["close"].rolling(self.fast).mean().iloc[-1]
        slow_ma = portfolio.history["close"].rolling(self.slow).mean().iloc[-1]

        if fast_ma > slow_ma and not portfolio.has_position:
            return [Order.market_buy(size=100)]
        elif fast_ma < slow_ma and portfolio.has_position:
            return [Order.market_sell(size=portfolio.position_size)]
        return []

# Run backtest
engine = BacktestEngine(data, MACrossStrategy(10, 30), initial_capital=10000)
result = engine.run()
print(result.summary())
```

## Packages

### qr_data

Data ingestion and caching for Bybit.

**Features:**
- Parquet storage (5-10x smaller than pickle)
- Auto-caching with intelligent range merging
- Data validation (gaps, OHLC relationships, duplicates)
- Clean provider interface for future exchanges

**API:**
```python
from qr_data import (
    OHLCVData,          # Data container
    DataCache,          # Caching layer
    BybitProvider,      # Bybit data fetcher
    validate_ohlcv,     # Data validation
)
```

### qr_backtester

Realistic event-driven backtesting engine.

**Features:**
- Volume-aware slippage (scales with order size vs bar volume)
- Intra-bar stop/take-profit detection using high/low
- Commission modeling (default 4 bps for Bybit taker)
- Walk-forward validation
- Strategy registry with decorator-based registration

**API:**
```python
from qr_backtester import (
    BacktestEngine,       # Main engine
    Strategy,             # Base class for strategies
    Order,                # Order types
    VolumeAwareSlippage,  # Slippage model
    register_strategy,    # Decorator for registry
    get_registry,         # Access strategy registry
)
```

**Built-in Strategies:**
- `SimpleMAStrategy` - Moving average crossover
- `RSIStrategy` - RSI mean reversion

### mft/ (Trading Bot)

Live trading system with Flask dashboard.

```bash
cd mft

# Run live trading
python3 src/run_trading_system.py  # Dashboard at http://localhost:5000

# Run backtest (legacy backtester)
python3 tools/backtester.py --symbols ETHUSDT --days 30
```

## AWS Deployment

```bash
# SSH into server
ssh -i "tgt-qr-key-oct-9.pem" ubuntu@16.171.196.141

# Deploy
cd /home/ubuntu/TGT-QR
git pull origin master
sudo systemctl restart tgt-trading.service tgt-dashboard.service

# Check status
sudo journalctl -u tgt-trading.service -f
```

Dashboard: http://16.171.196.141:5000

## Configuration

All trading parameters in `mft/config/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| MAX_POSITION_USDT | $10 | Max position size |
| LEVERAGE | 5x | Trading leverage |
| STOP_LOSS_PCT | 1% | Stop loss |
| TAKE_PROFIT_PCT | 2% | Take profit |
| TIMEFRAME | 15m | Candle timeframe |

## Research Methodology

1. **Hypothesis** - Form testable hypothesis
2. **Implementation** - Code the strategy
3. **Backtest** - Test on 60-day window
4. **Validation** - 60/40 train/test split
5. **Accept/Reject** - return > 0%, win rate >= 40%, trades >= 20

See `mft/RESEARCH_LOG.md` for documented experiments.

## License

MIT
