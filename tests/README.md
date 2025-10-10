# Test Utilities

This directory contains test and utility scripts for the trading system.

## Available Tests

### `test_system.py`
Comprehensive system test suite.
- Tests all core components (database, risk manager, strategy, dashboard)
- Validates API connections
- Checks module imports
- Verifies system integration

**Usage:**
```bash
python3 tests/test_system.py
```

### `symbol_check.py`
Utility to check available trading symbols on Bybit.
- Lists all available perpetual futures contracts
- Shows current prices and volumes
- Helps identify liquid trading pairs
- Useful for adding new symbols to the strategy

**Usage:**
```bash
python3 tests/symbol_check.py
```

## Running Tests

Make sure you have:
1. Activated the virtual environment: `source venv/bin/activate`
2. Set up API credentials in `.env` file
3. Installed all dependencies: `pip3 install -r requirements.txt`

