import os
from dotenv import load_dotenv

# Load .env file from the project root. This makes the config robust.
# It assumes this file is in /config and the .env is in the parent directory.
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)

# Bybit API Configuration
BYBIT_TESTNET = os.getenv('BYBIT_TESTNET', 'true').lower() == 'true'
# Try to get keys from environment, otherwise they will be None
BYBIT_API_KEY = os.getenv('BYBIT_API_KEY')
BYBIT_API_SECRET = os.getenv('BYBIT_API_SECRET')

# Trading Configuration - EXPANDED FOR MORE OPPORTUNITIES
# --- Trading Pairs ---
# List of symbols to trade
# Phase 2A: Expanded symbol list - Add BTCUSDT (worked well in previous backtest) and others
SYMBOLS = [
    'ETHUSDT', 'SOLUSDT', 'BTCUSDT', 'AVAXUSDT'
]

# --- Position Sizing & Leverage ---
MAX_POSITION_USDT = 10.00 # Margin per position (restored for $26 bankroll - meets 0.1 SOL min order with 5x leverage)
LEVERAGE = 5.0  # Higher leverage to reduce required margin per order
MAX_POSITIONS = 4  # Allow positions in all symbols (ETH, SOL, BTC, AVAX)
MIN_VOLUME_USDT = 200000  # Only trade reasonably liquid pairs (500K+ USD volume)
TIMEFRAME = '15'  # 15-minute candles per deployment decision
MA_PERIOD = 20  # A shorter MA period will be more sensitive to price changes
ATR_PERIOD = 14

# Strategy Configuration - RESTORED FOR $26 BANKROLL
STRATEGY_INTERVAL_MINUTES = 5  # Run strategy every 5 minutes
MAX_DAILY_LOSS_USDT = 10.00 # ~38% of bankroll daily loss limit (conservative)
MAX_TOTAL_LOSS_USDT = 15.00 # ~58% of bankroll total loss cap (conservative)

# Signal Filtering - More lenient for live trading vs backtesting
# Reduced MIN_TREND_STRENGTH to increase signal frequency (target: 1+ trade/day)
MIN_TREND_STRENGTH = 0.0002  # Reduced from 0.0005 for more trading opportunities
VOLATILITY_THRESHOLD_HIGH = 0.025  # Increased from 0.02
VOLATILITY_THRESHOLD_LOW = 0.015   # Increased from 0.01

# Database Configuration
DATABASE_URL = 'sqlite:///data/trading_data.db'

# Logging Configuration - DETAILED FOR RISK MONITORING
LOG_LEVEL = 'INFO'
LOG_FILE = 'logs/trading_system.log'
LOG_MAX_SIZE_MB = 10
LOG_BACKUP_COUNT = 5
LOG_ROTATION = 'daily'  # Daily log rotation for live trading
LOG_RETENTION_DAYS = 30  # Keep 30 days of logs

# Risk Management - ULTRA CONSERVATIVE APPROACH
STOP_LOSS_PCT = 0.01  # 1% stop loss (very tight)
TAKE_PROFIT_PCT = 0.02  # 2% take profit (2:1 reward:risk ratio)
MAX_POSITION_HOLD_HOURS = 24  # Maximum time to hold a position (24 hours)

# Exit Strategy - If strategy underperforms, exit positions when they go positive
EXIT_ON_PROFIT = True  # If True, close positions as soon as they become profitable (defensive mode)
EXIT_ON_PROFIT_MIN_PCT = 0.002  # Minimum profit % to exit (0.2% to cover fees)

# Trading Fees (Bybit Perpetual Futures)
# Taker fee: 0.055% (5.5 bps) per side, Maker fee: 0.02% (2 bps) per side
# Since we use market orders (taker), round-trip fee = 11 bps (5.5 bps entry + 5.5 bps exit)
TRADING_FEE_BPS = 11.0  # Round-trip trading fee in basis points (0.11% total)
SLIPPAGE_BPS = 2.0  # Estimated slippage in basis points (0.02%)

# Exchange Risk Limits & Order Attachments
RISK_LIMIT_ENABLED = True
# Note: Bybit v5 uses tier-based risk; value may be interpreted per account tier.
RISK_LIMIT_VALUE = 10000  # Target notional risk limit (exchange-tier specific)
USE_TP_SL_ON_ORDER = True  # Attach TP/SL to orders for risk management

# Dashboard Configuration
DASHBOARD_PORT = 5000
DASHBOARD_HOST = '0.0.0.0'
