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
SYMBOLS = [
    'ETHUSDT', 'SOLUSDT'
]

# --- Position Sizing & Leverage ---
MAX_POSITION_USDT = 10.00 # Margin per position (increased for $16 bankroll)
LEVERAGE = 5.0  # Higher leverage to reduce required margin per order
MAX_POSITIONS = 2  # ETH + SOL
MIN_VOLUME_USDT = 200000  # Only trade reasonably liquid pairs (500K+ USD volume)
TIMEFRAME = '15'  # 15-minute candles per deployment decision
MA_PERIOD = 20  # A shorter MA period will be more sensitive to price changes
ATR_PERIOD = 14

# Strategy Configuration - INCREASED LIMITS FOR $16 BANKROLL
STRATEGY_INTERVAL_MINUTES = 5  # Run strategy every 5 minutes
MAX_DAILY_LOSS_USDT = 6.00 # ~10% of bankroll daily loss limit
MAX_TOTAL_LOSS_USDT = 10.00 # ~10% of bankroll total loss cap

# Signal Filtering - More lenient for live trading vs backtesting
MIN_TREND_STRENGTH = 0.0005  # Reduced from 0.001 for more trading opportunities
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

# Exchange Risk Limits & Order Attachments
RISK_LIMIT_ENABLED = True
# Note: Bybit v5 uses tier-based risk; value may be interpreted per account tier.
RISK_LIMIT_VALUE = 10000  # Target notional risk limit (exchange-tier specific)
USE_TP_SL_ON_ORDER = True  # Attach TP/SL to orders when possible

# Dashboard Configuration
DASHBOARD_PORT = 5000
DASHBOARD_HOST = '0.0.0.0'
