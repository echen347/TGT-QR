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
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT', 
    'XRPUSDT', 'AVAXUSDT', 'MATICUSDT', 'LINKUSDT', 'UNIUSDT', 'DOTUSDT'
]

# --- Position Sizing & Leverage ---
MAX_POSITION_USDT = 0.25 # Max size of a single position in USDT
LEVERAGE = 5  # Reduced leverage for better risk management
MAX_POSITIONS = 5  # Allow up to 5 positions at once (broader diversification)
MIN_VOLUME_USDT = 500000  # Only trade reasonably liquid pairs (500K+ USD volume)
TIMEFRAME = '60'  # 1-hour candles for less noise
MA_PERIOD = 50  # 20-period MA on 1-hour data
ATR_PERIOD = 14

# Strategy Configuration - MODERATE RISK
STRATEGY_INTERVAL_MINUTES = 15  # Run strategy every 15 minutes
MAX_DAILY_LOSS_USDT = 0.50 # Max aggregate loss per day in USDT before pausing trading
MAX_TOTAL_LOSS_USDT = 1.0 # Max total loss from starting capital before stopping the bot

# Database Configuration
DATABASE_URL = 'sqlite:///data/trading_data.db'

# Logging Configuration - DETAILED FOR RISK MONITORING
LOG_LEVEL = 'INFO'
LOG_FILE = 'logs/trading_system.log'
LOG_MAX_SIZE_MB = 10
LOG_BACKUP_COUNT = 5
LOG_ROTATION = 'daily'  # Daily log rotation for live trading
LOG_RETENTION_DAYS = 30  # Keep 30 days of logs

# Risk Management - BALANCED APPROACH
STOP_LOSS_PCT = 0.02  # 2% stop loss
TAKE_PROFIT_PCT = 0.04  # 4% take profit (2:1 reward:risk ratio)

# Dashboard Configuration
DASHBOARD_PORT = 5000
DASHBOARD_HOST = '0.0.0.0'
