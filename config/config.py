import os
from dotenv import load_dotenv

load_dotenv()

# Bybit API Configuration
BYBIT_TESTNET = os.getenv('BYBIT_TESTNET', 'true').lower() == 'true'
BYBIT_API_KEY = os.getenv('BYBIT_API_KEY')
BYBIT_API_SECRET = os.getenv('BYBIT_API_SECRET')

# Trading Configuration - EXTREMELY CONSERVATIVE
SYMBOLS = ['BTCUSDT', 'ETHUSDT']  # Only 2 symbols initially
LEVERAGE = 1  # NO leverage for maximum safety
MAX_POSITION_USDT = 1.00  # Only $1 per position maximum
TIMEFRAME = '1m'  # 1 minute candles for strategy
MA_PERIOD = 60  # 1 hour moving average (60 minutes)

# Strategy Configuration - VERY CONSERVATIVE
STRATEGY_INTERVAL_MINUTES = 15  # Run strategy every 15 minutes (less frequent)
MAX_DAILY_LOSS_USDT = 2.00  # Maximum $2 daily loss
MAX_TOTAL_LOSS_USDT = 5.00  # Absolute maximum loss before stopping

# Database Configuration
DATABASE_URL = 'sqlite:///data/trading_data.db'

# Logging Configuration - DETAILED FOR RISK MONITORING
LOG_LEVEL = 'INFO'
LOG_FILE = 'logs/trading_system.log'
LOG_MAX_SIZE_MB = 10
LOG_BACKUP_COUNT = 5

# Risk Management - ULTRA CONSERVATIVE
STOP_LOSS_PCT = 0.01  # 1% stop loss (very tight)
TAKE_PROFIT_PCT = 0.02  # 2% take profit (quick profits)
MAX_POSITIONS = 1  # Only one position at a time
MIN_VOLUME_USDT = 1000000  # Only trade liquid pairs (1M+ USD volume)

# Dashboard Configuration
DASHBOARD_PORT = 5000
DASHBOARD_HOST = '0.0.0.0'
