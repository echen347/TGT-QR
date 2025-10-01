import os
from dotenv import load_dotenv

load_dotenv()

# Bybit API Configuration
BYBIT_TESTNET = os.getenv('BYBIT_TESTNET', 'true').lower() == 'true'
BYBIT_API_KEY = os.getenv('BYBIT_API_KEY')
BYBIT_API_SECRET = os.getenv('BYBIT_API_SECRET')

# Trading Configuration - ADJUSTED FOR 50¢ RISK PER TRADE
SYMBOLS = ['BTCUSDT', 'ETHUSDT']  # Start with 2 symbols
LEVERAGE = 10  # 10x leverage for smaller position sizes
MAX_POSITION_USDT = 0.50  # 50¢ risk per position (5¢ per 1% move with 10x leverage)
TIMEFRAME = '1m'  # 1 minute candles for strategy
MA_PERIOD = 60  # 1 hour moving average (60 minutes)

# Strategy Configuration - MODERATE RISK
STRATEGY_INTERVAL_MINUTES = 15  # Run strategy every 15 minutes
MAX_DAILY_LOSS_USDT = 5.00  # Maximum $5 daily loss (aligns with 5 USDT minimum)
MAX_TOTAL_LOSS_USDT = 10.00  # Absolute maximum loss before stopping

# Database Configuration
DATABASE_URL = 'sqlite:///data/trading_data.db'

# Logging Configuration - DETAILED FOR RISK MONITORING
LOG_LEVEL = 'INFO'
LOG_FILE = 'logs/trading_system.log'
LOG_MAX_SIZE_MB = 10
LOG_BACKUP_COUNT = 5

# Risk Management - BALANCED APPROACH
STOP_LOSS_PCT = 0.02  # 2% stop loss
TAKE_PROFIT_PCT = 0.04  # 4% take profit (2:1 reward:risk ratio)
MAX_POSITIONS = 1  # Only one position at a time
MIN_VOLUME_USDT = 1000000  # Only trade liquid pairs (1M+ USD volume)

# Dashboard Configuration
DASHBOARD_PORT = 5000
DASHBOARD_HOST = '0.0.0.0'
