import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from pybit.unified_trading import HTTP
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.config import (
    BYBIT_API_KEY, BYBIT_API_SECRET, BYBIT_TESTNET, SYMBOLS, LEVERAGE,
    MAX_POSITION_USDT, TIMEFRAME, MA_PERIOD, LOG_FILE, LOG_LEVEL,
    LOG_ROTATION, LOG_RETENTION_DAYS
)
import logging
from logging.handlers import TimedRotatingFileHandler
from database import db_manager
from risk_manager import risk_manager

# Create a global, shared instance of the strategy
strategy = MovingAverageStrategy()

class MovingAverageStrategy:
    """
    Simple Moving Average Strategy for Perpetual Futures
    Uses 1-hour MA on 1-minute price data as alpha signal
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(MovingAverageStrategy, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        self._initialized = True
        
        # Use the working API setup from day1_test.py
        self.session = HTTP(
            testnet=BYBIT_TESTNET,
            api_key=BYBIT_API_KEY,
            api_secret=BYBIT_API_SECRET
        )
        self.positions = {}
        self.price_history = {}
        self.signals = {}
        self.market_state = {} # New attribute for live data

        # Initialize price history for each symbol
        for symbol in SYMBOLS:
            self.price_history[symbol] = []
            self.signals[symbol] = 0
            self.market_state[symbol] = {
                'price': 0,
                'ma_value': 0,
                'signal': 0,
                'volume_24h': 0
            }

        # Setup logging
        self.logger = self._setup_logging()

    def _setup_logging(self):
        """Setup logging configuration with daily rotation"""
        logger = logging.getLogger('TradingStrategy')
        logger.setLevel(getattr(logging, LOG_LEVEL))

        # Daily rotating file handler for live trading
        if LOG_ROTATION == 'daily':
            # Format: logs/trading-YYYY-MM-DD.log
            log_filename = 'logs/trading.log'
            file_handler = TimedRotatingFileHandler(
                log_filename,
                when='midnight',
                interval=1,
                backupCount=LOG_RETENTION_DAYS,
                encoding='utf-8'
            )
            # Suffix format for archived logs
            file_handler.suffix = '%Y-%m-%d'
        else:
            # Fallback to regular FileHandler
            file_handler = logging.FileHandler(LOG_FILE)
        
        file_handler.setLevel(getattr(logging, LOG_LEVEL))

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    def get_historical_prices(self, symbol, limit=100):
        """Get historical price data"""
        try:
            response = self.session.get_kline(
                category="linear",
                symbol=symbol,
                interval=TIMEFRAME,
                limit=limit
            )

            if response['retCode'] == 0:
                klines = response['result']['list']
                prices = []

                for kline in reversed(klines):  # Reverse to get chronological order
                    try:
                        # Debug: print kline structure
                        self.logger.debug(f"Kline data: {kline}")

                        prices.append({
                            'timestamp': datetime.fromtimestamp(int(kline[0]) / 1000),
                            'open': float(kline[1]),
                            'high': float(kline[2]),
                            'low': float(kline[3]),
                            'close': float(kline[4]),
                            'volume': float(kline[5])
                        })
                    except Exception as e:
                        self.logger.error(f"Error processing kline {kline}: {str(e)}")
                        continue

                return prices
            else:
                self.logger.error(f"Error getting prices for {symbol}: {response['retMsg']}")
                return []

        except Exception as e:
            self.logger.error(f"Exception getting prices for {symbol}: {str(e)}")
            return []

    def calculate_ma_signal(self, symbol):
        """Calculate moving average signal"""
        if len(self.price_history[symbol]) < MA_PERIOD:
            return 0  # Not enough data

        prices = pd.DataFrame(self.price_history[symbol])
        ma = prices['close'].rolling(window=MA_PERIOD).mean().iloc[-1]
        current_price = prices['close'].iloc[-1]

        # Simple signal: 1 for buy (price > MA), -1 for sell (price < MA)
        if current_price > ma * 1.001:  # 0.1% threshold to avoid noise
            return 1
        elif current_price < ma * 0.999:  # 0.1% threshold to avoid noise
            return -1
        else:
            return 0

    def get_current_positions(self):
        """Get current open positions for all symbols"""
        try:
            response = self.session.get_positions(
                category="linear",
                settleCoin="USDT"
            )

            if response['retCode'] == 0:
                positions = response['result']['list']
                current_positions = {}

                for pos in positions:
                    symbol = pos['symbol']
                    position_size = float(pos['size'])
                    if symbol in SYMBOLS and position_size > 0:
                        current_positions[symbol] = {
                            'symbol': symbol,
                            'position_size': position_size,
                            'position_value': float(pos['positionValue']),
                            'entry_price': float(pos['avgPrice']),
                            'unrealized_pnl': float(pos['unrealisedPnl']),
                            'side': pos['side']
                        }

                return current_positions
            else:
                self.logger.error(f"Error getting positions: {response['retMsg']}")
                return {}

        except Exception as e:
            self.logger.error(f"Exception getting positions: {str(e)}")
            return {}

    def close_position(self, symbol, side, size):
        """Places a closing order for a specific position using base currency size."""
        try:
            self.logger.info(f"Attempting to place closing order for {symbol}: {side} {size}")
            response = self.session.place_order(
                category="linear",
                symbol=symbol,
                side=side,
                orderType="Market",
                qty=str(size),
                reduceOnly=True,
                marketUnit="baseCurrency"
            )
            if response.get('retCode') == 0:
                self.logger.info(f"Successfully placed closing order for {symbol} of size {size}.")
            else:
                self.logger.error(f"Failed to place closing order for {symbol}. Response: {response}")
        except Exception as e:
            self.logger.exception(f"Exception while placing closing order for {symbol}: {e}")

    def close_all_positions(self):
        """Close all open positions."""
        self.logger.warning("EMERGENCY STOP: Attempting to close all open positions.")
        open_positions = self.get_current_positions()

        if not open_positions:
            self.logger.info("No open positions to close.")
            return

        for symbol, position in open_positions.items():
            try:
                size = position['position_size']
                side = position['side']  # 'Buy' for long, 'Sell' for short

                if size > 0:
                    close_side = 'Sell' if side == 'Buy' else 'Buy'
                    self.logger.info(f"Closing {side} position for {symbol} of size {size} by placing a {close_side} order.")
                    self.close_position(symbol, close_side, size)
            except Exception as e:
                self.logger.exception(f"Failed to process and close position for {symbol}: {position}. Error: {e}")

    def get_symbol_volume(self, symbol):
        """Get 24h volume for symbol in USDT"""
        try:
            response = self.session.get_tickers(category="linear")
            if response['retCode'] == 0:
                tickers = response['result']['list']
                for ticker in tickers:
                    if ticker['symbol'] == symbol:
                        return float(ticker['volume24h'])
            return 0
        except Exception as e:
            self.logger.error(f"Exception getting volume for {symbol}: {str(e)}")
            return 0

    def place_order(self, symbol, side, qty_usdt, leverage=LEVERAGE):
        """Place a market order with comprehensive risk checks"""
        try:
            current_price = self.price_history[symbol][-1]['close']
            qty_contracts = qty_usdt * leverage / current_price

            # Comprehensive risk check before placing order
            current_volume = self.get_symbol_volume(symbol)
            if not risk_manager.can_open_position(symbol, qty_usdt, current_volume):
                self.logger.error(f"Risk management blocked order for {symbol}")
                return False

            # Additional safety checks
            if qty_usdt > MAX_POSITION_USDT:
                self.logger.error(f"Order size ${qty_usdt} exceeds maximum ${MAX_POSITION_USDT}")
                return False

            if leverage != LEVERAGE:
                self.logger.error(f"Leverage mismatch: requested {leverage}, configured {LEVERAGE}")
                return False

            # Use the working order placement from day1_test.py
            response = self.session.place_order(
                category="linear",
                symbol=symbol,
                side=side,
                orderType="Market",
                qty=str(qty_usdt),  # USDT amount for quote currency orders
                leverage=str(leverage),
                marketUnit="quoteCurrency"
            )

            if response['retCode'] == 0:
                self.logger.info(f"✅ Order placed: {side} {qty_usdt} USDT of {symbol} at ${current_price:.2f}")
                risk_manager.positions_count += 1

                # Save trade record to database
                db_manager.save_trade_record(
                    symbol=symbol,
                    side=side,
                    quantity=qty_contracts,
                    price=current_price,
                    value_usdt=qty_usdt
                )

                return True
            else:
                self.logger.error(f"❌ Order failed for {symbol}: {response['retMsg']}")
                return False

        except Exception as e:
            self.logger.error(f"❌ Exception placing order for {symbol}: {str(e)}")
            return False

    def run_strategy(self):
        """Main strategy execution loop"""
        try:
            self.logger.info("Running trading strategy...")
            # Check risk management first
            risk_status = risk_manager.get_risk_status()
            if not risk_status['can_trade']:
                self.logger.error("❌ Risk management prevents trading.")
                self.logger.error(f"Daily Loss: ${risk_status['daily_loss']:.2f}/${risk_status['max_daily_loss_usdt']:.2f}")
                self.logger.error(f"Total Loss: ${risk_status['total_loss']:.2f}/${risk_status['max_total_loss_usdt']:.2f}")
                return

            # Update price history for all symbols
            for symbol in SYMBOLS:
                prices = self.get_historical_prices(symbol, MA_PERIOD + 10)
                if not prices:
                    self.logger.warning(f"Could not fetch price data for {symbol}. Skipping this cycle.")
                    return # Exit the entire strategy run if any symbol fails
                    
                self.price_history[symbol] = prices[-MA_PERIOD:]
                # Save price data to database
                for price_data in prices[-10:]:  # Save last 10 candles
                    db_manager.save_price_data(symbol, price_data)

            # Calculate signals and execute trades
            for symbol in SYMBOLS:
                if len(self.price_history[symbol]) < MA_PERIOD:
                    self.logger.warning(f"Not enough historical data for {symbol} to calculate signal. Need {MA_PERIOD}, have {len(self.price_history[symbol])}.")
                    continue # Skip to the next symbol
                    
                signal = self.calculate_ma_signal(symbol)
                current_price = self.price_history[symbol][-1]['close']
                ma_value = pd.DataFrame(self.price_history[symbol])['close'].rolling(window=MA_PERIOD).mean().iloc[-1]

                    # Save signal to database
                    db_manager.save_signal_record(symbol, signal, ma_value, current_price)

                    # Get current positions
                    positions = self.get_current_positions()
                    current_position = positions.get(symbol, {'position_size': 0, 'position_value': 0})

                    self.logger.info(f"📊 {symbol}: Signal={signal}, Price=${current_price:.2f}, MA=${ma_value:.2f}")

                    # Trading logic with risk management
                    if signal == 1 and current_position['position_size'] == 0:
                        # Go long - only if we can open position
                        if risk_manager.can_open_position(symbol, MAX_POSITION_USDT):
                            success = self.place_order(symbol, "Buy", MAX_POSITION_USDT)
                            if success:
                                self.logger.info(f"🚀 LONG position opened for {symbol}")
                        else:
                            self.logger.warning(f"⚠️ LONG signal for {symbol} but risk management blocked")

                    elif signal == -1 and current_position['position_size'] == 0:
                        # Go short - only if we can open position
                        if risk_manager.can_open_position(symbol, MAX_POSITION_USDT):
                            success = self.place_order(symbol, "Sell", MAX_POSITION_USDT)
                            if success:
                                self.logger.info(f"🔻 SHORT position opened for {symbol}")
                        else:
                            self.logger.warning(f"⚠️ SHORT signal for {symbol} but risk management blocked")

                    elif signal == 0 and current_position['position_size'] > 0:
                        # Close position
                        position_value = abs(current_position['position_value'])
                        side = "Sell" if current_position['side'] == "Buy" else "Buy"
                        success = self.place_order(symbol, side, position_value)
                        if success:
                            self.logger.info(f"🔄 Position closed for {symbol}")
                            risk_manager.positions_count -= 1

                    self.signals[symbol] = signal
                    
                    # Update market state for the dashboard
                    self.market_state[symbol] = {
                        'price': current_price,
                        'ma_value': ma_value,
                        'signal': signal,
                        'volume_24h': self.get_symbol_volume(symbol)
                    }

            # Log current state
            self.log_trading_state()
            self.logger.info("=" * 60)
        except Exception as e:
            self.logger.exception(f"An unexpected error occurred during strategy execution: {e}")

    def log_trading_state(self):
        """Log the current state of all tracked symbols and risk"""
        self.logger.info("📈 TRADING STATE SUMMARY")
        self.logger.info("----------------------------------------")
        for symbol in SYMBOLS:
            price = self.price_history[symbol][-1]['close'] if self.price_history[symbol] else 0
            position_info = self.positions.get(symbol, {'size': 0, 'pnl': 0})
            self.logger.info(f"  {symbol}:")
            self.logger.info(f"    Signal: {self.signals.get(symbol, 0)}")
            self.logger.info(f"    Price: ${price:.2f}")
            self.logger.info(f"    Position: {position_info['size']}")
            self.logger.info(f"    PnL: ${position_info['pnl']:.4f}")

        self.logger.info("----------------------------------------")
        self.logger.info("🛡️ RISK STATUS")
        risk_status = risk_manager.get_risk_status()
        self.logger.info(f"  Daily Loss: ${risk_status['daily_loss']:.4f}/${risk_status['max_daily_loss_usdt']:.2f}")
        self.logger.info(f"  Total Loss: ${risk_status['total_loss']:.4f}/${risk_status['max_total_loss_usdt']:.2f}")
        self.logger.info(f"  Active Positions: {risk_status['positions_count']}/{risk_status['max_positions']}")
        self.logger.info(f"  Emergency Stopped: {risk_status['is_stopped']}")
        self.logger.info("----------------------------------------")

if __name__ == "__main__":
    # Note: This direct instantiation is for testing. 
    # The application uses the global 'strategy' instance.
    test_strategy = MovingAverageStrategy()

    # Run once for testing
    test_strategy.run_strategy()
