import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from pybit.unified_trading import HTTP
from config.config import *
import logging
from database import db_manager
from risk_manager import risk_manager

class MovingAverageStrategy:
    """
    Simple Moving Average Strategy for Perpetual Futures
    Uses 1-hour MA on 1-minute price data as alpha signal
    """

    def __init__(self):
        self.client = HTTP(
            testnet=BYBIT_TESTNET,
            api_key=BYBIT_API_KEY,
            api_secret=BYBIT_API_SECRET
        )
        self.positions = {}
        self.price_history = {}
        self.signals = {}
        self.session = HTTP(testnet=BYBIT_TESTNET)

        # Initialize price history for each symbol
        for symbol in SYMBOLS:
            self.price_history[symbol] = []
            self.signals[symbol] = 0

        # Setup logging
        self.logger = self._setup_logging()

    def _setup_logging(self):
        """Setup logging configuration"""
        logger = logging.getLogger('TradingStrategy')
        logger.setLevel(getattr(logging, LOG_LEVEL))

        # File handler
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
                    prices.append({
                        'timestamp': datetime.fromtimestamp(int(kline[0]) / 1000),
                        'open': float(kline[1]),
                        'high': float(kline[2]),
                        'low': float(kline[3]),
                        'close': float(kline[4]),
                        'volume': float(kline[5])
                    })

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
        """Get current positions for all symbols"""
        try:
            response = self.client.get_positions(
                category="linear",
                settleCoin="USDT"
            )

            if response['retCode'] == 0:
                positions = response['result']['list']
                current_positions = {}

                for pos in positions:
                    symbol = pos['symbol']
                    if symbol in SYMBOLS:
                        current_positions[symbol] = {
                            'position_size': float(pos['size']),
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

            response = self.client.place_order(
                category="linear",
                symbol=symbol,
                side=side,
                orderType="Market",
                qty=str(qty_contracts),
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
        """Main strategy execution with comprehensive risk management"""
        self.logger.info("=" * 60)
        self.logger.info("🔄 Running trading strategy...")

        # Check risk status first
        risk_status = risk_manager.get_risk_status()
        if not risk_status['can_trade']:
            self.logger.error("❌ Risk management prevents trading")
            self.logger.error(f"Daily Loss: ${risk_status['daily_loss']:.2f}/${risk_status['daily_loss_limit']:.2f}")
            self.logger.error(f"Total Loss: ${risk_status['total_loss']:.2f}/${risk_status['total_loss_limit']:.2f}")
            return

        # Update price history for all symbols
        for symbol in SYMBOLS:
            prices = self.get_historical_prices(symbol, MA_PERIOD + 10)
            if prices:
                self.price_history[symbol] = prices[-MA_PERIOD:]
                # Save price data to database
                for price_data in prices[-10:]:  # Save last 10 candles
                    db_manager.save_price_data(symbol, price_data)

        # Calculate signals and execute trades
        for symbol in SYMBOLS:
            if len(self.price_history[symbol]) >= MA_PERIOD:
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

        # Log current state
        self.log_trading_state()
        self.logger.info("=" * 60)

    def log_trading_state(self):
        """Log comprehensive trading state including risk metrics"""
        positions = self.get_current_positions()
        risk_status = risk_manager.get_risk_status()

        self.logger.info("📈 TRADING STATE SUMMARY")
        self.logger.info("-" * 40)

        total_pnl = 0
        total_value = 0
        open_positions = 0

        for symbol in SYMBOLS:
            pos = positions.get(symbol, {'position_size': 0, 'unrealized_pnl': 0, 'position_value': 0})
            pnl = pos.get('unrealized_pnl', 0)
            total_pnl += pnl

            position_size = pos.get('position_size', 0)
            if position_size > 0:
                open_positions += 1

            current_price = self.price_history[symbol][-1]['close'] if self.price_history[symbol] else 0

            self.logger.info(f"  {symbol}:")
            self.logger.info(f"    Signal: {self.signals[symbol]}")
            self.logger.info(f"    Price: ${current_price:.2f}")
            self.logger.info(f"    Position: {position_size}")
            self.logger.info(f"    PnL: ${pnl:.4f}")

            # Save position record if position exists
            if position_size > 0:
                db_manager.save_position_record(
                    symbol=symbol,
                    position_size=position_size,
                    entry_price=pos.get('entry_price', current_price),
                    current_price=current_price,
                    unrealized_pnl=pnl
                )

        # Risk status
        self.logger.info("-" * 40)
        self.logger.info("🛡️ RISK STATUS")
        self.logger.info(f"  Daily Loss: ${risk_status['daily_loss']:.4f}/${risk_status['daily_loss_limit']:.2f}")
        self.logger.info(f"  Total Loss: ${risk_status['total_loss']:.4f}/${risk_status['total_loss_limit']:.2f}")
        self.logger.info(f"  Open Positions: {open_positions}/{risk_status['max_positions']}")
        self.logger.info(f"  Can Trade: {'✅ YES' if risk_status['can_trade'] else '❌ NO'}")

        if total_pnl != 0:
            self.logger.info(f"💰 Total Unrealized PnL: ${total_pnl:.4f} USDT")

        # PnL summary from database
        pnl_summary = db_manager.get_pnl_summary()
        if pnl_summary['net_pnl'] != 0:
            self.logger.info(f"📊 24h Net PnL: ${pnl_summary['net_pnl']:.4f} USDT")

if __name__ == "__main__":
    strategy = MovingAverageStrategy()

    # Run once for testing
    strategy.run_strategy()
