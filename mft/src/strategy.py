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
    LOG_ROTATION, LOG_RETENTION_DAYS, MIN_TREND_STRENGTH,
    VOLATILITY_THRESHOLD_HIGH, VOLATILITY_THRESHOLD_LOW
)
import logging
from logging.handlers import TimedRotatingFileHandler
from database import db_manager
# Import the class, not the instance we removed
from risk_manager import RiskManager

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
        
        # This will be set by the main script after initialization
        self.risk_manager = None

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
                'signal': "NEUTRAL",
                'volume_24h': 0,
                'position_size': 0,
                'entry_price': 0,
                'stop_loss': 0,
                'take_profit': 0
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
    def set_risk_manager(self, risk_manager):
        """Allow the main script to set the shared risk manager instance."""
        self.risk_manager = risk_manager
    def get_historical_prices(self, symbol, limit=200):
        """Get historical price data. Reverted to get_kline for reliability."""
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
                for kline in reversed(klines):
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
        """Calculate moving average signal using unified signal calculator"""
        from signal_calculator import calculate_signal
        
        if len(self.price_history[symbol]) < MA_PERIOD + 10:
            self.market_state[symbol]['signal'] = "NEUTRAL"
            return 0
        
        prices_df = pd.DataFrame(self.price_history[symbol])
        signal_value, signal_name, metadata = calculate_signal(prices_df)
        
        # Update market_state with results
        self.market_state[symbol]['ma_value'] = metadata['ma']
        self.market_state[symbol]['price'] = metadata['price']
        self.market_state[symbol]['signal'] = signal_name
        
        # Enhanced logging for signal generation (INFO level for visibility)
        if signal_name != "NEUTRAL":
            self.logger.info(
                f"🔍 {symbol} {signal_name} signal generated: Price=${metadata['price']:.2f}, "
                f"MA=${metadata['ma']:.2f}, Deviation={metadata['deviation_pct']:.3f}%, "
                f"Threshold={metadata['threshold']:.3f}%, Vol={metadata['vol_category']}, "
                f"TrendStrength={metadata['trend_strength']:.6f}"
            )
        
        return signal_value

    def set_stop_loss_take_profit(self, symbol, entry_price, position_size, side):
        """Set stop loss and take profit levels"""
        # Use ATR-based stops for better risk management
        prices = pd.DataFrame(self.price_history[symbol][-20:])  # Last 20 candles for ATR
        atr = self.calculate_atr(prices, 14)

        # Dynamic stop loss based on volatility
        if atr > 0:
            stop_distance = max(atr * 1.5, entry_price * 0.01)  # At least 1%
        else:
            stop_distance = entry_price * 0.02  # 2% fallback

        # Take profit at 2:1 reward:risk ratio
        take_profit_distance = stop_distance * 2

        if side == "Buy":  # Long position
            stop_loss = entry_price - stop_distance
            take_profit = entry_price + take_profit_distance
        else:  # Short position
            stop_loss = entry_price + stop_distance
            take_profit = entry_price - take_profit_distance

        self.market_state[symbol].update({
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'position_size': position_size
        })

        return stop_loss, take_profit

    def calculate_atr(self, prices, period=14):
        """Calculate Average True Range"""
        high = prices['high']
        low = prices['low']
        close = prices['close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean().iloc[-1]

        return atr

    def sync_positions_with_bybit(self):
        """CRITICAL: Sync market_state with actual Bybit positions"""
        try:
            positions = self.get_current_positions()
            self.logger.info(f"🔄 Syncing positions with Bybit: Found {len(positions)} open positions")
            
            # Reset all position tracking first
            for symbol in SYMBOLS:
                self.market_state[symbol]['position_size'] = 0
                self.market_state[symbol]['position_value'] = 0
                self.market_state[symbol]['side'] = 'N/A'
                self.market_state[symbol]['entry_price'] = 0
                self.market_state[symbol]['stop_loss'] = 0
                self.market_state[symbol]['take_profit'] = 0
            
            # Update with actual positions
            for symbol, pos in positions.items():
                self.market_state[symbol]['position_size'] = pos['position_size']
                self.market_state[symbol]['position_value'] = pos['position_value']
                self.market_state[symbol]['side'] = pos['side']
                self.market_state[symbol]['entry_price'] = pos['entry_price']
                self.market_state[symbol]['current_price'] = pos['current_price']
                # Preserve entry time if it exists, otherwise set to now (for existing positions)
                if 'entry_time' not in self.market_state[symbol] or self.market_state[symbol]['entry_time'] is None:
                    from datetime import datetime
                    self.market_state[symbol]['entry_time'] = datetime.utcnow()
                self.logger.info(f"✅ Synced {symbol}: {pos['side']} {pos['position_size']} contracts")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to sync positions with Bybit: {e}")

    def check_exit_conditions(self, symbol, current_price):
        """Check if position should be closed due to stop loss, take profit, or time limit"""
        if self.market_state[symbol]['position_size'] == 0:
            return False, None

        entry_price = self.market_state[symbol]['entry_price']
        stop_loss = self.market_state[symbol]['stop_loss']
        take_profit = self.market_state[symbol]['take_profit']
        position_size = self.market_state[symbol]['position_size']
        
        # Check time limit (24 hours max hold)
        from datetime import datetime, timedelta
        entry_time = self.market_state[symbol].get('entry_time')
        if entry_time:
            time_held = datetime.utcnow() - entry_time
            if time_held > timedelta(hours=MAX_POSITION_HOLD_HOURS):
                self.logger.warning(f"⏰ Time limit reached for {symbol}: {time_held}")
                side = "Sell" if position_size > 0 else "Buy"
                return True, side

        # Check stop loss and take profit
        if position_size > 0:  # Long position
            if current_price <= stop_loss or current_price >= take_profit:
                return True, "Sell" if current_price <= stop_loss else "Sell"
        else:  # Short position
            if current_price >= stop_loss or current_price <= take_profit:
                return True, "Buy" if current_price >= stop_loss else "Buy"

        return False, None

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
                    # Include positions with any non-zero size (both LONG and SHORT)
                    if symbol in SYMBOLS and position_size != 0:
                        current_positions[symbol] = {
                            'symbol': symbol,
                            'position_size': position_size,
                            'position_value': float(pos['positionValue']),
                            'entry_price': float(pos['avgPrice']),
                            'unrealized_pnl': float(pos['unrealisedPnl']),
                            'side': pos['side'],
                            'current_price': float(pos.get('markPrice', pos['avgPrice']))  # Use mark price if available
                        }
                return current_positions
            else:
                self.logger.error(f"Error getting positions: {response['retMsg']}")
                return {}
        except Exception as e:
            self.logger.error(f"Exception getting positions: {str(e)}")
            return {}

    def update_market_data(self):
        """Update price history and market state for all symbols."""
        self.logger.info("Updating market data for all symbols...")
        for symbol in SYMBOLS:
            try:
                # Fetch latest k-line data
                prices = self.get_historical_prices(symbol, MA_PERIOD + 20)
                if not prices:
                    self.logger.warning(f"Could not fetch price data for {symbol}. Keeping stale data.")
                    # Ensure default state if it was never populated
                    if not self.price_history.get(symbol):
                        self.price_history[symbol] = []
                    continue

                self.price_history[symbol] = prices

                # Fetch ticker data for volume and live price
                ticker_response = self.session.get_tickers(category="linear", symbol=symbol)
                if ticker_response.get('retCode') == 0 and ticker_response['result']['list']:
                    ticker_data = ticker_response['result']['list'][0]
                    current_price = float(ticker_data.get('lastPrice', 0))
                    volume_24h = float(ticker_data.get('volume24h', 0))

                    self.market_state[symbol]['price'] = current_price
                    self.market_state[symbol]['volume_24h'] = volume_24h

                    # Update the latest price in our history for consistency
                    if self.price_history[symbol]:
                        self.price_history[symbol][-1]['close'] = current_price
                    
                    # Recalculate signal to keep dashboard updated
                    if len(self.price_history[symbol]) >= MA_PERIOD:
                        self.calculate_ma_signal(symbol)  # Updates market_state['signal']

                    # Save current price data to database for main dashboard charts (only if changed significantly)
                    if current_price > 0 and len(self.price_history[symbol]) > 0:
                        last_saved_price = self.market_state[symbol].get('last_saved_price', 0)
                        # Only save if price changed by more than 0.1% to reduce DB writes
                        if abs(current_price - last_saved_price) / max(last_saved_price, 0.01) > 0.001:
                            from datetime import datetime
                            db_manager.save_price_data(symbol, current_price, volume_24h)
                            self.market_state[symbol]['last_saved_price'] = current_price
                else:
                    self.logger.warning(f"Could not fetch ticker data for {symbol}. Using last k-line price.")
                    if self.price_history[symbol]:
                         self.market_state[symbol]['price'] = self.price_history[symbol][-1]['close']
                
            except Exception as e:
                self.logger.error(f"An unexpected error occurred while updating data for {symbol}: {e}")
                # Ensure a default state exists to prevent crashes
                if symbol not in self.market_state:
                    self.market_state[symbol] = {'price': 0, 'ma_value': 0, 'signal': 'NEUTRAL', 'volume_24h': 0}

    def close_position(self, symbol, side, size):
        """Places a closing order for a specific position using USDT amount."""
        try:
            # Calculate USDT value for the position
            position_value = abs(size)  # Use absolute size for USDT calculation
            self.logger.info(f"Attempting to place closing order for {symbol}: {side} {position_value} USDT")
            response = self.session.place_order(
                category="linear",
                symbol=symbol,
                side=side,
                orderType="Market",
                qty=str(position_value),
                reduceOnly=True,
                marketUnit="quoteCurrency"
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
            # Use leveraged notional to ensure min $5 notional is met
            # Clamp margin to available balance and config
            available_balance = 0
            try:
                balance_resp = self.session.get_wallet_balance(accountType="UNIFIED", coin="USDT")
                if balance_resp.get('retCode') == 0:
                    wallet_data = balance_resp['result']['list'][0]['coin'][0]
                    # Use walletBalance (total balance) since availableToWithdraw may be empty
                    available_balance = float(wallet_data.get('walletBalance') or 0)
            except Exception as e:
                self.logger.warning(f"Error getting balance: {e}")
                pass

            # Clamp to 80% of available to avoid 110007 and to MAX_POSITION_USDT (restored for $26 bankroll)
            from config.config import MAX_POSITION_USDT
            clamped_margin = min(float(qty_usdt), max(0.0, available_balance * 0.80), float(MAX_POSITION_USDT))
            
            # Calculate notional with leverage
            notional_usdt = clamped_margin * leverage
            
            # CRITICAL: Ensure we meet exchange minimum order size (0.1 SOL for SOLUSDT, etc.)
            # Calculate minimum notional based on min order qty (0.1 base currency)
            min_order_qty = 0.1  # Bybit minimum for most perpetuals
            min_notional_usdt = min_order_qty * current_price
            min_margin_for_order = min_notional_usdt / leverage
            
            if notional_usdt < min_notional_usdt:
                # We need to increase margin to meet minimum order size
                # Try to use full MAX_POSITION_USDT if balance allows
                required_margin = max(min_margin_for_order, MAX_POSITION_USDT)
                if available_balance * 0.80 >= required_margin:
                    # Use full MAX_POSITION_USDT (optimal sizing)
                    clamped_margin = MAX_POSITION_USDT
                    notional_usdt = clamped_margin * leverage
                    qty_base = notional_usdt / current_price
                    # Round to meet exchange step size
                    qty_base = round(qty_base * 10) / 10
                    if qty_base < min_order_qty:
                        qty_base = min_order_qty
                    notional_usdt = qty_base * current_price
                    clamped_margin = notional_usdt / leverage
                    self.logger.info(f"📈 Using optimal size: ${clamped_margin:.2f} margin (${notional_usdt:.2f} notional, {qty_base:.3f} {symbol.replace('USDT', '')})")
                elif available_balance * 0.80 >= min_margin_for_order:
                    # Use minimum needed to meet exchange requirement
                    clamped_margin = min_margin_for_order
                    notional_usdt = min_notional_usdt
                    qty_base = min_order_qty
                    self.logger.info(f"📈 Using minimum size: ${clamped_margin:.2f} margin (${notional_usdt:.2f} notional, {qty_base:.3f} {symbol.replace('USDT', '')})")
                else:
                    self.logger.warning(f"Insufficient balance for min order. Need ${min_margin_for_order:.2f} margin (${min_notional_usdt:.2f} notional, {min_order_qty} {symbol.replace('USDT', '')}), have ${available_balance:.2f} USDT")
                    return False
            
            if clamped_margin <= 0:
                self.logger.warning(f"Insufficient balance to place order for {symbol}. Avail={available_balance:.2f} USDT")
                return False

            qty_contracts = notional_usdt / current_price
            # Comprehensive risk check before placing order
            current_volume = self.get_symbol_volume(symbol)
            if not self.risk_manager.can_open_position(symbol, clamped_margin, current_volume):
                self.logger.error(f"Risk management blocked order for {symbol}")
                return False
            # Additional safety checks
            if clamped_margin > MAX_POSITION_USDT:
                self.logger.error(f"Order size ${clamped_margin} exceeds maximum ${MAX_POSITION_USDT}")
                return False
            if leverage != LEVERAGE:
                self.logger.error(f"Leverage mismatch: requested {leverage}, configured {LEVERAGE}")
                return False

            # Final check: Enforce exchange min notional ($5) guard
            if notional_usdt < 5.0:
                self.logger.warning(f"Notional ${notional_usdt:.2f} below exchange minimum $5. Skipping {symbol} order.")
                return False
            # Build optional attached TP/SL and risk limit
            # Calculate order quantity in base currency to meet minimum order size
            qty_base = notional_usdt / current_price
            # Ensure we meet minimum order qty (0.1 for most perpetuals)
            min_order_qty = 0.1
            if qty_base < min_order_qty:
                qty_base = min_order_qty
                notional_usdt = qty_base * current_price
                clamped_margin = notional_usdt / leverage
            
            # Round to 0.1 step size (qtyStep from exchange)
            qty_base = round(qty_base * 10) / 10  # Round to nearest 0.1
            # Ensure we still meet minimum after rounding
            if qty_base < min_order_qty:
                qty_base = min_order_qty
            
            order_kwargs = {
                "category": "linear",
                "symbol": symbol,
                "side": side,
                "orderType": "Market",
                # Use base currency qty to meet minimum order requirements (0.1 step size)
                "qty": str(qty_base),  # Format as string, e.g. "0.1", "0.2", etc.
                "leverage": str(leverage)
                # Removed marketUnit - using base currency qty instead
            }

            try:
                from config.config import USE_TP_SL_ON_ORDER, RISK_LIMIT_ENABLED, RISK_LIMIT_VALUE, STOP_LOSS_PCT, TAKE_PROFIT_PCT
                if USE_TP_SL_ON_ORDER:
                    # Prefer existing precomputed TP/SL from market_state; otherwise compute a conservative fallback
                    stop_loss = self.market_state[symbol].get('stop_loss')
                    take_profit = self.market_state[symbol].get('take_profit')
                    if not stop_loss or not take_profit:
                        try:
                            # Compute ATR-based distances on the fly
                            import pandas as pd
                            recent_prices = pd.DataFrame(self.price_history[symbol][-20:]) if len(self.price_history[symbol]) > 0 else None
                            atr_val = self.calculate_atr(recent_prices, 14) if recent_prices is not None else 0
                        except Exception:
                            atr_val = 0
                        # Fallback to config percentages if ATR not available
                        stop_distance = max(atr_val * 1.5, current_price * float(STOP_LOSS_PCT)) if atr_val and atr_val > 0 else current_price * float(STOP_LOSS_PCT)
                        take_profit_distance = max(stop_distance * 2, current_price * float(TAKE_PROFIT_PCT))
                        if side == "Buy":
                            stop_loss = current_price - stop_distance
                            take_profit = current_price + take_profit_distance
                        else:
                            stop_loss = current_price + stop_distance
                            take_profit = current_price - take_profit_distance
                    order_kwargs.update({
                        "takeProfit": str(take_profit),
                        "stopLoss": str(stop_loss)
                    })
                if RISK_LIMIT_ENABLED and RISK_LIMIT_VALUE:
                    order_kwargs.update({"riskLimitValue": str(RISK_LIMIT_VALUE)})
            except Exception:
                pass

            response = self.session.place_order(
                **order_kwargs
            )
            if response['retCode'] == 0:
                self.logger.info(f"✅ Order placed: {side} notional {notional_usdt:.2f} USDT ({clamped_margin:.2f} margin, {leverage}x) of {symbol} at ${current_price:.2f}")
                self.risk_manager.positions_count += 1
                # Save trade record to database
                db_manager.save_trade_record(
                    symbol=symbol,
                    side=side,
                    quantity=qty_contracts,
                    price=current_price,
                    value_usdt=notional_usdt
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
            if not self.risk_manager:
                self.logger.error("Risk manager has not been set. Cannot run strategy.")
                return

            self.logger.info("=" * 60)
            self.logger.info("🔄 Running trading strategy (Phase 1 Alpha Improvements)")
            self.logger.info(f"   Symbols: {', '.join(SYMBOLS)}")
            self.logger.info(f"   MA Period: {MA_PERIOD}, Timeframe: {TIMEFRAME}min")
            self.logger.info(f"   MIN_TREND_STRENGTH: {MIN_TREND_STRENGTH}")
            self.logger.info(f"   Thresholds: 0.2%/0.05%/0.03% (High/Normal/Low vol)")
            self.logger.info(f"   MA Slope Requirement: REMOVED (Phase 1)")
            self.logger.info("=" * 60)
            
            # CRITICAL: Sync positions with Bybit first
            self.sync_positions_with_bybit()
            
            # Check risk management first
            risk_status = self.risk_manager.get_risk_status()
            if not risk_status['can_trade']:
                self.logger.error("❌ Risk management prevents trading.")
                self.logger.error(f"Daily Loss: ${risk_status['daily_loss']:.2f}/${risk_status['max_daily_loss_usdt']:.2f}")
                self.logger.error(f"Total Loss: ${risk_status['total_loss']:.2f}/${risk_status['max_total_loss_usdt']:.2f}")
                return

            # Update all market data before making decisions
            self.update_market_data()

            # Save balance snapshot periodically (every minute) for PnL chart evolution
            from datetime import datetime, timedelta
            last_snapshot_time = getattr(self, '_last_balance_snapshot', None)
            if not last_snapshot_time or (datetime.utcnow() - last_snapshot_time).total_seconds() >= 60:
                try:
                    balance_resp = self.session.get_wallet_balance(accountType="UNIFIED", coin="USDT")
                    if balance_resp.get('retCode') == 0:
                        wallet_balance = float(balance_resp['result']['list'][0]['coin'][0]['walletBalance'])
                        current_positions = self.get_current_positions()
                        unrealized_pnl = sum(pos.get('unrealized_pnl', 0) for pos in current_positions.values() if pos.get('position_size', 0) != 0)
                        # Calculate realized PnL from closed trades
                        from database import TradeRecord
                        trades = db_manager.session.query(TradeRecord).all()
                        realized_pnl = sum(trade.pnl for trade in trades if hasattr(trade, 'pnl') and trade.pnl is not None)
                        total_pnl = realized_pnl + unrealized_pnl
                        db_manager.save_balance_snapshot(wallet_balance, unrealized_pnl, total_pnl)
                        self._last_balance_snapshot = datetime.utcnow()
                except Exception as e:
                    self.logger.warning(f"Failed to save balance snapshot: {e}")

            # --- TRADING LOGIC ---
            for symbol in SYMBOLS:
                if len(self.price_history.get(symbol, [])) < MA_PERIOD:
                    self.logger.warning(f"Not enough historical data for {symbol} to calculate signal. Need {MA_PERIOD}, have {len(self.price_history.get(symbol, []))}.")
                    continue

                signal = self.calculate_ma_signal(symbol) # This now also updates market_state
                current_price = self.market_state[symbol].get('price', 0)
                ma_value = self.market_state[symbol].get('ma_value', 0)
                signal_name = self.market_state[symbol].get('signal', 'NEUTRAL')  # Get string signal name

                if current_price == 0:
                    self.logger.warning(f"Skipping {symbol} due to zero price.")
                    continue

                # Save signal to database (use string signal name, not numeric value)
                db_manager.save_signal_record(symbol, signal_name, ma_value, current_price)
                
                # Use market_state for position tracking (synced with Bybit)
                current_position_size = self.market_state[symbol]['position_size']
                current_position_side = self.market_state[symbol]['side']

                # Enhanced logging: Always log signals, detailed logging for non-neutral
                if self.market_state[symbol]['signal'] != 'NEUTRAL':
                    price_deviation = ((current_price - ma_value) / ma_value) * 100 if ma_value > 0 else 0
                    self.logger.info(
                        f"📊 {symbol}: Price=${current_price:.2f}, MA=${ma_value:.2f}, "
                        f"Signal='{self.market_state[symbol]['signal']}', "
                        f"Deviation={price_deviation:.3f}%, Position={current_position_size} ({current_position_side})"
                    )
                elif current_position_size != 0:
                    # Log position even if signal is neutral
                    self.logger.info(f"📊 {symbol}: Price=${current_price:.2f}, MA=${ma_value:.2f}, Signal='NEUTRAL', Position={current_position_size} ({current_position_side})")

                # Check if we should exit current position due to stop loss/take profit
                should_exit, exit_side = self.check_exit_conditions(symbol, current_price)
                if should_exit and current_position_size != 0:
                    position_value = abs(self.market_state[symbol]['position_value'])
                    success = self.place_order(symbol, exit_side, position_value)
                    if success:
                        self.logger.info(f"🎯 {exit_side} position closed for {symbol} (Stop Loss/Take Profit)")
                        self.market_state[symbol]['position_size'] = 0  # Reset position tracking
                        self.risk_manager.positions_count -= 1
                    continue  # Skip to next symbol after exit

                # Trading logic with improved risk management
                if signal == 1 and current_position_size == 0:
                    # Go long - only if we can open position
                    current_volume = self.get_symbol_volume(symbol)
                    if self.risk_manager.can_open_position(symbol, MAX_POSITION_USDT, current_volume):
                        success = self.place_order(symbol, "Buy", MAX_POSITION_USDT)
                        if success:
                            self.logger.info(f"🚀 LONG position opened for {symbol}")
                            # ATOMIC: Update position tracking immediately after successful order
                            from datetime import datetime
                            self.market_state[symbol]['position_size'] = MAX_POSITION_USDT
                            self.market_state[symbol]['side'] = 'Buy'
                            self.market_state[symbol]['entry_price'] = current_price
                            self.market_state[symbol]['position_value'] = MAX_POSITION_USDT
                            self.market_state[symbol]['entry_time'] = datetime.utcnow()
                            # Set stop loss and take profit
                            self.set_stop_loss_take_profit(symbol, current_price, MAX_POSITION_USDT, "Buy")
                    else:
                        self.logger.warning(f"⚠️ LONG signal for {symbol} but risk management blocked")
                elif signal == -1 and current_position_size == 0:
                    # Go short - only if we can open position
                    current_volume = self.get_symbol_volume(symbol)
                    if self.risk_manager.can_open_position(symbol, MAX_POSITION_USDT, current_volume):
                        success = self.place_order(symbol, "Sell", MAX_POSITION_USDT)
                        if success:
                            self.logger.info(f"🔻 SHORT position opened for {symbol}")
                            # ATOMIC: Update position tracking immediately after successful order
                            from datetime import datetime
                            self.market_state[symbol]['position_size'] = -MAX_POSITION_USDT
                            self.market_state[symbol]['side'] = 'Sell'
                            self.market_state[symbol]['entry_price'] = current_price
                            self.market_state[symbol]['position_value'] = -MAX_POSITION_USDT
                            self.market_state[symbol]['entry_time'] = datetime.utcnow()
                            # Set stop loss and take profit
                            self.set_stop_loss_take_profit(symbol, current_price, -MAX_POSITION_USDT, "Sell")
                    else:
                        self.logger.warning(f"⚠️ SHORT signal for {symbol} but risk management blocked")
                elif signal == 0 and current_position_size != 0:
                    # Close position when signal goes neutral (MA crossover)
                    position_value = abs(self.market_state[symbol]['position_value'])
                    side = "Sell" if current_position_side == "Buy" else "Buy"
                    
                    # Use reduceOnly order with quote currency to close
                    self.logger.info(f"Attempting to close position for {symbol} due to neutral signal.")
                    try:
                        response = self.session.place_order(
                            category="linear",
                            symbol=symbol,
                            side=side,
                            orderType="Market",
                            qty=str(position_value),
                            reduceOnly=True,
                            marketUnit="quoteCurrency"
                        )
                        if response.get('retCode') == 0:
                            self.logger.info(f"🔄 Position closed for {symbol} (MA crossover)")
                            # ATOMIC: Reset position tracking immediately after successful close
                            self.market_state[symbol]['position_size'] = 0
                            self.market_state[symbol]['position_value'] = 0
                            self.market_state[symbol]['side'] = 'N/A'
                            self.market_state[symbol]['entry_price'] = 0
                            self.market_state[symbol]['entry_time'] = None
                            self.market_state[symbol]['stop_loss'] = 0
                            self.market_state[symbol]['take_profit'] = 0
                            self.risk_manager.positions_count -= 1
                        else:
                            self.logger.error(f"Failed to close position for {symbol} on neutral signal. Response: {response}")
                    except Exception as e:
                        self.logger.exception(f"Exception while closing position for {symbol}: {e}")

                self.signals[symbol] = signal
                    
            # Log current state after all symbols are processed
            self.log_trading_state()
            self.logger.info("=" * 60)
        except Exception as e:
            self.logger.exception(f"An unexpected error occurred during strategy execution: {e}")

    def get_market_state(self):
        """Returns the current market state for the dashboard."""
        return self.market_state
        
    def log_trading_state(self):
        """Log the current state - only show active signals/positions"""
        # Only log symbols with signals or positions
        active_symbols = [s for s in SYMBOLS if self.market_state.get(s, {}).get('signal', 'NEUTRAL') != 'NEUTRAL' 
                         or self.market_state.get(s, {}).get('position_size', 0) > 0]
        
        if active_symbols:
            self.logger.info("📈 ACTIVE SIGNALS/POSITIONS")
            for symbol in active_symbols:
                state = self.market_state.get(symbol, {})
                price = state.get('price', 0)
                ma = state.get('ma_value', 0)
                signal = state.get('signal', 'N/A')
                pos_size = state.get('position_size', 0)
                self.logger.info(f"  {symbol}: Price=${price:.2f}, MA=${ma:.2f}, Signal='{signal}', Position={pos_size}")
        
        # Always log risk status
        risk_status = self.risk_manager.get_risk_status()
        self.logger.info(f"🛡️ Risk: Daily ${risk_status['daily_loss']:.2f}/${risk_status['max_daily_loss_usdt']:.2f} | Positions: {risk_status['positions_count']}/{risk_status['max_positions']}")

# Create a global, shared instance of the strategy AFTER the class is defined
# strategy = MovingAverageStrategy()

if __name__ == "__main__":
    # Note: This direct instantiation is for testing. 
    # The application uses the global 'strategy' instance.
    test_strategy = MovingAverageStrategy()
    # Run once for testing
    test_strategy.run_strategy()
