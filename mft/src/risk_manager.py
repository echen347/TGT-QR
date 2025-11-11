import logging
from datetime import datetime, timedelta
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.config import *
from database import db_manager

# Initialize logger for this module
logger = logging.getLogger(__name__)

class RiskManager:
    """Ultra-conservative risk management for educational purposes"""

    def __init__(self):
        """Initialize risk manager with conservative defaults"""
        self.daily_loss = 0
        self.total_loss = 0
        self.is_stopped = False
        self.last_daily_reset = datetime.now().date()
        self.alerts = []
        self.positions_count = 0
        self.strategy = None  # To hold a reference to the strategy instance
        self.logger = logging.getLogger(__name__)

    def set_strategy(self, strategy):
        """Set the strategy instance to allow for closing positions."""
        self.strategy = strategy

    def add_alert(self, level, message):
        """Add a new alert."""
        # Prepend to show newest first
        self.alerts.insert(0, {'level': level, 'message': f"[{datetime.now().strftime('%H:%M:%S')}] {message}"})
        # Keep only the last 5 alerts
        self.alerts = self.alerts[:5]

    def get_alerts(self):
        """Return current alerts."""
        return self.alerts

    def reset_daily_loss(self):
        """Reset daily loss counter if it's a new day"""
        now = datetime.utcnow()
        if now.date() != self.last_daily_reset:
            self.logger.info(f"Resetting daily loss from ${self.daily_loss:.2f} to $0.00")
            self.daily_loss = 0.0
            self.last_daily_reset = now.date()

    def check_volume_filter(self, symbol, current_volume):
        """Check if symbol meets minimum volume requirements"""
        if current_volume < MIN_VOLUME_USDT:
            self.logger.warning(f"Symbol {symbol} volume ${current_volume:,.0f} below minimum ${MIN_VOLUME_USDT:,.0f}")
            return False
        return True

    def check_position_limit(self):
        """Check if we can open more positions"""
        # Get actual position count from strategy
        actual_positions_count = 0
        if self.strategy:
            try:
                positions = self.strategy.get_current_positions()
                actual_positions_count = len(positions)
            except Exception as e:
                self.logger.error(f"Error getting position count: {e}")
        
        if actual_positions_count >= MAX_POSITIONS:
            self.logger.warning(f"Position limit reached: {actual_positions_count}/{MAX_POSITIONS}")
            return False
        return True

    def check_daily_loss_limit(self, potential_loss=0):
        """Check if adding this loss would exceed daily limit"""
        if self.daily_loss + potential_loss > MAX_DAILY_LOSS_USDT:
            self.logger.error(f"Daily loss limit would be exceeded: ${self.daily_loss + potential_loss:.2f}/${MAX_DAILY_LOSS_USDT:.2f}")
            return False
        return True

    def check_total_loss_limit(self, potential_loss=0):
        """Check if adding this loss would exceed total limit"""
        if self.total_loss + potential_loss > MAX_TOTAL_LOSS_USDT:
            self.logger.critical(f"TOTAL LOSS LIMIT EXCEEDED: ${self.total_loss + potential_loss:.2f}/${MAX_TOTAL_LOSS_USDT:.2f}")
            self.logger.critical("TRADING SYSTEM SHOULD STOP IMMEDIATELY!")
            return False
        return True

    def check_position_size(self, position_value):
        """Check if position size is within limits"""
        if position_value > MAX_POSITION_USDT:
            self.logger.error(f"Position size ${position_value:.2f} exceeds maximum ${MAX_POSITION_USDT:.2f}")
            return False
        return True

    def can_open_position(self, symbol, position_value, current_volume=0):
        """Comprehensive check before opening any position"""
        self.reset_daily_loss()

        # Check all risk constraints
        checks = [
            ("Volume Filter", self.check_volume_filter(symbol, current_volume)),
            ("Position Limit", self.check_position_limit()),
            ("Daily Loss Limit", self.check_daily_loss_limit()),
            ("Total Loss Limit", self.check_total_loss_limit()),
            ("Position Size", self.check_position_size(position_value))
        ]

        for check_name, result in checks:
            if not result:
                message = f"Risk Block: {check_name} failed for {symbol}."
                self.logger.warning(message)
                self.add_alert('warning', message)
                # Add detailed logging for debugging
                if check_name == "Position Limit":
                    actual_count = len(self.strategy.get_current_positions()) if self.strategy else 0
                    self.logger.warning(f"Position Limit: {actual_count}/{MAX_POSITIONS} positions")
                elif check_name == "Volume Filter":
                    self.logger.warning(f"Volume Filter: {current_volume} < {MIN_VOLUME_USDT}")
                elif check_name == "Position Size":
                    self.logger.warning(f"Position Size: {position_value} > {MAX_POSITION_USDT}")
                return False

        return True

    def record_loss(self, loss_amount):
        """Record a loss and update counters"""
        self.daily_loss += loss_amount
        self.total_loss += loss_amount

        self.logger.warning(f"Recorded loss: ${loss_amount:.4f}")
        self.logger.warning(f"Daily loss: ${self.daily_loss:.4f}/${MAX_DAILY_LOSS_USDT:.2f}")
        self.logger.warning(f"Total loss: ${self.total_loss:.4f}/${MAX_TOTAL_LOSS_USDT:.2f}")

        # Emergency stop if limits exceeded
        if self.daily_loss >= MAX_DAILY_LOSS_USDT:
            self.logger.critical("DAILY LOSS LIMIT REACHED - STOPPING TRADING!")
            self.emergency_stop()

        if self.total_loss >= MAX_TOTAL_LOSS_USDT:
            self.logger.critical("TOTAL LOSS LIMIT REACHED - STOPPING TRADING!")
            self.emergency_stop()

    def record_profit(self, profit_amount):
        """Record a profit and update counters"""
        self.daily_loss = max(0, self.daily_loss - profit_amount)  # Reduce daily loss if profitable

        self.logger.info(f"Recorded profit: ${profit_amount:.4f}")
        self.logger.info(f"Daily loss: ${self.daily_loss:.4f}/${MAX_DAILY_LOSS_USDT:.2f}")

    def emergency_stop(self):
        """Trigger emergency stop and add a critical alert"""
        self.is_stopped = True
        message = f"EMERGENCY STOP! Total loss ${self.total_loss:.2f} exceeded limit of ${MAX_TOTAL_LOSS_USDT:.2f}"
        self.add_alert('danger', message)
        self.logger.critical(message)
        
        if self.strategy:
            self.logger.warning("Attempting to close all positions immediately.")
            self.strategy.close_all_positions()
        else:
            self.logger.error("Strategy object not set in RiskManager. Cannot close positions immediately.")

    def reset_stop(self):
        """Resets the emergency stop state."""
        self.is_stopped = False
        self.logger.warning("RISK MANAGER: Emergency stop has been reset. Trading is re-enabled.")

    def get_risk_status(self):
        """Get current risk status including leverage"""
        from config.config import LEVERAGE
        # Get actual position count from strategy
        actual_positions_count = 0
        if self.strategy:
            try:
                positions = self.strategy.get_current_positions()
                actual_positions_count = len(positions)
            except Exception as e:
                self.logger.error(f"Error getting position count: {e}")
        
        return {
            'daily_loss': self.daily_loss,
            'total_loss': self.total_loss,
            'positions_count': actual_positions_count,  # Use actual count, not cached
            'max_position_usdt': MAX_POSITION_USDT,
            'max_daily_loss_usdt': MAX_DAILY_LOSS_USDT,
            'max_total_loss_usdt': MAX_TOTAL_LOSS_USDT,
            'max_positions': MAX_POSITIONS,
            'leverage': int(LEVERAGE),  # Add leverage for dashboard display
            'is_stopped': self.is_stopped,
            'can_trade': not self.is_stopped and self.daily_loss < MAX_DAILY_LOSS_USDT and self.total_loss < MAX_TOTAL_LOSS_USDT
        }

# Global risk manager instance
# risk_manager = RiskManager()
