import logging
from datetime import datetime, timedelta
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.config import *
from database import db_manager

class RiskManager:
    """
    Ultra-conservative risk management system
    Designed to prevent significant losses at all costs
    """

    def __init__(self):
        self.logger = logging.getLogger('RiskManager')
        self.daily_loss = 0.0
        self.total_loss = 0.0
        self.positions_count = 0
        self.last_reset = datetime.utcnow()

    def reset_daily_loss(self):
        """Reset daily loss counter if it's a new day"""
        now = datetime.utcnow()
        if now.date() != self.last_reset.date():
            self.logger.info(f"Resetting daily loss from ${self.daily_loss:.2f} to $0.00")
            self.daily_loss = 0.0
            self.last_reset = now

    def check_volume_filter(self, symbol, current_volume):
        """Check if symbol meets minimum volume requirements"""
        if current_volume < MIN_VOLUME_USDT:
            self.logger.warning(f"Symbol {symbol} volume ${current_volume:,.0f} below minimum ${MIN_VOLUME_USDT:,.0f}")
            return False
        return True

    def check_position_limit(self):
        """Check if we can open more positions"""
        if self.positions_count >= MAX_POSITIONS:
            self.logger.warning(f"Position limit reached: {self.positions_count}/{MAX_POSITIONS}")
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

        all_passed = all(result for _, result in checks)

        if not all_passed:
            self.logger.error(f"Risk management blocked position for {symbol}")
            for check_name, result in checks:
                if not result:
                    self.logger.error(f"  - {check_name}: FAILED")

        return all_passed

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
        """Emergency stop - close all positions and halt trading"""
        self.logger.critical("=" * 80)
        self.logger.critical("EMERGENCY STOP ACTIVATED!")
        self.logger.critical(f"Daily Loss: ${self.daily_loss:.2f}")
        self.logger.critical(f"Total Loss: ${self.total_loss:.2f}")
        self.logger.critical("All trading operations should cease immediately!")
        self.logger.critical("=" * 80)

        # In a real system, this would close all positions
        # For now, just log the emergency
        return True

    def get_risk_status(self):
        """Get current risk status"""
        return {
            'daily_loss': self.daily_loss,
            'total_loss': self.total_loss,
            'positions_count': self.positions_count,
            'daily_loss_limit': MAX_DAILY_LOSS_USDT,
            'total_loss_limit': MAX_TOTAL_LOSS_USDT,
            'max_positions': MAX_POSITIONS,
            'can_trade': (self.daily_loss < MAX_DAILY_LOSS_USDT and
                         self.total_loss < MAX_TOTAL_LOSS_USDT and
                         self.positions_count < MAX_POSITIONS)
        }

# Global risk manager instance
risk_manager = RiskManager()
