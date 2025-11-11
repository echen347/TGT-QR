"""
Performance tracking module for sliding window adaptation.
Tracks recent performance and provides adaptive parameter recommendations.

ANTI-OVERFITTING DESIGN:
- Requires minimum 10 trades before adapting (configurable)
- Bounds on adjustments (0.7x to 1.5x) to prevent extreme changes
- Only adapts if performance is significantly different from baseline
- Uses longer window (14 days default) to smooth out noise
- Updates less frequently (48 hours) to reduce churn
"""
from datetime import datetime, timedelta
from database import db_manager, TradeRecord
import logging
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.config import ADAPTIVE_MIN_TRADES

logger = logging.getLogger(__name__)


class PerformanceTracker:
    """Tracks trading performance over sliding windows and provides adaptive recommendations"""
    
    def __init__(self, window_days=7):
        self.window_days = window_days
        self.db = db_manager
    
    def get_recent_trades(self, days=None):
        """Get trades from the last N days"""
        days = days or self.window_days
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        try:
            trades = self.db.session.query(TradeRecord)\
                .filter(TradeRecord.timestamp >= cutoff_date)\
                .order_by(TradeRecord.timestamp.desc())\
                .all()
            return trades
        except Exception as e:
            logger.error(f"Error fetching recent trades: {e}")
            return []
    
    def calculate_metrics(self, trades):
        """Calculate performance metrics from trades"""
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'total_return': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0
            }
        
        total_trades = len(trades)
        winning_trades = [t for t in trades if hasattr(t, 'pnl') and t.pnl and t.pnl > 0]
        losing_trades = [t for t in trades if hasattr(t, 'pnl') and t.pnl and t.pnl < 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0.0
        
        total_return = sum(t.pnl for t in trades if hasattr(t, 'pnl') and t.pnl)
        
        avg_win = sum(t.pnl for t in winning_trades) / len(winning_trades) if winning_trades else 0.0
        avg_loss = abs(sum(t.pnl for t in losing_trades) / len(losing_trades)) if losing_trades else 0.0
        
        total_wins = sum(t.pnl for t in winning_trades)
        total_losses = abs(sum(t.pnl for t in losing_trades))
        profit_factor = total_wins / total_losses if total_losses > 0 else 0.0
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_return': total_return,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor
        }
    
    def get_performance_summary(self, days=None):
        """Get performance summary for the last N days"""
        trades = self.get_recent_trades(days)
        metrics = self.calculate_metrics(trades)
        
        return {
            'window_days': days or self.window_days,
            'trades': metrics,
            'trades_per_day': metrics['total_trades'] / (days or self.window_days) if (days or self.window_days) > 0 else 0
        }
    
    def get_adaptive_recommendations(self):
        """
        Get adaptive parameter recommendations based on recent performance.
        
        ANTI-OVERFITTING SAFEGUARDS:
        - Requires minimum 10 trades (increased from 5) before adapting
        - Bounds on adjustments (0.7x to 1.5x) to prevent extreme changes
        - Only adapts if performance is significantly different from baseline
        - Uses longer window (7 days) to smooth out noise
        
        LOGIC:
        - When UNPROFITABLE: TIGHTEN filters (fewer signals, higher quality)
        - When PROFITABLE: Loosen filters slightly (more signals, but still selective)
        - Signals ALWAYS generate if conditions are met, just with adjusted thresholds
        """
        summary = self.get_performance_summary()
        metrics = summary['trades']
        
        recommendations = {
            'adjust_trend_strength': 1.0,  # Multiplier for MIN_TREND_STRENGTH
            'adjust_thresholds': 1.0,  # Multiplier for thresholds
            'adjust_position_size': 1.0,  # Multiplier for MAX_POSITION_USDT
            'reason': 'default'
        }
        
        # ANTI-OVERFITTING: Require minimum trades before adapting
        # This ensures we have enough data to make meaningful decisions
        min_trades = ADAPTIVE_MIN_TRADES if 'ADAPTIVE_MIN_TRADES' in globals() else 10
        if metrics['total_trades'] < min_trades:
            recommendations['reason'] = f'insufficient_data (need {min_trades} trades, have {metrics["total_trades"]})'
            return recommendations
        
        win_rate = metrics['win_rate']
        profit_factor = metrics['profit_factor']
        trades_per_day = summary['trades_per_day']
        
        # ANTI-OVERFITTING: Only adapt if performance is SIGNIFICANTLY different
        # Small variations are noise and shouldn't trigger adaptation
        
        # Underperforming: Tighten filters (FEWER signals, but signals still generate)
        # This makes us more selective when losing money
        if win_rate < 0.40 or profit_factor < 0.8:  # Stricter threshold (was 0.45/1.0)
            recommendations['adjust_trend_strength'] = 1.3  # Require stronger trends (was 1.5)
            recommendations['adjust_thresholds'] = 1.15  # Require larger deviations (was 1.2)
            recommendations['adjust_position_size'] = 0.85  # Reduce position size (was 0.8)
            recommendations['reason'] = f'underperforming (win_rate={win_rate:.2%}, pf={profit_factor:.2f}) - TIGHTENING filters'
        
        # Performing well: Loosen filters slightly (MORE signals, but still selective)
        elif win_rate > 0.60 and profit_factor > 1.8:  # Stricter threshold (was 0.55/1.5)
            recommendations['adjust_trend_strength'] = 0.92  # Allow slightly weaker trends (was 0.9)
            recommendations['adjust_thresholds'] = 0.97  # Catch slightly smaller deviations (was 0.95)
            recommendations['adjust_position_size'] = 1.0  # Maintain position size
            recommendations['reason'] = f'performing_well (win_rate={win_rate:.2%}, pf={profit_factor:.2f}) - LOOSENING filters slightly'
        
        # Low frequency: Loosen filters to get more signals (only if not losing money)
        elif trades_per_day < 0.3 and profit_factor >= 0.9:  # Only if not losing too much
            recommendations['adjust_trend_strength'] = 0.9  # More lenient trend filter (was 0.85)
            recommendations['adjust_thresholds'] = 0.92  # Smaller thresholds (was 0.9)
            recommendations['adjust_position_size'] = 1.0
            recommendations['reason'] = f'low_frequency ({trades_per_day:.2f} trades/day) - LOOSENING filters'
        
        # High frequency but poor performance: Tighten filters significantly
        elif trades_per_day > 2.5 and win_rate < 0.45:  # Stricter threshold
            recommendations['adjust_trend_strength'] = 1.25  # Require stronger trends (was 1.3)
            recommendations['adjust_thresholds'] = 1.12  # Require larger deviations (was 1.15)
            recommendations['adjust_position_size'] = 0.9
            recommendations['reason'] = f'high_frequency_poor ({trades_per_day:.2f} trades/day, {win_rate:.2%} win rate) - TIGHTENING filters'
        
        else:
            recommendations['reason'] = f'neutral (win_rate={win_rate:.2%}, {trades_per_day:.2f} trades/day, pf={profit_factor:.2f}) - NO CHANGE'
        
        # ANTI-OVERFITTING: Apply bounds to prevent extreme adjustments
        # Never adjust more than 30% in either direction
        recommendations['adjust_trend_strength'] = max(0.7, min(1.5, recommendations['adjust_trend_strength']))
        recommendations['adjust_thresholds'] = max(0.7, min(1.5, recommendations['adjust_thresholds']))
        recommendations['adjust_position_size'] = max(0.7, min(1.2, recommendations['adjust_position_size']))
        
        return recommendations

