"""
Performance tracking module for sliding window adaptation.
Tracks recent performance and provides adaptive parameter recommendations.
"""
from datetime import datetime, timedelta
from database import db_manager, TradeRecord
import logging

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
        """Get adaptive parameter recommendations based on recent performance"""
        summary = self.get_performance_summary()
        metrics = summary['trades']
        
        recommendations = {
            'adjust_trend_strength': 1.0,  # Multiplier for MIN_TREND_STRENGTH
            'adjust_thresholds': 1.0,  # Multiplier for thresholds
            'adjust_position_size': 1.0,  # Multiplier for MAX_POSITION_USDT
            'reason': 'default'
        }
        
        # Need at least 5 trades to make recommendations
        if metrics['total_trades'] < 5:
            recommendations['reason'] = 'insufficient_data'
            return recommendations
        
        win_rate = metrics['win_rate']
        profit_factor = metrics['profit_factor']
        trades_per_day = summary['trades_per_day']
        
        # Underperforming: Tighten filters, reduce position size
        if win_rate < 0.45 or profit_factor < 1.0:
            recommendations['adjust_trend_strength'] = 1.5  # Require stronger trends
            recommendations['adjust_thresholds'] = 1.2  # Require larger deviations
            recommendations['adjust_position_size'] = 0.8  # Reduce position size
            recommendations['reason'] = f'underperforming (win_rate={win_rate:.2%}, pf={profit_factor:.2f})'
        
        # Performing well: Loosen filters slightly, maintain position size
        elif win_rate > 0.55 and profit_factor > 1.5:
            recommendations['adjust_trend_strength'] = 0.9  # Allow weaker trends
            recommendations['adjust_thresholds'] = 0.95  # Catch smaller deviations
            recommendations['adjust_position_size'] = 1.0  # Maintain position size
            recommendations['reason'] = f'performing_well (win_rate={win_rate:.2%}, pf={profit_factor:.2f})'
        
        # Low frequency: Loosen filters to get more signals
        elif trades_per_day < 0.5:
            recommendations['adjust_trend_strength'] = 0.85  # More lenient trend filter
            recommendations['adjust_thresholds'] = 0.9  # Smaller thresholds
            recommendations['adjust_position_size'] = 1.0
            recommendations['reason'] = f'low_frequency ({trades_per_day:.2f} trades/day)'
        
        # High frequency but poor performance: Tighten filters
        elif trades_per_day > 2.0 and win_rate < 0.50:
            recommendations['adjust_trend_strength'] = 1.3
            recommendations['adjust_thresholds'] = 1.15
            recommendations['adjust_position_size'] = 0.9
            recommendations['reason'] = f'high_frequency_poor ({trades_per_day:.2f} trades/day, {win_rate:.2%} win rate)'
        
        else:
            recommendations['reason'] = f'neutral (win_rate={win_rate:.2%}, {trades_per_day:.2f} trades/day)'
        
        return recommendations

