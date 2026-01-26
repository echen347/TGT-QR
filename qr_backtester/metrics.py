"""Performance metrics calculation for backtesting."""

from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd

from .types import Trade, Metrics


def calculate_metrics(
    trades: list[Trade],
    equity_curve: pd.Series,
    initial_capital: float,
    risk_free_rate: float = 0.0,
    periods_per_year: Optional[int] = None,  # Auto-detect from data if None
) -> Metrics:
    """
    Calculate comprehensive performance metrics.

    Args:
        trades: List of completed trades
        equity_curve: Equity over time
        initial_capital: Starting capital
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year for annualization

    Returns:
        Metrics dataclass with all calculated metrics
    """
    if not trades or equity_curve.empty:
        return Metrics()

    # Auto-detect periods_per_year from equity curve frequency
    if periods_per_year is None:
        if len(equity_curve) >= 2:
            # Infer from median time between observations
            time_diffs = pd.Series(equity_curve.index).diff().dropna()
            median_seconds = time_diffs.dt.total_seconds().median()
            if median_seconds > 0:
                seconds_per_year = 365.25 * 24 * 3600
                periods_per_year = int(seconds_per_year / median_seconds)
            else:
                periods_per_year = 252 * 24  # Fallback to hourly
        else:
            periods_per_year = 252 * 24  # Fallback to hourly

    # Basic trade statistics
    num_trades = len(trades)
    winning_trades = [t for t in trades if t.pnl > 0]
    losing_trades = [t for t in trades if t.pnl < 0]
    num_winning = len(winning_trades)
    num_losing = len(losing_trades)

    # P&L calculations
    total_pnl = sum(t.pnl for t in trades)
    total_commission = sum(t.commission for t in trades)

    # Return calculations
    final_equity = equity_curve.iloc[-1] if len(equity_curve) > 0 else initial_capital
    total_return = (final_equity - initial_capital) / initial_capital

    # Win rate
    win_rate = num_winning / num_trades if num_trades > 0 else 0.0

    # Profit factor
    gross_profit = sum(t.pnl for t in winning_trades) if winning_trades else 0
    gross_loss = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0

    # Average trade P&L
    avg_trade_pnl = total_pnl / num_trades if num_trades > 0 else 0
    avg_winning_pnl = sum(t.pnl for t in winning_trades) / num_winning if num_winning > 0 else 0
    avg_losing_pnl = sum(t.pnl for t in losing_trades) / num_losing if num_losing > 0 else 0

    # Largest win/loss
    largest_win = max((t.pnl for t in trades), default=0)
    largest_loss = min((t.pnl for t in trades), default=0)

    # Average trade duration
    if trades:
        total_duration = sum((t.duration for t in trades), timedelta())
        avg_duration = total_duration / num_trades
    else:
        avg_duration = timedelta()

    # Returns series for Sharpe/Sortino
    returns = equity_curve.pct_change().dropna()

    # Sharpe ratio
    sharpe_ratio = calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year)

    # Sortino ratio
    sortino_ratio = calculate_sortino_ratio(returns, risk_free_rate, periods_per_year)

    # Drawdown
    max_dd, max_dd_duration = calculate_max_drawdown(equity_curve)

    # Exposure time (percentage of time with position)
    exposure_time = calculate_exposure_time(trades, equity_curve)

    return Metrics(
        total_return=total_return,
        total_pnl=total_pnl,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        max_drawdown=max_dd,
        max_drawdown_duration=max_dd_duration,
        win_rate=win_rate,
        profit_factor=profit_factor,
        num_trades=num_trades,
        num_winning=num_winning,
        num_losing=num_losing,
        avg_trade_pnl=avg_trade_pnl,
        avg_winning_pnl=avg_winning_pnl,
        avg_losing_pnl=avg_losing_pnl,
        avg_trade_duration=avg_duration,
        largest_win=largest_win,
        largest_loss=largest_loss,
        total_commission=total_commission,
        exposure_time=exposure_time,
    )


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252 * 24,
) -> float:
    """
    Calculate annualized Sharpe ratio.

    Args:
        returns: Series of periodic returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year

    Returns:
        Annualized Sharpe ratio
    """
    if len(returns) < 2:
        return 0.0

    excess_returns = returns - (risk_free_rate / periods_per_year)
    mean_return = excess_returns.mean()
    std_return = excess_returns.std()

    if std_return == 0:
        return 0.0

    sharpe = mean_return / std_return * np.sqrt(periods_per_year)
    return float(sharpe)


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252 * 24,
) -> float:
    """
    Calculate annualized Sortino ratio.

    Uses downside deviation instead of standard deviation.

    Args:
        returns: Series of periodic returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year

    Returns:
        Annualized Sortino ratio
    """
    if len(returns) < 2:
        return 0.0

    excess_returns = returns - (risk_free_rate / periods_per_year)
    mean_return = excess_returns.mean()

    # Downside deviation (only negative returns)
    downside_returns = returns[returns < 0]
    if len(downside_returns) == 0:
        return float('inf') if mean_return > 0 else 0.0

    downside_std = downside_returns.std()
    if downside_std == 0:
        return float('inf') if mean_return > 0 else 0.0

    sortino = mean_return / downside_std * np.sqrt(periods_per_year)
    return float(sortino)


def calculate_max_drawdown(equity_curve: pd.Series) -> tuple[float, timedelta]:
    """
    Calculate maximum drawdown and its duration.

    Args:
        equity_curve: Equity over time

    Returns:
        Tuple of (max drawdown percentage, duration)
    """
    if len(equity_curve) < 2:
        return 0.0, timedelta()

    # Running maximum
    running_max = equity_curve.expanding().max()

    # Drawdown series
    drawdown = (equity_curve - running_max) / running_max

    # Maximum drawdown
    max_dd = abs(drawdown.min())

    # Duration calculation
    if max_dd == 0:
        return 0.0, timedelta()

    # Find the drawdown start and end
    max_dd_idx = drawdown.idxmin()
    peak_idx = equity_curve[:max_dd_idx].idxmax()

    # Find recovery point (when equity returns to peak)
    after_trough = equity_curve[max_dd_idx:]
    recovered = after_trough[after_trough >= equity_curve[peak_idx]]

    if len(recovered) > 0:
        recovery_idx = recovered.index[0]
        duration = recovery_idx - peak_idx
    else:
        # Not yet recovered
        duration = equity_curve.index[-1] - peak_idx

    return float(max_dd), duration


def calculate_exposure_time(
    trades: list[Trade],
    equity_curve: pd.Series,
) -> float:
    """
    Calculate percentage of time with open position.

    Args:
        trades: List of completed trades
        equity_curve: Equity over time

    Returns:
        Exposure time as percentage (0-1)
    """
    if not trades or equity_curve.empty:
        return 0.0

    total_period = equity_curve.index[-1] - equity_curve.index[0]
    if total_period.total_seconds() == 0:
        return 0.0

    # Sum up all trade durations
    total_exposure = sum((t.duration for t in trades), timedelta())

    return total_exposure.total_seconds() / total_period.total_seconds()


def calculate_calmar_ratio(
    total_return: float,
    max_drawdown: float,
    period_years: float,
) -> float:
    """
    Calculate Calmar ratio (annualized return / max drawdown).

    Args:
        total_return: Total return percentage
        max_drawdown: Maximum drawdown percentage
        period_years: Period in years

    Returns:
        Calmar ratio
    """
    if max_drawdown == 0 or period_years == 0:
        return float('inf') if total_return > 0 else 0.0

    annualized_return = (1 + total_return) ** (1 / period_years) - 1
    return annualized_return / max_drawdown


def calculate_recovery_factor(
    total_pnl: float,
    max_drawdown_value: float,
) -> float:
    """
    Calculate recovery factor (net profit / max drawdown).

    Args:
        total_pnl: Total P&L
        max_drawdown_value: Maximum drawdown in currency

    Returns:
        Recovery factor
    """
    if max_drawdown_value == 0:
        return float('inf') if total_pnl > 0 else 0.0

    return total_pnl / max_drawdown_value


def calculate_risk_adjusted_return(
    returns: pd.Series,
    target_return: float = 0.0,
) -> float:
    """
    Calculate risk-adjusted return (return / risk).

    Args:
        returns: Series of periodic returns
        target_return: Target return (default 0)

    Returns:
        Risk-adjusted return
    """
    if len(returns) < 2:
        return 0.0

    mean_return = returns.mean()
    std_return = returns.std()

    if std_return == 0:
        return 0.0

    return (mean_return - target_return) / std_return
