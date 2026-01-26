"""Core types for qr_backtester package."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

import pandas as pd


class Side(Enum):
    """Order/position side."""
    LONG = "long"
    SHORT = "short"


class OrderType(Enum):
    """Order type."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    TAKE_PROFIT = "take_profit"


class OrderStatus(Enum):
    """Order status."""
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Bar:
    """
    Single OHLCV bar.

    Attributes:
        timestamp: Bar timestamp
        open: Open price
        high: High price
        low: Low price
        close: Close price
        volume: Trading volume
        index: Bar index in the data series
    """
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    index: int = 0

    @classmethod
    def from_series(cls, row: pd.Series, index: int = 0) -> "Bar":
        """Create Bar from pandas Series."""
        return cls(
            timestamp=row["timestamp"],
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]),
            index=index,
        )


@dataclass
class Order:
    """
    Trading order.

    Attributes:
        id: Unique order ID
        side: Order side (LONG/SHORT)
        order_type: Order type (MARKET/LIMIT/STOP/TAKE_PROFIT)
        size: Order size in base currency
        price: Limit price (None for market orders)
        stop_price: Stop trigger price (for stop orders)
        reduce_only: If True, only reduces position
        created_at: Order creation timestamp
        status: Order status
    """
    id: str
    side: Side
    order_type: OrderType
    size: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    reduce_only: bool = False
    created_at: Optional[datetime] = None
    status: OrderStatus = OrderStatus.PENDING

    @classmethod
    def market_buy(cls, size: float, **kwargs) -> "Order":
        """Create market buy order."""
        return cls(
            id=kwargs.get("id", ""),
            side=Side.LONG,
            order_type=OrderType.MARKET,
            size=size,
            **{k: v for k, v in kwargs.items() if k != "id"},
        )

    @classmethod
    def market_sell(cls, size: float, **kwargs) -> "Order":
        """Create market sell order."""
        return cls(
            id=kwargs.get("id", ""),
            side=Side.SHORT,
            order_type=OrderType.MARKET,
            size=size,
            **{k: v for k, v in kwargs.items() if k != "id"},
        )

    @classmethod
    def limit_buy(cls, size: float, price: float, **kwargs) -> "Order":
        """Create limit buy order."""
        return cls(
            id=kwargs.get("id", ""),
            side=Side.LONG,
            order_type=OrderType.LIMIT,
            size=size,
            price=price,
            **{k: v for k, v in kwargs.items() if k != "id"},
        )

    @classmethod
    def limit_sell(cls, size: float, price: float, **kwargs) -> "Order":
        """Create limit sell order."""
        return cls(
            id=kwargs.get("id", ""),
            side=Side.SHORT,
            order_type=OrderType.LIMIT,
            size=size,
            price=price,
            **{k: v for k, v in kwargs.items() if k != "id"},
        )

    @classmethod
    def stop_loss(cls, side: Side, size: float, stop_price: float, **kwargs) -> "Order":
        """Create stop loss order."""
        # Stop loss side is opposite of position side
        order_side = Side.SHORT if side == Side.LONG else Side.LONG
        return cls(
            id=kwargs.get("id", ""),
            side=order_side,
            order_type=OrderType.STOP,
            size=size,
            stop_price=stop_price,
            reduce_only=True,
            **{k: v for k, v in kwargs.items() if k != "id"},
        )

    @classmethod
    def take_profit(cls, side: Side, size: float, price: float, **kwargs) -> "Order":
        """Create take profit order."""
        # Take profit side is opposite of position side
        order_side = Side.SHORT if side == Side.LONG else Side.LONG
        return cls(
            id=kwargs.get("id", ""),
            side=order_side,
            order_type=OrderType.TAKE_PROFIT,
            size=size,
            price=price,
            reduce_only=True,
            **{k: v for k, v in kwargs.items() if k != "id"},
        )


@dataclass
class Fill:
    """
    Order fill (execution).

    Attributes:
        order_id: Original order ID
        side: Fill side
        size: Filled size
        price: Fill price
        commission: Commission paid
        slippage: Slippage incurred
        timestamp: Fill timestamp
        bar_index: Bar index at fill time
    """
    order_id: str
    side: Side
    size: float
    price: float
    commission: float
    slippage: float
    timestamp: datetime
    bar_index: int


@dataclass
class Position:
    """
    Open position.

    Attributes:
        side: Position side (LONG/SHORT)
        size: Position size
        entry_price: Average entry price
        entry_time: Entry timestamp
        unrealized_pnl: Current unrealized P&L
        stop_loss: Stop loss price (optional)
        take_profit: Take profit price (optional)
    """
    side: Side
    size: float
    entry_price: float
    entry_time: datetime
    unrealized_pnl: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    def update_pnl(self, current_price: float) -> float:
        """Update and return unrealized P&L."""
        if self.side == Side.LONG:
            self.unrealized_pnl = (current_price - self.entry_price) * self.size
        else:
            self.unrealized_pnl = (self.entry_price - current_price) * self.size
        return self.unrealized_pnl


@dataclass
class Trade:
    """
    Completed trade (entry + exit).

    Attributes:
        id: Unique trade ID
        side: Trade side
        entry_price: Entry price
        exit_price: Exit price
        size: Trade size
        entry_time: Entry timestamp
        exit_time: Exit timestamp
        pnl: Realized P&L (after commissions)
        commission: Total commission paid
        return_pct: Return percentage
        duration: Trade duration
        exit_reason: Reason for exit (signal, stop_loss, take_profit)
    """
    id: str
    side: Side
    entry_price: float
    exit_price: float
    size: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    commission: float
    return_pct: float
    duration: timedelta
    exit_reason: str = "signal"


@dataclass
class Metrics:
    """
    Backtest performance metrics.

    Attributes:
        total_return: Total return percentage
        total_pnl: Total P&L in currency
        sharpe_ratio: Annualized Sharpe ratio
        sortino_ratio: Annualized Sortino ratio
        max_drawdown: Maximum drawdown percentage
        max_drawdown_duration: Duration of max drawdown
        win_rate: Win rate percentage
        profit_factor: Gross profit / gross loss
        num_trades: Total number of trades
        num_winning: Number of winning trades
        num_losing: Number of losing trades
        avg_trade_pnl: Average P&L per trade
        avg_winning_pnl: Average winning trade P&L
        avg_losing_pnl: Average losing trade P&L
        avg_trade_duration: Average trade duration
        largest_win: Largest winning trade
        largest_loss: Largest losing trade
        total_commission: Total commission paid
        exposure_time: Percentage of time in position
    """
    total_return: float = 0.0
    total_pnl: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: timedelta = field(default_factory=lambda: timedelta(0))
    win_rate: float = 0.0
    profit_factor: float = 0.0
    num_trades: int = 0
    num_winning: int = 0
    num_losing: int = 0
    avg_trade_pnl: float = 0.0
    avg_winning_pnl: float = 0.0
    avg_losing_pnl: float = 0.0
    avg_trade_duration: timedelta = field(default_factory=lambda: timedelta(0))
    largest_win: float = 0.0
    largest_loss: float = 0.0
    total_commission: float = 0.0
    exposure_time: float = 0.0


@dataclass
class BacktestResult:
    """
    Complete backtest result.

    Attributes:
        trades: List of completed trades
        equity_curve: Equity over time
        metrics: Performance metrics
        orders: All orders placed
        fills: All order fills
        parameters: Strategy parameters used
    """
    trades: list[Trade]
    equity_curve: pd.Series
    metrics: Metrics
    orders: list[Order] = field(default_factory=list)
    fills: list[Fill] = field(default_factory=list)
    parameters: dict = field(default_factory=dict)

    def summary(self) -> str:
        """Return formatted summary string."""
        m = self.metrics
        return f"""
Backtest Results
================
Total Return: {m.total_return:.2%}
Total P&L: ${m.total_pnl:.2f}
Sharpe Ratio: {m.sharpe_ratio:.2f}
Max Drawdown: {m.max_drawdown:.2%}
Win Rate: {m.win_rate:.2%}
Profit Factor: {m.profit_factor:.2f}
Number of Trades: {m.num_trades}
Avg Trade P&L: ${m.avg_trade_pnl:.2f}
Total Commission: ${m.total_commission:.2f}
"""

    def to_returns(self) -> pd.Series:
        """
        Convert equity curve to returns Series.

        Returns:
            pandas Series of periodic returns (compatible with QuantStats)
        """
        return self.equity_curve.pct_change().dropna()

    def tearsheet(
        self,
        output: str = "tearsheet.html",
        benchmark: Optional[str] = None,
    ) -> None:
        """
        Generate QuantStats HTML tearsheet.

        Requires quantstats to be installed: pip install quantstats

        Args:
            output: Output HTML file path
            benchmark: Benchmark ticker for comparison (e.g., "SPY", "BTC-USD")
        """
        try:
            import quantstats as qs
        except ImportError:
            raise ImportError(
                "quantstats is required for tearsheet generation. "
                "Install with: pip install quantstats"
            )

        returns = self.to_returns()
        qs.reports.html(returns, benchmark=benchmark, output=output)
