"""Portfolio and position management for backtesting."""

from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from .types import Bar, Order, Fill, Position, Trade, Side, OrderType, OrderStatus


class Portfolio:
    """
    Portfolio manager for tracking positions, orders, and P&L.

    Provides the strategy with access to:
    - Current position state
    - Historical price data
    - Account equity
    - Order management
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        leverage: float = 1.0,
    ):
        """
        Initialize portfolio.

        Args:
            initial_capital: Starting capital
            leverage: Maximum leverage multiplier
        """
        self.initial_capital = initial_capital
        self.leverage = leverage

        # Account state
        self.cash = initial_capital
        self.equity = initial_capital

        # Position tracking
        self.position: Optional[Position] = None
        self.pending_orders: list[Order] = []

        # History
        self.trades: list[Trade] = []
        self.fills: list[Fill] = []
        self.all_orders: list[Order] = []
        self.equity_history: list[tuple[datetime, float]] = []

        # Price history for strategy calculations
        # Pre-allocate DataFrame for efficiency on long backtests
        self._history: Optional[pd.DataFrame] = None
        self._history_list: list[dict] = []
        self._history_dirty: bool = False

        # State
        self._order_counter = 0
        self._trade_counter = 0
        self._current_bar: Optional[Bar] = None

    @property
    def has_position(self) -> bool:
        """Check if there is an open position."""
        return self.position is not None

    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.position is not None and self.position.side == Side.LONG

    @property
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.position is not None and self.position.side == Side.SHORT

    @property
    def position_size(self) -> float:
        """Get current position size (0 if no position)."""
        return self.position.size if self.position else 0.0

    @property
    def position_value(self) -> float:
        """Get current position value."""
        if self.position is None or self._current_bar is None:
            return 0.0
        return self.position.size * self._current_bar.close

    @property
    def history(self) -> pd.DataFrame:
        """Get price history as DataFrame."""
        if self._history is None or self._history_dirty:
            self._history = pd.DataFrame(self._history_list)
            self._history_dirty = False
        return self._history

    @property
    def current_price(self) -> float:
        """Get current bar close price."""
        return self._current_bar.close if self._current_bar else 0.0

    def _generate_order_id(self) -> str:
        """Generate unique order ID."""
        self._order_counter += 1
        return f"order_{self._order_counter}"

    def _generate_trade_id(self) -> str:
        """Generate unique trade ID."""
        self._trade_counter += 1
        return f"trade_{self._trade_counter}"

    def update_bar(self, bar: Bar) -> None:
        """
        Update portfolio with new bar data.

        Args:
            bar: New bar data
        """
        self._current_bar = bar

        # Add to history
        self._history_list.append({
            "timestamp": bar.timestamp,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
        })
        self._history_dirty = True  # Mark for rebuild on next access

        # Update position P&L
        if self.position:
            self.position.update_pnl(bar.close)

        # Update equity
        self._update_equity(bar)

        # Record equity history
        self.equity_history.append((bar.timestamp, self.equity))

    def _update_equity(self, bar: Bar) -> None:
        """Update account equity."""
        unrealized_pnl = self.position.unrealized_pnl if self.position else 0.0
        self.equity = self.cash + unrealized_pnl

    def submit_order(self, order: Order) -> Order:
        """
        Submit an order.

        Args:
            order: Order to submit

        Returns:
            Order with assigned ID
        """
        if not order.id:
            order.id = self._generate_order_id()

        if order.created_at is None:
            order.created_at = self._current_bar.timestamp if self._current_bar else datetime.now()

        self.all_orders.append(order)

        if order.order_type == OrderType.MARKET:
            # Market orders are filled immediately (handled by engine)
            pass
        else:
            # Pending orders (stop, limit, take profit)
            self.pending_orders.append(order)

        return order

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a pending order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if order was cancelled
        """
        for i, order in enumerate(self.pending_orders):
            if order.id == order_id:
                order.status = OrderStatus.CANCELLED
                self.pending_orders.pop(i)
                return True
        return False

    def cancel_all_orders(self) -> int:
        """
        Cancel all pending orders.

        Returns:
            Number of orders cancelled
        """
        count = len(self.pending_orders)
        for order in self.pending_orders:
            order.status = OrderStatus.CANCELLED
        self.pending_orders.clear()
        return count

    def process_fill(self, fill: Fill) -> Optional[Trade]:
        """
        Process an order fill.

        Args:
            fill: Fill to process

        Returns:
            Trade if position was closed, None otherwise
        """
        self.fills.append(fill)

        # Update order status
        for order in self.all_orders:
            if order.id == fill.order_id:
                order.status = OrderStatus.FILLED
                break

        # Remove from pending if present
        self.pending_orders = [o for o in self.pending_orders if o.id != fill.order_id]

        # Handle position changes
        if self.position is None:
            # Opening new position
            self._open_position(fill)
            return None

        # Check if fill is on same side (adding to position) or opposite (closing)
        if fill.side == self.position.side:
            # Adding to position
            self._add_to_position(fill)
            return None
        else:
            # Closing position
            return self._close_position(fill)

    def _open_position(self, fill: Fill) -> None:
        """Open a new position from a fill."""
        self.position = Position(
            side=fill.side,
            size=fill.size,
            entry_price=fill.price,
            entry_time=fill.timestamp,
        )

        # Deduct commission from cash
        self.cash -= fill.commission

    def _add_to_position(self, fill: Fill) -> None:
        """Add to existing position."""
        if self.position is None:
            return

        # Calculate new average entry price
        total_value = (self.position.size * self.position.entry_price) + (fill.size * fill.price)
        total_size = self.position.size + fill.size

        self.position.size = total_size
        self.position.entry_price = total_value / total_size

        # Deduct commission
        self.cash -= fill.commission

    def _close_position(self, fill: Fill) -> Optional[Trade]:
        """Close position and record trade."""
        if self.position is None:
            return None

        close_size = min(fill.size, self.position.size)

        # Calculate P&L
        if self.position.side == Side.LONG:
            pnl = (fill.price - self.position.entry_price) * close_size
        else:
            pnl = (self.position.entry_price - fill.price) * close_size

        # Deduct commission
        pnl -= fill.commission

        # Calculate return percentage
        entry_value = self.position.entry_price * close_size
        return_pct = pnl / entry_value if entry_value > 0 else 0.0

        # Determine exit reason
        exit_reason = "signal"
        for order in self.all_orders:
            if order.id == fill.order_id:
                if order.order_type == OrderType.STOP:
                    exit_reason = "stop_loss"
                elif order.order_type == OrderType.TAKE_PROFIT:
                    exit_reason = "take_profit"
                break

        # Create trade record
        trade = Trade(
            id=self._generate_trade_id(),
            side=self.position.side,
            entry_price=self.position.entry_price,
            exit_price=fill.price,
            size=close_size,
            entry_time=self.position.entry_time,
            exit_time=fill.timestamp,
            pnl=pnl,
            commission=fill.commission,
            return_pct=return_pct,
            duration=fill.timestamp - self.position.entry_time,
            exit_reason=exit_reason,
        )

        self.trades.append(trade)

        # Update cash
        self.cash += pnl + (self.position.entry_price * close_size)

        # Update position
        remaining_size = self.position.size - close_size
        if remaining_size <= 0:
            self.position = None
            # Cancel any remaining stop/TP orders
            self.pending_orders = [
                o for o in self.pending_orders
                if not o.reduce_only
            ]
        else:
            self.position.size = remaining_size

        return trade

    def get_equity_curve(self) -> pd.Series:
        """Get equity curve as pandas Series."""
        if not self.equity_history:
            return pd.Series(dtype=float)

        timestamps, values = zip(*self.equity_history)
        return pd.Series(values, index=pd.DatetimeIndex(timestamps))

    def get_position_summary(self) -> dict:
        """Get current position summary."""
        if self.position is None:
            return {
                "has_position": False,
                "side": None,
                "size": 0,
                "entry_price": 0,
                "unrealized_pnl": 0,
                "current_price": self.current_price,
            }

        return {
            "has_position": True,
            "side": self.position.side.value,
            "size": self.position.size,
            "entry_price": self.position.entry_price,
            "unrealized_pnl": self.position.unrealized_pnl,
            "current_price": self.current_price,
            "stop_loss": self.position.stop_loss,
            "take_profit": self.position.take_profit,
        }

    def reset(self) -> None:
        """Reset portfolio to initial state."""
        self.cash = self.initial_capital
        self.equity = self.initial_capital
        self.position = None
        self.pending_orders = []
        self.trades = []
        self.fills = []
        self.all_orders = []
        self.equity_history = []
        self._history = None
        self._history_list = []
        self._history_dirty = False
        self._order_counter = 0
        self._trade_counter = 0
        self._current_bar = None
