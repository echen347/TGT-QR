"""Execution simulation with realistic slippage and fill models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from .types import Bar, Order, Fill, Side, OrderType


class SlippageModel(ABC):
    """Abstract base class for slippage models."""

    @abstractmethod
    def calculate_slippage_bps(
        self,
        order: Order,
        bar: Bar,
    ) -> float:
        """
        Calculate slippage in basis points.

        Args:
            order: Order to fill
            bar: Current bar data

        Returns:
            Slippage in basis points
        """
        ...


class FixedSlippage(SlippageModel):
    """Fixed slippage model - same slippage for all orders."""

    def __init__(self, slippage_bps: float = 2.0):
        """
        Initialize with fixed slippage.

        Args:
            slippage_bps: Fixed slippage in basis points
        """
        self.slippage_bps = slippage_bps

    def calculate_slippage_bps(self, order: Order, bar: Bar) -> float:
        return self.slippage_bps


class VolumeAwareSlippage(SlippageModel):
    """
    Volume-aware slippage model.

    Slippage increases with order size relative to bar volume:
    - Small orders (< 1% of volume): ~base_bps
    - Large orders (> 10% of volume): up to 5x base_bps
    """

    def __init__(
        self,
        base_bps: float = 2.0,
        max_multiplier: float = 5.0,
        volume_threshold: float = 0.10,
    ):
        """
        Initialize volume-aware slippage.

        Args:
            base_bps: Base slippage in basis points
            max_multiplier: Maximum slippage multiplier
            volume_threshold: Volume ratio threshold for max slippage
        """
        self.base_bps = base_bps
        self.max_multiplier = max_multiplier
        self.volume_threshold = volume_threshold

    def calculate_slippage_bps(self, order: Order, bar: Bar) -> float:
        if bar.volume <= 0:
            return self.base_bps * self.max_multiplier

        # Estimate order value relative to bar volume
        order_value = order.size * bar.close
        bar_value = bar.volume * bar.close

        volume_ratio = order_value / bar_value if bar_value > 0 else 1.0

        # Scale slippage based on volume ratio
        multiplier = 1.0 + (self.max_multiplier - 1.0) * min(
            volume_ratio / self.volume_threshold, 1.0
        )

        return self.base_bps * multiplier


class SpreadSlippage(SlippageModel):
    """
    Spread-based slippage model.

    Estimates slippage from bid-ask spread inferred from high-low range.
    """

    def __init__(self, spread_fraction: float = 0.5, min_bps: float = 1.0):
        """
        Initialize spread-based slippage.

        Args:
            spread_fraction: Fraction of high-low range to use as spread
            min_bps: Minimum slippage in basis points
        """
        self.spread_fraction = spread_fraction
        self.min_bps = min_bps

    def calculate_slippage_bps(self, order: Order, bar: Bar) -> float:
        if bar.close <= 0:
            return self.min_bps

        # Estimate spread from high-low range
        range_pct = (bar.high - bar.low) / bar.close
        spread_bps = range_pct * self.spread_fraction * 10000 / 2  # Half spread

        return max(spread_bps, self.min_bps)


@dataclass
class ExecutionEngine:
    """
    Order execution simulation.

    Handles:
    - Slippage calculation
    - Commission calculation
    - Fill price determination
    - Stop/limit order trigger detection
    """

    slippage_model: SlippageModel
    commission_bps: float = 4.0  # Bybit taker fee (0.04%)
    enable_partial_fills: bool = False
    max_fill_fraction: float = 0.10  # Max 10% of bar volume

    def calculate_fill_price(
        self,
        order: Order,
        bar: Bar,
        trigger_price: Optional[float] = None,
    ) -> float:
        """
        Calculate the fill price including slippage.

        Args:
            order: Order to fill
            bar: Current bar
            trigger_price: Price that triggered the order (for stops)

        Returns:
            Fill price after slippage
        """
        base_price = trigger_price if trigger_price else bar.open
        slippage_bps = self.slippage_model.calculate_slippage_bps(order, bar)
        slippage_pct = slippage_bps / 10000

        # Apply slippage in direction unfavorable to trader
        if order.side == Side.LONG:
            fill_price = base_price * (1 + slippage_pct)
        else:
            fill_price = base_price * (1 - slippage_pct)

        return fill_price

    def calculate_commission(self, size: float, price: float) -> float:
        """Calculate commission for a fill."""
        notional = size * price
        return notional * self.commission_bps / 10000

    def calculate_fill_size(self, order: Order, bar: Bar) -> float:
        """
        Calculate actual fill size (handles partial fills).

        Args:
            order: Order to fill
            bar: Current bar

        Returns:
            Fill size (may be less than order size if partial fills enabled)
        """
        if not self.enable_partial_fills:
            return order.size

        # Limit fill to fraction of bar volume
        max_fill_value = bar.volume * bar.close * self.max_fill_fraction
        max_fill_size = max_fill_value / bar.close if bar.close > 0 else order.size

        return min(order.size, max_fill_size)

    def try_fill_market_order(
        self,
        order: Order,
        bar: Bar,
        bar_index: int,
    ) -> Optional[Fill]:
        """
        Try to fill a market order.

        Args:
            order: Market order
            bar: Current bar
            bar_index: Current bar index

        Returns:
            Fill if order can be filled, None otherwise
        """
        if order.order_type != OrderType.MARKET:
            return None

        fill_size = self.calculate_fill_size(order, bar)
        fill_price = self.calculate_fill_price(order, bar)
        commission = self.calculate_commission(fill_size, fill_price)

        slippage_bps = self.slippage_model.calculate_slippage_bps(order, bar)
        slippage = fill_size * fill_price * slippage_bps / 10000

        return Fill(
            order_id=order.id,
            side=order.side,
            size=fill_size,
            price=fill_price,
            commission=commission,
            slippage=slippage,
            timestamp=bar.timestamp,
            bar_index=bar_index,
        )

    def check_stop_trigger(
        self,
        order: Order,
        bar: Bar,
        position_side: Side,
    ) -> Optional[float]:
        """
        Check if a stop order triggers on this bar.

        Uses intra-bar high/low to detect stop triggers.

        Args:
            order: Stop order
            bar: Current bar
            position_side: Side of the position (for determining trigger direction)

        Returns:
            Trigger price if stop triggered, None otherwise
        """
        if order.order_type != OrderType.STOP or order.stop_price is None:
            return None

        stop_price = order.stop_price

        # For long position, stop triggers when price drops below stop
        # For short position, stop triggers when price rises above stop
        if position_side == Side.LONG:
            if bar.low <= stop_price:
                # Check if it gapped down through stop
                if bar.open <= stop_price:
                    return bar.open  # Gap down - fill at open
                return stop_price  # Normal stop trigger
        else:
            if bar.high >= stop_price:
                # Check if it gapped up through stop
                if bar.open >= stop_price:
                    return bar.open  # Gap up - fill at open
                return stop_price  # Normal stop trigger

        return None

    def check_take_profit_trigger(
        self,
        order: Order,
        bar: Bar,
        position_side: Side,
    ) -> Optional[float]:
        """
        Check if a take profit order triggers on this bar.

        Args:
            order: Take profit order
            bar: Current bar
            position_side: Side of the position

        Returns:
            Trigger price if TP triggered, None otherwise
        """
        if order.order_type != OrderType.TAKE_PROFIT or order.price is None:
            return None

        tp_price = order.price

        # For long position, TP triggers when price rises above TP
        # For short position, TP triggers when price drops below TP
        if position_side == Side.LONG:
            if bar.high >= tp_price:
                if bar.open >= tp_price:
                    return bar.open  # Gap up - fill at open
                return tp_price
        else:
            if bar.low <= tp_price:
                if bar.open <= tp_price:
                    return bar.open  # Gap down - fill at open
                return tp_price

        return None

    def try_fill_stop_order(
        self,
        order: Order,
        bar: Bar,
        bar_index: int,
        position_side: Side,
    ) -> Optional[Fill]:
        """
        Try to fill a stop order.

        Args:
            order: Stop order
            bar: Current bar
            bar_index: Current bar index
            position_side: Side of the position

        Returns:
            Fill if stop triggered, None otherwise
        """
        trigger_price = self.check_stop_trigger(order, bar, position_side)
        if trigger_price is None:
            return None

        fill_size = self.calculate_fill_size(order, bar)
        fill_price = self.calculate_fill_price(order, bar, trigger_price)
        commission = self.calculate_commission(fill_size, fill_price)

        slippage = abs(fill_price - trigger_price) * fill_size

        return Fill(
            order_id=order.id,
            side=order.side,
            size=fill_size,
            price=fill_price,
            commission=commission,
            slippage=slippage,
            timestamp=bar.timestamp,
            bar_index=bar_index,
        )

    def try_fill_take_profit_order(
        self,
        order: Order,
        bar: Bar,
        bar_index: int,
        position_side: Side,
    ) -> Optional[Fill]:
        """
        Try to fill a take profit order.

        Args:
            order: Take profit order
            bar: Current bar
            bar_index: Current bar index
            position_side: Side of the position

        Returns:
            Fill if TP triggered, None otherwise
        """
        trigger_price = self.check_take_profit_trigger(order, bar, position_side)
        if trigger_price is None:
            return None

        fill_size = self.calculate_fill_size(order, bar)
        # For TP, we get the target price (favorable to us)
        fill_price = trigger_price
        commission = self.calculate_commission(fill_size, fill_price)

        return Fill(
            order_id=order.id,
            side=order.side,
            size=fill_size,
            price=fill_price,
            commission=commission,
            slippage=0.0,  # No slippage on favorable fills
            timestamp=bar.timestamp,
            bar_index=bar_index,
        )
