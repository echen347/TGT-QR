"""Abstract base class for trading strategies."""

from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..types import Bar, Order, Fill
    from ..portfolio import Portfolio


class Strategy(ABC):
    """
    Abstract base class for trading strategies.

    Subclass this to implement your trading logic. The main method
    to implement is `on_bar()` which is called for each new bar.

    Example:
        class MACrossStrategy(Strategy):
            def __init__(self, fast: int = 10, slow: int = 30):
                self.fast = fast
                self.slow = slow

            @property
            def name(self) -> str:
                return f"MA_{self.fast}_{self.slow}"

            def on_bar(self, bar, portfolio):
                if len(portfolio.history) < self.slow:
                    return []

                fast_ma = portfolio.history["close"].rolling(self.fast).mean().iloc[-1]
                slow_ma = portfolio.history["close"].rolling(self.slow).mean().iloc[-1]

                if fast_ma > slow_ma and not portfolio.has_position:
                    return [Order.market_buy(size=100)]
                elif fast_ma < slow_ma and portfolio.has_position:
                    return [Order.market_sell(size=portfolio.position_size)]
                return []
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Return strategy name.

        Should be unique and descriptive. Used for logging and identification.
        """
        ...

    @abstractmethod
    def on_bar(self, bar: "Bar", portfolio: "Portfolio") -> list["Order"]:
        """
        Called on each new bar.

        This is the main method where trading logic should be implemented.

        Args:
            bar: Current bar data (timestamp, OHLCV)
            portfolio: Portfolio state (positions, equity, history)

        Returns:
            List of orders to execute (empty list if no action)
        """
        ...

    @property
    def params(self) -> dict:
        """
        Return strategy parameters.

        Override to return meaningful parameters for logging/optimization.
        Default implementation returns instance __dict__.
        """
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }

    def on_start(self, portfolio: "Portfolio") -> None:
        """
        Called before backtest starts.

        Override to initialize strategy state.

        Args:
            portfolio: Portfolio instance
        """
        pass

    def on_end(self, portfolio: "Portfolio") -> None:
        """
        Called after backtest ends.

        Override to cleanup or log final state.

        Args:
            portfolio: Portfolio instance
        """
        pass

    def on_fill(self, fill: "Fill") -> None:
        """
        Called when an order is filled.

        Override to track fills or update strategy state.

        Args:
            fill: Fill details
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, params={self.params})"


class SimpleMAStrategy(Strategy):
    """
    Simple moving average crossover strategy.

    Goes long when fast MA crosses above slow MA,
    closes position when fast MA crosses below slow MA.
    """

    def __init__(
        self,
        fast_period: int = 10,
        slow_period: int = 30,
        position_size: float = 100.0,
    ):
        """
        Initialize MA strategy.

        Args:
            fast_period: Fast MA period
            slow_period: Slow MA period
            position_size: Size per trade
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.position_size = position_size

    @property
    def name(self) -> str:
        return f"MA_{self.fast_period}_{self.slow_period}"

    def on_bar(self, bar: "Bar", portfolio: "Portfolio") -> list["Order"]:
        from ..types import Order

        # Need enough history for slow MA
        if len(portfolio.history) < self.slow_period:
            return []

        closes = portfolio.history["close"]
        fast_ma = closes.rolling(self.fast_period).mean().iloc[-1]
        slow_ma = closes.rolling(self.slow_period).mean().iloc[-1]

        # Check for crossover
        if len(closes) < 2:
            return []

        prev_fast_ma = closes.rolling(self.fast_period).mean().iloc[-2]
        prev_slow_ma = closes.rolling(self.slow_period).mean().iloc[-2]

        # Bullish crossover - go long
        if fast_ma > slow_ma and prev_fast_ma <= prev_slow_ma:
            if not portfolio.has_position:
                return [Order.market_buy(self.position_size)]

        # Bearish crossover - close long
        elif fast_ma < slow_ma and prev_fast_ma >= prev_slow_ma:
            if portfolio.is_long:
                return [Order.market_sell(portfolio.position_size)]

        return []


class RSIStrategy(Strategy):
    """
    RSI mean-reversion strategy.

    Goes long when RSI is oversold, closes when RSI is overbought.
    """

    def __init__(
        self,
        period: int = 14,
        oversold: float = 30,
        overbought: float = 70,
        position_size: float = 100.0,
    ):
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self.position_size = position_size

    @property
    def name(self) -> str:
        return f"RSI_{self.period}_{self.oversold}_{self.overbought}"

    def _calculate_rsi(self, closes) -> float:
        """Calculate RSI."""
        delta = closes.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()

        rs = gain / loss.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]

    def on_bar(self, bar: "Bar", portfolio: "Portfolio") -> list["Order"]:
        from ..types import Order

        if len(portfolio.history) < self.period + 1:
            return []

        rsi = self._calculate_rsi(portfolio.history["close"])

        # Oversold - go long
        if rsi < self.oversold and not portfolio.has_position:
            return [Order.market_buy(self.position_size)]

        # Overbought - close long
        elif rsi > self.overbought and portfolio.is_long:
            return [Order.market_sell(portfolio.position_size)]

        return []
