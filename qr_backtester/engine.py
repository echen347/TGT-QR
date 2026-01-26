"""Main backtesting engine."""

from datetime import datetime
from typing import Optional, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from qr_data import OHLCVData

from .types import Bar, Order, BacktestResult, OrderType, OrderStatus
from .execution import ExecutionEngine, SlippageModel, VolumeAwareSlippage
from .portfolio import Portfolio
from .metrics import calculate_metrics


class BacktestEngine:
    """
    Event-driven backtesting engine.

    Simulates bar-by-bar trading with realistic execution including:
    - Volume-aware slippage
    - Intra-bar stop/take-profit detection
    - Commission modeling
    - Position tracking
    """

    def __init__(
        self,
        data: "OHLCVData",
        strategy: "Strategy",
        initial_capital: float = 10000.0,
        commission_bps: float = 4.0,
        slippage_model: Optional[SlippageModel] = None,
        leverage: float = 1.0,
        enable_partial_fills: bool = False,
    ):
        """
        Initialize backtest engine.

        Args:
            data: OHLCV data to backtest on
            strategy: Strategy instance to test
            initial_capital: Starting capital
            commission_bps: Commission in basis points
            slippage_model: Slippage model (default: VolumeAwareSlippage)
            leverage: Maximum leverage
            enable_partial_fills: Enable partial fill simulation
        """
        self.data = data
        self.strategy = strategy
        self.initial_capital = initial_capital

        # Execution engine
        if slippage_model is None:
            slippage_model = VolumeAwareSlippage()

        self.execution = ExecutionEngine(
            slippage_model=slippage_model,
            commission_bps=commission_bps,
            enable_partial_fills=enable_partial_fills,
        )

        # Portfolio
        self.portfolio = Portfolio(
            initial_capital=initial_capital,
            leverage=leverage,
        )

    def run(self) -> BacktestResult:
        """
        Run the backtest.

        Returns:
            BacktestResult with trades, equity curve, and metrics
        """
        # Reset portfolio
        self.portfolio.reset()

        # Initialize strategy
        if hasattr(self.strategy, 'on_start'):
            self.strategy.on_start(self.portfolio)

        # Process each bar
        for idx in range(len(self.data)):
            bar = Bar.from_series(self.data.df.iloc[idx], idx)

            # Update portfolio with new bar
            self.portfolio.update_bar(bar)

            # Check and fill pending orders (stops, TPs)
            self._process_pending_orders(bar, idx)

            # Get strategy orders
            orders = self.strategy.on_bar(bar, self.portfolio)

            # Process strategy orders
            if orders:
                for order in orders:
                    self._process_order(order, bar, idx)

        # Close any open position at end
        if self.portfolio.has_position:
            self._close_position_at_end()

        # Finalize strategy
        if hasattr(self.strategy, 'on_end'):
            self.strategy.on_end(self.portfolio)

        # Calculate metrics
        equity_curve = self.portfolio.get_equity_curve()
        metrics = calculate_metrics(
            self.portfolio.trades,
            equity_curve,
            self.initial_capital,
        )

        return BacktestResult(
            trades=self.portfolio.trades,
            equity_curve=equity_curve,
            metrics=metrics,
            orders=self.portfolio.all_orders,
            fills=self.portfolio.fills,
            parameters=self.strategy.params if hasattr(self.strategy, 'params') else {},
        )

    def _process_pending_orders(self, bar: Bar, bar_index: int) -> None:
        """Process pending stop and take profit orders."""
        if not self.portfolio.position:
            return

        orders_to_remove = []
        position_side = self.portfolio.position.side

        for order in self.portfolio.pending_orders:
            fill = None

            if order.order_type == OrderType.STOP:
                fill = self.execution.try_fill_stop_order(
                    order, bar, bar_index, position_side
                )
            elif order.order_type == OrderType.TAKE_PROFIT:
                fill = self.execution.try_fill_take_profit_order(
                    order, bar, bar_index, position_side
                )

            if fill:
                trade = self.portfolio.process_fill(fill)
                orders_to_remove.append(order.id)

                # Notify strategy
                if hasattr(self.strategy, 'on_fill'):
                    self.strategy.on_fill(fill)

                # If position closed, cancel remaining orders
                if not self.portfolio.has_position:
                    break

        # Remove filled orders
        self.portfolio.pending_orders = [
            o for o in self.portfolio.pending_orders
            if o.id not in orders_to_remove
        ]

    def _process_order(self, order: Order, bar: Bar, bar_index: int) -> None:
        """Process a new order from the strategy."""
        # Assign ID if not set
        submitted_order = self.portfolio.submit_order(order)

        if order.order_type == OrderType.MARKET:
            # Fill market orders immediately
            fill = self.execution.try_fill_market_order(submitted_order, bar, bar_index)
            if fill:
                self.portfolio.process_fill(fill)

                # Notify strategy
                if hasattr(self.strategy, 'on_fill'):
                    self.strategy.on_fill(fill)

    def _close_position_at_end(self) -> None:
        """Close any remaining position at the last bar."""
        if not self.portfolio.position:
            return

        last_bar = Bar.from_series(
            self.data.df.iloc[-1],
            len(self.data) - 1
        )

        # Create closing order
        close_order = Order.market_sell(self.portfolio.position.size)
        if self.portfolio.position.side.value == "short":
            close_order = Order.market_buy(self.portfolio.position.size)

        submitted = self.portfolio.submit_order(close_order)
        fill = self.execution.try_fill_market_order(
            submitted, last_bar, len(self.data) - 1
        )

        if fill:
            self.portfolio.process_fill(fill)

    def run_walkforward(
        self,
        train_pct: float = 0.6,
        n_folds: int = 1,
    ) -> dict:
        """
        Run walk-forward validation.

        Args:
            train_pct: Percentage of data for training
            n_folds: Number of folds (1 = simple train/test split)

        Returns:
            Dictionary with train and test results
        """
        df = self.data.df
        total_bars = len(df)

        if n_folds == 1:
            # Simple train/test split
            split_idx = int(total_bars * train_pct)

            train_df = df.iloc[:split_idx].reset_index(drop=True)
            test_df = df.iloc[split_idx:].reset_index(drop=True)

            # Import here to avoid circular import
            from qr_data import OHLCVData

            train_data = OHLCVData(
                symbol=self.data.symbol,
                timeframe=self.data.timeframe,
                df=train_df,
            )

            test_data = OHLCVData(
                symbol=self.data.symbol,
                timeframe=self.data.timeframe,
                df=test_df,
            )

            # Run train
            train_engine = BacktestEngine(
                data=train_data,
                strategy=self.strategy,
                initial_capital=self.initial_capital,
                slippage_model=self.execution.slippage_model,
                commission_bps=self.execution.commission_bps,
            )
            train_result = train_engine.run()

            # Run test
            test_engine = BacktestEngine(
                data=test_data,
                strategy=self.strategy,
                initial_capital=self.initial_capital,
                slippage_model=self.execution.slippage_model,
                commission_bps=self.execution.commission_bps,
            )
            test_result = test_engine.run()

            return {
                "train": train_result,
                "test": test_result,
                "train_period": (train_df["timestamp"].iloc[0], train_df["timestamp"].iloc[-1]),
                "test_period": (test_df["timestamp"].iloc[0], test_df["timestamp"].iloc[-1]),
            }

        # Multiple folds walk-forward
        fold_size = total_bars // n_folds
        results = []

        for i in range(n_folds):
            # Train on all data before this fold
            train_end = fold_size * (i + 1)
            train_start = 0

            # Test on this fold
            test_start = train_end
            test_end = min(test_start + fold_size, total_bars)

            if test_start >= total_bars:
                break

            train_df = df.iloc[train_start:train_end].reset_index(drop=True)
            test_df = df.iloc[test_start:test_end].reset_index(drop=True)

            from qr_data import OHLCVData

            train_data = OHLCVData(
                symbol=self.data.symbol,
                timeframe=self.data.timeframe,
                df=train_df,
            )

            test_data = OHLCVData(
                symbol=self.data.symbol,
                timeframe=self.data.timeframe,
                df=test_df,
            )

            train_engine = BacktestEngine(
                data=train_data,
                strategy=self.strategy,
                initial_capital=self.initial_capital,
                slippage_model=self.execution.slippage_model,
                commission_bps=self.execution.commission_bps,
            )

            test_engine = BacktestEngine(
                data=test_data,
                strategy=self.strategy,
                initial_capital=self.initial_capital,
                slippage_model=self.execution.slippage_model,
                commission_bps=self.execution.commission_bps,
            )

            results.append({
                "fold": i + 1,
                "train": train_engine.run(),
                "test": test_engine.run(),
                "train_period": (train_df["timestamp"].iloc[0], train_df["timestamp"].iloc[-1]),
                "test_period": (test_df["timestamp"].iloc[0], test_df["timestamp"].iloc[-1]),
            })

        return {"folds": results, "n_folds": n_folds}


# Import Strategy here to avoid circular import issues
from .strategy.base import Strategy
