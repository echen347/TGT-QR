"""
qr_backtester - Realistic backtesting engine for quantitative research.

This package provides:
- Event-driven backtesting engine
- Realistic execution simulation with volume-aware slippage
- Intra-bar stop/take-profit detection
- Comprehensive performance metrics
- Strategy base class and registry

Example usage:
    from qr_backtester import BacktestEngine, Strategy, Order
    from qr_data import DataCache, BybitProvider
    from pathlib import Path
    from datetime import datetime, timedelta

    # Define a strategy
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

    # Get data
    cache = DataCache(Path("./data_cache"), BybitProvider())
    data = cache.get("ETHUSDT", "15m", datetime.now() - timedelta(days=30), datetime.now())

    # Run backtest
    engine = BacktestEngine(data, MACrossStrategy(10, 30))
    result = engine.run()

    print(result.summary())
"""

# Types
from .types import (
    Side,
    OrderType,
    OrderStatus,
    Bar,
    Order,
    Fill,
    Position,
    Trade,
    Metrics,
    BacktestResult,
)

# Execution
from .execution import (
    SlippageModel,
    FixedSlippage,
    VolumeAwareSlippage,
    SpreadSlippage,
    ExecutionEngine,
)

# Portfolio
from .portfolio import Portfolio

# Metrics
from .metrics import (
    calculate_metrics,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_exposure_time,
)

# Engine
from .engine import BacktestEngine

# Strategy
from .strategy import (
    Strategy,
    StrategyRegistry,
    register_strategy,
    get_registry,
)
from .strategy.base import SimpleMAStrategy, RSIStrategy

__version__ = "0.1.0"

__all__ = [
    # Types
    "Side",
    "OrderType",
    "OrderStatus",
    "Bar",
    "Order",
    "Fill",
    "Position",
    "Trade",
    "Metrics",
    "BacktestResult",
    # Execution
    "SlippageModel",
    "FixedSlippage",
    "VolumeAwareSlippage",
    "SpreadSlippage",
    "ExecutionEngine",
    # Portfolio
    "Portfolio",
    # Metrics
    "calculate_metrics",
    "calculate_sharpe_ratio",
    "calculate_sortino_ratio",
    "calculate_max_drawdown",
    "calculate_exposure_time",
    # Engine
    "BacktestEngine",
    # Strategy
    "Strategy",
    "StrategyRegistry",
    "register_strategy",
    "get_registry",
    "SimpleMAStrategy",
    "RSIStrategy",
]
