"""Strategy module for qr_backtester."""

from .base import Strategy
from .registry import StrategyRegistry, register_strategy, get_registry

__all__ = [
    "Strategy",
    "StrategyRegistry",
    "register_strategy",
    "get_registry",
]
