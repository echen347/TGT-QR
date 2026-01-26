"""Strategy registry for discovery and registration."""

from typing import Type, Optional, Callable

from .base import Strategy


class StrategyRegistry:
    """
    Registry for strategy discovery and instantiation.

    Strategies can be registered via:
    1. @register_strategy decorator
    2. registry.register("name", StrategyClass)

    Example:
        # Using decorator
        @register_strategy
        class MyStrategy(Strategy):
            ...

        # Using explicit registration
        registry = get_registry()
        registry.register("my_strategy", MyStrategy)

        # Getting a strategy
        strategy = registry.get("my_strategy", param1=value1)

        # Listing strategies
        for name in registry.list_strategies():
            print(name)
    """

    def __init__(self):
        self._strategies: dict[str, Type[Strategy]] = {}

    def register(
        self,
        name: str,
        strategy_class: Type[Strategy],
        overwrite: bool = False,
    ) -> None:
        """
        Register a strategy class.

        Args:
            name: Strategy name (lowercase, underscores)
            strategy_class: Strategy class to register
            overwrite: If True, overwrite existing registration

        Raises:
            ValueError: If name already registered and overwrite=False
            TypeError: If strategy_class is not a Strategy subclass
        """
        if not isinstance(strategy_class, type) or not issubclass(strategy_class, Strategy):
            raise TypeError(f"{strategy_class} is not a Strategy subclass")

        name = name.lower().replace("-", "_")

        if name in self._strategies and not overwrite:
            raise ValueError(
                f"Strategy '{name}' already registered. "
                f"Use overwrite=True to replace."
            )

        self._strategies[name] = strategy_class

    def unregister(self, name: str) -> bool:
        """
        Unregister a strategy.

        Args:
            name: Strategy name

        Returns:
            True if strategy was unregistered, False if not found
        """
        name = name.lower().replace("-", "_")
        if name in self._strategies:
            del self._strategies[name]
            return True
        return False

    def get(self, name: str, **kwargs) -> Strategy:
        """
        Get a strategy instance.

        Args:
            name: Strategy name
            **kwargs: Parameters to pass to strategy constructor

        Returns:
            Strategy instance

        Raises:
            KeyError: If strategy not found
        """
        name = name.lower().replace("-", "_")

        if name not in self._strategies:
            available = ", ".join(sorted(self._strategies.keys()))
            raise KeyError(
                f"Strategy '{name}' not found. "
                f"Available: {available or 'none'}"
            )

        return self._strategies[name](**kwargs)

    def get_class(self, name: str) -> Type[Strategy]:
        """
        Get a strategy class (not instantiated).

        Args:
            name: Strategy name

        Returns:
            Strategy class
        """
        name = name.lower().replace("-", "_")

        if name not in self._strategies:
            raise KeyError(f"Strategy '{name}' not found")

        return self._strategies[name]

    def list_strategies(self) -> list[str]:
        """
        List all registered strategy names.

        Returns:
            Sorted list of strategy names
        """
        return sorted(self._strategies.keys())

    def is_registered(self, name: str) -> bool:
        """Check if a strategy is registered."""
        return name.lower().replace("-", "_") in self._strategies

    def clear(self) -> None:
        """Clear all registrations."""
        self._strategies.clear()

    def __len__(self) -> int:
        return len(self._strategies)

    def __contains__(self, name: str) -> bool:
        return self.is_registered(name)

    def __iter__(self):
        return iter(self._strategies.keys())


# Global registry instance
_global_registry: Optional[StrategyRegistry] = None


def get_registry() -> StrategyRegistry:
    """
    Get the global strategy registry.

    Returns:
        Global StrategyRegistry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = StrategyRegistry()
        _register_builtin_strategies(_global_registry)
    return _global_registry


def _register_builtin_strategies(registry: StrategyRegistry) -> None:
    """Register built-in strategies."""
    from .base import SimpleMAStrategy, RSIStrategy

    registry.register("simple_ma", SimpleMAStrategy)
    registry.register("ma_crossover", SimpleMAStrategy)  # Alias
    registry.register("rsi", RSIStrategy)


def register_strategy(
    cls: Optional[Type[Strategy]] = None,
    *,
    name: Optional[str] = None,
) -> Callable:
    """
    Decorator to register a strategy class.

    Can be used with or without arguments:

        @register_strategy
        class MyStrategy(Strategy):
            ...

        @register_strategy(name="custom_name")
        class AnotherStrategy(Strategy):
            ...

    Args:
        cls: Strategy class (when used without parentheses)
        name: Custom name (defaults to class name in snake_case)

    Returns:
        Decorated class
    """
    def decorator(strategy_class: Type[Strategy]) -> Type[Strategy]:
        # Determine strategy name
        strategy_name = name
        if strategy_name is None:
            # Convert CamelCase to snake_case
            class_name = strategy_class.__name__
            strategy_name = ""
            for i, char in enumerate(class_name):
                if char.isupper() and i > 0:
                    strategy_name += "_"
                strategy_name += char.lower()

        # Register with global registry
        registry = get_registry()
        registry.register(strategy_name, strategy_class, overwrite=True)

        return strategy_class

    # Handle both @register_strategy and @register_strategy()
    if cls is not None:
        return decorator(cls)
    return decorator
