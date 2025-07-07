from typing import Dict, Type, List
import pkgutil
import importlib
import logging
import time

logger = logging.getLogger(__name__)


class MetricRegistry:
    """Decorator-based metric registry for automatic metric discovery."""

    def __init__(self):
        self.metrics: Dict[str, Type] = {}
        self._discovered = False

    def register_metric(self, name: str):
        """
        Decorator to register a metric class.

        Args:
            name: Name to register the metric under

        Returns:
            Decorator function

        Example:
            @register_metric("bleu")
            class BleuMetric(BaseMetric):
                pass
        """

        def decorator(metric_class: Type) -> Type:
            # Import BaseMetric here to avoid circular imports
            from karma.metrics.base_metric_abs import BaseMetric

            if not issubclass(metric_class, BaseMetric):
                raise ValueError(
                    f"{metric_class.__name__} must inherit from BaseMetric"
                )

            if name in self.metrics:
                logger.warning(
                    f"Metric '{name}' is already registered. Overriding with {metric_class.__name__}"
                )

            self.metrics[name] = metric_class
            logger.debug(f"Registered metric: {name} -> {metric_class.__name__}")
            return metric_class

        return decorator

    def get_metric_class(self, name: str) -> Type:
        """
        Get metric class by name.

        Args:
            name: Name of the metric to retrieve

        Returns:
            Metric class

        Raises:
            ValueError: If metric is not found
        """
        if not self._discovered:
            self.discover_metrics()

        if name not in self.metrics:
            # check if it's supported by the hf-evaluate library.
            try:
                from karma.metrics.common_metrics import HfMetric

                # check if the class can be initalised.
                metric = HfMetric(name)
                self.metrics[name] = HfMetric
                return metric
            except ValueError:
                available = list(self.metrics.keys())
                raise ValueError(
                    f"Metric '{name}' not found in KARMA or evaluate library. Available metrics: {available}"
                )
        return self.metrics[name]()

    def list_metrics(self) -> List[str]:
        """
        List available metric names.

        Returns:
            List of registered metric names
        """
        if not self._discovered:
            self.discover_metrics()
        return list(self.metrics.keys())

    def discover_metrics(self, use_cache: bool = True):
        """
        Automatically discover and import all metric modules.

        This method imports all modules in the karma.metrics package,
        which triggers the decorator registration. Uses caching for performance.
        
        Args:
            use_cache: Whether to use cached discovery results (default: True)
        """
        if self._discovered:
            return

        # Try to load from cache first
        if use_cache:
            cached_data = self._load_from_cache()
            if cached_data:
                logger.debug("Loaded metrics registry from cache")
                self._discovered = True
                return

        # Perform discovery
        start_time = time.time()
        logger.debug("Starting metrics discovery...")
        
        try:
            import karma.metrics

            # Import all modules in karma.metrics package
            for finder, name, ispkg in pkgutil.iter_modules(
                karma.metrics.__path__, karma.metrics.__name__ + "."
            ):
                # Skip base classes and utility modules
                if not name.endswith((".base_metric_abs", ".asr_wer_preprocessor")):
                    try:
                        importlib.import_module(name)
                        logger.debug(f"Imported metric module: {name}")
                    except ImportError as e:
                        logger.warning(f"Could not import metric module {name}: {e}")
        except ImportError as e:
            logger.error(f"Could not import karma.metrics package: {e}")

        discovery_time = time.time() - start_time
        logger.debug(f"Metrics discovery completed in {discovery_time:.2f}s")
        
        self._discovered = True
        
        # Cache the results
        if use_cache:
            self._save_to_cache()
    
    def _load_from_cache(self) -> bool:
        """
        Load registry data from cache.
        
        Returns:
            True if cache was loaded successfully, False otherwise
        """
        try:
            from karma.registries.cache_manager import get_cache_manager
            
            cache_manager = get_cache_manager()
            cached_data = cache_manager.get_cached_discovery("metrics")
            
            if cached_data:
                self.metrics = cached_data.get("metrics", {})
                return True
                
        except Exception as e:
            logger.debug(f"Failed to load metrics registry from cache: {e}")
            
        return False
    
    def _save_to_cache(self) -> None:
        """
        Save current registry data to cache.
        """
        try:
            from karma.registries.cache_manager import get_cache_manager
            
            cache_manager = get_cache_manager()
            cache_data = {
                "metrics": self.metrics
            }
            
            cache_manager.set_cached_discovery("metrics", cache_data)
            logger.debug("Saved metrics registry to cache")
            
        except Exception as e:
            logger.debug(f"Failed to save metrics registry to cache: {e}")

    def is_registered(self, name: str) -> bool:
        """
        Check if a metric is registered.

        Args:
            name: Name of the metric to check

        Returns:
            True if metric is registered, False otherwise
        """
        if not self._discovered:
            self.discover_metrics()
        return name in self.metrics

    def unregister_metric(self, name: str) -> bool:
        """
        Unregister a metric.

        Args:
            name: Name of the metric to unregister

        Returns:
            True if metric was unregistered, False if it wasn't registered
        """
        if name in self.metrics:
            del self.metrics[name]
            logger.debug(f"Unregistered metric: {name}")
            return True
        return False

    def clear_registry(self):
        """Clear all registered metrics. Mainly for testing purposes."""
        self.metrics.clear()
        self._discovered = False
        logger.debug("Cleared metric registry")


# Global registry instance
metric_registry = MetricRegistry()

# Convenience decorator function
register_metric = metric_registry.register_metric
