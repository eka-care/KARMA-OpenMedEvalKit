from typing import Dict, Type, List, Optional, Any
import pkgutil
import importlib
import logging
import time

from karma.utils.argument_processing import (
    validate_registry_args,
    prepare_registry_metadata,
)

logger = logging.getLogger(__name__)


class MetricRegistry:
    """Decorator-based metric registry for automatic metric discovery."""

    def __init__(self):
        self.metrics: Dict[str, Dict[str, Any]] = {}
        self._discovered = False

    def register_metric(
        self,
        name: str,
        required_args: Optional[List[str]] = None,
        optional_args: Optional[List[str]] = None,
        default_args: Optional[Dict[str, Any]] = None,
    ):
        """
        Decorator to register a metric class with its argument metadata.

        Args:
            name: Name to register the metric under
            required_args: List of required argument names for metric instantiation
            optional_args: List of optional argument names for metric instantiation
            default_args: Dictionary of default values for arguments

        Returns:
            Decorator function

        Examples:
            @register_metric("bleu")
            class BleuMetric(BaseMetric):
                pass

            @register_metric(
                name="f1",
                required_args=["average"],
                optional_args=["labels", "pos_label"],
                default_args={"average": "binary"}
            )
            class F1Metric(BaseMetric):
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

            # Prepare metadata using shared utility
            metadata = prepare_registry_metadata(
                required_args=required_args,
                optional_args=optional_args,
                default_args=default_args,
            )

            # Store metric class and metadata
            self.metrics[name] = {
                "class": metric_class,
                "module": metric_class.__module__,
                "class_name": metric_class.__name__,
                "required_args": metadata["required_args"],
                "optional_args": metadata["optional_args"],
                "default_args": metadata["default_args"],
            }

            logger.debug(f"Registered metric: {name} -> {metric_class.__name__}")
            return metric_class

        return decorator

    def get_metric_class(self, name: str, **kwargs):
        """
        Get metric instance by name with optional constructor arguments.

        Args:
            name: Name of the metric to retrieve
            **kwargs: Additional arguments to pass to the metric constructor

        Returns:
            Metric instance

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
                metric = HfMetric(name, **kwargs)
                # Store as simple class reference for HF metrics (no metadata)
                self.metrics[name] = {
                    "class": HfMetric,
                    "module": HfMetric.__module__,
                    "class_name": HfMetric.__name__,
                    "required_args": [],
                    "optional_args": [],
                    "default_args": {},
                }
                return metric
            except ValueError:
                available = list(self.metrics.keys())
                raise ValueError(
                    f"Metric '{name}' not found in KARMA or evaluate library. Available metrics: {available}"
                )
        # Get the metric class from the registry
        metric_info = self.metrics[name]
        metric_class = metric_info["class"]
        return metric_class(name, **kwargs)

    def list_metrics(self) -> List[str]:
        """
        List available metric names.

        Returns:
            List of registered metric names
        """
        if not self._discovered:
            self.discover_metrics()
        return list(self.metrics.keys())

    def validate_metric_args(
        self, metric_name: str, provided_args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate metric arguments against registry requirements.

        Args:
            metric_name: Name of the metric
            provided_args: Arguments provided by user

        Returns:
            Validated and merged arguments

        Raises:
            ValueError: If validation fails
        """
        if not self._discovered:
            self.discover_metrics()

        if metric_name not in self.metrics:
            raise ValueError(f"Metric '{metric_name}' not found in registry")

        metric_info = self.metrics[metric_name]

        return validate_registry_args(
            registry_name="metric",
            item_name=metric_name,
            provided_args=provided_args,
            required_args=metric_info["required_args"],
            optional_args=metric_info["optional_args"],
            default_args=metric_info["default_args"],
        )

    def get_metric_required_args(self, metric_name: str) -> List[str]:
        """
        Get required arguments for a metric.

        Args:
            metric_name: Name of the metric

        Returns:
            List of required argument names

        Raises:
            ValueError: If metric is not found
        """
        if not self._discovered:
            self.discover_metrics()

        if metric_name not in self.metrics:
            raise ValueError(f"Metric '{metric_name}' not found in registry")

        return self.metrics[metric_name]["required_args"].copy()

    def get_metric_optional_args(self, metric_name: str) -> List[str]:
        """
        Get optional arguments for a metric.

        Args:
            metric_name: Name of the metric

        Returns:
            List of optional argument names

        Raises:
            ValueError: If metric is not found
        """
        if not self._discovered:
            self.discover_metrics()

        if metric_name not in self.metrics:
            raise ValueError(f"Metric '{metric_name}' not found in registry")

        return self.metrics[metric_name]["optional_args"].copy()

    def get_metric_default_args(self, metric_name: str) -> Dict[str, Any]:
        """
        Get default arguments for a metric.

        Args:
            metric_name: Name of the metric

        Returns:
            Dictionary of default argument values

        Raises:
            ValueError: If metric is not found
        """
        if not self._discovered:
            self.discover_metrics()

        if metric_name not in self.metrics:
            raise ValueError(f"Metric '{metric_name}' not found in registry")

        return self.metrics[metric_name]["default_args"].copy()

    def get_metric_all_args(self, metric_name: str) -> Dict[str, Any]:
        """
        Get all argument information for a metric.

        Args:
            metric_name: Name of the metric

        Returns:
            Dictionary containing required, optional, and default arguments

        Raises:
            ValueError: If metric is not found
        """
        if not self._discovered:
            self.discover_metrics()

        if metric_name not in self.metrics:
            raise ValueError(f"Metric '{metric_name}' not found in registry")

        info = self.metrics[metric_name]
        return {
            "required": info["required_args"].copy(),
            "optional": info["optional_args"].copy(),
            "defaults": info["default_args"].copy(),
        }

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

            # Import all modules in karma.metrics package recursively
            for finder, name, ispkg in pkgutil.walk_packages(
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
            cache_data = {"metrics": self.metrics}

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
