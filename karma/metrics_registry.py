from typing import Dict, Type, List
import pkgutil
import importlib
import logging

logger = logging.getLogger(__name__)

class BaseMetric:
    def __init__(self, metric_name: str):
        self.metric_name = metric_name

    def evaluate(self, predictions, references):
        raise NotImplementedError

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
            if not issubclass(metric_class, BaseMetric):
                raise ValueError(f"{metric_class.__name__} must inherit from BaseMetric")
            
            if name in self.metrics:
                logger.warning(f"Metric '{name}' is already registered. Overriding with {metric_class.__name__}")
            
            self.metrics[name] = metric_class
            logger.debug(f"Registered metric: {name} -> {metric_class.__name__}")
            return metric_class
        return decorator
    
    def get_metric(self, name: str) -> Type:
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
            available = list(self.metrics.keys())
            raise ValueError(f"Metric '{name}' not found. Available metrics: {available}")
        return self.metrics[name]
    
    def list_metrics(self) -> List[str]:
        """
        List available metric names.
        
        Returns:
            List of registered metric names
        """
        if not self._discovered:
            self.discover_metrics()
        return list(self.metrics.keys())
    
    def discover_metrics(self):
        """
        Automatically discover and import all metric modules.
        
        This method imports the karma.metrics module,
        which triggers the decorator registration.
        """
        if self._discovered:
            return
            
        try:
            # Import the metrics module to trigger decorator registration
            importlib.import_module("karma.metrics")
            logger.debug("Imported karma.metrics module")
        except ImportError as e:
            logger.error(f"Could not import karma.metrics package: {e}")
            
        self._discovered = True
    
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