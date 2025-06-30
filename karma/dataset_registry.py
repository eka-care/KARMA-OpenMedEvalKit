"""
Dataset registry for automatic dataset discovery and registration.

This module provides a decorator-based registry system that allows datasets
to register themselves automatically when imported, along with their metadata.
"""

import importlib
import pkgutil
from typing import Dict, Type, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class DatasetRegistry:
    """Decorator-based registry for datasets with their metrics and metadata."""
    
    def __init__(self):
        self.datasets: Dict[str, Dict[str, Any]] = {}
        self._discovered = False
    
    def register_dataset(
        self, 
        name: str, 
        metrics: List[str], 
        task_type: str = "mcqa",
        required_args: Optional[List[str]] = None,
        optional_args: Optional[List[str]] = None,
        default_args: Optional[Dict[str, Any]] = None
    ):
        """
        Decorator to register a dataset class with its metrics and metadata.
        
        Args:
            name: Name to register the dataset under
            metrics: List of metrics that can be computed for this dataset
            task_type: Type of task (e.g., "mcqa", "vqa", "qa", "translation")
            required_args: List of required argument names for dataset instantiation
            optional_args: List of optional argument names for dataset instantiation
            default_args: Dictionary of default values for arguments
            
        Returns:
            Decorator function
            
        Examples:
            @register_dataset("pubmedqa", metrics=["accuracy"], task_type="mcqa")
            class PubMedQADataset(BaseMultimodalDataset):
                pass
                
            @register_dataset(
                "in22conv", 
                metrics=["bleu", "accuracy"], 
                task_type="translation",
                required_args=["source_language", "target_language"],
                optional_args=["domain", "variant"],
                default_args={"domain": "general"}
            )
            class In22ConvDataset(BaseMultimodalDataset):
                pass
        """
        def decorator(dataset_class: Type) -> Type:
            # Import BaseMultimodalDataset here to avoid circular imports
            from karma.eval_datasets.base_dataset import BaseMultimodalDataset
            
            if not issubclass(dataset_class, BaseMultimodalDataset):
                raise ValueError(f"{dataset_class.__name__} must inherit from BaseMultimodalDataset")
            
            if name in self.datasets:
                logger.warning(f"Dataset '{name}' is already registered. Overriding with {dataset_class.__name__}")
            
            # Process arguments
            processed_metrics = metrics.copy() if isinstance(metrics, list) else [metrics]
            processed_required_args = required_args.copy() if required_args else []
            processed_optional_args = optional_args.copy() if optional_args else []
            processed_default_args = default_args.copy() if default_args else {}
            
            self.datasets[name] = {
                'class': dataset_class,
                'metrics': processed_metrics,
                'task_type': task_type,
                'required_args': processed_required_args,
                'optional_args': processed_optional_args,
                'default_args': processed_default_args,
                'module': dataset_class.__module__,
                'class_name': dataset_class.__name__
            }
            logger.debug(f"Registered dataset: {name} -> {dataset_class.__name__}")
            return dataset_class
        return decorator
    
    def get_dataset_info(self, name: str) -> Dict[str, Any]:
        """
        Get dataset information including class, metrics, and metadata.
        
        Args:
            name: Name of the dataset to retrieve
            
        Returns:
            Dictionary containing dataset information
            
        Raises:
            ValueError: If dataset is not found
        """
        if not self._discovered:
            self.discover_datasets()
            
        if name not in self.datasets:
            available = list(self.datasets.keys())
            raise ValueError(f"Dataset '{name}' not found. Available datasets: {available}")
        return self.datasets[name].copy()
    
    def get_dataset_class(self, name: str) -> Type:
        """
        Get dataset class by name.
        
        Args:
            name: Name of the dataset to retrieve
            
        Returns:
            Dataset class
            
        Raises:
            ValueError: If dataset is not found
        """
        info = self.get_dataset_info(name)
        return info['class']
    
    def list_datasets(self) -> List[str]:
        """
        List available dataset names.
        
        Returns:
            List of registered dataset names
        """
        if not self._discovered:
            self.discover_datasets()
        return list(self.datasets.keys())
    
    def create_dataset(self, name: str, validate_args: bool = True, **kwargs) -> Any:
        """
        Create dataset instance with given parameters.
        
        Args:
            name: Name of the dataset to create
            validate_args: Whether to validate arguments against dataset requirements
            **kwargs: Parameters to pass to dataset constructor
            
        Returns:
            Dataset instance
            
        Raises:
            ValueError: If dataset is not found or required arguments are missing
            TypeError: If argument validation fails
        """
        info = self.get_dataset_info(name)
        dataset_class = info['class']
        
        if validate_args:
            # Validate and merge arguments
            validated_kwargs = self.validate_dataset_args(name, kwargs)
            logger.debug(f"Creating dataset '{name}' with validated args: {list(validated_kwargs.keys())}")
            return dataset_class(**validated_kwargs)
        else:
            # Use arguments as provided (backwards compatibility)
            logger.debug(f"Creating dataset '{name}' without validation with args: {list(kwargs.keys())}")
            return dataset_class(**kwargs)
    
    def get_dataset_metrics(self, name: str) -> List[str]:
        """
        Get available metrics for a dataset.
        
        Args:
            name: Name of the dataset
            
        Returns:
            List of available metrics
            
        Raises:
            ValueError: If dataset is not found
        """
        info = self.get_dataset_info(name)
        return info['metrics'].copy()
    
    def get_dataset_task_type(self, name: str) -> str:
        """
        Get task type for a dataset.
        
        Args:
            name: Name of the dataset
            
        Returns:
            Task type string
            
        Raises:
            ValueError: If dataset is not found
        """
        info = self.get_dataset_info(name)
        return info['task_type']
    
    def discover_datasets(self):
        """
        Automatically discover and import all dataset modules.
        
        This method imports all modules in the karma.eval_datasets package,
        which triggers the decorator registration.
        """
        if self._discovered:
            return
            
        try:
            import karma.eval_datasets
            
            # Import all modules in karma.eval_datasets package
            for finder, name, ispkg in pkgutil.iter_modules(
                karma.eval_datasets.__path__, 
                karma.eval_datasets.__name__ + "."
            ):
                if not name.endswith('.base_dataset'):  # Skip base module
                    try:
                        importlib.import_module(name)
                        logger.debug(f"Imported dataset module: {name}")
                    except ImportError as e:
                        logger.warning(f"Could not import dataset module {name}: {e}")
        except ImportError as e:
            logger.error(f"Could not import karma.eval_datasets package: {e}")
            
        self._discovered = True
    
    def is_registered(self, name: str) -> bool:
        """
        Check if a dataset is registered.
        
        Args:
            name: Name of the dataset to check
            
        Returns:
            True if dataset is registered, False otherwise
        """
        if not self._discovered:
            self.discover_datasets()
        return name in self.datasets
    
    def unregister_dataset(self, name: str) -> bool:
        """
        Unregister a dataset.
        
        Args:
            name: Name of the dataset to unregister
            
        Returns:
            True if dataset was unregistered, False if it wasn't registered
        """
        if name in self.datasets:
            del self.datasets[name]
            logger.debug(f"Unregistered dataset: {name}")
            return True
        return False
    
    def clear_registry(self):
        """Clear all registered datasets. Mainly for testing purposes."""
        self.datasets.clear()
        self._discovered = False
        logger.debug("Cleared dataset registry")
    
    def list_datasets_by_task_type(self, task_type: str) -> List[str]:
        """
        List datasets filtered by task type.
        
        Args:
            task_type: Task type to filter by
            
        Returns:
            List of dataset names matching the task type
        """
        if not self._discovered:
            self.discover_datasets()
        return [name for name, info in self.datasets.items() 
                if info['task_type'] == task_type]
    
    def list_datasets_by_metric(self, metric: str) -> List[str]:
        """
        List datasets that support a specific metric.
        
        Args:
            metric: Metric name to filter by
            
        Returns:
            List of dataset names that support the metric
        """
        if not self._discovered:
            self.discover_datasets()
        return [name for name, info in self.datasets.items() 
                if metric in info['metrics']]
    
    def get_dataset_required_args(self, name: str) -> List[str]:
        """
        Get required arguments for a dataset.
        
        Args:
            name: Name of the dataset
            
        Returns:
            List of required argument names
            
        Raises:
            ValueError: If dataset is not found
        """
        info = self.get_dataset_info(name)
        return info['required_args'].copy()
    
    def get_dataset_optional_args(self, name: str) -> List[str]:
        """
        Get optional arguments for a dataset.
        
        Args:
            name: Name of the dataset
            
        Returns:
            List of optional argument names
            
        Raises:
            ValueError: If dataset is not found
        """
        info = self.get_dataset_info(name)
        return info['optional_args'].copy()
    
    def get_dataset_default_args(self, name: str) -> Dict[str, Any]:
        """
        Get default arguments for a dataset.
        
        Args:
            name: Name of the dataset
            
        Returns:
            Dictionary of default argument values
            
        Raises:
            ValueError: If dataset is not found
        """
        info = self.get_dataset_info(name)
        return info['default_args'].copy()
    
    def get_dataset_all_args(self, name: str) -> Dict[str, List[str]]:
        """
        Get all argument information for a dataset.
        
        Args:
            name: Name of the dataset
            
        Returns:
            Dictionary with 'required' and 'optional' argument lists
            
        Raises:
            ValueError: If dataset is not found
        """
        info = self.get_dataset_info(name)
        return {
            'required': info['required_args'].copy(),
            'optional': info['optional_args'].copy(),
            'defaults': info['default_args'].copy()
        }
    
    def validate_dataset_args(self, name: str, provided_args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate provided arguments against dataset requirements.
        
        Args:
            name: Name of the dataset
            provided_args: Dictionary of arguments provided by user
            
        Returns:
            Dictionary of validated and merged arguments (with defaults applied)
            
        Raises:
            ValueError: If dataset is not found or required arguments are missing
            TypeError: If argument types are invalid
        """
        if not self._discovered:
            self.discover_datasets()
            
        info = self.get_dataset_info(name)
        required_args = info['required_args']
        optional_args = info['optional_args']
        default_args = info['default_args']
        
        # Check for missing required arguments
        missing_required = [arg for arg in required_args if arg not in provided_args]
        if missing_required:
            raise ValueError(
                f"Missing required arguments for dataset '{name}': {missing_required}. "
                f"Required: {required_args}"
            )
        
        # Check for unexpected arguments
        all_valid_args = set(required_args + optional_args + list(default_args.keys()))
        # Add common dataset arguments that are always allowed
        all_valid_args.update(['dataset_name', 'split', 'config', 'stream', 'commit_hash'])
        
        unexpected_args = [arg for arg in provided_args.keys() if arg not in all_valid_args]
        if unexpected_args:
            logger.warning(
                f"Unexpected arguments for dataset '{name}': {unexpected_args}. "
                f"Valid arguments: {sorted(all_valid_args)}"
            )
        
        # Merge arguments: defaults + provided
        final_args = default_args.copy()
        final_args.update(provided_args)
        
        return final_args


# Global registry instance
dataset_registry = DatasetRegistry()

# Convenience decorator function
register_dataset = dataset_registry.register_dataset