"""
Model registry for automatic model discovery and registration.

This module provides a decorator-based registry system that allows models
to register themselves automatically when imported.
"""

import importlib
import pkgutil
from typing import Dict, Type, List, Optional
import logging

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Decorator-based model registry for automatic model discovery."""
    
    def __init__(self):
        self.models: Dict[str, Type] = {}
        self._discovered = False
    
    def register_model(self, name: str):
        """
        Decorator to register a model class.
        
        Args:
            name: Name to register the model under
            
        Returns:
            Decorator function
            
        Example:
            @register_model("qwen")
            class QwenModel(BaseLLM):
                pass
        """
        def decorator(model_class: Type) -> Type:
            # Import BaseLLM here to avoid circular imports
            from karma.models.base import BaseLLM
            
            if not issubclass(model_class, BaseLLM):
                raise ValueError(f"{model_class.__name__} must inherit from BaseLLM")
            
            if name in self.models:
                logger.warning(f"Model '{name}' is already registered. Overriding with {model_class.__name__}")
            
            self.models[name] = model_class
            logger.debug(f"Registered model: {name} -> {model_class.__name__}")
            return model_class
        return decorator
    
    def get_model(self, name: str) -> Type:
        """
        Get model class by name.
        
        Args:
            name: Name of the model to retrieve
            
        Returns:
            Model class
            
        Raises:
            ValueError: If model is not found
        """
        if not self._discovered:
            self.discover_models()
            
        if name not in self.models:
            available = list(self.models.keys())
            raise ValueError(f"Model '{name}' not found. Available models: {available}")
        return self.models[name]
    
    def list_models(self) -> List[str]:
        """
        List available model names.
        
        Returns:
            List of registered model names
        """
        if not self._discovered:
            self.discover_models()
        return list(self.models.keys())
    
    def discover_models(self):
        """
        Automatically discover and import all model modules.
        
        This method imports all modules in the karma.models package,
        which triggers the decorator registration.
        """
        if self._discovered:
            return
            
        try:
            import karma.models
            
            # Import all modules in karma.models package
            for finder, name, ispkg in pkgutil.iter_modules(
                karma.models.__path__, 
                karma.models.__name__ + "."
            ):
                if not name.endswith('.base'):  # Skip base module to avoid issues
                    try:
                        importlib.import_module(name)
                        logger.debug(f"Imported model module: {name}")
                    except ImportError as e:
                        logger.warning(f"Could not import model module {name}: {e}")
        except ImportError as e:
            logger.error(f"Could not import karma.models package: {e}")
            
        self._discovered = True
    
    def is_registered(self, name: str) -> bool:
        """
        Check if a model is registered.
        
        Args:
            name: Name of the model to check
            
        Returns:
            True if model is registered, False otherwise
        """
        if not self._discovered:
            self.discover_models()
        return name in self.models
    
    def unregister_model(self, name: str) -> bool:
        """
        Unregister a model.
        
        Args:
            name: Name of the model to unregister
            
        Returns:
            True if model was unregistered, False if it wasn't registered
        """
        if name in self.models:
            del self.models[name]
            logger.debug(f"Unregistered model: {name}")
            return True
        return False
    
    def clear_registry(self):
        """Clear all registered models. Mainly for testing purposes."""
        self.models.clear()
        self._discovered = False
        logger.debug("Cleared model registry")


# Global registry instance
model_registry = ModelRegistry()

# Convenience decorator function
register_model = model_registry.register_model