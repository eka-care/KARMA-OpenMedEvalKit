"""
Processor registry for automatic processor discovery and registration.

This module provides a decorator-based registry system that allows processors
to register themselves automatically when imported.
"""

import importlib
import pkgutil
from typing import Dict, Type, List, Optional
import logging

from karma.processors.base import BaseProcessor

logger = logging.getLogger(__name__)


class ProcessorRegistry:
    """Decorator-based processor registry for automatic processor discovery."""
    
    def __init__(self):
        self.processors: Dict[str, Type] = {}
        self._discovered = False
    
    def register_processor(self, name: str):
        """
        Decorator to register a processor class.
        
        Args:
            name: Name to register the processor under
            
        Returns:
            Decorator function
            
        Example:
            @register_processor("devnagari_transliterator")
            class DevanagariTransliterator(BaseProcessor):
                pass
        """
        def decorator(processor_class: Type) -> Type:
            if not issubclass(processor_class, BaseProcessor):
                raise ValueError(f"{processor_class.__name__} must inherit from BaseProcessor")
            
            if name in self.processors:
                logger.warning(f"Processor '{name}' is already registered. Overriding with {processor_class.__name__}")
            
            self.processors[name] = processor_class
            logger.debug(f"Registered processor: {name} -> {processor_class.__name__}")
            return processor_class
        return decorator
    
    def get_processor(self, name: str) -> BaseProcessor:
        """
        Get processor instance by name.
        
        Args:
            name: Name of the processor to retrieve
            
        Returns:
            Processor instance
            
        Raises:
            ValueError: If processor is not found
        """
        if not self._discovered:
            self.discover_processors()
            
        if name not in self.processors:
            available = list(self.processors.keys())
            raise ValueError(f"Processor '{name}' not found. Available processors: {available}")
        return self.processors[name]()
    
    def get_processor_class(self, name: str) -> Type:
        """
        Get processor class by name.
        
        Args:
            name: Name of the processor to retrieve
            
        Returns:
            Processor class
            
        Raises:
            ValueError: If processor is not found
        """
        if not self._discovered:
            self.discover_processors()
            
        if name not in self.processors:
            available = list(self.processors.keys())
            raise ValueError(f"Processor '{name}' not found. Available processors: {available}")
        return self.processors[name]
    
    def list_processors(self) -> List[str]:
        """
        List available processor names.
        
        Returns:
            List of registered processor names
        """
        if not self._discovered:
            self.discover_processors()
        return list(self.processors.keys())
    
    def discover_processors(self):
        """
        Automatically discover and import all processor modules.
        
        This method imports all modules in the karma.processors package,
        which triggers the decorator registration.
        """
        if self._discovered:
            return
            
        try:
            import karma.processors
            
            # Import all modules in karma.processors package
            for finder, name, ispkg in pkgutil.iter_modules(
                karma.processors.__path__, 
                karma.processors.__name__ + "."
            ):
                try:
                    importlib.import_module(name)
                    logger.debug(f"Imported processor module: {name}")
                except ImportError as e:
                    logger.warning(f"Could not import processor module {name}: {e}")
        except ImportError as e:
            logger.error(f"Could not import karma.processors package: {e}")
            
        self._discovered = True
    
    def is_registered(self, name: str) -> bool:
        """
        Check if a processor is registered.
        
        Args:
            name: Name of the processor to check
            
        Returns:
            True if processor is registered, False otherwise
        """
        if not self._discovered:
            self.discover_processors()
        return name in self.processors
    
    def unregister_processor(self, name: str) -> bool:
        """
        Unregister a processor.
        
        Args:
            name: Name of the processor to unregister
            
        Returns:
            True if processor was unregistered, False if it wasn't registered
        """
        if name in self.processors:
            del self.processors[name]
            logger.debug(f"Unregistered processor: {name}")
            return True
        return False
    
    def clear_registry(self):
        """Clear all registered processors. Mainly for testing purposes."""
        self.processors.clear()
        self._discovered = False
        logger.debug("Cleared processor registry")


# Global registry instance
processor_registry = ProcessorRegistry()

# Convenience decorator function
register_processor = processor_registry.register_processor 