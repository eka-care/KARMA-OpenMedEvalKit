"""
Processor registry for automatic processor discovery and registration.

This module provides a decorator-based registry system that allows processors
to register themselves automatically when imported.
"""

import importlib
import pkgutil
from typing import Dict, Type, List, Optional, Any
import logging
import time

from karma.processors.base import BaseProcessor
from karma.utils.argument_processing import validate_registry_args, prepare_registry_metadata

logger = logging.getLogger(__name__)


class ProcessorRegistry:
    """Decorator-based processor registry for automatic processor discovery."""
    
    def __init__(self):
        self.processors: Dict[str, Dict[str, Any]] = {}
        self._discovered = False
    
    def register_processor(
        self, 
        name: str,
        required_args: Optional[List[str]] = None,
        optional_args: Optional[List[str]] = None,
        default_args: Optional[Dict[str, Any]] = None
    ):
        """
        Decorator to register a processor class with its argument metadata.
        
        Args:
            name: Name to register the processor under
            required_args: List of required argument names for processor instantiation
            optional_args: List of optional argument names for processor instantiation
            default_args: Dictionary of default values for arguments
            
        Returns:
            Decorator function
            
        Examples:
            @register_processor("devnagari_transliterator")
            class DevanagariTransliterator(BaseProcessor):
                pass
                
            @register_processor(
                "text_normalizer",
                required_args=["language"],
                optional_args=["case_sensitive", "remove_punctuation"],
                default_args={"case_sensitive": True, "remove_punctuation": False}
            )
            class TextNormalizer(BaseProcessor):
                pass
        """
        def decorator(processor_class: Type) -> Type:
            if not issubclass(processor_class, BaseProcessor):
                raise ValueError(f"{processor_class.__name__} must inherit from BaseProcessor")
            
            if name in self.processors:
                logger.warning(f"Processor '{name}' is already registered. Overriding with {processor_class.__name__}")
            
            # Prepare argument metadata
            metadata = prepare_registry_metadata(required_args, optional_args, default_args)
            
            self.processors[name] = {
                'class': processor_class,
                'module': processor_class.__module__,
                'class_name': processor_class.__name__,
                'required_args': metadata['required_args'],
                'optional_args': metadata['optional_args'],
                'default_args': metadata['default_args']
            }
            logger.debug(f"Registered processor: {name} -> {processor_class.__name__}")
            return processor_class
        return decorator
    
    def get_processor(self, name: str, **kwargs) -> BaseProcessor:
        """
        Get processor instance by name with optional arguments.
        
        Args:
            name: Name of the processor to retrieve
            **kwargs: Arguments to pass to processor constructor
            
        Returns:
            Processor instance
            
        Raises:
            ValueError: If processor is not found or arguments are invalid
        """
        if not self._discovered:
            self.discover_processors()
            
        if name not in self.processors:
            available = list(self.processors.keys())
            raise ValueError(f"Processor '{name}' not found. Available processors: {available}")
        
        processor_info = self.processors[name]
        processor_class = processor_info['class']
        
        # Validate arguments if any are provided
        if kwargs:
            validated_args = validate_registry_args(
                registry_name="processor",
                item_name=name,
                provided_args=kwargs,
                required_args=processor_info['required_args'],
                optional_args=processor_info['optional_args'],
                default_args=processor_info['default_args']
            )
            return processor_class(**validated_args)
        else:
            # Use default arguments only
            return processor_class(**processor_info['default_args'])
    
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
        return self.processors[name]['class']
    
    def list_processors(self) -> List[str]:
        """
        List available processor names.
        
        Returns:
            List of registered processor names
        """
        if not self._discovered:
            self.discover_processors()
        return list(self.processors.keys())
    
    def discover_processors(self, use_cache: bool = True):
        """
        Automatically discover and import all processor modules.
        
        This method imports all modules in the karma.processors package,
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
                logger.debug("Loaded processor registry from cache")
                self._discovered = True
                return

        # Perform discovery
        start_time = time.time()
        logger.debug("Starting processor discovery...")
        
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
        
        discovery_time = time.time() - start_time
        logger.debug(f"Processor discovery completed in {discovery_time:.2f}s")
        
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
            cached_data = cache_manager.get_cached_discovery("processor")
            
            if cached_data:
                self.processors = cached_data.get("processors", {})
                return True
                
        except Exception as e:
            logger.debug(f"Failed to load processor registry from cache: {e}")
            
        return False
    
    def _save_to_cache(self) -> None:
        """
        Save current registry data to cache.
        """
        try:
            from karma.registries.cache_manager import get_cache_manager
            
            cache_manager = get_cache_manager()
            cache_data = {
                "processors": self.processors
            }
            
            cache_manager.set_cached_discovery("processor", cache_data)
            logger.debug("Saved processor registry to cache")
            
        except Exception as e:
            logger.debug(f"Failed to save processor registry to cache: {e}")
    
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
    
    def get_processor_info(self, name: str) -> Dict[str, Any]:
        """
        Get processor information including metadata.
        
        Args:
            name: Name of the processor
            
        Returns:
            Dictionary containing processor metadata
            
        Raises:
            ValueError: If processor is not found
        """
        if not self._discovered:
            self.discover_processors()
            
        if name not in self.processors:
            available = list(self.processors.keys())
            raise ValueError(f"Processor '{name}' not found. Available processors: {available}")
        
        return self.processors[name].copy()
    
    def get_processor_required_args(self, name: str) -> List[str]:
        """
        Get required arguments for a processor.
        
        Args:
            name: Name of the processor
            
        Returns:
            List of required argument names
            
        Raises:
            ValueError: If processor is not found
        """
        info = self.get_processor_info(name)
        return info['required_args'].copy()
    
    def get_processor_optional_args(self, name: str) -> List[str]:
        """
        Get optional arguments for a processor.
        
        Args:
            name: Name of the processor
            
        Returns:
            List of optional argument names
            
        Raises:
            ValueError: If processor is not found
        """
        info = self.get_processor_info(name)
        return info['optional_args'].copy()
    
    def get_processor_default_args(self, name: str) -> Dict[str, Any]:
        """
        Get default arguments for a processor.
        
        Args:
            name: Name of the processor
            
        Returns:
            Dictionary of default argument values
            
        Raises:
            ValueError: If processor is not found
        """
        info = self.get_processor_info(name)
        return info['default_args'].copy()
    
    def get_processor_all_args(self, name: str) -> Dict[str, Any]:
        """
        Get all argument information for a processor.
        
        Args:
            name: Name of the processor
            
        Returns:
            Dictionary with 'required', 'optional', and 'defaults' argument info
            
        Raises:
            ValueError: If processor is not found
        """
        info = self.get_processor_info(name)
        return {
            'required': info['required_args'].copy(),
            'optional': info['optional_args'].copy(),
            'defaults': info['default_args'].copy()
        }
    
    def validate_processor_args(self, name: str, provided_args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate provided arguments against processor requirements.
        
        Args:
            name: Name of the processor
            provided_args: Dictionary of arguments provided by user
            
        Returns:
            Dictionary of validated and merged arguments (with defaults applied)
            
        Raises:
            ValueError: If processor is not found or required arguments are missing
        """
        if not self._discovered:
            self.discover_processors()
            
        info = self.get_processor_info(name)
        
        return validate_registry_args(
            registry_name="processor",
            item_name=name,
            provided_args=provided_args,
            required_args=info['required_args'],
            optional_args=info['optional_args'],
            default_args=info['default_args']
        )


# Global registry instance
processor_registry = ProcessorRegistry()

# Convenience decorator function
register_processor = processor_registry.register_processor 