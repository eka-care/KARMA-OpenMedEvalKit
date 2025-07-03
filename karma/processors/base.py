"""
Base processor class for all text processors in KARMA.

This module provides the base class that all processors
should inherit from to ensure a consistent interface.
"""

from typing import List


class BaseProcessor:
    """Base class for all processors."""
    
    def __init__(self, **kwargs):
        """
        Initialize the processor with a default name and optional arguments.
        
        Args:
            **kwargs: Optional keyword arguments for processor configuration
        """
        self.name = self.__class__.__name__.lower()
        
        # Store any additional arguments provided
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def process(self, texts: List[str]) -> List[str]:
        """
        Process the input texts.
        
        Args:
            texts: List of input texts to process
            
        Returns:
            List of processed texts
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement the process method")
    
    def __str__(self) -> str:
        """String representation of the processor."""
        return f"{self.__class__.__name__}(name='{self.name}')"
    
    def __repr__(self) -> str:
        """Detailed representation of the processor."""
        return self.__str__() 