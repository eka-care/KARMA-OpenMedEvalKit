"""
Base processor class for all text processors in KARMA.

This module provides the base class that all processors
should inherit from to ensure a consistent interface.
"""


class BaseProcessor:
    """Base class for all processors."""
    
    def __init__(self):
        """Initialize the processor with a default name."""
        self.name = self.__class__.__name__.lower()
    
    def process(self, text: str) -> str:
        """
        Process the input text.
        
        Args:
            text: Input text to process
            
        Returns:
            Processed text
            
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