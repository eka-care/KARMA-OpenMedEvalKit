"""
Devanagari Transliterator using indic-transliteration library.

This module provides a simple class to transliterate text from any Indic script
to Devanagari script using the indic-transliteration library with built-in 
script detection.
"""

import logging
import unicodedata

from indic_transliteration import sanscript
from indic_transliteration.detect import detect
from karma.processors.base import BaseProcessor
from karma.registries.processor_registry import register_processor

logger = logging.getLogger(__name__)


@register_processor("devnagari_transliterator")
class DevanagariTransliterator(BaseProcessor):
    """
    Simple transliterator that converts any Indic text to Devanagari script
    using automatic script detection.
    """
    
    def __init__(self):
        """Initialize the transliterator."""
        super().__init__()
        self.name = "devnagari_transliterator"
    
    def process(self, text: str) -> str:
        """
        Transliterate any Indic text to Devanagari script with automatic script detection.
        
        Args:
            text: Input text to transliterate
            
        Returns:
            Text transliterated to Devanagari script
        """
        
        try:
            # Apply Unicode normalization
            text = unicodedata.normalize('NFC', text)
            
            # Detect the script/scheme of input text
            detected_scheme = detect(text)
            
            # If already Devanagari, return as is
            if detected_scheme == 'Devanagari':
                return text.strip()
            
            # If detection fails or unsupported, try to transliterate from common schemes
            if detected_scheme is None:
                logger.debug(f"Could not detect scheme for text: {text[:50]}...")
                return text.strip()
            
            # Transliterate to Devanagari
            transliterated = sanscript.transliterate(text, detected_scheme, sanscript.DEVANAGARI)
            
            # Apply Unicode normalization to result
            transliterated = unicodedata.normalize('NFC', transliterated)
            
            return transliterated
            
        except Exception as e:
            logger.warning(f"Transliteration failed: {e}")
            return text.strip()  # Return original text if transliteration fails
        