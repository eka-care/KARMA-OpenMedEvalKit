"""
Devanagari Transliterator using indic-transliteration library.

This module provides a simple class to transliterate text from any Indic script
to Devanagari script using the indic-transliteration library with built-in 
script detection.
"""

import logging
import unicodedata
from typing import List

from indic_transliteration import sanscript
from indic_transliteration.detect import detect
from karma.processors.base import BaseProcessor
from karma.registries.processor_registry import register_processor

logger = logging.getLogger(__name__)


@register_processor(
    "devnagari_transliterator",
    optional_args=["normalize", "fallback_scheme"],
    default_args={"normalize": True, "fallback_scheme": None}
)
class DevanagariTransliterator(BaseProcessor):
    """
    Simple transliterator that converts any Indic text to Devanagari script
    using automatic script detection.
    """
    
    def __init__(self, **kwargs):
        """Initialize the transliterator."""
        super().__init__(**kwargs)
        self.name = "devnagari_transliterator"
    
    def process(self, texts: List[str]) -> List[str]:
        """
        Transliterate any Indic text to Devanagari script with automatic script detection.
        
        Args:
            texts: List of input texts to transliterate
            
        Returns:
            List of texts transliterated to Devanagari script
        """
        
        results = []
        
        for text in texts:
            try:
                # Apply Unicode normalization if enabled
                if getattr(self, 'normalize', True):
                    text = unicodedata.normalize('NFC', text)
                
                # Detect the script/scheme of input text
                detected_scheme = detect(text)
                
                # If already Devanagari, return as is
                if detected_scheme == 'Devanagari':
                    results.append(text.strip())
                    continue
                
                # If detection fails or unsupported, try fallback scheme if provided
                if detected_scheme is None:
                    fallback = getattr(self, 'fallback_scheme', None)
                    if fallback:
                        logger.debug(f"Using fallback scheme '{fallback}' for text: {text[:50]}...")
                        detected_scheme = fallback
                    else:
                        logger.debug(f"Could not detect scheme for text: {text[:50]}...")
                        results.append(text.strip())
                        continue
                
                # Transliterate to Devanagari
                transliterated = sanscript.transliterate(text, detected_scheme, sanscript.DEVANAGARI)
                
                # Apply Unicode normalization to result if enabled
                if getattr(self, 'normalize', True):
                    transliterated = unicodedata.normalize('NFC', transliterated)
                
                results.append(transliterated)
                
            except Exception as e:
                logger.warning(f"Transliteration failed: {e}")
                results.append(text.strip())  # Return original text if transliteration fails
        
        return results
        