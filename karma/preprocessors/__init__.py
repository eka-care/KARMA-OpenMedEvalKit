"""
Preprocessors for text normalization and language-specific processing.

This module contains preprocessors for different languages and scripts,
focusing on proper tokenization, normalization, and standardization
for multilingual benchmarking.
"""

from .indiclang import DevanagariTransliterator

__all__ = [
    'DevanagariTransliterator',
] 