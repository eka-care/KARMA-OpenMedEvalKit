"""
Authentication utilities for KARMA.
Handles HuggingFace token management with proper error handling.
"""
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Cache the token at module level to avoid repeated environment lookups
_HF_TOKEN_CACHE: Optional[str] = None


def get_hf_token() -> Optional[str]:
    """
    Get HuggingFace token from environment variables.
    
    Checks both HUGGINGFACE_TOKEN (recommended, matches documentation) 
    and HF_TOKEN (legacy) for backwards compatibility.
    
    Returns:
        Optional[str]: The HuggingFace token if found, None otherwise
    
    Example:
        >>> from karma.utils.auth import get_hf_token
        >>> token = get_hf_token()
        >>> if token:
        >>>     dataset = load_dataset("private/dataset", token=token)
    """
    global _HF_TOKEN_CACHE
    
    # Return cached token if already retrieved
    if _HF_TOKEN_CACHE is not None:
        return _HF_TOKEN_CACHE
    
    # Check both environment variable names for compatibility
    token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
    
    if not token:
        logger.debug(
            "HuggingFace token not found in environment variables. "
            "Set HUGGINGFACE_TOKEN or HF_TOKEN to access private models/datasets. "
            "Add it to your .env file or export it in your shell:\n"
            "  Linux/Mac: export HUGGINGFACE_TOKEN=your_token_here\n"
            "  Windows: $env:HUGGINGFACE_TOKEN='your_token_here'"
        )
    
    # Cache the result (even if None) to avoid repeated lookups and log spam
    _HF_TOKEN_CACHE = token
    return token


def ensure_hf_login() -> bool:
    """
    Attempt to login to HuggingFace using token from environment.
    
    This function will try to authenticate with HuggingFace Hub using
    the token obtained from get_hf_token(). If no token is available
    or login fails, it will log a warning but not raise an exception.
    
    Returns:
        bool: True if login successful, False otherwise
    
    Example:
        >>> from karma.utils.auth import ensure_hf_login
        >>> if ensure_hf_login():
        >>>     print("Ready to use private models!")
        >>> else:
        >>>     print("Continuing without authentication")
    """
    from huggingface_hub import login
    
    token = get_hf_token()
    
    if not token:
        logger.warning(
            "HuggingFace token not found. Some models/datasets may not be accessible. "
            "Set HUGGINGFACE_TOKEN in your environment to enable authentication."
        )
        return False
    
    try:
        login(token=token)
        logger.info("Successfully authenticated with HuggingFace")
        return True
    except ValueError as e:
        logger.warning(f"HuggingFace login failed: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during HuggingFace login: {e}")
        return False