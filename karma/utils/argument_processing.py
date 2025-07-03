"""
Common argument processing utilities for all registries.

This module provides reusable functions for argument validation, merging,
and processing that can be used by dataset, processor, and model registries.
"""

import logging
from typing import Dict, List, Any, Optional, Union

logger = logging.getLogger(__name__)


def validate_registry_args(
    registry_name: str,
    item_name: str,
    provided_args: Dict[str, Any],
    required_args: List[str],
    optional_args: List[str],
    default_args: Dict[str, Any],
    common_args: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Validate provided arguments against registry item requirements.
    
    Args:
        registry_name: Name of the registry (for error messages)
        item_name: Name of the registry item (dataset, processor, etc.)
        provided_args: Dictionary of arguments provided by user
        required_args: List of required argument names
        optional_args: List of optional argument names
        default_args: Dictionary of default values for arguments
        common_args: List of common arguments that are always allowed
        
    Returns:
        Dictionary of validated and merged arguments (with defaults applied)
        
    Raises:
        ValueError: If required arguments are missing
        TypeError: If argument types are invalid
    """
    # Check for missing required arguments
    missing_required = [arg for arg in required_args if arg not in provided_args]
    if missing_required:
        raise ValueError(
            f"Missing required arguments for {registry_name} '{item_name}': {missing_required}. "
            f"Required: {required_args}"
        )
    
    # Check for unexpected arguments
    all_valid_args = set(required_args + optional_args + list(default_args.keys()))
    
    # Add common arguments that are always allowed
    if common_args:
        all_valid_args.update(common_args)
    
    unexpected_args = [arg for arg in provided_args.keys() if arg not in all_valid_args]
    if unexpected_args:
        logger.warning(
            f"Unexpected arguments for {registry_name} '{item_name}': {unexpected_args}. "
            f"Valid arguments: {sorted(all_valid_args)}"
        )
    
    # Merge arguments: defaults + provided
    final_args = default_args.copy()
    final_args.update(provided_args)
    
    return final_args


def merge_registry_kwargs(
    defaults: Dict[str, Any],
    provided: Dict[str, Any],
    overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Merge kwargs with precedence: defaults < provided < overrides.
    
    Args:
        defaults: Default argument values
        provided: User-provided argument values
        overrides: Override argument values (highest precedence)
        
    Returns:
        Merged dictionary with proper precedence
    """
    result = defaults.copy()
    result.update(provided)
    if overrides:
        result.update(overrides)
    return result


def process_arg_list(
    args: Union[List[str], str, None],
    default: Optional[List[str]] = None
) -> List[str]:
    """
    Process argument lists, handling string->list conversion.
    
    Args:
        args: Arguments as list, string, or None
        default: Default value if args is None
        
    Returns:
        Processed argument list
    """
    if args is None:
        return default.copy() if default else []
    
    if isinstance(args, str):
        return [args]
    
    if isinstance(args, list):
        return args.copy()
    
    raise TypeError(f"Arguments must be string, list, or None, got {type(args)}")


def validate_arg_types(
    args: Dict[str, Any],
    type_specs: Dict[str, type],
    item_name: str
) -> Dict[str, Any]:
    """
    Validate argument types against specifications.
    
    Args:
        args: Arguments to validate
        type_specs: Dictionary mapping argument names to expected types
        item_name: Name of the item (for error messages)
        
    Returns:
        Validated arguments dictionary
        
    Raises:
        TypeError: If argument types don't match specifications
    """
    validated = {}
    
    for arg_name, arg_value in args.items():
        if arg_name in type_specs:
            expected_type = type_specs[arg_name]
            if not isinstance(arg_value, expected_type):
                raise TypeError(
                    f"Argument '{arg_name}' for '{item_name}' must be of type {expected_type.__name__}, "
                    f"got {type(arg_value).__name__}"
                )
        validated[arg_name] = arg_value
    
    return validated


def format_argument_error(
    registry_name: str,
    item_name: str,
    error_type: str,
    details: str,
    suggestions: Optional[List[str]] = None
) -> str:
    """
    Format argument error messages consistently.
    
    Args:
        registry_name: Name of the registry
        item_name: Name of the registry item
        error_type: Type of error (e.g., "missing_required", "invalid_type")
        details: Detailed error description
        suggestions: Optional list of suggestions
        
    Returns:
        Formatted error message
    """
    message = f"Argument error for {registry_name} '{item_name}': {details}"
    
    if suggestions:
        message += f"\nSuggestions: {', '.join(suggestions)}"
    
    return message


def get_missing_required_args(
    provided_args: Dict[str, Any],
    required_args: List[str]
) -> List[str]:
    """
    Get list of missing required arguments.
    
    Args:
        provided_args: Arguments provided by user
        required_args: List of required argument names
        
    Returns:
        List of missing required argument names
    """
    return [arg for arg in required_args if arg not in provided_args]


def get_unexpected_args(
    provided_args: Dict[str, Any],
    valid_args: List[str]
) -> List[str]:
    """
    Get list of unexpected arguments.
    
    Args:
        provided_args: Arguments provided by user
        valid_args: List of valid argument names
        
    Returns:
        List of unexpected argument names
    """
    valid_set = set(valid_args)
    return [arg for arg in provided_args.keys() if arg not in valid_set]


def prepare_registry_metadata(
    required_args: Optional[List[str]] = None,
    optional_args: Optional[List[str]] = None,
    default_args: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Prepare and validate registry metadata.
    
    Args:
        required_args: List of required argument names
        optional_args: List of optional argument names
        default_args: Dictionary of default values
        
    Returns:
        Processed metadata dictionary
    """
    processed_required = process_arg_list(required_args, [])
    processed_optional = process_arg_list(optional_args, [])
    processed_defaults = default_args.copy() if default_args else {}
    
    return {
        'required_args': processed_required,
        'optional_args': processed_optional,
        'default_args': processed_defaults
    }