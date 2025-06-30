"""
Utility functions for the Karma CLI.

This module provides common utilities for parsing arguments, validation,
and other CLI-related functionality.
"""

import json
import re
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

import click
from rich.console import Console
from rich.prompt import Prompt, Confirm

from karma.dataset_registry import dataset_registry


def parse_dataset_args(dataset_args_str: str) -> Dict[str, Dict[str, Any]]:
    """
    Parse dataset arguments from CLI string format.
    
    Args:
        dataset_args_str: String in format "dataset1:key=val,key2=val2;dataset2:key=val"
        
    Returns:
        Dictionary mapping dataset names to their arguments
        
    Examples:
        >>> parse_dataset_args("in22conv:source_language=en,target_language=hi")
        {'in22conv': {'source_language': 'en', 'target_language': 'hi'}}
        
        >>> parse_dataset_args("ds1:arg1=val1;ds2:arg2=val2,arg3=val3")
        {'ds1': {'arg1': 'val1'}, 'ds2': {'arg2': 'val2', 'arg3': 'val3'}}
    """
    dataset_args = {}
    if not dataset_args_str:
        return dataset_args
    
    # Split by semicolon for different datasets
    for dataset_spec in dataset_args_str.split(';'):
        if ':' not in dataset_spec:
            continue
            
        dataset_name, args_str = dataset_spec.split(':', 1)
        args = {}
        
        # Split by comma for different arguments
        for arg_pair in args_str.split(','):
            if '=' in arg_pair:
                key, value = arg_pair.split('=', 1)
                args[key.strip()] = value.strip()
        
        dataset_args[dataset_name.strip()] = args
    
    return dataset_args


def validate_dataset_args(
    dataset_name: str, 
    provided_args: Dict[str, Any], 
    console: Optional[Console] = None
) -> Dict[str, Any]:
    """
    Validate dataset arguments against registry requirements.
    
    Args:
        dataset_name: Name of the dataset
        provided_args: Arguments provided by user
        console: Console for output (optional)
        
    Returns:
        Validated and merged arguments
        
    Raises:
        click.ClickException: If validation fails
    """
    if console is None:
        console = Console()
    
    try:
        # Use the registry's validation
        validated_args = dataset_registry.validate_dataset_args(dataset_name, provided_args)
        return validated_args
    except ValueError as e:
        # Convert to click exception for better CLI error handling
        raise click.ClickException(str(e))


def parse_datasets_list(datasets_str: str) -> List[str]:
    """
    Parse comma-separated list of dataset names.
    
    Args:
        datasets_str: Comma-separated dataset names
        
    Returns:
        List of dataset names
    """
    if not datasets_str:
        return []
    
    return [name.strip() for name in datasets_str.split(',') if name.strip()]


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string (e.g., "1.2 MB", "3.4 GB")
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string (e.g., "1m 30s", "2h 15m")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.0f}s"
    else:
        hours = int(seconds // 3600)
        remaining_minutes = int((seconds % 3600) // 60)
        return f"{hours}h {remaining_minutes}m"


def save_results(
    results: Dict[str, Any], 
    output_path: str, 
    format_type: str = "json",
    console: Optional[Console] = None
) -> None:
    """
    Save results to file in specified format.
    
    Args:
        results: Results dictionary to save
        output_path: Path to save file
        format_type: Format type ("json", "yaml", "csv")
        console: Console for output messages
    """
    if console is None:
        console = Console()
    
    output_file = Path(output_path)
    
    try:
        if format_type.lower() == "json":
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        elif format_type.lower() == "yaml":
            import yaml
            with open(output_file, 'w') as f:
                yaml.dump(results, f, default_flow_style=False, allow_unicode=True)
        elif format_type.lower() == "csv":
            import pandas as pd
            # Flatten results for CSV format
            rows = []
            for dataset_name, dataset_result in results.items():
                if dataset_name.startswith('_'):
                    continue
                if isinstance(dataset_result, dict) and 'metrics' in dataset_result:
                    for metric_name, metric_data in dataset_result['metrics'].items():
                        rows.append({
                            'dataset': dataset_name,
                            'task_type': dataset_result.get('task_type', ''),
                            'metric': metric_name,
                            'score': metric_data.get('score', 0),
                            'num_samples': metric_data.get('num_samples', 0),
                            'evaluation_time': metric_data.get('evaluation_time', 0)
                        })
            
            df = pd.DataFrame(rows)
            df.to_csv(output_file, index=False)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
        
        console.print(f"[green]Results saved to {output_file}[/green]")
        
    except Exception as e:
        console.print(f"[red]Error saving results: {e}[/red]")
        raise click.ClickException(f"Failed to save results: {e}")


def prompt_for_missing_args(
    dataset_name: str, 
    missing_args: List[str],
    console: Optional[Console] = None
) -> Dict[str, str]:
    """
    Interactively prompt user for missing required arguments.
    
    Args:
        dataset_name: Name of the dataset
        missing_args: List of missing argument names
        console: Console for output
        
    Returns:
        Dictionary of user-provided arguments
    """
    if console is None:
        console = Console()
    
    console.print(f"\n[yellow]Missing required arguments for dataset '{dataset_name}'[/yellow]")
    
    args = {}
    for arg_name in missing_args:
        value = Prompt.ask(f"Enter value for [cyan]{arg_name}[/cyan]")
        args[arg_name] = value
    
    return args


def get_cache_info(cache_path: str) -> Dict[str, Any]:
    """
    Get information about the cache database.
    
    Args:
        cache_path: Path to cache database
        
    Returns:
        Dictionary with cache information
    """
    cache_file = Path(cache_path)
    
    if not cache_file.exists():
        return {
            'exists': False,
            'size': 0,
            'size_formatted': '0 B',
            'path': str(cache_file)
        }
    
    size_bytes = cache_file.stat().st_size
    
    return {
        'exists': True,
        'size': size_bytes,
        'size_formatted': format_file_size(size_bytes),
        'path': str(cache_file)
    }


def validate_model_path(model_path: str) -> bool:
    """
    Validate if model path exists (local) or looks like valid HuggingFace model ID.
    
    Args:
        model_path: Path or model ID to validate
        
    Returns:
        True if valid, False otherwise
    """
    # Check if it's a local path
    if Path(model_path).exists():
        return True
    
    # Check if it looks like a HuggingFace model ID (org/model format)
    hf_pattern = re.compile(r'^[a-zA-Z0-9][\w\-\.]*\/[\w\-\.]+$')
    if hf_pattern.match(model_path):
        return True
    
    # Check if it's a simple model name (might be valid HF model)
    simple_pattern = re.compile(r'^[a-zA-Z0-9][\w\-\.]*$')
    if simple_pattern.match(model_path):
        return True
    
    return False


class ClickFormatter:
    """Utility class for consistent CLI formatting."""
    
    @staticmethod
    def success(message: str) -> str:
        """Format success message."""
        return f"[green]✓[/green] {message}"
    
    @staticmethod
    def error(message: str) -> str:
        """Format error message."""
        return f"[red]✗[/red] {message}"
    
    @staticmethod
    def warning(message: str) -> str:
        """Format warning message."""
        return f"[yellow]⚠[/yellow] {message}"
    
    @staticmethod
    def info(message: str) -> str:
        """Format info message."""
        return f"[blue]ℹ[/blue] {message}"
    
    @staticmethod
    def highlight(text: str) -> str:
        """Highlight important text."""
        return f"[cyan]{text}[/cyan]"