"""
Info command for the Karma CLI.

This module implements the 'info' command which provides detailed information
about models, datasets, and system status.
"""

import platform
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel

from karma.cli.formatters.table import ModelFormatter, DatasetFormatter, SystemFormatter
from karma.cli.utils import ClickFormatter, get_cache_info
from karma.registries.model_registry import model_registry
from karma.registries.dataset_registry import dataset_registry
from karma.registries.registry_manager import discover_all_registries


@click.group(name="info")
@click.pass_context
def info_cmd(ctx):
    """
    Get detailed information about models, datasets, and system status.

    Use 'karma info model <name>' or 'karma info dataset <name>' for details.
    """
    pass


@info_cmd.command(name="model")
@click.argument("model_name")
@click.option(
    "--show-code", is_flag=True, help="Show model class code location and basic info"
)
@click.pass_context
def info_model(ctx, model_name, show_code):
    """
    Get detailed information about a specific model.

    This command shows comprehensive information about a registered model,
    including its class details, module location, and implementation info.

    Examples:
        karma info model "Qwen/Qwen3-0.6B"
        karma info model medgemma --show-code
    """
    console = ctx.obj["console"]

    try:
        # Discover all registries for better performance
        discover_all_registries(use_cache=True, parallel=True)

        # Check if model exists
        if not model_registry.is_registered(model_name):
            available_models = model_registry.list_models()
            console.print(
                ClickFormatter.error(f"Model '{model_name}' not found in registry")
            )
            console.print(f"Available models: {', '.join(available_models)}")
            raise click.Abort()

        # Get model meta and class
        model_meta = model_registry.get_model_meta(model_name)
        model_class = model_meta.get_loader_class()

        # Prepare model info
        model_info = {
            "class_name": model_class.__name__,
            "module": model_class.__module__,
            "file": getattr(model_class, "__file__", "Unknown"),
            "doc": model_class.__doc__ or "No documentation available",
        }

        # Show basic info table
        console.print(f"\n[bold cyan]Model Information: {model_name}[/bold cyan]")
        console.print("─" * 50)

        table = ModelFormatter.format_model_info(model_name, model_info)
        console.print(table)

        # Show documentation
        if model_info["doc"] and model_info["doc"] != "No documentation available":
            console.print(f"\n[cyan]Description:[/cyan]")
            # Clean up docstring formatting
            doc_lines = model_info["doc"].strip().split("\n")
            cleaned_doc = "\n".join(line.strip() for line in doc_lines)
            console.print(Panel(cleaned_doc, border_style="dim"))

        # Show code location if requested
        if show_code:
            console.print(f"\n[cyan]Code Location:[/cyan]")
            file_path = model_info.get("file", "Unknown")
            if file_path != "Unknown" and Path(file_path).exists():
                console.print(f"  File: {file_path}")
                console.print(
                    f"  Line: {getattr(model_class, '__qualname__', 'Unknown')}"
                )
            else:
                console.print("  File location not available")

        # Show initialization signature
        try:
            import inspect

            sig = inspect.signature(model_class.__init__)
            console.print(f"\n[cyan]Constructor Signature:[/cyan]")
            console.print(f"  {model_class.__name__}{sig}")
        except Exception:
            pass

        # Show usage examples
        _show_model_examples(console, model_name)

        console.print(
            f"\n{ClickFormatter.success('Model information retrieved successfully')}"
        )

    except Exception as e:
        console.print(ClickFormatter.error(f"Failed to get model info: {str(e)}"))
        raise e
        raise click.Abort()


@info_cmd.command(name="dataset")
@click.argument("dataset_name")
@click.option(
    "--show-examples", is_flag=True, help="Show usage examples with arguments"
)
@click.option("--show-code", is_flag=True, help="Show dataset class code location")
@click.pass_context
def info_dataset(ctx, dataset_name, show_examples, show_code):
    """
    Get detailed information about a specific dataset.

    This command shows comprehensive information about a registered dataset,
    including its requirements, supported metrics, and usage examples.

    Examples:
        karma info dataset openlifescienceai/pubmedqa
        karma info dataset in22conv --show-examples
        karma info dataset slake --show-code
    """
    console = ctx.obj["console"]

    try:
        # Discover all registries for better performance
        discover_all_registries(use_cache=True, parallel=True)

        # Check if dataset exists
        if not dataset_registry.is_registered(dataset_name):
            available_datasets = dataset_registry.list_datasets()
            console.print(
                ClickFormatter.error(f"Dataset '{dataset_name}' not found in registry")
            )
            console.print(f"Available datasets: {', '.join(available_datasets)}")
            raise click.Abort()

        # Get dataset info
        dataset_info = dataset_registry.get_dataset_info(dataset_name)
        dataset_class = dataset_info["class"]

        # Show basic info table
        console.print(f"\n[bold cyan]Dataset Information: {dataset_name}[/bold cyan]")
        console.print("─" * 50)

        table = DatasetFormatter.format_dataset_info(dataset_name, dataset_info)
        console.print(table)

        # Show class documentation
        if dataset_class.__doc__:
            console.print(f"\n[cyan]Description:[/cyan]")
            doc_lines = dataset_class.__doc__.strip().split("\n")
            cleaned_doc = "\n".join(line.strip() for line in doc_lines)
            console.print(Panel(cleaned_doc, border_style="dim"))

        # Show usage examples
        if (
            show_examples
            or dataset_info.get("required_args")
            or dataset_info.get("optional_args")
        ):
            _show_dataset_examples(console, dataset_name, dataset_info)

        # Show code location if requested
        if show_code:
            console.print(f"\n[cyan]Code Location:[/cyan]")
            file_path = getattr(dataset_class, "__file__", "Unknown")
            if file_path != "Unknown" and Path(file_path).exists():
                console.print(f"  File: {file_path}")
                console.print(f"  Module: {dataset_info.get('module', 'Unknown')}")
            else:
                console.print("  File location not available")

        console.print(
            f"\n{ClickFormatter.success('Dataset information retrieved successfully')}"
        )

    except Exception as e:
        console.print(ClickFormatter.error(f"Failed to get dataset info: {str(e)}"))
        raise click.Abort()


@info_cmd.command(name="system")
@click.option(
    "--cache-path", default="./cache.db", help="Path to cache database to check"
)
@click.pass_context
def info_system(ctx, cache_path):
    """
    Get system information and status.

    This command shows information about the Karma system, including
    available resources, cache status, and environment details.

    Examples:
        karma info system
        karma info system --cache-path /path/to/cache.db
    """
    console = ctx.obj["console"]

    try:
        # Discover resources using optimized registry manager
        console.print("[cyan]Discovering system resources...[/cyan]")
        discover_all_registries(use_cache=True, parallel=True)

        # Get counts
        models_count = len(model_registry.list_models())
        datasets_count = len(dataset_registry.list_datasets())

        # Get cache info
        cache_info = get_cache_info(cache_path)

        # Show system info table
        console.print(f"\n[bold cyan]System Information[/bold cyan]")
        console.print("─" * 50)

        system_table = SystemFormatter.format_system_info(
            cache_info, models_count, datasets_count
        )
        console.print(system_table)

        # Show environment info
        console.print(f"\n[cyan]Environment:[/cyan]")
        console.print(f"  Python: {sys.version.split()[0]}")
        console.print(f"  Platform: {platform.platform()}")
        console.print(f"  Architecture: {platform.machine()}")

        # Show package info
        try:
            import karma

            console.print(
                f"  Karma CLI: {getattr(karma, '__version__', 'development')}"
            )
        except:
            console.print(f"  Karma CLI: development")

        # Show dependency status
        console.print(f"\n[cyan]Dependencies:[/cyan]")
        _check_dependencies(console)

        # Show usage examples
        _show_system_examples(console)

        console.print(
            f"\n{ClickFormatter.success('System information retrieved successfully')}"
        )

    except Exception as e:
        console.print(ClickFormatter.error(f"Failed to get system info: {str(e)}"))
        raise click.Abort()


def _show_model_examples(console: Console, model_name: str) -> None:
    """
    Show usage examples for a model.

    Args:
        console: Rich console for output
        model_name: Name of the model
    """
    console.print(f"\n[cyan]Usage Examples:[/cyan]")

    # Basic evaluation
    console.print(f"\n[green]Basic evaluation:[/green]")
    console.print(
        f'  karma eval --model "{model_name}" --datasets openlifescienceai/pubmedqa'
    )

    # With multiple datasets
    console.print(f"\n[blue]With multiple datasets:[/blue]")
    console.print(f'  karma eval --model "{model_name}" \\')
    console.print(
        f"    --datasets openlifescienceai/pubmedqa,openlifescienceai/mmlu_professional_medicine"
    )

    # With custom arguments
    console.print(f"\n[yellow]With custom arguments:[/yellow]")
    console.print(f'  karma eval --model "{model_name}" \\')
    console.print(f"    --datasets openlifescienceai/pubmedqa \\")
    console.print(f"    --max-samples 100 --batch-size 4")

    # Interactive mode
    console.print(f"\n[magenta]Interactive mode:[/magenta]")
    console.print(f'  karma eval --model "{model_name}" --interactive')


def _show_system_examples(console: Console) -> None:
    """
    Show usage examples for system commands.

    Args:
        console: Rich console for output
    """
    console.print(f"\n[cyan]Usage Examples:[/cyan]")

    # List available resources
    console.print(f"\n[green]List available resources:[/green]")
    console.print(f"  karma list models")
    console.print(f"  karma list datasets")

    # Get detailed information
    console.print(f"\n[blue]Get detailed information:[/blue]")
    console.print(f'  karma info model "Qwen/Qwen3-0.6B"')
    console.print(f"  karma info dataset openlifescienceai/pubmedqa")

    # Run evaluation
    console.print(f"\n[yellow]Run evaluation:[/yellow]")
    console.print(
        f'  karma eval --model "Qwen/Qwen3-0.6B" --datasets openlifescienceai/pubmedqa'
    )

    # Check cache status
    console.print(f"\n[magenta]Check cache status:[/magenta]")
    console.print(f"  karma info system --cache-path ./cache.db")


def _show_dataset_examples(
    console: Console, dataset_name: str, dataset_info: dict
) -> None:
    """
    Show usage examples for a dataset.

    Args:
        console: Rich console for output
        dataset_name: Name of the dataset
        dataset_info: Dataset information dictionary
    """
    console.print(f"\n[cyan]Usage Examples:[/cyan]")

    required_args = dataset_info.get("required_args", [])
    optional_args = dataset_info.get("optional_args", [])
    default_args = dataset_info.get("default_args", {})

    # Basic usage (no arguments required)
    if not required_args:
        console.print(f"\n[green]Basic usage:[/green]")
        console.print(
            f'  karma eval --model "Qwen/Qwen3-0.6B"  --datasets {dataset_name}'
        )

    # Usage with required arguments
    if required_args:
        console.print(f"\n[yellow]With required arguments:[/yellow]")
        example_args = []
        for arg in required_args:
            if arg in ["source_language", "target_language"]:
                example_args.append(f"{arg}={'en' if 'source' in arg else 'hi'}")
            elif arg in ["domain"]:
                example_args.append(f"{arg}=medical")
            else:
                example_args.append(f"{arg}=<value>")

        args_str = ",".join(example_args)
        console.print(f'  karma eval --model "Qwen/Qwen3-0.6B"  \\')
        console.print(f"    --datasets {dataset_name} \\")
        console.print(f'    --dataset-args "{dataset_name}:{args_str}"')

    # Usage with optional arguments
    if optional_args:
        console.print(f"\n[blue]With optional arguments:[/blue]")
        all_args = []

        # Add required args
        for arg in required_args:
            if arg in ["source_language", "target_language"]:
                all_args.append(f"{arg}={'en' if 'source' in arg else 'hi'}")
            else:
                all_args.append(f"{arg}=<value>")

        # Add optional args
        for arg in optional_args:
            if arg == "domain":
                all_args.append(f"{arg}=conversational")
            else:
                all_args.append(f"{arg}=<optional_value>")

        if all_args:
            args_str = ",".join(all_args)
            console.print(f'  karma eval --model "Qwen/Qwen3-0.6B" \\')
            console.print(f"    --datasets {dataset_name} \\")
            console.print(f'    --dataset-args "{dataset_name}:{args_str}"')

    # Show interactive mode
    if required_args:
        console.print(f"\n[magenta]Interactive mode (prompts for arguments):[/magenta]")
        console.print(f'  karma eval --model "Qwen/Qwen3-0.6B" \\')
        console.print(f"    --datasets {dataset_name} --interactive")


def _check_dependencies(console: Console) -> None:
    """
    Check status of key dependencies.

    Args:
        console: Rich console for output
    """
    dependencies = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("datasets", "HuggingFace Datasets"),
        ("rich", "Rich"),
        ("click", "Click"),
        ("weave", "Weave"),
        ("duckdb", "DuckDB"),
    ]

    for module_name, display_name in dependencies:
        try:
            module = __import__(module_name)
            version = getattr(module, "__version__", "unknown")
            console.print(f"  [green]✓[/green] {display_name}: {version}")
        except ImportError:
            console.print(f"  [red]✗[/red] {display_name}: not installed")
        except Exception as e:
            console.print(f"  [yellow]?[/yellow] {display_name}: {str(e)}")
