"""
List command for the Karma CLI.

This module implements the 'list' command which displays available models,
datasets, and other resources in the Karma framework.
"""

import click
import csv
import json
import io
from typing import Optional
from rich.console import Console
from rich.panel import Panel

from karma.cli.formatters.table import ModelFormatter, DatasetFormatter
from karma.cli.utils import ClickFormatter
from karma.registries.model_registry import model_registry
from karma.registries.dataset_registry import dataset_registry
from karma.registries.metrics_registry import metric_registry
from karma.registries.registry_manager import discover_all_registries


def format_models_csv(models):
    """Format models list as CSV."""
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow(['name'])
    
    # Write data
    for model in sorted(models):
        writer.writerow([model])
    
    return output.getvalue()


def format_datasets_csv(datasets_info):
    """Format datasets info as CSV."""
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow([
        'name', 'task_type', 'metrics', 'required_args', 'optional_args', 
        'processors', 'split', 'commit_hash'
    ])
    
    # Write data
    for dataset_name in sorted(datasets_info.keys()):
        info = datasets_info[dataset_name]
        
        # Format list fields as JSON for proper parsing
        metrics = json.dumps(info.get('metrics', []))
        required_args = json.dumps(info.get('required_args', []))
        optional_args = json.dumps(info.get('optional_args', []))
        processors = json.dumps(info.get('processors', []))
        
        writer.writerow([
            dataset_name,
            info.get('task_type', ''),
            metrics,
            required_args,
            optional_args,
            processors,
            info.get('split', ''),
            info.get('commit_hash', '')
        ])
    
    return output.getvalue()


def format_metrics_csv(metrics):
    """Format metrics list as CSV."""
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow(['name'])
    
    # Write data
    for metric in sorted(metrics):
        writer.writerow([metric])
    
    return output.getvalue()


@click.group(name="list")
@click.pass_context
def list_cmd(ctx):
    """
    List available models, datasets, and other resources.

    Use 'karma list models' or 'karma list datasets' to see available resources.
    """
    pass


@list_cmd.command(name="models")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "simple", "csv"], case_sensitive=False),
    default="table",
    help="Output format",
)
@click.pass_context
def list_models(ctx, output_format):
    """
    List all available models in the registry.

    This command discovers and displays all models that have been registered
    with the Karma framework.

    Examples:
        karma list models
        karma list models --format simple
    """
    console = ctx.obj["console"]

    try:
        # Discover all registries for better performance
        console.print("[cyan]Discovering models...[/cyan]")
        discover_all_registries(use_cache=True, parallel=True)

        # Get models list
        models = model_registry.list_models()

        if not models:
            console.print(ClickFormatter.warning("No models found in registry"))
            console.print(
                "\nTo register a model, add the @register_model decorator to your model class."
            )
            return

        # Display results
        if output_format == "table":
            table = ModelFormatter.format_models_list(models)
            console.print("\n")
            console.print(table)
        elif output_format == "csv":
            csv_output = format_models_csv(models)
            print(csv_output, end='')  # Use plain print to avoid rich formatting
            return  # Don't print success message for CSV
        else:
            console.print(f"\n[cyan]Available Models ({len(models)}):[/cyan]")
            for model in sorted(models):
                console.print(f"  {model}")

        console.print(f"\n{ClickFormatter.success(f'Found {len(models)} models')}")

    except Exception as e:
        raise e
        console.print(ClickFormatter.error(f"Failed to list models: {str(e)}"))
        raise click.Abort()


@list_cmd.command(name="datasets")
@click.option(
    "--task-type", help="Filter by task type (e.g., 'mcqa', 'vqa', 'translation')"
)
@click.option("--metric", help="Filter by supported metric (e.g., 'accuracy', 'bleu')")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "simple", "csv"], case_sensitive=False),
    default="table",
    help="Output format",
)
@click.option("--show-args", is_flag=True, help="Show detailed argument information")
@click.pass_context
def list_datasets(ctx, task_type, metric, output_format, show_args):
    """
    List all available datasets in the registry.

    This command discovers and displays all datasets that have been registered
    with the Karma framework, with optional filtering capabilities.

    Examples:
        karma list datasets
        karma list datasets --task-type translation
        karma list datasets --metric bleu
        karma list datasets --show-args
    """
    console = ctx.obj["console"]

    try:
        # Discover all registries for better performance
        console.print("[cyan]Discovering datasets...[/cyan]")
        discover_all_registries(use_cache=True, parallel=True)

        # Get datasets list
        all_datasets = dataset_registry.list_datasets()

        if not all_datasets:
            console.print(ClickFormatter.warning("No datasets found in registry"))
            console.print(
                "\nTo register a dataset, add the @register_dataset decorator to your dataset class."
            )
            return

        # Build dataset info dictionary
        datasets_info = {}
        for dataset_name in all_datasets:
            try:
                info = dataset_registry.get_dataset_info(dataset_name)
                datasets_info[dataset_name] = info
            except Exception as e:
                console.print(
                    ClickFormatter.warning(
                        f"Could not get info for dataset '{dataset_name}': {e}"
                    )
                )

        # Apply filters
        filtered_datasets = _apply_dataset_filters(datasets_info, task_type, metric)

        if not filtered_datasets:
            filter_desc = []
            if task_type:
                filter_desc.append(f"task-type='{task_type}'")
            if metric:
                filter_desc.append(f"metric='{metric}'")

            filter_str = " and ".join(filter_desc)
            console.print(
                ClickFormatter.warning(
                    f"No datasets found matching filters: {filter_str}"
                )
            )
            return

        # Display results
        if output_format == "table":
            table = DatasetFormatter.format_datasets_list(filtered_datasets)
            console.print("\n")
            console.print(table)

            # Show detailed argument info if requested
            if show_args:
                _show_detailed_args(console, filtered_datasets)

        elif output_format == "csv":
            csv_output = format_datasets_csv(filtered_datasets)
            print(csv_output, end='')  # Use plain print to avoid rich formatting
            return  # Don't print success message for CSV
        else:
            console.print(
                f"\n[cyan]Available Datasets ({len(filtered_datasets)}):[/cyan]"
            )
            for dataset_name in sorted(filtered_datasets.keys()):
                info = filtered_datasets[dataset_name]
                task_type_str = info.get("task_type", "unknown")
                metrics_str = ", ".join(info.get("metrics", []))
                processors_str = ", ".join(info.get("processors") or [])

                if processors_str:
                    console.print(
                        f"  {dataset_name} ({task_type_str}) - Metrics: {metrics_str} - Processors: {processors_str}"
                    )
                else:
                    console.print(
                        f"  {dataset_name} ({task_type_str}) - Metrics: {metrics_str}"
                    )

        # Show summary
        filter_parts = []
        if task_type:
            filter_parts.append(f"task-type '{task_type}'")
        if metric:
            filter_parts.append(f"metric '{metric}'")

        if filter_parts:
            filter_str = " and ".join(filter_parts)
            console.print(
                f"\n{ClickFormatter.success(f'Found {len(filtered_datasets)} datasets matching {filter_str}')}"
            )
        else:
            console.print(
                f"\n{ClickFormatter.success(f'Found {len(filtered_datasets)} datasets')}"
            )

    except Exception as e:
        console.print(ClickFormatter.error(f"Failed to list datasets: {str(e)}"))
        raise e
        raise click.Abort()


@list_cmd.command(name="metrics")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "simple", "csv"], case_sensitive=False),
    default="table",
    help="Output format",
)
@click.pass_context
def list_metrics(ctx, output_format):
    """
    List all available metrics in the registry.

    This command discovers and displays all metrics that have been registered
    with the Karma framework.

    Examples:
        karma list metrics
        karma list metrics --format simple
    """
    console = ctx.obj["console"]

    try:
        # Discover all registries for better performance
        console.print("[cyan]Discovering metrics...[/cyan]")
        discover_all_registries(use_cache=True, parallel=True)

        # Get metrics list
        metrics = metric_registry.list_metrics()

        if not metrics:
            console.print(ClickFormatter.warning("No metrics found in registry"))
            console.print(
                "\nTo register a metric, add the @register_metric decorator to your metric class."
            )
            return

        # Display results
        if output_format == "table":
            from rich.table import Table

            table = Table(
                title="Available Metrics", show_header=True, header_style="bold cyan"
            )
            table.add_column("Metric Name", style="green", width=20)

            for metric in sorted(metrics):
                table.add_row(metric)

            console.print("\n")
            console.print(table)
        elif output_format == "csv":
            csv_output = format_metrics_csv(metrics)
            print(csv_output, end='')  # Use plain print to avoid rich formatting
            return  # Don't print success message for CSV
        else:
            console.print(f"\n[cyan]Available Metrics ({len(metrics)}):[/cyan]")
            for metric in sorted(metrics):
                console.print(f"  {metric}")

        console.print(f"\n{ClickFormatter.success(f'Found {len(metrics)} metrics')}")

    except Exception as e:
        console.print(ClickFormatter.error(f"Failed to list metrics: {str(e)}"))
        raise click.Abort()


def _apply_dataset_filters(
    datasets_info: dict, task_type: Optional[str] = None, metric: Optional[str] = None
) -> dict:
    """
    Apply filters to dataset information.

    Args:
        datasets_info: Dictionary of dataset information
        task_type: Task type filter
        metric: Metric filter

    Returns:
        Filtered datasets dictionary
    """
    filtered = datasets_info.copy()

    if task_type:
        filtered = {
            name: info
            for name, info in filtered.items()
            if info.get("task_type") == task_type
        }

    if metric:
        filtered = {
            name: info
            for name, info in filtered.items()
            if metric in info.get("metrics", [])
        }

    return filtered


def _show_detailed_args(console: Console, datasets_info: dict) -> None:
    """
    Show detailed argument information for datasets.

    Args:
        console: Rich console for output
        datasets_info: Dictionary of dataset information
    """
    datasets_with_args = {
        name: info
        for name, info in datasets_info.items()
        if (
            info.get("required_args")
            or info.get("optional_args")
            or info.get("default_args")
        )
    }

    if not datasets_with_args:
        console.print("\n[dim]No datasets require special arguments[/dim]")
        return

    console.print("\n[bold cyan]Dataset Arguments Detail[/bold cyan]")
    console.print("─" * 50)

    for dataset_name, info in datasets_with_args.items():
        console.print(f"\n[cyan]{dataset_name}:[/cyan]")

        required_args = info.get("required_args", [])
        optional_args = info.get("optional_args", [])
        default_args = info.get("default_args", {})
        processors = info.get("processors") or []

        if required_args:
            console.print(f"  [red]Required:[/red] {', '.join(required_args)}")

        if optional_args:
            console.print(f"  [yellow]Optional:[/yellow] {', '.join(optional_args)}")

        if default_args:
            defaults_str = ", ".join([f"{k}={v}" for k, v in default_args.items()])
            console.print(f"  [green]Defaults:[/green] {defaults_str}")

        if processors:
            console.print(
                f"  [bright_magenta]Processors:[/bright_magenta] {', '.join(processors)}"
            )

        # Show example usage
        if required_args:
            example_args = []
            for arg in required_args:
                if arg in ["source_language", "target_language"]:
                    example_args.append(f"{arg}={'en' if 'source' in arg else 'hi'}")
                else:
                    example_args.append(f"{arg}=<value>")

            example = f"{dataset_name}:" + ",".join(example_args)
            console.print(f'  [dim]Example: --dataset-args "{example}"[/dim]')


# Add subcommands to the list group
@list_cmd.command(name="all")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "simple", "csv"], case_sensitive=False),
    default="table",
    help="Output format",
)
@click.pass_context
def list_all(ctx, output_format):
    """
    List both models and datasets.

    This is a convenience command that displays both available models
    and datasets in one go.
    """
    console = ctx.obj["console"]

    if output_format == "csv":
        # For CSV output, we need to handle it differently
        # Let's create a combined CSV output
        console.print(ClickFormatter.error("CSV format not yet supported for 'list all' command. Use individual commands instead."))
        console.print("Try: karma list models --format csv")
        console.print("     karma list datasets --format csv")
        console.print("     karma list metrics --format csv")
        return

    # Show header
    console.print(
        Panel.fit("[bold cyan]Karma Registry Overview[/bold cyan]", border_style="cyan")
    )

    # List models
    console.print("\n[bold cyan]MODELS[/bold cyan]")
    console.print("─" * 20)
    ctx.invoke(list_models, output_format=output_format)

    # List datasets
    console.print("\n[bold cyan]DATASETS[/bold cyan]")
    console.print("─" * 20)
    ctx.invoke(list_datasets, output_format=output_format)

    # List metrics
    console.print("\n[bold cyan]METRICS[/bold cyan]")
    console.print("─" * 20)
    ctx.invoke(list_metrics, output_format=output_format)
