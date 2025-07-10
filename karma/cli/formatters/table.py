"""
Table formatters for displaying structured data in the CLI.

This module provides Rich table formatters for evaluation results,
model listings, dataset information, and other tabular data.
"""

from typing import Dict, Any, List, Optional
from rich.table import Table
from rich.console import Console
from rich.text import Text
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
)

from karma.cli.utils import format_duration, format_file_size


class ResultsFormatter:
    """Formatter for evaluation results."""

    @staticmethod
    def format_results_table(
        results: Dict[str, Any], title: str = "Evaluation Results"
    ) -> Table:
        """
        Format evaluation results as a rich table.

        Args:
            results: Results dictionary from evaluation
            title: Table title

        Returns:
            Rich Table object
        """
        table = Table(title=title, show_header=True, header_style="bold magenta")

        # Add columns
        table.add_column("Dataset", style="cyan", no_wrap=True)
        table.add_column("Task Type", style="blue")
        table.add_column("Metric", style="yellow")
        table.add_column("Score", style="green", justify="right")
        table.add_column("Samples", style="dim", justify="right")
        table.add_column("Time", style="dim", justify="right")
        table.add_column("Status", style="white")

        # Process results
        for dataset_name, dataset_result in results.items():
            if dataset_name.startswith("_"):  # Skip summary
                continue

            if not isinstance(dataset_result, dict):
                continue

            if dataset_result.get("status") == "completed":
                task_type = dataset_result.get("task_type", "unknown")
                metrics = dataset_result.get("metrics", {})

                for metric_name, metric_data in metrics.items():
                    score = metric_data.get("score", 0)
                    num_samples = metric_data.get("num_samples", 0)
                    eval_time = metric_data.get("evaluation_time", 0)

                    # Format score
                    if isinstance(score, float):
                        score_text = f"{score:.3f}"
                    else:
                        score_text = str(score)

                    # Add row
                    table.add_row(
                        dataset_name,
                        task_type,
                        metric_name,
                        score_text,
                        str(num_samples),
                        format_duration(eval_time),
                        "[green]✓ Completed[/green]",
                    )
            elif dataset_result.get("status") == "failed":
                error_msg = dataset_result.get("error", "Unknown error")
                table.add_row(
                    dataset_name, "—", "—", "—", "—", "—", f"[red]✗ Failed[/red]"
                )

        return table

    @staticmethod
    def format_summary_table(results: Dict[str, Any]) -> Table:
        """
        Format evaluation summary as a rich table.

        Args:
            results: Results dictionary with summary

        Returns:
            Rich Table object
        """
        summary = results.get("_summary", {})

        table = Table(title="Evaluation Summary", show_header=False, box=None)
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="white")

        # Add summary rows
        if "model" in summary:
            table.add_row("Model", summary["model"])
        if "model_path" in summary:
            table.add_row("Model Path", summary["model_path"])
        if "total_datasets" in summary and "successful_datasets" in summary:
            success_rate = (
                summary["successful_datasets"] / summary["total_datasets"] * 100
            )
            table.add_row(
                "Datasets",
                f"{summary['successful_datasets']}/{summary['total_datasets']} ({success_rate:.1f}%)",
            )
        if "total_evaluation_time" in summary:
            table.add_row(
                "Total Time", format_duration(summary["total_evaluation_time"])
            )
        if "timestamp" in summary:
            table.add_row("Completed", summary["timestamp"])

        return table


class ModelFormatter:
    """Formatter for model information."""

    @staticmethod
    def format_models_list(models: List[str], title: str = "Available Models") -> Table:
        """
        Format list of models as a table.

        Args:
            models: List of model names
            title: Table title

        Returns:
            Rich Table object
        """
        table = Table(title=title, show_header=True, header_style="bold cyan")
        table.add_column("Model Name", style="cyan")

        for model in sorted(models):
            table.add_row(model)

        return table

    @staticmethod
    def format_model_info(model_name: str, model_info: Dict[str, Any]) -> Table:
        """
        Format detailed model information.

        Args:
            model_name: Name of the model
            model_info: Model information dictionary

        Returns:
            Rich Table object
        """
        table = Table(title=f"Model: {model_name}", show_header=False, box=None)
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Name", model_name)
        table.add_row("Class", model_info.get("class_name", "Unknown"))
        table.add_row("Module", model_info.get("module", "Unknown"))

        return table


class DatasetFormatter:
    """Formatter for dataset information."""

    @staticmethod
    def format_datasets_list(
        datasets_info: Dict[str, Dict[str, Any]], title: str = "Available Datasets"
    ) -> Table:
        """
        Format list of datasets as a table.

        Args:
            datasets_info: Dictionary mapping dataset names to their info
            title: Table title

        Returns:
            Rich Table object
        """
        table = Table(title=title, show_header=True, header_style="bold cyan")
        table.add_column("Dataset", style="cyan")
        table.add_column("Task Type", style="blue")
        table.add_column("Metrics", style="yellow")
        table.add_column("Processors", style="magenta")
        table.add_column("Required Args", style="green")
        table.add_column("Commit Hash", style="dim cyan")
        table.add_column("Split", style="dim blue")

        for dataset_name in sorted(datasets_info.keys()):
            info = datasets_info[dataset_name]

            metrics = ", ".join(info.get("metrics", []))
            processors = ", ".join(info.get("processors") or [])
            required_args = ", ".join(info.get("required_args", []))
            commit_hash = info.get("commit_hash", "")
            split = info.get("split", "")

            if not metrics:
                metrics = "—"
            if not processors:
                processors = "—"
            if not required_args:
                required_args = "—"
            if not commit_hash:
                commit_hash = "—"
            else:
                # Truncate commit hash to first 8 characters for better display
                commit_hash = commit_hash[:8]
            if not split:
                split = "—"

            table.add_row(
                dataset_name,
                info.get("task_type", "unknown"),
                metrics,
                processors,
                required_args,
                commit_hash,
                split,
            )

        return table

    @staticmethod
    def format_dataset_info(dataset_name: str, dataset_info: Dict[str, Any]) -> Table:
        """
        Format detailed dataset information.

        Args:
            dataset_name: Name of the dataset
            dataset_info: Dataset information dictionary

        Returns:
            Rich Table object
        """
        table = Table(title=f"Dataset: {dataset_name}", show_header=False, box=None)
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Name", dataset_name)
        table.add_row("Class", dataset_info.get("class_name", "Unknown"))
        table.add_row("Module", dataset_info.get("module", "Unknown"))
        table.add_row("Task Type", dataset_info.get("task_type", "Unknown"))

        # Metrics
        metrics = dataset_info.get("metrics", [])
        table.add_row("Metrics", ", ".join(metrics) if metrics else "None")

        # Processors
        processors = dataset_info.get("processors") or []
        table.add_row("Processors", ", ".join(processors) if processors else "None")

        # Arguments
        required_args = dataset_info.get("required_args", [])
        optional_args = dataset_info.get("optional_args", [])
        default_args = dataset_info.get("default_args", {})

        if required_args:
            table.add_row("Required Args", ", ".join(required_args))
        else:
            table.add_row("Required Args", "None")

        if optional_args:
            table.add_row("Optional Args", ", ".join(optional_args))

        if default_args:
            defaults_str = ", ".join([f"{k}={v}" for k, v in default_args.items()])
            table.add_row("Default Args", defaults_str)

        return table


class SystemFormatter:
    """Formatter for system information."""

    @staticmethod
    def format_system_info(
        cache_info: Dict[str, Any], models_count: int, datasets_count: int
    ) -> Table:
        """
        Format system information table.

        Args:
            cache_info: Cache information dictionary
            models_count: Number of available models
            datasets_count: Number of available datasets

        Returns:
            Rich Table object
        """
        table = Table(title="System Information", show_header=False, box=None)
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="white")

        table.add_row("Available Models", str(models_count))
        table.add_row("Available Datasets", str(datasets_count))

        # Cache information
        if cache_info["exists"]:
            cache_status = f"✓ Available ({cache_info['size_formatted']})"
        else:
            cache_status = "✗ Not found"

        table.add_row("Cache Database", cache_status)
        table.add_row("Cache Path", cache_info["path"])

        return table


def create_progress_bar(description: str = "Processing") -> Progress:
    """
    Create a standardized progress bar for CLI operations.

    Args:
        description: Description text for the progress bar

    Returns:
        Rich Progress object
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=Console(),
    )
