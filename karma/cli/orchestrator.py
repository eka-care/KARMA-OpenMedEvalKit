"""
Multi-Dataset Orchestrator with CLI support.

This module provides enhanced orchestration capabilities for evaluating models
across multiple datasets with rich progress display and argument validation.
"""

import json
import time
import logging
from typing import List, Dict, Any, Optional

from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
)

from karma.benchmark import Benchmark
from karma.registries.model_registry import model_registry
from karma.registries.dataset_registry import dataset_registry
from karma.registries.metrics_registry import metric_registry
from karma.registries.processor_registry import processor_registry
from karma.cli.utils import format_duration, validate_dataset_args
from karma.cache import CacheManager


logger = logging.getLogger(__name__)


class MultiDatasetOrchestrator:
    """Enhanced orchestrator for multi-dataset evaluation with CLI support."""

    def __init__(
        self,
        model_name: str,
        model_path: str,
        console: Optional[Console] = None,
        **model_kwargs,
    ):
        """
        Initialize the orchestrator.

        Args:
            model_name: Name of the model in the registry
            model_path: Path to the model (local path or HuggingFace model ID)
            console: Rich console for output (optional)
            **model_kwargs: Additional model-specific parameters
        """
        self.model_name = model_name
        self.model_path = model_path
        self.model_kwargs = model_kwargs
        self.console = console or Console()
        self.results = {}

        # Validate model exists in registry
        if not model_registry.is_registered(model_name):
            available_models = model_registry.list_models()
            raise ValueError(
                f"Model '{model_name}' not found in registry. "
                f"Available models: {available_models}"
            )

    def evaluate_all_datasets(
        self,
        dataset_names: Optional[List[str]] = None,
        dataset_args: Optional[Dict[str, Dict[str, Any]]] = None,
        processor_args: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
        metric_args: Optional[Dict[str, Dict[str, Any]]] = None,
        batch_size: int = 1,
        use_cache: bool = True,
        show_progress: bool = True,
        max_samples: Optional[int] = None,
        verbose: bool = False,
        dry_run: bool = False,
        refresh_cache: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluate model on multiple datasets with enhanced CLI support.

        Args:
            dataset_names: List of dataset names to evaluate (None for all)
            dataset_args: Dictionary mapping dataset names to their arguments
            processor_args: Dictionary mapping dataset names to processor names to their arguments
            metric_args: Dictionary mapping metric names to their arguments
            batch_size: Batch size for evaluation
            use_cache: Whether to use caching for evaluation
            show_progress: Whether to show progress bars
            max_samples: Maximum number of samples to evaluate
            verbose: Whether to display verbose output
            dry_run: Whether to run in dry-run mode
            refresh_cache: Whether to skip cache lookup and force regeneration

        Returns:
            Dictionary containing evaluation results
        """
        # # Discover models and datasets
        # model_registry.discover_models()
        # dataset_registry.discover_datasets()
        # metric_registry.discover_metrics()

        # Get dataset list
        if dataset_names is None:
            dataset_names = dataset_registry.list_datasets()

        dataset_args = dataset_args or {}

        # Validate all dataset arguments before starting
        self._validate_all_dataset_args(dataset_names, dataset_args)

        # Initialize model once
        self.console.print(
            f"\n[cyan]Initializing model: {self.model_name} with {self.model_kwargs.get('model_kwargs')}[/cyan]"
        )
        model_meta = model_registry.get_model_meta(self.model_name)
        model_meta.loader_kwargs = self.model_kwargs.get("model_kwargs")
        model_meta.model_path = self.model_path

        model = model_registry.get_model(
            self.model_name, **self.model_kwargs.get("model_kwargs")
        )

        # try:
        #     model = model_class(self.model_path, **self.model_kwargs)
        #     self.console.print("[green]✓ Model initialized successfully[/green]")
        # except Exception as e:
        #     self.console.print(f"[red]✗ Failed to initialize model: {e}[/red]")
        #     raise

        # Initialize cache manager once for all datasets
        cache_manager = None
        if use_cache:
            self.console.print("\n[cyan]Initializing cache manager[/cyan]")
            try:
                # Use a generic dataset name for the shared cache manager
                cache_manager = CacheManager(str(dataset_names), model_meta)
                self.console.print(
                    "[green]✓ Cache manager initialized successfully[/green]"
                )
            except Exception as e:
                self.console.print(
                    f"[red]✗ Failed to initialize cache manager: {e}[/red]"
                )
                raise

        # Start evaluation
        overall_start_time = time.time()

        self.console.print(
            f"\n[yellow]Starting evaluation on {len(dataset_names)} datasets[/yellow]"
        )

        # Create progress bar if requested
        if show_progress:
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("({task.completed}/{task.total})"),
                TimeElapsedColumn(),
                console=self.console,
            )

            with progress:
                main_task = progress.add_task(
                    "Evaluating datasets", total=len(dataset_names)
                )

                for i, dataset_name in enumerate(dataset_names):
                    progress.update(main_task, description=f"Evaluating {dataset_name}")

                    self._evaluate_single_dataset(
                        dataset_name,
                        dataset_args.get(dataset_name, {}),
                        processor_args.get(dataset_name, {}) if processor_args else {},
                        metric_args if metric_args else {},
                        model,
                        batch_size,
                        use_cache,
                        progress,
                        cache_manager,
                        max_samples,
                        verbose,
                        dry_run=dry_run,
                        refresh_cache=refresh_cache,
                    )

                    progress.advance(main_task)
        else:
            for dataset_name in dataset_names:
                self._evaluate_single_dataset(
                    dataset_name,
                    dataset_args.get(dataset_name, {}),
                    processor_args.get(dataset_name, {}) if processor_args else {},
                    metric_args if metric_args else {},
                    model,
                    batch_size,
                    use_cache,
                    None,
                    cache_manager,
                    max_samples,
                    dry_run=dry_run,
                    refresh_cache=refresh_cache,
                )

        # Add summary
        total_time = time.time() - overall_start_time
        successful_datasets = len(
            [
                r
                for r in self.results.values()
                if isinstance(r, dict) and r.get("status") == "completed"
            ]
        )

        self.results["_summary"] = {
            "model": self.model_name,
            "model_path": self.model_path,
            "total_datasets": len(dataset_names),
            "successful_datasets": successful_datasets,
            "total_evaluation_time": total_time,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        return self.results

    def _validate_all_dataset_args(
        self, dataset_names: List[str], dataset_args: Dict[str, Dict[str, Any]]
    ) -> None:
        """
        Validate arguments for all datasets before starting evaluation.

        Args:
            dataset_names: List of dataset names
            dataset_args: Dictionary of dataset arguments

        Raises:
            ValueError: If any dataset argument validation fails
        """
        self.console.print("[cyan]Validating dataset arguments...[/cyan]")

        validation_errors = []

        for dataset_name in dataset_names:
            try:
                # Get dataset info
                if not dataset_registry.is_registered(dataset_name):
                    validation_errors.append(
                        f"Dataset '{dataset_name}' not found in registry"
                    )
                    continue

                # Validate arguments if provided
                provided_args = dataset_args.get(dataset_name, {})
                validated_args = validate_dataset_args(
                    dataset_name, provided_args, self.console
                )

                # Update with validated args
                dataset_args[dataset_name] = validated_args

            except Exception as e:
                validation_errors.append(f"Dataset '{dataset_name}': {str(e)}")

        if validation_errors:
            self.console.print("[red]Validation errors found:[/red]")
            for error in validation_errors:
                self.console.print(f"  [red]✗[/red] {error}")
            raise ValueError("Dataset argument validation failed")

        self.console.print("[green]✓ All dataset arguments validated[/green]")

    def _evaluate_single_dataset(
        self,
        dataset_name: str,
        dataset_args: Dict[str, Any],
        processor_args: Dict[str, Dict[str, Any]],
        metric_args: Optional[Dict[str, Dict[str, Any]]] = None,
        model: Any = None,
        batch_size: int = 1,
        use_cache: bool = True,
        progress: Optional[Progress] = None,
        cache_manager: Optional[CacheManager] = None,
        max_samples: Optional[int] = None,
        verbose: bool = False,
        dry_run: bool = False,
        refresh_cache: bool = False,
    ) -> None:
        """
        Evaluate model on a single dataset.

        Args:
            dataset_name: Name of the dataset
            dataset_args: Arguments for dataset creation
            processor_args: Arguments for processor creation (mapping processor names to their args)
            metric_args: Arguments for metric creation (mapping metric names to their args)
            model: Model instance
            batch_size: Batch size for evaluation
            use_cache: Whether to use caching for evaluation
            progress: Progress bar (optional)
            cache_manager: Optional pre-initialized cache manager instance
        """
        dataset_start_time = time.time()

        if not progress:
            self.console.print(
                f"\n[bold cyan]Evaluating {dataset_name.upper()}[/bold cyan]"
            )

        try:
            # Get dataset info
            dataset_info = dataset_registry.get_dataset_info(dataset_name)

            # Get processors from dataset registry metadata and processor registry
            processor_names = dataset_info.get("processors", [])
            processor_instances = []

            if processor_names:
                for processor_name in processor_names:
                    try:
                        # Get processor arguments if provided
                        proc_args = processor_args.get(processor_name, {})

                        if proc_args:
                            # Validate processor arguments
                            from karma.cli.utils import validate_processor_args

                            validated_args = validate_processor_args(
                                dataset_name, processor_name, proc_args, self.console
                            )
                            processor_instance = processor_registry.get_processor(
                                processor_name, **validated_args
                            )
                            self.console.print(
                                f"\nLoaded processor '{processor_name}' with arguments {validated_args} for dataset '{dataset_name}'"
                            )
                        else:
                            processor_instance = processor_registry.get_processor(
                                processor_name
                            )
                            self.console.print(
                                f"\nLoaded processor '{processor_name}' for dataset '{dataset_name}'"
                            )

                        processor_instances.append(processor_instance)
                    except ValueError as e:
                        self.console.print(
                            f"\nCould not load processor '{processor_name}' for dataset '{dataset_name}': {e}"
                        )

            # Pass processors to dataset creation
            final_dataset_args = dataset_args.copy()
            if processor_instances:
                final_dataset_args["processors"] = processor_instances

            if max_samples:
                try:
                    final_dataset_args["max_samples"] = int(max_samples)
                    if int(final_dataset_args["max_samples"]) < 0:
                        raise ValueError
                except ValueError:
                    self.console.print(
                        f"[red]Invalid max_samples argument: {max_samples}, needs to be a positive integer[/red]"
                    )

            # Create dataset with validated arguments
            dataset = dataset_registry.create_dataset(
                dataset_name, validate_args=True, **final_dataset_args
            )

            # Run evaluation for each metric
            dataset_results = {}
            metrics = dataset_info["metrics"]
            metrics_classes = []
            for metric_name in metrics:
                # Get metric class from registry
                metric_kwargs = metric_args.get(metric_name, {}) if metric_args else {}
                if metric_kwargs == {}:
                    logger.info(f"No metric kwargs provided for {metric_name}, using default values only")
                metric_instance = metric_registry.get_metric_class(
                    metric_name, **metric_kwargs
                )
                metrics_classes.append(metric_instance)
                # Create benchmark instance
            benchmark = Benchmark(
                model=model,
                dataset=dataset,
                cache_manager=cache_manager,
                progress=progress,
                # console=self.console,
                verbose_mode=verbose,
                refresh_cache=refresh_cache,
            )

            # Run evaluation
            result = benchmark.evaluate(
                metrics=metrics_classes, batch_size=batch_size, dry_run=dry_run
            )

            for metric_key, score in result["overall_score"].items():
                dataset_results[metric_key] = {
                    "score": score,
                    "evaluation_time": result["summary"]["evaluation_time"],
                    "num_samples": len(result["predictions"]),
                }

                if progress:
                    metric_task = progress.add_task(f"Computing {metric_key}", total=1)
                    progress.remove_task(metric_task)
                else:
                    self.console.print(f"  Computing [yellow]{metric_key}[/yellow]...")
                    self.console.print(f"    [green]{metric_key}: {score:.3f}[/green]")

            # Store results
            self.results[dataset_name] = {
                "metrics": dataset_results,
                "task_type": dataset_info["task_type"],
                "status": "completed",
                "dataset_args": dataset_args,
                "evaluation_time": time.time() - dataset_start_time,
            }

            if not progress:
                self.console.print(
                    f"  [green]✓ Completed in {format_duration(time.time() - dataset_start_time)}[/green]"
                )

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error evaluating {dataset_name}: {error_msg}")
            raise e

            self.results[dataset_name] = {
                "error": error_msg,
                "status": "failed",
                "dataset_args": dataset_args,
            }

            if not progress:
                self.console.print(f"  [red]✗ Failed: {error_msg}[/red]")

    def save_results(self, output_path: str, format_type: str = "json") -> None:
        """
        Save results to file.

        Args:
            output_path: Path to save results
            format_type: Format type (json, yaml, csv)
        """
        from karma.cli.utils import save_results

        save_results(self.results, output_path, format_type, self.console)

    def print_summary(self, format_type: str = "table") -> None:
        """
        Print evaluation summary.

        Args:
            format_type: Format type (table, json)
        """
        if format_type == "table":
            from karma.cli.formatters.table import ResultsFormatter

            # Print results table
            results_table = ResultsFormatter.format_results_table(self.results)
            self.console.print("\n")
            self.console.print(results_table)

            # Print summary table
            summary_table = ResultsFormatter.format_summary_table(self.results)
            self.console.print("\n")
            self.console.print(summary_table)

        elif format_type == "json":
            self.console.print("\n[cyan]Results Summary:[/cyan]")
            self.console.print(json.dumps(self.results, indent=2))

    def get_results(self) -> Dict[str, Any]:
        """
        Get evaluation results.

        Returns:
            Dictionary containing evaluation results
        """
        return self.results.copy()
