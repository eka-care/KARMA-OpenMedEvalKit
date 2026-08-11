"""
Evaluation command for the Karma CLI.

This module implements the 'eval' command which evaluates models across
multiple healthcare datasets with support for dataset-specific arguments.
"""

from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel

from karma.cli.orchestrator import MultiDatasetOrchestrator
from karma.cli.utils import (
    parse_dataset_args,
    parse_processor_args,
    parse_metric_args,
    parse_datasets_list,
    prepare_model_overrides,
    validate_model_path,
    get_cache_info,
    ClickFormatter,
    prompt_for_missing_metric_args,
)
from dotenv import load_dotenv
from karma.registries.model_registry import model_registry
from karma.registries.dataset_registry import dataset_registry
from karma.registries.metrics_registry import metric_registry
from karma.registries.registry_manager import discover_all_registries
import os

from karma.registries.processor_registry import processor_registry


def _mark_run_failed(error: Optional[BaseException] = None) -> None:
    """Update job status to 'failed' in DB, keyed by KARMA_JOB_ID or KARMA_CONFIG_HASH."""
    import os
    import logging
    job_id = os.environ.get("KARMA_JOB_ID", "").strip()
    config_hash = os.environ.get("KARMA_CONFIG_HASH", "").strip()
    reason = None
    if error is not None:
        last_line = str(error).strip().splitlines()[-1] if str(error).strip() else ""
        reason = f"{type(error).__name__}: {last_line}" if last_line else type(error).__name__
    try:
        from karma.cache.cache_manager import PROCESS_JOB_IDS, create_cache_io
        cache_io = create_cache_io()
        if job_id or config_hash:
            payload = {"status": "failed", "error_message": reason}
            if job_id:
                payload["job_id"] = job_id
            if config_hash:
                payload["config_hash"] = config_hash
            cache_io.save_run_metadata(payload)
        # mark_failed only flips running/submitted rows — completed jobs stay untouched.
        marker = getattr(cache_io, "mark_failed", None)
        if callable(marker):
            for jid in PROCESS_JOB_IDS:
                if jid != job_id:
                    marker(jid, reason)
    except Exception:
        logging.getLogger(__name__).warning("Could not mark run as failed in DB", exc_info=True)


@click.command(name="eval")
@click.option(
    "--model", required=True, help="Model name from registry (e.g., 'qwen', 'medgemma')"
)
@click.option(
    "--model-path",
    help="Model path (local path or HuggingFace model ID). If not provided, uses path from model metadata.",
)
@click.option(
    "--datasets",
    help="Comma-separated dataset names (default: evaluate on all datasets)",
)
@click.option(
    "--dataset-args",
    help="Dataset arguments in format 'dataset:key=val,key2=val2;dataset2:key=val'",
)
@click.option(
    "--processor-args",
    help="Processor arguments in format 'dataset.processor:key=val,key2=val2;dataset2.processor:key=val'",
)
@click.option(
    "--metric-args",
    help="Metric arguments in format 'metric_name:key=val,key2=val2;metric2:key=val'",
)
@click.option(
    "--batch-size",
    default=8,
    type=click.IntRange(1, 128),
    help="Batch size for evaluation",
)
@click.option(
    "--cache/--no-cache",
    default=True,
    help="Enable or disable caching for evaluation",
)
@click.option("--output", default="results.json", help="Output file path")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json"], case_sensitive=False),
    default="table",
    help="Results display format",
)
@click.option(
    "--save-format",
    type=click.Choice(["json", "yaml", "csv"], case_sensitive=False),
    default="json",
    help="Results save format",
)
@click.option(
    "--progress/--no-progress",
    default=True,
    help="Show progress bars during evaluation",
)
@click.option(
    "--interactive",
    is_flag=True,
    help="Interactively prompt for missing dataset, processor, and metric arguments",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Validate arguments and show what would be evaluated without running",
)
@click.option(
    "--model-config",
    help="Path to model configuration file (JSON/YAML) with model-specific parameters",
)
@click.option(
    "--model-args",
    help='Model parameter overrides as JSON string (e.g., \'{"temperature": 0.7, "top_p": 0.9}\')',
)
@click.option(
    "--max-samples",
    help="Maximum number of samples to use for evaluation, this is helpful for running on a few samples and checking if everything is working.",
)
@click.option(
    "--verbose",
    default=False,
    help="Pass this argument to have a verbose output",
)
@click.option(
    "--refresh-cache",
    is_flag=True,
    help="Skip cache lookup and force regeneration of all results",
)
@click.option(
    "--cached-only",
    is_flag=True,
    help=(
        "Score only samples that already have cached model outputs; never "
        "initialize/run the evaluated model (no GPU needed). Configs with no "
        "cache are reported and skipped. Useful for re-running the LLM judge."
    ),
)
@click.pass_context
def eval_cmd(
    ctx,
    model,
    model_path,
    datasets,
    dataset_args,
    processor_args,
    metric_args,
    batch_size,
    cache,
    output,
    output_format,
    save_format,
    progress,
    interactive,
    dry_run,
    model_config,
    model_args,
    max_samples,
    verbose,
    refresh_cache,
    cached_only,
):
    """
    Evaluate a model on healthcare datasets.

    This command evaluates a specified model across one or more healthcare
    datasets, with support for dataset-specific arguments and rich output.

    Examples:

        # Basic evaluation on all datasets
        karma eval --model "Qwen/Qwen2.5-0.5B-Instruct"

        # Evaluate specific datasets
        karma eval --model "path/to/model" --datasets "pubmedqa,medmcqa"

        # With dataset and processor arguments
        karma eval --model "path" --datasets "in22conv" \\
          --dataset-args "in22conv:source_language=en,target_language=hi" \\
          --processor-args "in22conv.devnagari_transliterator:source_script=en,target_script=hi"

        # With metric arguments
        karma eval --model "path" --datasets "pubmedqa" \\
          --metric-args "accuracy:normalize=true,sample_weight=none;bleu:max_order=4"
    """
    console = ctx.obj["console"]
    verbose = ctx.obj.get("verbose", False)

    load_dotenv()

    try:
        # Discover available models and datasets using optimized registry manager
        console.print("[cyan]Discovering models and datasets...[/cyan]")
        discover_all_registries(use_cache=True, parallel=True)

        # Validate model (check by ModelMeta name, not short alias)
        model_names = model_registry.list_models()

        # Check if the provided model name matches any registered model
        if model not in model_names:
            # Try to find partial matches or suggest alternatives
            partial_matches = [
                name for name in model_names if model.lower() in name.lower()
            ]

            console.print(
                ClickFormatter.error(f"Model '{model}' not found in registry")
            )

            if partial_matches:
                console.print(
                    f"Did you mean one of these? {', '.join(partial_matches[:5])}"
                )
            else:
                console.print(f"Available models: {', '.join(model_names[:10])}")
                if len(model_names) > 10:
                    console.print(f"... and {len(model_names) - 10} more")

            raise click.Abort()

        # Handle model configuration and parameter overrides
        # This either loads the file or else provided args from CLI.
        model_overrides = prepare_model_overrides(
            model, model_path, model_config, model_args, console
        )

        # Extract final model path (could come from meta config)
        final_model_path = model_overrides.get("model_name_or_path", model_path)

        # # Validate final model path if provided
        # if final_model_path and not validate_model_path(final_model_path):
        #     console.print(
        #         ClickFormatter.warning(
        #             f"Model path '{final_model_path}' may not be valid"
        #         )
        #     )
        #     if not click.confirm("Continue anyway?"):
        #         raise click.Abort()

        # Parse datasets list
        dataset_names = parse_datasets_list(datasets) if datasets else None
        if dataset_names:
            # Validate datasets exist
            for dataset_name in dataset_names:
                if not dataset_registry.is_registered(dataset_name):
                    available_datasets = dataset_registry.list_datasets()
                    console.print(
                        ClickFormatter.error(
                            f"Dataset '{dataset_name}' not found in registry"
                        )
                    )
                    console.print(
                        f"Available datasets: {', '.join(available_datasets)}"
                    )
                    raise click.Abort()
        else:
            dataset_names = dataset_registry.list_datasets()

        # Parse dataset arguments
        parsed_dataset_args = parse_dataset_args(dataset_args) if dataset_args else {}

        # Parse processor arguments
        parsed_processor_args = (
            parse_processor_args(processor_args) if processor_args else {}
        )

        # Parse metric arguments
        parsed_metric_args = parse_metric_args(metric_args) if metric_args else {}

        # Interactive mode for missing arguments
        if interactive:
            parsed_dataset_args = _handle_interactive_args(
                dataset_names, parsed_dataset_args, console
            )

            # Interactive mode for processor arguments
            parsed_processor_args = _handle_interactive_processor_args(
                dataset_names, parsed_processor_args, console
            )

            # Interactive mode for metric arguments
            parsed_metric_args = _handle_interactive_metric_args(
                dataset_names, parsed_metric_args, console
            )

        # Show evaluation plan
        _show_evaluation_plan(
            console,
            model,
            final_model_path,
            dataset_names,
            parsed_dataset_args,
            parsed_processor_args,
            parsed_metric_args,
            model_overrides,
            batch_size,
            cache,
            output,
            refresh_cache,
        )

        # Dry run mode
        # if dry_run:
        #     console.print(
        #         "\n[yellow]Dry run completed. No evaluation performed.[/yellow]"
        #     )
        #     return

        # Confirm if not in quiet mode
        # if not ctx.obj.get("quiet", False):
        #     if not click.confirm("\nProceed with evaluation?"):
        #         console.print("[yellow]Evaluation cancelled.[/yellow]")
        #         return

        # Create orchestrator and run evaluation
        console.print("\n" + "=" * 60)

        orchestrator = MultiDatasetOrchestrator(
            model_name=model,
            model_path=final_model_path,
            model_kwargs=model_overrides,
            console=console,
        )

        # Run evaluation
        results = orchestrator.evaluate_all_datasets(
            dataset_names=dataset_names,
            dataset_args=parsed_dataset_args,
            processor_args=parsed_processor_args,
            metric_args=parsed_metric_args,
            batch_size=batch_size,
            use_cache=cache,
            show_progress=progress,
            max_samples=max_samples,
            verbose=verbose,
            dry_run=dry_run,
            refresh_cache=refresh_cache,
            cached_only=cached_only,
        )

        # Display results
        console.print("\n" + "=" * 60)
        orchestrator.print_summary(format_type=output_format)

        # Save results
        orchestrator.save_results(output, save_format)

        # Show completion message
        console.print(
            f"\n{ClickFormatter.success('Evaluation completed successfully!')}"
        )

        if verbose:
            console.print(f"Results saved to: {output}")
            if cache:
                console.print("Cache: Enabled")
            else:
                console.print("Cache: Disabled")

    except KeyboardInterrupt as e:
        console.print("\n[yellow]Evaluation interrupted by user[/yellow]")
        _mark_run_failed(e)
        raise click.Abort()
    except Exception as e:
        console.print(f"\n{ClickFormatter.error(f'Evaluation failed: {str(e)}')}")
        _mark_run_failed(e)
        raise e


def _handle_interactive_args(
    dataset_names: list, existing_args: dict, console: Console
) -> dict:
    """
    Handle interactive argument collection for datasets.

    Args:
        dataset_names: List of dataset names
        existing_args: Already provided arguments
        console: Rich console for output

    Returns:
        Complete dataset arguments dictionary
    """
    from karma.cli.utils import prompt_for_missing_args

    complete_args = existing_args.copy()

    for dataset_name in dataset_names:
        try:
            # Get required arguments for this dataset
            required_args = dataset_registry.get_dataset_required_args(dataset_name)

            if required_args:
                provided_args = complete_args.get(dataset_name, {})
                missing_args = [
                    arg for arg in required_args if arg not in provided_args
                ]

                if missing_args:
                    console.print(
                        f"\n[cyan]Dataset '{dataset_name}' requires additional arguments[/cyan]"
                    )
                    new_args = prompt_for_missing_args(
                        dataset_name, missing_args, console
                    )

                    if dataset_name not in complete_args:
                        complete_args[dataset_name] = {}
                    complete_args[dataset_name].update(new_args)

        except Exception as e:
            console.print(
                ClickFormatter.warning(
                    f"Could not get requirements for dataset '{dataset_name}': {e}"
                )
            )

    return complete_args


def _handle_interactive_processor_args(
    dataset_names: list, existing_processor_args: dict, console: Console
) -> dict:
    """
    Handle interactive processor argument collection for datasets.

    Args:
        dataset_names: List of dataset names
        existing_processor_args: Already provided processor arguments
        console: Rich console for output

    Returns:
        Complete processor arguments dictionary
    """
    from karma.cli.utils import prompt_for_missing_processor_args

    complete_processor_args = existing_processor_args.copy()

    for dataset_name in dataset_names:
        try:
            # Get dataset info to find associated processors
            dataset_info = dataset_registry.get_dataset_info(dataset_name)
            processor_names = dataset_info.get("processors", [])

            if processor_names:
                # Check if user wants to configure processors
                if not click.confirm(
                    f"\nDataset '{dataset_name}' uses processors: {', '.join(processor_names)}. "
                    f"Configure processor arguments?"
                ):
                    continue

                # Initialize dataset processor args if not exists
                if dataset_name not in complete_processor_args:
                    complete_processor_args[dataset_name] = {}

                for processor_name in processor_names:
                    try:
                        # Get processor argument information
                        processor_info = processor_registry.get_processor_all_args(
                            processor_name
                        )
                        required_args = processor_info["required"]
                        optional_args = processor_info["optional"]
                        default_args = processor_info["defaults"]

                        # Check what arguments are already provided
                        provided_args = complete_processor_args[dataset_name].get(
                            processor_name, {}
                        )
                        missing_required = [
                            arg for arg in required_args if arg not in provided_args
                        ]

                        # Prompt for arguments if needed
                        if missing_required or (
                            optional_args
                            and click.confirm(
                                f"Configure optional arguments for processor '{processor_name}'?"
                            )
                        ):
                            new_args = prompt_for_missing_processor_args(
                                dataset_name,
                                processor_name,
                                missing_required,
                                optional_args,
                                default_args,
                                console,
                            )

                            if new_args:
                                if (
                                    processor_name
                                    not in complete_processor_args[dataset_name]
                                ):
                                    complete_processor_args[dataset_name][
                                        processor_name
                                    ] = {}
                                complete_processor_args[dataset_name][
                                    processor_name
                                ].update(new_args)

                    except Exception as e:
                        console.print(
                            ClickFormatter.warning(
                                f"Could not get requirements for processor '{processor_name}': {e}"
                            )
                        )

        except Exception as e:
            console.print(
                ClickFormatter.warning(
                    f"Could not get processor info for dataset '{dataset_name}': {e}"
                )
            )

    return complete_processor_args


def _handle_interactive_metric_args(
    dataset_names: list, existing_metric_args: dict, console: Console
) -> dict:
    """
    Handle interactive metric argument collection.

    Args:
        dataset_names: List of dataset names
        existing_metric_args: Already provided metric arguments
        console: Rich console for output

    Returns:
        Complete metric arguments dictionary
    """
    complete_metric_args = existing_metric_args.copy()

    # Get all metrics that will be used across datasets
    all_metrics = set()
    for dataset_name in dataset_names:
        try:
            dataset_info = dataset_registry.get_dataset_info(dataset_name)
            metrics = dataset_info.get("metrics", [])
            all_metrics.update(metrics)
        except Exception as e:
            console.print(
                ClickFormatter.warning(
                    f"Could not get metrics for dataset '{dataset_name}': {e}"
                )
            )

    if not all_metrics:
        return complete_metric_args

    # Check if user wants to configure metrics
    console.print(
        f"\n[blue]Found {len(all_metrics)} unique metrics: {', '.join(sorted(all_metrics))}[/blue]"
    )
    if not click.confirm(f"Configure metric arguments?"):
        return complete_metric_args

    # Handle each metric
    for metric_name in sorted(all_metrics):
        try:
            # Get metric argument information
            metric_info = metric_registry.get_metric_all_args(metric_name)
            required_args = metric_info["required"]
            optional_args = metric_info["optional"]
            default_args = metric_info["defaults"]

            # Check what arguments are already provided
            provided_args = complete_metric_args.get(metric_name, {})
            missing_required = [
                arg for arg in required_args if arg not in provided_args
            ]

            # Show metric information to user
            if required_args or optional_args:
                console.print(f"\n[cyan]Metric '{metric_name}' configuration:[/cyan]")
                if required_args:
                    console.print(f"  Required arguments: {', '.join(required_args)}")
                if optional_args:
                    console.print(
                        f"  Optional arguments: {', '.join(optional_args)} (defaults: {default_args})"
                    )

            # Prompt for arguments if needed
            if missing_required or (
                optional_args
                and click.confirm(
                    f"Configure optional arguments for metric '{metric_name}'?"
                )
            ):
                new_args = prompt_for_missing_metric_args(
                    metric_name,
                    missing_required,
                    optional_args,
                    default_args,
                    console,
                )

                if new_args:
                    if metric_name not in complete_metric_args:
                        complete_metric_args[metric_name] = {}
                    complete_metric_args[metric_name].update(new_args)

        except Exception as e:
            console.print(
                ClickFormatter.warning(
                    f"Could not get requirements for metric '{metric_name}': {e}"
                )
            )

    return complete_metric_args


def _show_evaluation_plan(
    console: Console,
    model: str,
    model_path: str,
    dataset_names: list,
    dataset_args: dict,
    processor_args: dict,
    metric_args: dict,
    model_overrides: dict,
    batch_size: int,
    use_cache: bool,
    output: str,
    refresh_cache: bool,
) -> None:
    """
    Display the evaluation plan to the user.

    Args:
        console: Rich console for output
        model: Model name
        model_path: Model path
        dataset_names: List of dataset names
        dataset_args: Dataset arguments
        processor_args: Processor arguments
        metric_args: Metric arguments
        model_overrides: Model overrides
        batch_size: Batch size
        use_cache: Whether to use caching
        output: Output file path
        refresh_cache: Whether to refresh cache
    """
    console.print("\n[bold cyan]Evaluation Plan[/bold cyan]")
    console.print("─" * 50)

    console.print(f"[cyan]Model:[/cyan] {model}")
    console.print(f"[cyan]Model Path:[/cyan] {model_path}")
    console.print(f"[cyan]Datasets:[/cyan] {len(dataset_names)} datasets")

    if len(dataset_names) <= 10:
        console.print(f"  {', '.join(dataset_names)}")
    else:
        console.print(
            f"  {', '.join(dataset_names[:8])}, ... and {len(dataset_names) - 8} more"
        )

    console.print(f"[cyan]Batch Size:[/cyan] {batch_size}")
    cache_status = "Enabled" if use_cache else "Disabled"
    if use_cache and refresh_cache:
        cache_status += " (Refresh mode)"
    console.print(f"[cyan]Cache:[/cyan] {cache_status}")
    console.print(f"[cyan]Output File:[/cyan] {output}")

    # Show dataset arguments if any
    if dataset_args:
        console.print(f"[cyan]Dataset Arguments:[/cyan]")
        for dataset_name, args in dataset_args.items():
            if args:
                args_str = ", ".join([f"{k}={v}" for k, v in args.items()])
                console.print(f"  {dataset_name}: {args_str}")

    # Show processor arguments if any
    if processor_args:
        console.print(f"[cyan]Processor Arguments:[/cyan]")
        for dataset_name, processors in processor_args.items():
            for processor_name, args in processors.items():
                if args:
                    args_str = ", ".join([f"{k}={v}" for k, v in args.items()])
                    console.print(f"  {dataset_name}.{processor_name}: {args_str}")

    # Show metric arguments if any
    if metric_args:
        console.print(f"[cyan]Metric Arguments:[/cyan]")
        for metric_name, args in metric_args.items():
            if args:
                args_str = ", ".join([f"{k}={v}" for k, v in args.items()])
                console.print(f"  {metric_name}: {args_str}")

    # Show model overrides if any
    if model_overrides:
        console.print(f"[cyan]Model Configuration:[/cyan]")
        # Filter out internal keys
        display_overrides = {
            k: v
            for k, v in model_overrides.items()
            if not k.startswith("_") and k != "model_name_or_path"
        }
        if display_overrides:
            for key, value in display_overrides.items():
                console.print(f"  {key}: {value}")

    # Show cache info if caching is enabled
    if use_cache:
        cache_path = os.getenv("KARMA_CACHE_PATH", "./cache.db")
        cache_info = get_cache_info(cache_path)
        if cache_info["exists"]:
            console.print(
                f"[cyan]Cache Status:[/cyan] Available ({cache_info['size_formatted']})"
            )
        else:
            console.print(f"[cyan]Cache Status:[/cyan] New cache will be created")
    else:
        console.print(f"[cyan]Cache Status:[/cyan] Disabled")


def _get_model_specific_help(model_name: str) -> str:
    """
    Get model-specific parameter help text.

    Args:
        model_name: Name of the model

    Returns:
        Help text for model-specific parameters
    """
    try:
        model_meta = model_registry.get_model_meta(model_name)
        help_lines = [
            f"Model: {model_meta.name}",
            f"Type: {model_meta.model_type.value}",
        ]

        if model_meta.loader_kwargs:
            help_lines.append("Default parameters:")
            for key, value in model_meta.loader_kwargs.items():
                help_lines.append(f"  {key}: {value}")

        if model_meta.medical_domains:
            help_lines.append(
                f"Medical domains: {', '.join(model_meta.medical_domains)}"
            )

        return "\n".join(help_lines)

    except Exception:
        return "Could not retrieve model parameter information."
