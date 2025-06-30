"""
Evaluation command for the Karma CLI.

This module implements the 'eval' command which evaluates models across
multiple healthcare datasets with support for dataset-specific arguments.
"""

import click
from rich.console import Console
from rich.panel import Panel

from karma.cli.orchestrator import MultiDatasetOrchestrator
from karma.cli.utils import (
    parse_dataset_args,
    parse_datasets_list,
    validate_model_path,
    get_cache_info,
    ClickFormatter,
)
from dotenv import load_dotenv
from karma.registries.model_registry import model_registry
from karma.registries.dataset_registry import dataset_registry


@click.command(name="eval")
@click.option(
    "--model", required=True, help="Model name from registry (e.g., 'qwen', 'medgemma')"
)
@click.option(
    "--model-path",
    required=True,
    help="Model path (local path or HuggingFace model ID)",
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
    help="Interactively prompt for missing dataset arguments",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Validate arguments and show what would be evaluated without running",
)
@click.pass_context
def eval_cmd(
    ctx,
    model,
    model_path,
    datasets,
    dataset_args,
    batch_size,
    cache,
    output,
    output_format,
    save_format,
    progress,
    interactive,
    dry_run,
):
    """
    Evaluate a model on healthcare datasets.

    This command evaluates a specified model across one or more healthcare
    datasets, with support for dataset-specific arguments and rich output.

    Examples:

        # Basic evaluation on all datasets
        karma eval --model qwen --model-path "Qwen/Qwen2.5-0.5B-Instruct"

        # Evaluate specific datasets
        karma eval --model qwen --model-path "path/to/model" --datasets "pubmedqa,medmcqa"

        # With dataset arguments
        karma eval --model qwen --model-path "path" --datasets "in22conv" \\
          --dataset-args "in22conv:source_language=en,target_language=hi"
    """
    console = ctx.obj["console"]
    verbose = ctx.obj.get("verbose", False)

    load_dotenv()

    # Show header
    console.print(
        Panel.fit(
            "[bold cyan]KARMA: Knowledge Assessment and Reasoning for Medical Applications[/bold cyan]",
            border_style="cyan",
        )
    )

    try:
        # Discover available models and datasets
        console.print("[cyan]Discovering models and datasets...[/cyan]")
        model_registry.discover_models()
        dataset_registry.discover_datasets()

        # Validate model
        if not model_registry.is_registered(model):
            available_models = model_registry.list_models()
            console.print(
                ClickFormatter.error(f"Model '{model}' not found in registry")
            )
            console.print(f"Available models: {', '.join(available_models)}")
            raise click.Abort()

        # Validate model path
        if not validate_model_path(model_path):
            console.print(
                ClickFormatter.warning(f"Model path '{model_path}' may not be valid")
            )
            if not click.confirm("Continue anyway?"):
                raise click.Abort()

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

        # Interactive mode for missing arguments
        if interactive:
            parsed_dataset_args = _handle_interactive_args(
                dataset_names, parsed_dataset_args, console
            )

        # Show evaluation plan
        _show_evaluation_plan(
            console,
            model,
            model_path,
            dataset_names,
            parsed_dataset_args,
            batch_size,
            cache,
            output,
        )

        # Dry run mode
        if dry_run:
            console.print(
                "\n[yellow]Dry run completed. No evaluation performed.[/yellow]"
            )
            return

        # Confirm if not in quiet mode
        # if not ctx.obj.get("quiet", False):
        #     if not click.confirm("\nProceed with evaluation?"):
        #         console.print("[yellow]Evaluation cancelled.[/yellow]")
        #         return

        # Create orchestrator and run evaluation
        console.print("\n" + "=" * 60)

        orchestrator = MultiDatasetOrchestrator(
            model_name=model, model_path=model_path, console=console
        )

        # Run evaluation
        results = orchestrator.evaluate_all_datasets(
            dataset_names=dataset_names,
            dataset_args=parsed_dataset_args,
            batch_size=batch_size,
            use_cache=cache,
            show_progress=progress,
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

    except KeyboardInterrupt:
        console.print("\n[yellow]Evaluation interrupted by user[/yellow]")
        raise click.Abort()
    except Exception as e:
        console.print(f"\n{ClickFormatter.error(f'Evaluation failed: {str(e)}')}")
        raise e
        if verbose:
            console.print_exception()
        raise click.Abort()


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


def _show_evaluation_plan(
    console: Console,
    model: str,
    model_path: str,
    dataset_names: list,
    dataset_args: dict,
    batch_size: int,
    use_cache: bool,
    output: str,
) -> None:
    """
    Display the evaluation plan to the user.

    Args:
        console: Rich console for output
        model: Model name
        model_path: Model path
        dataset_names: List of dataset names
        dataset_args: Dataset arguments
        batch_size: Batch size
        use_cache: Whether to use caching
        output: Output file path
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
    console.print(f"[cyan]Cache:[/cyan] {'Enabled' if use_cache else 'Disabled'}")
    console.print(f"[cyan]Output File:[/cyan] {output}")

    # Show dataset arguments if any
    if dataset_args:
        console.print(f"[cyan]Dataset Arguments:[/cyan]")
        for dataset_name, args in dataset_args.items():
            if args:
                args_str = ", ".join([f"{k}={v}" for k, v in args.items()])
                console.print(f"  {dataset_name}: {args_str}")

    # Show cache info if caching is enabled
    if use_cache:
        import os
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
