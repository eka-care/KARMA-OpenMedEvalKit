"""
Interactive CLI command for the Karma healthcare model evaluation framework.

This module provides an interactive command-line interface that allows users to
select models, datasets, and processors through a guided experience with
iterative planning based on compatibility.
"""

import click
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt, Confirm, IntPrompt
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from typing import List, Dict, Any, Optional
import json
import platform
from pathlib import Path

# Try to import simple-term-menu for arrow key navigation
ARROW_KEY_SUPPORT = False
try:
    from simple_term_menu import TerminalMenu

    # Only enable on Unix-like systems (Linux, macOS)
    if platform.system() in ["Linux", "Darwin"]:
        ARROW_KEY_SUPPORT = True
except ImportError:
    TerminalMenu = None

from karma.registries.model_registry import model_registry
from karma.registries.dataset_registry import dataset_registry
from karma.registries.processor_registry import processor_registry
from karma.registries.registry_manager import discover_all_registries
from karma.cli.orchestrator import MultiDatasetOrchestrator
from karma.cli.utils import get_compatible_datasets_for_model

console = Console()


class InteractiveSession:
    """Manages the interactive session state and configuration."""

    def __init__(self):
        self.selected_model: Optional[str] = None
        self.model_args: Dict[str, Any] = {}
        self.selected_datasets: List[str] = []
        self.dataset_args: Dict[str, Dict[str, Any]] = {}
        self.dataset_processors: Dict[
            str, List[str]
        ] = {}  # Auto-assigned processors per dataset
        self.processor_args: Dict[
            str, Dict[str, Any]
        ] = {}  # Processor arguments if needed
        self.config_file: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary for saving."""
        return {
            "model": self.selected_model,
            "model_args": self.model_args,
            "datasets": self.selected_datasets,
            "dataset_args": self.dataset_args,
            "dataset_processors": self.dataset_processors,
            "processor_args": self.processor_args,
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        """Load session from dictionary."""
        self.selected_model = data.get("model")
        self.model_args = data.get("model_args", {})
        self.selected_datasets = data.get("datasets", [])
        self.dataset_args = data.get("dataset_args", {})
        self.dataset_processors = data.get("dataset_processors", {})
        self.processor_args = data.get("processor_args", {})


def display_model_table(models: List[str]) -> None:
    """Display available models in a table format."""
    table = Table(title="Available Models")
    table.add_column("Index", style="cyan", no_wrap=True)
    table.add_column("Model Name", style="magenta")
    table.add_column("Type", style="green")
    table.add_column("Modalities", style="yellow")
    table.add_column("Description", style="blue")

    for idx, model_name in enumerate(models, 1):
        try:
            model_meta = model_registry.get_model_meta(model_name)
            modalities = ", ".join(
                [
                    mod.value if hasattr(mod, "value") else str(mod)
                    for mod in model_meta.modalities
                ]
            )
            model_type = (
                model_meta.model_type.value
                if hasattr(model_meta.model_type, "value")
                else str(model_meta.model_type)
            )
            description = model_meta.description or "No description"
            table.add_row(str(idx), model_name, model_type, modalities, description)
        except Exception as e:
            table.add_row(str(idx), model_name, "Unknown", "Unknown", f"Error: {e}")

    console.print(table)


def display_dataset_table(datasets: List[str]) -> None:
    """Display available datasets in a table format."""
    table = Table(title="Available Datasets")
    table.add_column("Index", style="cyan", no_wrap=True)
    table.add_column("Dataset Name", style="magenta")
    table.add_column("Task Type", style="green")
    table.add_column("Metrics", style="yellow")
    table.add_column("Required Args", style="blue")

    for idx, dataset_name in enumerate(datasets, 1):
        try:
            dataset_info = dataset_registry.get_dataset_info(dataset_name)
            metrics = ", ".join(dataset_info.get("metrics", []))
            required_args = ", ".join(dataset_info.get("required_args", []))
            table.add_row(
                str(idx),
                dataset_name,
                dataset_info.get("task_type", "Unknown"),
                metrics,
                required_args if required_args else "None",
            )
        except Exception as e:
            table.add_row(str(idx), dataset_name, "Unknown", "Unknown", f"Error: {e}")

    console.print(table)


# Processor table display removed - processors are now auto-assigned


def create_model_menu_items(models: List[str]) -> List[str]:
    """Create formatted menu items for model selection."""
    menu_items = []
    used_display_names = {}  # Track used display names to avoid duplicates

    for model_name in models:
        try:
            model_meta = model_registry.get_model_meta(model_name)
            modalities = ", ".join(
                [
                    mod.value if hasattr(mod, "value") else str(mod)
                    for mod in model_meta.modalities
                ]
            )
            model_type = (
                model_meta.model_type.value
                if hasattr(model_meta.model_type, "value")
                else str(model_meta.model_type)
            )
            description = model_meta.description or "No description"

            # Smart truncation to avoid duplicates
            if len(model_name) <= 30:
                display_name = model_name
            else:
                # Try different truncation strategies to avoid duplicates
                base_truncated = model_name[:27] + "..."
                display_name = base_truncated

                # If this truncated name already exists, try to make it unique
                counter = 1
                while (
                    display_name in used_display_names
                    and used_display_names[display_name] != model_name
                ):
                    # Try truncating at word boundaries or use different lengths
                    if counter == 1:
                        # Try truncating at last slash or underscore
                        last_sep = max(model_name.rfind("/"), model_name.rfind("_"))
                        if last_sep > 15 and last_sep < len(model_name) - 3:
                            display_name = (
                                model_name[:last_sep] + "/..."
                                if last_sep == model_name.rfind("/")
                                else model_name[:last_sep] + "_..."
                            )
                        else:
                            display_name = model_name[:25] + f"...({counter})"
                    else:
                        display_name = model_name[:25] + f"...({counter})"
                    counter += 1
                    if counter > 5:  # Prevent infinite loop
                        display_name = model_name[:20] + f"...{counter}"
                        break

            used_display_names[display_name] = model_name
            menu_items.append(f"{display_name:<32} [{model_type}, {modalities}]")
        except Exception:
            # Fallback for models without metadata
            if len(model_name) <= 30:
                display_name = model_name
            else:
                display_name = model_name[:27] + "..."
                # Handle duplicates in fallback case too
                counter = 1
                while (
                    display_name in used_display_names
                    and used_display_names[display_name] != model_name
                ):
                    display_name = model_name[:25] + f"...({counter})"
                    counter += 1

            used_display_names[display_name] = model_name
            menu_items.append(f"{display_name:<32} [Unknown type]")

    return menu_items


def create_dataset_menu_items(datasets: List[str]) -> List[str]:
    """Create formatted menu items for dataset selection."""
    menu_items = []
    used_display_names = {}  # Track used display names to avoid duplicates

    for dataset_name in datasets:
        try:
            dataset_info = dataset_registry.get_dataset_info(dataset_name)
            task_type = dataset_info.get("task_type", "unknown")
            metrics = ", ".join(dataset_info.get("metrics", []))
            required_args = dataset_info.get("required_args", [])

            # Smart truncation to avoid duplicates
            if len(dataset_name) <= 30:
                display_name = dataset_name
            else:
                # Try different truncation strategies to avoid duplicates
                base_truncated = dataset_name[:27] + "..."
                display_name = base_truncated

                # If this truncated name already exists, try to make it unique
                counter = 1
                while (
                    display_name in used_display_names
                    and used_display_names[display_name] != dataset_name
                ):
                    # Try truncating at word boundaries or use different lengths
                    if counter == 1:
                        # Try truncating at last slash or underscore
                        last_sep = max(dataset_name.rfind("/"), dataset_name.rfind("_"))
                        if last_sep > 15 and last_sep < len(dataset_name) - 3:
                            display_name = (
                                dataset_name[:last_sep] + "/..."
                                if last_sep == dataset_name.rfind("/")
                                else dataset_name[:last_sep] + "_..."
                            )
                        else:
                            display_name = dataset_name[:25] + f"...({counter})"
                    else:
                        display_name = dataset_name[:25] + f"...({counter})"
                    counter += 1
                    if counter > 5:  # Prevent infinite loop
                        display_name = dataset_name[:20] + f"...{counter}"
                        break

            used_display_names[display_name] = dataset_name

            # # Add indicators for required args
            # args_indicator = " ⚠️" if required_args else ""
            #
            menu_items.append(f"{display_name:<32} [{task_type}, {metrics}]")
        except Exception:
            # Fallback for datasets without info
            if len(dataset_name) <= 30:
                display_name = dataset_name
            else:
                display_name = dataset_name[:27] + "..."
                # Handle duplicates in fallback case too
                counter = 1
                while (
                    display_name in used_display_names
                    and used_display_names[display_name] != dataset_name
                ):
                    display_name = dataset_name[:25] + f"...({counter})"
                    counter += 1

            used_display_names[display_name] = dataset_name
            menu_items.append(f"{display_name:<32} [Unknown]")

    return menu_items


# Auto-assign processors function moved to be part of session management


def arrow_key_model_selection(models: List[str]) -> Optional[int]:
    """Arrow key-based model selection."""
    if not ARROW_KEY_SUPPORT:
        return None

    console.print("\n[bold cyan]🎯 Model Selection[/bold cyan]")
    console.print(f"[dim]Found {len(models)} available models[/dim]")
    console.print(
        "[dim]Use ↑↓ arrows or j/k to navigate, ENTER to select, / to search, ESC to cancel[/dim]"
    )

    menu_items = create_model_menu_items(models)

    terminal_menu = TerminalMenu(
        menu_items,
        menu_cursor="▶ ",
        menu_cursor_style=("fg_cyan", "bold"),
        menu_highlight_style=("bg_cyan", "fg_black"),
        cycle_cursor=True,
        clear_screen=False,
        show_search_hint=True,
        search_key="/",
    )

    choice = terminal_menu.show()
    return choice


def arrow_key_dataset_selection(datasets: List[str]) -> Optional[List[int]]:
    """Arrow key-based multi-select dataset selection."""
    if not ARROW_KEY_SUPPORT:
        return None

    console.print("\n[bold cyan]📊 Dataset Selection[/bold cyan]")
    console.print(f"[dim]Found {len(datasets)} compatible datasets[/dim]")
    console.print(
        "[dim]Use ↑↓ arrows to navigate, SPACE to toggle, ENTER to confirm, / to search, ESC to cancel[/dim]"
    )

    menu_items = create_dataset_menu_items(datasets)
    terminal_menu = TerminalMenu(
        menu_items,
        menu_cursor="▶ ",
        menu_cursor_style=("fg_green", "bold"),
        menu_highlight_style=("bg_green", "fg_black"),
        multi_select=True,
        # show_multi_select_hint=True,
    )

    choices = terminal_menu.show()
    return choices if choices is not None else []


def select_model(
    session: InteractiveSession,
    initial_model: Optional[str] = None,
    auto_mode: bool = False,
) -> bool:
    """Interactive model selection."""
    if initial_model:
        if model_registry.is_registered(initial_model):
            session.selected_model = initial_model
            console.print(f"[green]✓[/green] Using pre-selected model: {initial_model}")
            return configure_model_args(session)
        else:
            console.print(f"[red]Error: Model '{initial_model}' not found[/red]")
            return False

    models = model_registry.list_models()

    if not models:
        console.print("[red]No models available[/red]")
        return False

    if auto_mode:
        # In auto mode, select the first model
        session.selected_model = models[0]
        console.print("[yellow]Auto mode: selecting first model[/yellow]")
        console.print(f"[green]✓[/green] Selected model: {session.selected_model}")
        return configure_model_args(session)

    # Try arrow key selection first
    choice = arrow_key_model_selection(models)

    if choice is not None:
        # Arrow key selection successful
        session.selected_model = models[choice]
        console.print(f"[green]✓[/green] Selected model: {session.selected_model}")
        return configure_model_args(session)

    # Fallback to table-based selection
    console.print("\n[bold cyan]Model Selection[/bold cyan]")
    display_model_table(models)

    while True:
        try:
            choice = IntPrompt.ask(
                "Select a model by index", default=1, show_default=True
            )

            if 1 <= choice <= len(models):
                session.selected_model = models[choice - 1]
                console.print(
                    f"[green]✓[/green] Selected model: {session.selected_model}"
                )
                return configure_model_args(session)
            else:
                console.print("[red]Invalid selection. Please try again.[/red]")

        except (ValueError, KeyboardInterrupt, EOFError):
            # Default to first model in case of error
            session.selected_model = models[0]
            console.print("[yellow]Using first model as default[/yellow]")
            console.print(f"[green]✓[/green] Selected model: {session.selected_model}")
            return configure_model_args(session)


def configure_model_args(session: InteractiveSession) -> bool:
    """Configure model arguments interactively."""
    try:
        model_meta = model_registry.get_model_meta(session.selected_model)
        default_args = model_meta.loader_kwargs or {}

        console.print(
            f"\n[bold cyan]Configure Model Arguments for {session.selected_model}[/bold cyan]"
        )

        if not default_args:
            console.print("[yellow]No configurable arguments for this model[/yellow]")
            return True

        console.print(
            "[dim]Press Enter to use default values, or type 'skip' to use all defaults[/dim]"
        )

        try:
            # Check if we want to configure arguments
            configure_choice = Prompt.ask(
                "Configure model arguments?",
                choices=["yes", "no", "skip"],
                default="skip",
            )

            if configure_choice in ["no", "skip"]:
                # Use all default values
                session.model_args = default_args.copy()
                console.print("[yellow]Using all default values[/yellow]")
                return True
        except (EOFError, KeyboardInterrupt):
            # Non-interactive mode or user cancelled
            session.model_args = default_args.copy()
            console.print("[yellow]Using all default values[/yellow]")
            return True

        # Interactive configuration
        for arg_name, default_value in default_args.items():
            try:
                current_value = Prompt.ask(
                    f"{arg_name}", default=str(default_value), show_default=True
                )

                # Try to convert to appropriate type
                try:
                    if isinstance(default_value, bool):
                        session.model_args[arg_name] = current_value.lower() in (
                            "true",
                            "1",
                            "yes",
                            "on",
                        )
                    elif isinstance(default_value, int):
                        session.model_args[arg_name] = int(current_value)
                    elif isinstance(default_value, float):
                        session.model_args[arg_name] = float(current_value)
                    else:
                        session.model_args[arg_name] = current_value
                except ValueError:
                    session.model_args[arg_name] = current_value

            except (EOFError, KeyboardInterrupt):
                # Use default for this argument
                session.model_args[arg_name] = default_value

        return True

    except Exception as e:
        console.print(f"[red]Error configuring model arguments: {e}[/red]")
        # Use defaults as fallback
        try:
            model_meta = model_registry.get_model_meta(session.selected_model)
            session.model_args = model_meta.loader_kwargs or {}
        except:
            session.model_args = {}
        return True


def select_datasets(
    session: InteractiveSession,
    initial_datasets: Optional[List[str]] = None,
    auto_mode: bool = False,
) -> bool:
    """Interactive dataset selection with multi-select support."""
    if initial_datasets:
        valid_datasets = []
        for dataset in initial_datasets:
            if dataset_registry.is_registered(dataset):
                valid_datasets.append(dataset)
            else:
                console.print(f"[red]Warning: Dataset '{dataset}' not found[/red]")

        if valid_datasets:
            session.selected_datasets = valid_datasets
            console.print(
                f"[green]✓[/green] Using pre-selected datasets: {', '.join(valid_datasets)}"
            )
            return configure_dataset_args(session, auto_mode)

    # Get compatible datasets based on selected model
    compatible_datasets = get_compatible_datasets_for_model(session.selected_model)

    if not compatible_datasets:
        console.print("[red]No compatible datasets available[/red]")
        return False

    if session.selected_model:
        console.print(
            f"[dim]Showing datasets compatible with {session.selected_model}[/dim]"
        )

    selected_indices = []

    if auto_mode:
        # In auto mode, prefer datasets without required args
        preferred_idx = None
        for idx, dataset_name in enumerate(compatible_datasets):
            try:
                dataset_info = dataset_registry.get_dataset_info(dataset_name)
                required_args = dataset_info.get("required_args", [])
                if not required_args:  # Prefer datasets with no required args
                    preferred_idx = idx
                    break
            except:
                continue

        selected_idx = preferred_idx if preferred_idx is not None else 0
        session.selected_datasets = [compatible_datasets[selected_idx]]
        console.print(
            f"[yellow]Auto mode: selecting {session.selected_datasets[0]}[/yellow]"
        )
    else:
        # Try arrow key selection first
        choices = arrow_key_dataset_selection(compatible_datasets)

        if choices is not None and len(choices) > 0:
            # Arrow key selection successful
            session.selected_datasets = [compatible_datasets[idx] for idx in choices]
            console.print(
                f"[green]✓[/green] Selected datasets: {', '.join(session.selected_datasets)}"
            )

    return configure_dataset_args(session, auto_mode)


def configure_dataset_args(
    session: InteractiveSession, auto_mode: bool = False
) -> bool:
    """Configure dataset arguments interactively."""
    for dataset_name in session.selected_datasets:
        try:
            dataset_info = dataset_registry.get_dataset_info(dataset_name)
            required_args = dataset_info.get("required_args", [])
            optional_args = dataset_info.get("optional_args", [])
            default_args = dataset_info.get("default_args", {})

            if not required_args and not optional_args:
                console.print(
                    f"[yellow]No configurable arguments for dataset {dataset_name}[/yellow]"
                )
                continue

            console.print(
                f"\n[bold cyan]Configure Arguments for {dataset_name}[/bold cyan]"
            )

            dataset_config = {}

            # Configure required arguments
            for arg_name in required_args:
                if auto_mode:
                    # In auto mode, provide sensible defaults for common required args
                    if arg_name == "source_language":
                        value = "en"
                    elif arg_name == "target_language":
                        value = "hi"
                    elif arg_name == "language":
                        value = "hi"
                    else:
                        value = "default"
                    console.print(
                        f"[yellow]Auto mode: using '{value}' for {arg_name}[/yellow]"
                    )
                else:
                    try:
                        value = Prompt.ask(f"{arg_name} (required)")
                    except (EOFError, KeyboardInterrupt):
                        console.print(
                            f"[yellow]Using default value for {arg_name}[/yellow]"
                        )
                        # Provide sensible defaults
                        if arg_name == "source_language":
                            value = "en"
                        elif arg_name == "target_language":
                            value = "hi"
                        elif arg_name == "language":
                            value = "hi"
                        else:
                            value = "default"
                dataset_config[arg_name] = value

            # Configure optional arguments
            if not auto_mode:  # Skip optional args in auto mode
                for arg_name in optional_args:
                    default_value = default_args.get(arg_name, "")
                    try:
                        value = Prompt.ask(
                            f"{arg_name} (optional)",
                            default=str(default_value) if default_value else "",
                            show_default=bool(default_value),
                        )
                        if value:
                            dataset_config[arg_name] = value
                    except (EOFError, KeyboardInterrupt):
                        console.print(
                            f"[yellow]Skipping optional argument {arg_name}[/yellow]"
                        )
                        break

            session.dataset_args[dataset_name] = dataset_config

        except Exception as e:
            console.print(f"[red]Error configuring dataset {dataset_name}: {e}[/red]")
            return False

    return True


def auto_assign_processors(session: InteractiveSession) -> bool:
    """Automatically assign processors based on dataset configurations."""
    session.dataset_processors = {}

    for dataset_name in session.selected_datasets:
        try:
            dataset_info = dataset_registry.get_dataset_info(dataset_name)
            dataset_processors = dataset_info.get("processors", [])

            if dataset_processors:
                # Validate that all processors are registered
                valid_processors = []
                for processor in dataset_processors:
                    if processor_registry.is_registered(processor):
                        valid_processors.append(processor)
                    else:
                        console.print(
                            f"[yellow]Warning: Processor '{processor}' for dataset '{dataset_name}' not found[/yellow]"
                        )

                if valid_processors:
                    session.dataset_processors[dataset_name] = valid_processors
                    console.print(
                        f"[green]✓[/green] Auto-assigned processors for {dataset_name}: {', '.join(valid_processors)}"
                    )
                else:
                    console.print(
                        f"[yellow]No valid processors found for dataset {dataset_name}[/yellow]"
                    )
            else:
                console.print(
                    f"[dim]No processors configured for dataset {dataset_name}[/dim]"
                )

        except Exception as e:
            console.print(
                f"[red]Error getting processor info for dataset {dataset_name}: {e}[/red]"
            )

    return True


# Processor configuration is now handled automatically based on dataset configurations


def display_configuration_summary(session: InteractiveSession) -> None:
    """Display a summary of the current configuration."""
    console.print("\n[bold cyan]Configuration Summary[/bold cyan]")

    # Model section
    model_panel = Panel(
        f"Model: {session.selected_model}\n"
        + f"Arguments: {json.dumps(session.model_args, indent=2) if session.model_args else 'None'}",
        title="Model Configuration",
        border_style="blue",
    )

    # Datasets section
    datasets_text = f"Datasets: {', '.join(session.selected_datasets)}\n"
    if session.dataset_args:
        datasets_text += "Arguments:\n"
        for dataset, args in session.dataset_args.items():
            datasets_text += f"  {dataset}: {json.dumps(args, indent=2)}\n"
    else:
        datasets_text += "Arguments: None"

    datasets_panel = Panel(
        datasets_text, title="Dataset Configuration", border_style="green"
    )

    # Processors section
    if session.dataset_processors:
        processors_text = "Auto-assigned processors:\n"
        for dataset, processors in session.dataset_processors.items():
            processors_text += f"  {dataset}: {', '.join(processors)}\n"
        if session.processor_args:
            processors_text += "Arguments:\n"
            for processor, args in session.processor_args.items():
                processors_text += f"  {processor}: {json.dumps(args, indent=2)}\n"
    else:
        processors_text = "No processors configured"

    processors_panel = Panel(
        processors_text, title="Processor Configuration", border_style="yellow"
    )

    console.print(model_panel)
    console.print(datasets_panel)
    console.print(processors_panel)


def save_configuration(session: InteractiveSession) -> bool:
    """Save the current configuration to a file."""
    try:
        config_dir = Path.home() / ".karma" / "configs"
        config_dir.mkdir(parents=True, exist_ok=True)

        try:
            config_name = Prompt.ask("Enter configuration name", default="default")
        except (EOFError, KeyboardInterrupt):
            # Use default name in non-interactive mode
            config_name = "default"
            console.print("[yellow]Using default configuration name[/yellow]")

        config_file = config_dir / f"{config_name}.json"

        with open(config_file, "w") as f:
            json.dump(session.to_dict(), f, indent=2)

        console.print(f"[green]✓[/green] Configuration saved to {config_file}")
        session.config_file = str(config_file)
        return True

    except Exception as e:
        console.print(f"[red]Error saving configuration: {e}[/red]")
        return False


def load_configuration(session: InteractiveSession) -> bool:
    """Load a configuration from a file."""
    try:
        config_dir = Path.home() / ".karma" / "configs"

        if not config_dir.exists():
            console.print("[yellow]No saved configurations found[/yellow]")
            return False

        config_files = list(config_dir.glob("*.json"))

        if not config_files:
            console.print("[yellow]No saved configurations found[/yellow]")
            return False

        console.print("\n[bold cyan]Available Configurations[/bold cyan]")
        for idx, config_file in enumerate(config_files, 1):
            console.print(f"{idx}. {config_file.stem}")

        choice = IntPrompt.ask(
            "Select configuration to load", default=1, show_default=True
        )

        if 1 <= choice <= len(config_files):
            selected_config = config_files[choice - 1]

            with open(selected_config, "r") as f:
                config_data = json.load(f)

            session.from_dict(config_data)
            session.config_file = str(selected_config)
            console.print(
                f"[green]✓[/green] Configuration loaded from {selected_config}"
            )
            return True
        else:
            console.print("[red]Invalid selection[/red]")
            return False

    except Exception as e:
        console.print(f"[red]Error loading configuration: {e}[/red]")
        return False


def execute_evaluation(session: InteractiveSession) -> bool:
    """Execute the evaluation with the current configuration."""
    try:
        console.print("\n[bold cyan]Executing Evaluation[/bold cyan]")
        console.print(f"Session model args {session.model_args}")
        # Create orchestrator with model information
        orchestrator = MultiDatasetOrchestrator(
            model_name=session.selected_model,
            model_path=session.selected_model,  # Using model name as path (registry will handle it)
            console=console,
            **{"model_kwargs": session.model_args},
        )

        # Prepare processor arguments in the format expected by orchestrator
        processor_args = {}
        for dataset_name in session.selected_datasets:
            if dataset_name in session.dataset_processors:
                dataset_processor_args = {}
                for processor_name in session.dataset_processors[dataset_name]:
                    if processor_name in session.processor_args:
                        dataset_processor_args[processor_name] = session.processor_args[
                            processor_name
                        ]
                if dataset_processor_args:
                    processor_args[dataset_name] = dataset_processor_args

        # Execute evaluation
        results = orchestrator.evaluate_all_datasets(
            dataset_names=session.selected_datasets,
            dataset_args=session.dataset_args,
            processor_args=processor_args,
            batch_size=1,
            use_cache=True,
            show_progress=True,
            verbose=False,
        )

        console.print("[green]✓[/green] Evaluation completed successfully!")

        # Display results summary
        orchestrator.print_summary("table")

        return True

    except Exception as e:
        console.print(f"[red]Error during evaluation: {e}[/red]")
        import traceback

        console.print(f"[red]Details: {traceback.format_exc()}[/red]")
        return False


@click.command()
@click.option("--model", help="Pre-select a model to start with")
@click.option("--dataset", multiple=True, help="Pre-select datasets to start with")
@click.option("--load-config", help="Load configuration from file")
@click.option("--auto", is_flag=True, help="Auto-mode: use defaults for all selections")
@click.pass_context
def interactive_cmd(ctx, model, dataset, load_config, auto):
    """
    Interactive mode for model evaluation with iterative planning.

    This command provides a guided experience for selecting models and datasets
    with intelligent compatibility filtering. Processors are automatically assigned
    based on dataset configurations.

    Examples:
        karma interactive
        karma interactive --model qwen
        karma interactive --dataset pubmedqa --dataset medqa
        karma interactive --load-config my_config
    """
    # Show interface mode
    interface_mode = (
        "Arrow Key Navigation" if ARROW_KEY_SUPPORT else "Table-based Selection"
    )

    console.print(
        Panel.fit(
            f"[bold cyan]KARMA Interactive Mode[/bold cyan]\n"
            f"Welcome to the interactive evaluation setup!\n"
            f"[dim]Interface: {interface_mode}[/dim]",
            border_style="blue",
        )
    )

    session = InteractiveSession()

    try:
        # Load configuration if specified
        if load_config:
            config_file = Path(load_config)
            if config_file.exists():
                with open(config_file, "r") as f:
                    config_data = json.load(f)
                session.from_dict(config_data)
                console.print(
                    f"[green]✓[/green] Configuration loaded from {config_file}"
                )
                display_configuration_summary(session)

                try:
                    if auto or Confirm.ask("Use this configuration?"):
                        return execute_evaluation(session)
                except (EOFError, KeyboardInterrupt):
                    if auto:
                        return execute_evaluation(session)
                    console.print("[yellow]Configuration load cancelled[/yellow]")
                    return

        # Initialize registries
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Initializing registries...", total=None)

            # Use optimized parallel discovery
            discover_all_registries(use_cache=True, parallel=True)

            progress.update(task, description="Registries initialized!")

        # Step 1: Model Selection
        if not select_model(session, model, auto):
            return

        # Step 2: Dataset Selection
        if not select_datasets(session, list(dataset) if dataset else None, auto):
            return

        # Step 3: Automatic Processor Assignment
        if not auto_assign_processors(session):
            return

        # Step 4: Configuration Review
        display_configuration_summary(session)

        # Step 5: Save Configuration (optional)
        try:
            if auto or Confirm.ask("Save this configuration?"):
                save_configuration(session)
        except (EOFError, KeyboardInterrupt):
            if not auto:
                console.print("[yellow]Configuration save skipped[/yellow]")

        # Step 6: Execute Evaluation
        try:
            if auto or Confirm.ask("Execute evaluation now?"):
                execute_evaluation(session)
            else:
                console.print(
                    "[yellow]Evaluation not executed. Configuration is ready for use.[/yellow]"
                )
                if session.config_file:
                    console.print(
                        f"[dim]You can run it later with: karma interactive --load-config {session.config_file}[/dim]"
                    )
        except (EOFError, KeyboardInterrupt):
            if auto:
                execute_evaluation(session)
            else:
                console.print("[yellow]Evaluation cancelled[/yellow]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Interactive session cancelled[/yellow]")
    except Exception as e:
        console.print(f"[red]Error in interactive mode: {e}[/red]")
        raise
