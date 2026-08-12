"""
Main CLI entry point for the Karma healthcare model evaluation framework.

This module provides the main command-line interface using Click for
subcommand organization and Rich for beautiful output formatting.
"""

import click
from rich.console import Console
from rich.panel import Panel
from rich.traceback import install

from karma.cli.commands.list import list_cmd
from karma.cli.commands.info import info_cmd

import logging
from rich.logging import RichHandler

FORMAT = "%(message)s"
logging.basicConfig(
    level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

for _noisy in (
    "httpx", "httpcore", "urllib3", "botocore", "boto3", "openai", "google", "deepgram"
):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

# # Install rich traceback handler for better error display
install()

# Global console instance
console = Console()


class LazyGroup(click.Group):
    def get_command(self, ctx, cmd_name):
        if cmd_name == "eval":
            from karma.cli.commands.eval import eval_cmd
            return eval_cmd
        if cmd_name == "interactive":
            from karma.cli.commands.interactive import interactive_cmd
            return interactive_cmd
        return super().get_command(ctx, cmd_name)

    def list_commands(self, ctx):
        return ["eval", "list", "info", "interactive"]


@click.group(cls=LazyGroup)
@click.version_option(version="0.1.0", prog_name="karma")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--quiet", "-q", is_flag=True, help="Suppress non-essential output")
@click.pass_context
def karma(ctx, verbose, quiet):
    """
    Karma - Healthcare AI Model Evaluation Framework

    A comprehensive toolkit for evaluating healthcare AI models across
    multiple India centric datasets with automatic discovery and rich output formatting.

    Examples:
        karma eval --model "Qwen/Qwen2.5-0.5B-Instruct"
        karma list models
        karma info dataset pubmedqa
    """
    # Ensure context object exists
    ctx.ensure_object(dict)

    # Store global options in context
    ctx.obj["verbose"] = verbose
    ctx.obj["quiet"] = quiet
    ctx.obj["console"] = console

    # Adjust console verbosity
    root_logger = logging.getLogger()
    if verbose:
        root_logger.setLevel(logging.DEBUG)
    elif quiet:
        root_logger.setLevel(logging.WARNING)
        console.quiet = True
    else:
        root_logger.setLevel(logging.INFO)

        # Show header
    console.print(
        Panel.fit(
            "[bold cyan]KARMA: Knowledge Assessment and Reasoning for Medical Applications[/bold cyan]",
            border_style="cyan",
        )
    )


karma.add_command(list_cmd)
karma.add_command(info_cmd)


def main():
    """Main entry point for the CLI."""
    try:
        karma()
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        raise click.Abort()
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise


if __name__ == "__main__":
    main()
