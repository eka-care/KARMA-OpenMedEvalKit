"""
OutputAdapter: Unified logger/console interface for Karma.

Provides a drop-in replacement for logging that can route output to either a standard logger or a rich Console.
"""

import logging
from typing import Optional, Any

try:
    from rich.console import Console
except ImportError:
    Console = None


class OutputAdapter:
    def __init__(
        self, logger: Optional[logging.Logger] = None, console: Optional[Console] = None
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.console = console

    def info(self, msg: Any, *args, **kwargs):
        if self.console:
            self.console.log(msg)
        else:
            self.logger.info(msg, *args, **kwargs)

    def warning(self, msg: Any, *args, **kwargs):
        if self.console:
            self.console.log(f"[yellow]{msg}[/yellow]")
        else:
            self.logger.warning(msg, *args, **kwargs)

    def error(self, msg: Any, *args, **kwargs):
        if self.console:
            self.console.log(f"[red]{msg}[/red]")
        else:
            self.logger.error(msg, *args, **kwargs)

    def debug(self, msg: Any, *args, **kwargs):
        if self.console:
            self.console.log(f"[dim]{msg}[/dim]")
        else:
            self.logger.debug(msg, *args, **kwargs)

    def success(self, msg: Any, *args, **kwargs):
        if self.console:
            self.console.log(f"[green]{msg}[/green]")
        else:
            self.logger.info(msg, *args, **kwargs)

    def print(self, msg: Any, *args, **kwargs):
        # For compatibility with print-style output
        if self.console:
            self.console.print(msg)
        else:
            print(msg)

    def exception(self, msg: Any, *args, **kwargs):
        if self.console:
            self.console.log(f"[red]{msg}[/red]")
        self.logger.exception(msg, *args, **kwargs)

    def bind(
        self, logger: Optional[logging.Logger] = None, console: Optional[object] = None
    ):
        """
        Return a new OutputAdapter with updated logger/console.
        """
        return OutputAdapter(logger or self.logger, console or self.console)
