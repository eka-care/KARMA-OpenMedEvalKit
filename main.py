"""
Main entry point for Karma CLI - kept for backwards compatibility.

The main CLI entry point has been moved to karma.cli.main:main and is 
accessible via the 'karma' command when installed.
"""

import sys
from karma.cli.main import main as cli_main


def main():
    """Legacy main function - redirects to CLI."""
    print("Note: The main CLI has moved to 'karma' command.")
    print("Use 'karma --help' for usage information.")
    print("Running CLI now...\n")
    
    # Run the CLI main function
    cli_main()


if __name__ == "__main__":
    main()
