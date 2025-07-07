# CLI Basics

KARMA provides a comprehensive CLI built with Click and Rich for an excellent user experience.

## Basic Commands

```bash
# Get help
karma --help

# Check version
karma --version

# List all available models
karma list models

# List all available datasets
karma list datasets

# Get detailed information about a model
karma info model qwen

# Get detailed information about a dataset
karma info dataset openlifescienceai/pubmedqa
```

## CLI Structure

The KARMA CLI is organized into several main commands:

- **`karma eval`** - Run model evaluations
- **`karma list`** - List available resources (models, datasets, metrics)
- **`karma info`** - Get detailed information about specific resources
- **`karma --help`** - Get help for any command

## Getting Help

You can get help for any command by adding `--help`:

```bash
# General help
karma --help

# Help for evaluation command
karma eval --help

# Help for list command
karma list --help

# Help for info command
karma info --help
```

## Next Steps

- **Run your first evaluation**: See [Running Evaluations](running-evaluations.md)
- **Learn about models**: Check out the [Models Guide](../models/overview.md)
- **Explore datasets**: Read the [Datasets Guide](../datasets/overview.md)