# CLI API Reference

This section documents KARMA's command-line interface, including main commands, orchestration, and utility functions.

## Main CLI Entry Point

### main

Main CLI entry point using Click for subcommand organization.

::: karma.cli.main.main
    options:
      show_source: false
      show_root_heading: true

### karma

Root command group for KARMA CLI.

::: karma.cli.main.karma
    options:
      show_source: false
      show_root_heading: true

## Commands

### Evaluation Command

#### eval_command

Main evaluation command for running model evaluations.

::: karma.cli.commands.eval
    options:
      show_source: false
      show_root_heading: true

### List Commands

#### list_models

List all available models in the registry.

::: karma.cli.commands.list
    options:
      show_source: false
      show_root_heading: true

### Info Commands

#### info_commands

Get detailed information about models and datasets.

::: karma.cli.commands.info
    options:
      show_source: false
      show_root_heading: true

## Orchestration

### MultiDatasetOrchestrator

Enhanced orchestrator for multi-dataset evaluation with CLI support.

::: karma.cli.orchestrator.MultiDatasetOrchestrator
    options:
      show_source: false
      show_root_heading: true
      show_category_heading: true
      members_order: source
      filters: ["!^_"]
      merge_init_into_class: true

## Output and Formatting

#### ResultsFormatter

Rich table formatter for evaluation results.

::: karma.cli.formatters.table.ResultsFormatter
    options:
      show_source: false
      show_root_heading: true
      show_category_heading: true
      members_order: source
      filters: ["!^_"]

#### ModelFormatter

Rich table formatter for model information.

::: karma.cli.formatters.table.ModelFormatter
    options:
      show_source: false
      show_root_heading: true
      show_category_heading: true
      members_order: source
      filters: ["!^_"]

#### DatasetFormatter

Rich table formatter for dataset information.

::: karma.cli.formatters.table.DatasetFormatter
    options:
      show_source: false
      show_root_heading: true
      show_category_heading: true
      members_order: source
      filters: ["!^_"]

## Utilities

### CLI Utilities

Utility functions for CLI operations.

::: karma.cli.utils
    options:
      show_source: false
      show_root_heading: true

## Usage Examples

### Basic CLI Usage

```bash
# Get help
karma --help

# Check version
karma --version

# List available models
karma list models

# List available datasets  
karma list datasets

# Get model information
karma info model qwen

# Get dataset information
karma info dataset openlifescienceai/pubmedqa
```

### Evaluation Commands

```bash
# Basic evaluation
karma eval --model qwen --model-path "Qwen/Qwen3-0.6B"

# Evaluate specific datasets
karma eval --model qwen --model-path "Qwen/Qwen3-0.6B" --datasets "openlifescienceai/pubmedqa,openlifescienceai/medmcqa"

# Save results to file
karma eval --model qwen --model-path "Qwen/Qwen3-0.6B" --output results.json

# Custom batch size
karma eval --model qwen --model-path "Qwen/Qwen3-0.6B" --batch-size 16

# Disable caching
karma eval --model qwen --model-path "Qwen/Qwen3-0.6B" --no-cache

# Clear cache before evaluation
karma eval --model qwen --model-path "Qwen/Qwen3-0.6B" --clear-cache
```

### Advanced CLI Usage

```bash
# Dataset-specific arguments
karma eval --model qwen --model-path "Qwen/Qwen3-0.6B" \
  --datasets "ai4bharat/IN22-Conv" \
  --dataset-args "ai4bharat/IN22-Conv:source_language=en,target_language=hi"

# Custom model parameters
karma eval --model qwen --model-path "Qwen/Qwen3-0.6B" \
  --model-kwargs '{"temperature":0.5,"max_tokens":512,"top_p":0.9}'

# Multiple dataset arguments
karma eval --model qwen --model-path "Qwen/Qwen3-0.6B" \
  --datasets "openlifescienceai/medmcqa,openlifescienceai/pubmedqa" \
  --dataset-args "openlifescienceai/medmcqa:split=validation" \
  --dataset-args "openlifescienceai/pubmedqa:subset=pqa_labeled"

# Verbose output with logging
karma eval --model qwen --model-path "Qwen/Qwen3-0.6B" --verbose
```

### Programmatic CLI Usage

```python
from karma.cli.orchestrator import MultiDatasetOrchestrator
from karma.cli.output_adapter import OutputAdapter

# Initialize orchestrator
orchestrator = MultiDatasetOrchestrator(
    use_cache=True,
    batch_size=8,
    verbose=True
)

# Run evaluation
results = orchestrator.evaluate_all_datasets(
    model_name="qwen",
    model_path="Qwen/Qwen3-0.6B",
    datasets=["openlifescienceai/pubmedqa", "openlifescienceai/medmcqa"],
    dataset_args={
        "openlifescienceai/medmcqa": {"split": "validation"},
        "openlifescienceai/pubmedqa": {"subset": "pqa_labeled"}
    },
    model_kwargs={"temperature": 0.7}
)

# Handle output
output_adapter = OutputAdapter(output_path="results.json")
output_adapter.save_results(results)

```
## CLI Configuration

### Environment Variables

```bash
# Cache configuration
export KARMA_CACHE_TYPE=duckdb
export KARMA_CACHE_PATH=./cache.db

# Model configuration
export HUGGINGFACE_TOKEN=your_token
export OPENAI_API_KEY=your_openai_key

# Performance settings
export KARMA_DEFAULT_BATCH_SIZE=8
export KARMA_MAX_WORKERS=4

# Logging
export LOG_LEVEL=INFO
export LOG_FILE=karma.log
```

### Configuration Files

Create a `karma_config.yaml` file:

```yaml
# Default CLI settings
defaults:
  batch_size: 8
  use_cache: true
  verbose: false

# Model-specific settings
models:
  qwen:
    default_kwargs:
      temperature: 0.7
      max_tokens: 512
    
# Dataset-specific settings
datasets:
  openlifescienceai/pubmedqa:
    default_args:
      split: test
  openlifescienceai/medmcqa:
    default_args:
      split: validation

# Output settings
output:
  format: json
  save_individual_results: true
  include_metadata: true
```

## Command Reference

### Global Options

- `--help`: Show help message
- `--version`: Show version information
- `--verbose`: Enable verbose logging
- `--config`: Specify configuration file

### eval Command Options

- `--model`: Model name (required)
- `--model-path`: Path to model weights (required)
- `--datasets`: Comma-separated list of datasets
- `--dataset-args`: Dataset-specific arguments
- `--model-kwargs`: Model-specific parameters (JSON)
- `--batch-size`: Batch size for evaluation
- `--output`: Output file path
- `--cache/--no-cache`: Enable/disable caching
- `--clear-cache`: Clear cache before evaluation

### list Command Options

- `models`: List available models
- `datasets`: List available datasets
- `--format`: Output format (table, json, yaml)
- `--filter`: Filter by type or category

### info Command Options

#### Model Information
- `model <name>`: Get model information
- `model <name> --show-code`: Show model class code location and basic info

#### Dataset Information  
- `dataset <name>`: Get dataset information
- `dataset <name> --show-examples`: Show usage examples with arguments
- `dataset <name> --show-code`: Show dataset class code location

#### System Information
- `system`: Get system information and status
- `system --cache-path <path>`: Path to cache database to check

#### Examples
```bash
# Get model information with code location
karma info model qwen --show-code

# Get dataset information with usage examples
karma info dataset openlifescienceai/pubmedqa --show-examples

# Get system information with custom cache path
karma info system --cache-path /path/to/cache.db
```

## See Also

- [Models API](models.md) - Model implementation details
- [Datasets API](datasets.md) - Dataset implementation details
- [Metrics API](metrics.md) - Metric implementation details
- [Registries API](registries.md) - Registry system