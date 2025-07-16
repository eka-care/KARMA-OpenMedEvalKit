# KARMA: Knowledge Assessment and Reasoning for Medical Applications

<p align="center">
    <em>Karma is a bench</em>
</p>

KARMA-OpenMedEvalKit is a toolkit to evaluate medical application datasets across multiple modalities.
Currently, KARMA supports over 12 datasets spanning text, image and audio modalities.

## Quick Start

Get started with KARMA in minutes:

```bash
# Clone the repository
git clone https://github.com/eka-care/KARMA-OpenMedEvalKit.git
cd KARMA-OpenMedEvalKit

# Install with uv (recommended)
uv sync

# Run your first evaluation on 3 samples on a MCQA task.
karma eval --model "Qwen/Qwen3-0.6B" --datasets openlifescienceai/pubmedqa --max-samples 3
```

### Explore Available Resources

```console
$ karma --help
Usage: karma [OPTIONS] COMMAND [ARGS]...

  Karma - Healthcare AI Model Evaluation Framework

  A comprehensive toolkit for evaluating healthcare AI models across multiple
  India centric datasets with automatic discovery and rich output formatting.

  Examples:
      karma eval --model "Qwen/Qwen3-0.6B" --datasets pubmedqa
      karma list models
      karma info dataset pubmedqa

Options:
  --version      Show the version and exit.
  -v, --verbose  Enable verbose output
  -q, --quiet    Suppress non-essential output
  --help         Show this message and exit.

Commands:
  eval  Evaluate a model on healthcare datasets.
  info  Get detailed information about models, datasets, and system status.
  list  List available models, datasets, and other resources.
```

### Discover Models and Datasets

KARMA supports a wide range of models, datasets, and metrics for medical AI evaluation. To see the complete list of currently supported resources, visit our [**Supported Resources**](supported-resources.md) page, which is automatically updated with each release.

**Quick commands to explore available resources:**

```bash
# List all resources
karma list all

# List specific resource types
karma list models
karma list datasets
karma list metrics
karma list processors
```

### Preview Your Evaluation

```console
$ karma eval --model "Qwen/Qwen3-0.6B" --datasets "openlifescienceai/pubmedqa" --dry-run
╭────────────────────────────────────────────────────────────────────╮
│ KARMA: Knowledge Assessment and Reasoning for Medical Applications │
╰────────────────────────────────────────────────────────────────────╯

Evaluation Plan
──────────────────────────────────────────────────
Model: Qwen/Qwen3-0.6B
Model Path: Qwen/Qwen3-0.6B
Datasets: 1 datasets
  openlifescienceai/pubmedqa
Batch Size: 8
Cache: Enabled
Output File: results.json
Model Configuration:
  temperature: 0.7
  top_k: 50
  top_p: 0.9
  enable_thinking: True
  max_tokens: 256

Dry run completed. No evaluation performed.
```

The CLI provides rich formatting, auto-discovery of models and datasets, and clear feedback - making it easy to get started with medical AI evaluation.

## Architecture Overview

KARMA is built around four core components:

1. **[Models](user-guide/models/built-in-models.md)** - Unified interface for medical AI models
2. **[Datasets](user-guide/datasets/datasets_overview.md)** - Standardized medical evaluation datasets
3. **[Metrics](user-guide/metrics/metrics_overview.md)** - Comprehensive evaluation metrics
5. **[Processors](user-guide/processors/processors_overview.md)** - A way to post process the output of the model 

## License

This project is licensed under the terms of the MIT license.