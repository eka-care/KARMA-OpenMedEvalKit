---
title: CLI Basics
---

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
- **`karma interactive`** - Interactive mode of the CLI
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
## Evaluate With Additional Args

This guide explains how to pass additional arguments to control datasets, models, processors, and metrics during evaluation using the `karma eval` command.

KARMA CLI supports fine-grained control using the following flags:

- `--dataset-args`
- `--model-args`
- `--processor-args`
- `--metrics-args`

These arguments let you filter subsets, customize generation parameters, modify input processing, and tune evaluation metrics.

#### General Syntax
```bash
# Test with Additional Args
karma eval \
  --model <model_name> \
  --datasets <dataset_name> \
  --dataset-args "<dataset_name>:param1=value1,param2=value2" \
  --model-args "param=value" \
  --processor-args "<dataset_name>:param=value" \
  --metrics-args "<metric_name>:param=value"
```

### Example
#### Dataset Args
```bash
--dataset-args "ekacare/MedMCQA-Indic:subset=as"
```

#### Model Args
```bash
--model-args "temperature=0.7,max_tokens=256"
```

#### Processor Args
```bash
--processor-args "ai4bharat/IN22-Conv.devnagari_transliterator:source_script=en,target_script=hi"
```

#### Metrics Args
```bash
--metrics-args "accuracy:threshold=0.8"
```
