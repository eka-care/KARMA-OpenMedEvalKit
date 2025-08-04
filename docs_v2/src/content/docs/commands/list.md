---
title: karma list
description: Complete reference for the karma list commands
---

The `karma list` command group provides discovery and listing functionality for all KARMA resources.

## Usage

```bash
karma list [COMMAND] [OPTIONS]
```

## Subcommands

- `karma list models` - List all available models
- `karma list datasets` - List all available datasets  
- `karma list metrics` - List all available metrics
- `karma list all` - List all resources (models, datasets, and metrics)

---

## karma list models

List all available models in the registry.

### Usage
```bash
karma list models [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--format` | table\|simple\|csv | table | Output format |

### Examples

```bash
# Table format (default)
karma list models

# Simple text format
karma list models --format simple

# CSV format
karma list models --format csv
```

### Output

The table format shows:
- Model Name
- Status (Available/Unavailable)
- Modality (Text, Audio, Vision, etc.)

---

## karma list datasets

List all available datasets in the registry with optional filtering.

### Usage
```bash
karma list datasets [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--task-type TEXT` | TEXT | - | Filter by task type (e.g., 'mcqa', 'vqa', 'translation') |
| `--metric TEXT` | TEXT | - | Filter by supported metric (e.g., 'accuracy', 'bleu') |
| `--format` | table\|simple\|csv | table | Output format |
| `--show-args` | FLAG | false | Show detailed argument information |

### Examples

```bash
# List all datasets
karma list datasets

# Filter by task type
karma list datasets --task-type mcqa

# Filter by metric
karma list datasets --metric bleu

# Show detailed argument information
karma list datasets --show-args

# Multiple filters
karma list datasets --task-type translation --metric bleu

# CSV output
karma list datasets --format csv
```

### Output

The table format shows:
- Dataset Name
- Task Type
- Metrics
- Required Args
- Processors
- Split
- Commit Hash

With `--show-args`, additional details are shown:
- Required arguments with examples
- Optional arguments with defaults
- Processor information
- Usage examples

---

## karma list metrics

List all available metrics in the registry.

### Usage
```bash
karma list metrics [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--format` | table\|simple\|csv | table | Output format |

### Examples

```bash
# Table format (default)
karma list metrics

# Simple text format
karma list metrics --format simple

# CSV format
karma list metrics --format csv
```

### Output

Shows all registered metrics including:
- KARMA native metrics
- HuggingFace Evaluate metrics (as fallback)

---

## karma list all

List both models, datasets, and metrics in one command.

### Usage
```bash
karma list all [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--format` | table\|simple | table | Output format (CSV not supported) |

### Examples

```bash
# Show all resources
karma list all

# Simple format
karma list all --format simple
```

### Output

Displays:
1. **MODELS** section with all available models
2. **DATASETS** section with all available datasets  
3. **METRICS** section with all available metrics

## Common Usage Patterns

### Discovery Workflow
```bash
# 1. See what models are available
karma list models

# 2. See what datasets work with medical tasks
karma list datasets --task-type mcqa

# 3. Check what metrics are available
karma list metrics

# 4. Get detailed info about a specific dataset
karma info dataset openlifescienceai/pubmedqa
```

### Integration Workflow
```bash
# Export for scripts
karma list models --format csv > models.csv
karma list datasets --format csv > datasets.csv

# Check compatibility
karma list datasets --metric exact_match
```

### Development Workflow
```bash
# Quick overview
karma list all

# Detailed dataset analysis
karma list datasets --show-args --format table
```

## Output Formats

### Table Format
- Rich formatted tables with colors and styling
- Best for interactive use
- Default format

### Simple Format  
- Plain text, one item per line
- Good for scripting and piping
- Minimal formatting

### CSV Format
- Comma-separated values
- Best for data processing and exports
- Machine-readable format
