---
title: karma eval
description: Complete reference for the karma eval command
---

The `karma eval` command is the core of KARMA, used to evaluate models on healthcare datasets.

## Usage

```bash
karma eval [OPTIONS]
```

## Description

Evaluate a model on healthcare datasets. This command evaluates a specified model across one or more healthcare datasets, with support for dataset-specific arguments and rich output.

## Required Options

| Option | Description |
|--------|-------------|
| `--model TEXT` | Model name from registry (e.g., 'Qwen/Qwen3-0.6B', 'google/medgemma-4b-it') **[required]** |

## Optional Arguments

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--model-path TEXT` | TEXT | - | Model path (local path or HuggingFace model ID). If not provided, uses path from model metadata |
| `--datasets TEXT` | TEXT | all | Comma-separated dataset names (default: evaluate on all datasets) |
| `--dataset-args TEXT` | TEXT | - | Dataset arguments in format 'dataset:key=val,key2=val2;dataset2:key=val' |
| `--processor-args TEXT` | TEXT | - | Processor arguments in format 'dataset.processor:key=val,key2=val2;dataset2.processor:key=val' |
| `--metric-args TEXT` | TEXT | - | Metric arguments in format 'metric_name:key=val,key2=val2;metric2:key=val' |
| `--batch-size INTEGER` | 1-128 | 8 | Batch size for evaluation |
| `--cache / --no-cache` | FLAG | enabled | Enable or disable caching for evaluation |
| `--output TEXT` | TEXT | results.json | Output file path |
| `--format` | table\|json | table | Results display format |
| `--save-format` | json\|yaml\|csv | json | Results save format |
| `--progress / --no-progress` | FLAG | enabled | Show progress bars during evaluation |
| `--interactive` | FLAG | false | Interactively prompt for missing dataset, processor, and metric arguments |
| `--dry-run` | FLAG | false | Validate arguments and show what would be evaluated without running |
| `--model-config TEXT` | TEXT | - | Path to model configuration file (JSON/YAML) with model-specific parameters |
| `--model-args TEXT` | TEXT | - | Model parameter overrides as JSON string (e.g., '{"temperature": 0.7, "top_p": 0.9}') |
| `--max-samples TEXT` | TEXT | - | Maximum number of samples to use for evaluation (helpful for testing) |
| `--verbose` | FLAG | false | Enable verbose output |
| `--refresh-cache` | FLAG | false | Skip cache lookup and force regeneration of all results |

## Examples

### Basic Evaluation
```bash
karma eval --model "Qwen/Qwen3-0.6B" --datasets "openlifescienceai/pubmedqa"
```

### Multiple Datasets
```bash
karma eval --model "Qwen/Qwen3-0.6B" --datasets "openlifescienceai/pubmedqa,openlifescienceai/medmcqa"
```

### With Dataset Arguments
```bash
karma eval --model "ai4bharat/indic-conformer-600m-multilingual" \
  --datasets "ai4bharat/IN22-Conv" \
  --dataset-args "ai4bharat/IN22-Conv:source_language=en,target_language=hi"
```

### With Processor Arguments
```bash
karma eval --model "ai4bharat/indic-conformer-600m-multilingual" \
  --datasets "ai4bharat/IN22-Conv" \
  --processor-args "ai4bharat/IN22-Conv.devnagari_transliterator:source_script=en,target_script=hi"
```

### With Metric Arguments
```bash
karma eval --model "Qwen/Qwen3-0.6B" \
  --datasets "Tonic/Health-Bench-Eval-OSS-2025-07" \
  --metric-args "rubric_evaluation:provider_to_use=openai,model_id=gpt-4o-mini,batch_size=5"
```

### With Model Configuration File
```bash
karma eval --model "Qwen/Qwen3-0.6B" \
  --datasets "openlifescienceai/pubmedqa" \
  --model-config "config/qwen_medical.json"
```

### With Model Parameter Overrides
```bash
karma eval --model "Qwen/Qwen3-0.6B" \
  --datasets "openlifescienceai/pubmedqa" \
  --model-args '{"temperature": 0.3, "max_tokens": 1024, "enable_thinking": true}'
```

### Testing with Limited Samples
```bash
karma eval --model "Qwen/Qwen3-0.6B" \
  --datasets "openlifescienceai/pubmedqa" \
  --max-samples 10 --verbose
```

### Interactive Mode
```bash
karma eval --model "Qwen/Qwen3-0.6B" --interactive
```

### Dry Run Validation
```bash
karma eval --model "Qwen/Qwen3-0.6B" \
  --datasets "openlifescienceai/pubmedqa" \
  --dry-run --model-args '{"temperature": 0.5}'
```

### Force Cache Refresh
```bash
karma eval --model "Qwen/Qwen3-0.6B" \
  --datasets "openlifescienceai/pubmedqa" \
  --refresh-cache
```

## Configuration Priority

Model parameters are applied in the following priority order (highest to lowest):

1. **CLI `--model-args`** - Highest priority
2. **Config file (`--model-config`)** - Overrides metadata defaults  
3. **Model metadata defaults** - From registry
4. **CLI `--model-path`** - Sets model path if metadata doesn't provide one

## Configuration File Formats

### JSON Format
```json
{
  "temperature": 0.7,
  "max_tokens": 2048,
  "top_p": 0.9,
  "enable_thinking": true
}
```

### YAML Format
```yaml
temperature: 0.7
max_tokens: 2048
top_p: 0.9
enable_thinking: true
```

## Common Issues

### Model Not Found
```bash
karma list models
```

### Dataset Not Found
```bash
karma list datasets
```

### Invalid JSON in model-args
```bash
# Wrong
--model-args '{temperature: 0.7}'

# Correct
--model-args '{"temperature": 0.7}'
```
