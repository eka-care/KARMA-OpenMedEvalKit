---
title: Basic Usage
---

This guide covers the fundamental usage patterns of KARMA for medical AI evaluation.

## Command Line Interface

KARMA provides a comprehensive CLI built with Click and Rich for an excellent user experience.

### Basic Commands

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
karma info model Qwen/Qwen3-0.6B

# Get detailed information about a dataset
karma info dataset openlifescienceai/pubmedqa
```

## Running Evaluations

### Simple Evaluation

The most basic evaluation requires specifying a model and model path:

```bash
karma eval --model Qwen/Qwen3-0.6B
```

This will:
- Load the Qwen3-0.6B model
- Run evaluation on all supported datasets
- Display results with progress tracking
- Cache results for faster re-runs

### Evaluate Specific Datasets

```bash
# Single dataset
karma eval --model Qwen/Qwen3-0.6B --datasets openlifescienceai/pubmedqa

# Multiple datasets
karma eval --model Qwen/Qwen3-0.6B --datasets "openlifescienceai/pubmedqa,openlifescienceai/medmcqa,openlifescienceai/medqa"
```

### Save Results

```bash
# Save to JSON file
karma eval --model Qwen/Qwen3-0.6B --output results.json

# Save to custom path
karma eval --model Qwen/Qwen3-0.6B --output /path/to/results.json
```

## Working with Different Models

### Built-in Models

KARMA includes several pre-configured models:

```bash
# Qwen models
karma eval --model Qwen/Qwen3-0.6B
karma eval --model Qwen/Qwen3-0.6B --model-path "Qwen/Qwen3-1.7B"

# MedGemma models
karma eval --model medgemma --model-path "google/medgemma-4b-it"
```

### Custom Model Parameters

```bash
# Adjust generation parameters
karma eval --model Qwen/Qwen3-0.6B \
  --model-kwargs '{"temperature":0.5,"max_tokens":512,"top_p":0.9}'

# Disable thinking mode (for Qwen)
karma eval --model Qwen/Qwen3-0.6B \
  --model-kwargs '{"enable_thinking":false}'
```

## Dataset Configuration

### Dataset-Specific Arguments

Some datasets require additional configuration:

```bash
# Translation datasets with language pairs
karma eval --model Qwen/Qwen3-0.6B \
    --datasets "ai4bharat/IN22-Conv" \
    --dataset-args "ai4bharat/IN22-Conv:source_language=en,target_language=hi"

# Datasets with specific splits
karma eval --model Qwen/Qwen3-0.6B --datasets "openlifescienceai/medmcqa" \
  --dataset-args "openlifescienceai/medmcqa:split=validation"
```

### Supported Datasets

KARMA supports 12+ medical datasets:

- **openlifescienceai/pubmedqa** - PubMed Question Answering
- **openlifescienceai/medmcqa** - Medical Multiple Choice QA
- **openlifescienceai/medqa** - Medical Question Answering
- **ChuGyouk/MedXpertQA** - Medical Expert QA
- **mdwiratathya/SLAKE-vqa-english** - Structured Language And Knowledge Extraction
- **flaviagiammarino/vqa-rad** - Visual Question Answering for Radiology
- **ai4bharat/IN22-Conv** - Indic Language Conversation Translation
- **ai4bharat/indicvoices_r** - Indic Voices Recognition
- **openlifescienceai/mmlu_professional_medicine** - Medical MMLU benchmarks

## Performance Optimization

### Batch Processing

```bash
# Adjust batch size for your hardware
karma eval --model Qwen/Qwen3-0.6B --batch-size 8

# Smaller batch for limited memory
karma eval --model Qwen/Qwen3-0.6B --batch-size 2

# Larger batch for high-end hardware
karma eval --model Qwen/Qwen3-0.6B --batch-size 16
```

### Caching

KARMA uses intelligent caching to speed up repeated evaluations:

```bash
# Use cache (default)
karma eval --model Qwen/Qwen3-0.6B --cache

# Force fresh evaluation
karma eval --model Qwen/Qwen3-0.6B --no-cache

# Clear cache before evaluation
karma eval --model Qwen/Qwen3-0.6B --clear-cache
```

## Understanding Results

### Result Format

KARMA outputs comprehensive evaluation results:

```json
{
  "model": "qwen",
  "model_path": "Qwen/Qwen3-0.6B",
  "results": {
    "openlifescienceai/pubmedqa": {
      "metrics": {
        "exact_match": 0.745,
        "accuracy": 0.745
      },
      "num_examples": 1000,
      "runtime_seconds": 45.2,
      "cache_hit_rate": 0.8
    },
    "openlifescienceai/medmcqa": {
      "metrics": {
        "exact_match": 0.623,
        "accuracy": 0.623
      },
      "num_examples": 4183,
      "runtime_seconds": 120.5,
      "cache_hit_rate": 0.2
    }
  },
  "total_runtime": 165.7,
  "timestamp": "2025-01-15T10:30:00Z"
}
```

### Metrics Explained

- **exact_match**: Percentage of predictions that exactly match the ground truth
- **accuracy**: Overall accuracy (same as exact_match for most datasets)
- **bleu**: BLEU score for text generation tasks
- **wer**: Word Error Rate for speech recognition tasks
- **cer**: Character Error Rate for speech recognition tasks

## Common Workflows

### Model Comparison

```bash
# Compare different model sizes
karma eval --model Qwen/Qwen3-0.6B --output qwen_0.6b.json
karma eval --model Qwen/Qwen3-0.6B --model-path "Qwen/Qwen3-1.7B" --output qwen_1.7b.json

# Compare different models
karma eval --model Qwen/Qwen3-0.6B --output qwen_results.json
karma eval --model medgemma --model-path "google/medgemma-4b-it" --output medgemma_results.json
```

### Dataset-Specific Evaluation

```bash
# Focus on specific medical domains
karma eval --model Qwen/Qwen3-0.6B \
  --datasets "openlifescienceai/pubmedqa,openlifescienceai/medmcqa,openlifescienceai/medqa"  # Text-based QA

karma eval --model Qwen/Qwen3-0.6B \
  --datasets "mdwiratathya/SLAKE-vqa-english,flaviagiammarino/vqa-rad"  # Vision-language tasks
```

### Parameter Tuning

```bash
# Test different temperature settings
karma eval --model Qwen/Qwen3-0.6B \
  --model-kwargs '{"temperature":0.1}' --output temp_0.1.json

karma eval --model Qwen/Qwen3-0.6B \
  --model-kwargs '{"temperature":0.7}' --output temp_0.7.json

karma eval --model Qwen/Qwen3-0.6B \
  --model-kwargs '{"temperature":1.0}' --output temp_1.0.json
```

## Error Handling

### Common Issues

#### Model Loading Errors

```bash
# Check if model exists
karma info model qwen

# Verify model path
karma eval --model Qwen/Qwen3-0.6B --dry-run
```

#### Dataset Loading Errors

```bash
# Check dataset availability
karma info dataset openlifescienceai/pubmedqa

# Verify dataset arguments
karma eval --model Qwen/Qwen3-0.6B --datasets openlifescienceai/pubmedqa --dry-run
```

#### Memory Issues

```bash
# Reduce batch size
karma eval --model Qwen/Qwen3-0.6B --batch-size 1

# Use smaller model
karma eval --model Qwen/Qwen3-0.6B --datasets openlifescienceai/pubmedqa
```

## Next Steps

- **Custom models?** See the [API Reference](../api-reference/models)
- **Issues?** Visit our [GitHub Issues](https://github.com/eka-care/KARMA-OpenMedEvalKit/issues)