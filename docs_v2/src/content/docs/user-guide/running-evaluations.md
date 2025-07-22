---
title: Running evaluations
---

This guide covers the fundamental usage patterns of KARMA for medical AI evaluation.

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
  --model-args '{"temperature":0.5,"max_tokens":512,"top_p":0.9}'

# Disable thinking mode (for Qwen)
karma eval --model Qwen/Qwen3-0.6B \
  --model-args '{"enable_thinking":false}'
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

# Refresh cache
karma eval --model Qwen/Qwen3-0.6B --refresh-cache
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
## Common Workflows

### Model Comparison

```bash
# Compare different model sizes
karma eval --model Qwen/Qwen3-0.6B --output qwen_0.6b.json
karma eval --model "Qwen/Qwen3-1.7B" --output qwen_1.7b.json

# Compare different models
karma eval --model Qwen/Qwen3-0.6B --output qwen_results.json
karma eval --model "google/medgemma-4b-it" --output medgemma_results.json
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
  --model-args '{"temperature":0.1}' --output temp_0.1.json

karma eval --model Qwen/Qwen3-0.6B \
  --model-args '{"temperature":0.7}' --output temp_0.7.json

karma eval --model Qwen/Qwen3-0.6B \
  --model-args '{"temperature":1.0}' --output temp_1.0.json
```
