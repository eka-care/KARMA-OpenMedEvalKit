# Running Evaluations

This guide covers how to run model evaluations using the KARMA CLI.

## Simple Evaluation

The most basic evaluation requires specifying a model and model path:

```bash
karma eval --model "Qwen/Qwen3-0.6B"
```

This will:
- Load the Qwen3-0.6B model
- Run evaluation on all supported datasets
- Display results with progress tracking
- Cache results for faster re-runs

## Evaluate Specific Datasets

```bash
# Single dataset
karma eval --model "Qwen/Qwen3-0.6B" --datasets openlifescienceai/pubmedqa

# Multiple datasets
karma eval --model "Qwen/Qwen3-0.6B" \
  --datasets "openlifescienceai/pubmedqa,openlifescienceai/medmcqa,openlifescienceai/medqa"
```

## Common Workflows

### Model Comparison

```bash
# Compare different model sizes
karma eval --model "Qwen/Qwen3-0.6B" --output qwen_0.6b.json
karma eval --model "Qwen/Qwen3-1.7B" --output qwen_1.7b.json

# Compare different models
karma eval --model "Qwen/Qwen3-0.6B" --output qwen_results.json
karma eval --model medgemma --model-path "google/medgemma-4b-it" --output medgemma_results.json
```

### Dataset-Specific Evaluation

```bash
# Focus on specific medical domains
karma eval --model "Qwen/Qwen3-0.6B" \
  --datasets "openlifescienceai/pubmedqa,openlifescienceai/medmcqa,openlifescienceai/medqa"  # Text-based QA

karma eval --model "Qwen/Qwen3-0.6B" \
  --datasets "mdwiratathya/SLAKE-vqa-english,flaviagiammarino/vqa-rad"  # Vision-language tasks
```

### Parameter Tuning

```bash
# Test different temperature settings
karma eval --model "Qwen/Qwen3-0.6B" \
  --model-kwargs '{"temperature":0.1}' --output temp_0.1.json

karma eval --model "Qwen/Qwen3-0.6B" \
  --model-kwargs '{"temperature":0.7}' --output temp_0.7.json

karma eval --model "Qwen/Qwen3-0.6B" \
  --model-kwargs '{"temperature":1.0}' --output temp_1.0.json
```
### Evaluation With Additional Args
```bash
# Test with Additional Args
karma eval --model "Qwen/Qwen3-0.6B" --datasets ekacare/MedMCQA-Indic --dataset-args "ekacare/MedMCQA-Indic:subset=as"
```

## Next Steps

- **Learn about models**: Check out the [Models Guide](../models/overview.md)
- **Configure datasets**: Read the [Datasets Guide](../datasets/datasets_overview.md)
- **Optimize performance**: See [Performance Optimization](performance-optimization.md)
- **Understand results**: Read the [Metrics Guide](../metrics/metrics_overview.md)