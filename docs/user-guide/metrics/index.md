# Metrics

Learn how to understand and work with evaluation metrics in KARMA, from interpreting results to creating custom metrics.

## Quick Start

```bash
# List available metrics
karma list metrics

# Run evaluation and check results
karma eval --model qwen --model-path "Qwen/Qwen3-0.6B" \
  --datasets openlifescienceai/pubmedqa --output results.json

# View results
cat results.json
```

## Guide

**[Metrics Overview](overview.md)** - Complete guide to understanding metrics, interpreting results, and creating custom metrics.

## Key Topics Covered

- **Understanding Results** - How to interpret evaluation output
- **Available Metrics** - Built-in metrics for different task types
- **Metric Interpretation** - What each metric means and when to use it
- **Custom Metrics** - Creating domain-specific evaluation metrics
- **Advanced Analysis** - Comparing models and statistical considerations
- **Best Practices** - Guidelines for metric selection and interpretation

## Common Metrics

- **exact_match**: Exact string matching for classification tasks
- **bleu**: Text generation quality for open-ended responses
- **wer/cer**: Speech recognition accuracy metrics
- **accuracy**: Overall classification accuracy

## Next Steps

Once you understand metrics, explore these related topics:

- **[Models](../models/overview.md)** - Work with different model types
- **[Datasets](../datasets/overview.md)** - Understand evaluation datasets
- **[Running Evaluations](../basic-usage/running-evaluations.md)** - Execute comprehensive evaluations