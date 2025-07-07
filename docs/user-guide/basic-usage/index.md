# Basic Usage Guide

This section covers the fundamental usage patterns of KARMA for medical AI evaluation.

## Getting Started

If you're new to KARMA, start with these guides in order:

1. **[CLI Basics](cli-basics.md)** - Learn the basic CLI commands and structure
2. **[Running Evaluations](running-evaluations.md)** - Run your first model evaluations
3. **[Performance Optimization](performance-optimization.md)** - Optimize evaluations for your hardware

## Quick Reference

### Essential Commands

```bash
# List available resources
karma list models
karma list datasets

# Get information
karma info model qwen
karma info dataset openlifescienceai/pubmedqa

# Run evaluation
karma eval --model qwen --model-path "Qwen/Qwen3-0.6B" \
  --datasets openlifescienceai/pubmedqa
```

### Common Workflows

- **First-time users**: Start with [CLI Basics](cli-basics.md)
- **Running evaluations**: Go to [Running Evaluations](running-evaluations.md)
- **Performance issues**: Check [Performance Optimization](performance-optimization.md)

## Next Steps

Once you're comfortable with the basics, explore these advanced topics:

- **[Models](../models/overview.md)** - Work with different models and create custom ones
- **[Datasets](../datasets/overview.md)** - Understand datasets and add your own
- **[Processors](../processors/overview.md)** - Use and create text processors
- **[Metrics](../metrics/overview.md)** - Understand evaluation metrics
- **[Configuration](../configuration/environment-setup.md)** - Advanced configuration options