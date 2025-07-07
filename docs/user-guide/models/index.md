# Models Guide

Learn how to work with models in KARMA, from using built-in models to creating your own custom implementations.

## Getting Started

If you're new to models in KARMA, start with these guides in order:

1. **[Built-in Models](built-in-models.md)** - Explore pre-configured models ready to use
2. **[Model Configuration](model-configuration.md)** - Customize model parameters and behavior
3. **[Custom Models](custom-models.md)** - Create and integrate your own models
4. **[Performance Optimization](performance-optimization.md)** - Optimize models for your hardware

## Quick Reference

### Essential Commands

```bash
# List available models
karma list models

# Get model information
karma info model qwen

# Use a model
karma eval --model qwen --model-path "Qwen/Qwen3-0.6B" \
  --datasets openlifescienceai/pubmedqa
```

### Common Workflows

- **First-time users**: Start with [Built-in Models](built-in-models.md)
- **Need customization**: Check [Model Configuration](model-configuration.md)
- **Performance issues**: See [Performance Optimization](performance-optimization.md)
- **Advanced integration**: Read [Custom Models](custom-models.md)

## Next Steps

Once you're comfortable with models, explore these related topics:

- **[Datasets](../datasets/overview.md)** - Work with evaluation datasets
- **[Metrics](../metrics/overview.md)** - Understand evaluation metrics
- **[Running Evaluations](../basic-usage/running-evaluations.md)** - Execute model evaluations