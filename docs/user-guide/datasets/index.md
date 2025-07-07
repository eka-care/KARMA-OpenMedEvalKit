# Datasets

Learn how to work with datasets in KARMA, from using built-in datasets to creating your own custom implementations.

## Quick Start

```bash
# List available datasets
karma list datasets

# Get dataset information
karma info dataset openlifescienceai/pubmedqa

# Use a dataset
karma eval --model qwen --model-path "Qwen/Qwen3-0.6B" \
  --datasets openlifescienceai/pubmedqa
```

## Guide

**[Datasets Overview](overview.md)** - Complete guide to working with datasets, including built-in datasets, configuration, and creating custom datasets.

## Key Topics Covered

- **Built-in Datasets** - Pre-configured medical datasets ready to use
- **Dataset Configuration** - Customizing dataset arguments and behavior
- **Custom Dataset Integration** - Creating and registering your own datasets
- **Data Formatting** - Understanding dataset structure and requirements
- **Best Practices** - Tips for dataset selection and development

## Available Dataset Types

- **Text-based** - Question answering and classification tasks
- **Vision-language** - Medical image analysis and VQA
- **Audio** - Speech recognition and transcription
- **Translation** - Multi-language medical text translation

## Next Steps

Once you understand datasets, explore these related topics:

- **[Models](../models/overview.md)** - Work with different model types
- **[Metrics](../metrics/overview.md)** - Understand evaluation metrics
- **[Running Evaluations](../basic-usage/running-evaluations.md)** - Execute comprehensive evaluations