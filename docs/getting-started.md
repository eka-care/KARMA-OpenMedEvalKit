# Getting Started

This guide will help you get up and running with KARMA in just a few minutes.

## Prerequisites

- Python 3.12 or higher
- Git
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## Installation

### Option 1: Using uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/eka-care/KARMA-OpenMedEvalKit.git
cd KARMA-OpenMedEvalKit

# Install with uv
uv install

# Install with documentation dependencies (optional)
uv install --group docs
```

### Option 2: Using pip

```bash
# Clone the repository
git clone https://github.com/eka-care/KARMA-OpenMedEvalKit.git
cd KARMA-OpenMedEvalKit

# Install in editable mode
pip install -e .

# Install with audio support (optional)
pip install -e ".[audio]"
```

## Your First Evaluation

Let's run a simple evaluation using the Qwen model on the PubMedQA dataset:

```bash
karma eval --model qwen --model-path "Qwen/Qwen3-0.6B" --datasets pubmedqa
```

This command will:
1. Load the Qwen3-0.6B model
2. Run evaluation on the PubMedQA dataset
3. Display results with progress tracking
4. Cache results for faster re-runs

## Basic Commands

### List Available Resources

```bash
# List all available models
karma list models

# List all available datasets
karma list datasets

# Get detailed information about a specific model
karma info model qwen

# Get detailed information about a specific dataset
karma info dataset pubmedqa
```

### Run Evaluations

```bash
# Basic evaluation
karma eval --model qwen --model-path "Qwen/Qwen3-0.6B"

# Evaluate specific datasets
karma eval --model qwen --model-path "Qwen/Qwen3-0.6B" --datasets "pubmedqa,medmcqa"

# Save results to file
karma eval --model qwen --model-path "Qwen/Qwen3-0.6B" --output results.json

# Disable caching (for fresh runs)
karma eval --model qwen --model-path "Qwen/Qwen3-0.6B" --no-cache
```

### Advanced Usage

```bash
# Custom batch size
karma eval --model qwen --model-path "Qwen/Qwen3-0.6B" --batch-size 8

# Dataset-specific arguments
karma eval --model qwen --model-path "Qwen/Qwen3-0.6B" --datasets "in22conv" \
  --dataset-args "in22conv:source_language=en,target_language=hi"

# Custom model parameters
karma eval --model qwen --model-path "Qwen/Qwen3-0.6B" \
  --model-kwargs '{"temperature":0.5,"max_tokens":512}'
```

## Configuration

KARMA supports environment-based configuration. Create a `.env` file in your project root:

```bash
# Cache configuration
KARMA_CACHE_TYPE=duckdb
KARMA_CACHE_PATH=./cache.db

# Model configuration
HUGGINGFACE_TOKEN=your_token_here
LOG_LEVEL=INFO

# Optional: DynamoDB configuration for production
# KARMA_CACHE_TYPE=dynamodb
# AWS_REGION=us-east-1
# DYNAMODB_TABLE_NAME=karma-cache
```

## Understanding Results

KARMA provides comprehensive evaluation results:

```json
{
  "model": "qwen",
  "model_path": "Qwen/Qwen3-0.6B",
  "dataset": "pubmedqa",
  "metrics": {
    "exact_match": 0.745,
    "accuracy": 0.745
  },
  "num_examples": 1000,
  "runtime_seconds": 45.2,
  "cache_hit_rate": 0.0
}
```

## Next Steps

- **Explore datasets**: Check out our [Dataset Guide](user-guide/basic-usage.md) to learn about available medical datasets
- **Add custom models**: See the [API Reference](api-reference/models.md) for integrating your own models
- **Advanced features**: Learn about caching, batch processing, and more in [Advanced Usage](user-guide/advanced-usage.md)
- **Contributing**: Help improve KARMA by reading our [Contributing Guide](contributing.md)

## Getting Help

- **Documentation**: Browse the complete [API Reference](api-reference/models.md)
- **Issues**: Report bugs or request features on [GitHub](https://github.com/eka-care/KARMA-OpenMedEvalKit/issues)
- **Examples**: Check out the `examples/` directory for more usage patterns