# Getting Started

This guide will help you get up and running with KARMA in just a few minutes.

## Prerequisites

- Python 3.12 or higher
- Git
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## Installation

### Option 1: Using uv (Recommended)
1. Install UV if not present
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Install karma 
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

Let's run a simple evaluation using the Qwen3-0.6B model on the PubMedQA dataset:

```bash
karma eval --model "Qwen/Qwen3-0.6B" \
--datasets openlifescienceai/pubmedqa
```

This command will:
1. Load the Qwen3-0.6B model
2. Run evaluation on the PubMedQA dataset (openlifescienceai/pubmedqa)
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
karma info dataset openlifescienceai/pubmedqa
```

### Run Evaluations

```bash
# Basic evaluation
karma eval --model qwen --model-path "Qwen/Qwen3-0.6B"

# Evaluate specific datasets
karma eval --model qwen --model-path "Qwen/Qwen3-0.6B" --datasets "openlifescienceai/pubmedqa,openlifescienceai/medmcqa"

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
# This is a translation task on the in22conv dataset, which needs the source language and target language.
# This is the provided through the dataset-args.
karma eval --model qwen --model-path "Qwen/Qwen3-0.6B" --datasets "ai4bharat/IN22-Conv" \
  --dataset-args "ai4bharat/IN22-Conv:source_language=en,target_language=hi"

# Custom model parameters
# There is also an option to provide model specific arguemnts. 
# You can see what the supported arguments for the model are through the karma info command

karma info model "Qwen/Qwen3-0.6B"

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

KARMA provides comprehensive evaluation results.

The output is saved in results.json and also in the printed to the console.


```json
>$ karma eval --model qwen --model-path "Qwen/Qwen3-0.6B" \
  --datasets "openlifescienceai/pubmedqa" --batch-size 1 \
  --model-kwargs '{"temperature":0.5, "enable_thinking": false}' --max-samples 3


                                       Evaluation Results                                        
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━┓
┃ Dataset                    ┃ Task Type ┃ Metric      ┃ Score ┃ Samples ┃   Time ┃ Status      ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━┩
│ openlifescienceai/pubmedqa │ mcqa      │ exact_match │ 1.000 │       3 │ 1m 17s │ ✓ Completed │
└────────────────────────────┴───────────┴─────────────┴───────┴─────────┴────────┴─────────────┘


       Evaluation Summary        
 Model       Qwen/Qwen3-0.6B     
 Model Path  Qwen/Qwen3-0.6B     
 Datasets    1/1 (100.0%)        
 Total Time  1m 25s              
 Completed   2025-07-04 11:58:32 
Results saved to results.json

✓ Evaluation completed successfully!
```
The results.json file will look like this.
```json
{
  "openlifescienceai/pubmedqa": {
    "metrics": {
      "exact_match": {
        "score": 1.0,
        "evaluation_time": 76.95517897605896,
        "num_samples": 3
      }
    },
    "task_type": "mcqa",
    "status": "completed",
    "dataset_args": {},
    "evaluation_time": 85.19215798377991
  },
  "_summary": {
    "model": "Qwen/Qwen3-0.6B",
    "model_path": "Qwen/Qwen3-0.6B",
    "total_datasets": 1,
    "successful_datasets": 1,
    "total_evaluation_time": 85.19616675376892,
    "timestamp": "2025-07-04 11:58:32"
  }
}
```

## Next Steps

- **Explore datasets**: Check out our [Dataset Guide](user-guide/basic-usage.md) to learn about available medical datasets
- **Add custom models**: See the [API Reference](api-reference/models.md) for integrating your own models
