---
title: Installation Guide
---

This guide provides detailed installation instructions for KARMA on different platforms and environments.

## Installation Methods

### Method 1: Using PyPI

```bash
# Install from PyPI
pip install karma-medeval

# Verify installation
karma --version
```

### Method 2: Development Installation

For contributors and developers:

```bash
# Clone the repository
git clone https://github.com/eka-care/KARMA-OpenMedEvalKit.git
cd KARMA-OpenMedEvalKit

# Install with development dependencies
uv install --group dev

# Install with all optional dependencies
uv install --group dev --group audio
```

## Optional Dependencies

### Audio Support

For audio-related datasets and ASR metrics:

```bash
# From PyPI
pip install karma-medeval[audio]

# From source
pip install -e ".[audio]"
```

This includes:
- `jiwer` - Word Error Rate calculations
- `num2words` - Number to word conversion
- `torchaudio` - Audio processing

### Install with all dependencies
```bash
# From PyPI
pip install karma-medeval[all]

# From source
pip install -e ".[all]"
```

## Environment Configuration

Create a `.env` file in your project root:

```bash
# Required: HuggingFace token for model downloads
HUGGINGFACE_TOKEN=your_token_here

# Cache configuration
KARMA_CACHE_TYPE=duckdb
KARMA_CACHE_PATH=./cache.db

# Logging
LOG_LEVEL=INFO

# Optional: OpenAI API key (for certain metrics)
OPENAI_API_KEY=your_openai_key

# Optional: DynamoDB configuration (for production)
# KARMA_CACHE_TYPE=dynamodb
# AWS_REGION=us-east-1
# DYNAMODB_TABLE_NAME=karma-cache
```

#### HuggingFace Token
To access gated models or datasets, set this environment variable with your Huggingface token.

You can see the guide to create tokens [here](https://huggingface.co/docs/hub/en/security-tokens)
```bash
# Login to HuggingFace
huggingface-cli login

# Or set environment variable
export HUGGINGFACE_TOKEN=your_token_here
```
