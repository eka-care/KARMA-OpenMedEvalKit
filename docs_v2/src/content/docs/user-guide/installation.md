---
title: Installation Guide
---

This guide provides detailed installation instructions for KARMA on different platforms and environments.

## Installation Methods

### Method 1: Using uv (Recommended)

[uv](https://docs.astral.sh/uv/) is the fastest Python package manager and is our recommended installation method.

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/eka-care/KARMA-OpenMedEvalKit.git
cd KARMA-OpenMedEvalKit

# Install KARMA
uv sync

# Verify installation
karma --version
```

### Method 2: Using pip

```bash
# Clone the repository
git clone https://github.com/eka-care/KARMA-OpenMedEvalKit.git
cd KARMA-OpenMedEvalKit

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode
pip install -e .

# Verify installation
karma --version
```

### Method 3: Development Installation

For contributors and developers:

```bash
# Clone the repository
git clone https://github.com/eka-care/KARMA-OpenMedEvalKit.git
cd KARMA-OpenMedEvalKit

# Install with development dependencies
uv install --group dev

# Install with all optional dependencies
uv install --group dev --group docs --group audio
```

## Optional Dependencies

### Audio Support

For audio-related datasets and ASR metrics:

```bash
# With uv
uv install --group audio

# With pip
pip install -e ".[audio]"
```

This includes:
- `jiwer` - Word Error Rate calculations
- `num2words` - Number to word conversion
- `torchaudio` - Audio processing

### Install with all dependencies in developer mode
```bash
# With uv
uv sync --all-extras

# with pip
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
