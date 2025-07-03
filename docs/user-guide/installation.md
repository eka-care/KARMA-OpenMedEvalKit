# Installation Guide

This guide provides detailed installation instructions for KARMA on different platforms and environments.

## System Requirements

- **Python**: 3.12 or higher
- **Operating System**: Linux, macOS, or Windows
- **Memory**: 8GB RAM minimum, 16GB recommended
- **Storage**: 10GB free space for models and cache
- **GPU**: Optional but recommended for faster inference

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
uv install

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

### Documentation Dependencies

For building documentation locally:

```bash
# With uv
uv install --group docs

# With pip
pip install mkdocs mkdocs-material mkdocstrings[python] mkdocs-llmstxt
```

## Platform-Specific Instructions

### macOS

```bash
# Install Homebrew if you haven't already
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python 3.12
brew install python@3.12

# Install uv
brew install uv

# Follow standard installation
git clone https://github.com/eka-care/KARMA-OpenMedEvalKit.git
cd KARMA-OpenMedEvalKit
uv install
```

### Linux (Ubuntu/Debian)

```bash
# Update package list
sudo apt update

# Install Python 3.12
sudo apt install python3.12 python3.12-venv python3.12-dev

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Follow standard installation
git clone https://github.com/eka-care/KARMA-OpenMedEvalKit.git
cd KARMA-OpenMedEvalKit
uv install
```

### Windows

```powershell
# Install Python 3.12 from python.org or Microsoft Store

# Install uv
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Follow standard installation
git clone https://github.com/eka-care/KARMA-OpenMedEvalKit.git
cd KARMA-OpenMedEvalKit
uv install
```

## GPU Support

KARMA automatically detects and uses available GPUs. For optimal performance:

### NVIDIA GPUs

```bash
# Install CUDA-enabled PyTorch (if not already installed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Apple Silicon (M1/M2/M3)

```bash
# PyTorch with Metal Performance Shaders support is included by default
# No additional installation needed
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

## Verification

Test your installation:

```bash
# Check version
karma --version

# List available models
karma list models

# List available datasets
karma list datasets

# Run a quick test evaluation
karma eval --model qwen --model-path "Qwen/Qwen3-0.6B" --datasets pubmedqa --batch-size 1
```

## Troubleshooting

### Common Issues

#### ImportError: No module named 'karma'

```bash
# Make sure you're in the correct directory
cd KARMA-OpenMedEvalKit

# Reinstall in editable mode
pip install -e .
```

#### CUDA/GPU Issues

```bash
# Check PyTorch GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# For Apple Silicon
python -c "import torch; print(torch.backends.mps.is_available())"
```

#### HuggingFace Token Issues

```bash
# Login to HuggingFace
huggingface-cli login

# Or set environment variable
export HUGGINGFACE_TOKEN=your_token_here
```

#### Permission Errors

```bash
# On Linux/macOS, ensure proper permissions
chmod +x karma

# If using pip, try with --user flag
pip install --user -e .
```

### Getting Help

If you encounter issues:

1. Check the [FAQ](https://github.com/eka-care/KARMA-OpenMedEvalKit/wiki/FAQ)
2. Search existing [Issues](https://github.com/eka-care/KARMA-OpenMedEvalKit/issues)
3. Create a new issue with:
   - Your operating system
   - Python version
   - Installation method used
   - Full error message

## Next Steps

- **First time user?** Continue to [Basic Usage](basic-usage.md)
- **Need advanced features?** Check out [Advanced Usage](advanced-usage.md)
- **Want to contribute?** See our [Contributing Guide](../contributing.md)