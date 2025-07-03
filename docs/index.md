# KARMA: Knowledge Assessment and Reasoning for Medical Applications

<p align="center">
    <em>High-performance, easy to learn, fast to benchmark, ready for production</em>
</p>

---

**Documentation**: [https://karma-docs.example.com](https://karma-docs.example.com)

**Source Code**: [https://github.com/eka-care/KARMA-OpenMedEvalKit](https://github.com/eka-care/KARMA-OpenMedEvalKit)

---

KARMA is a comprehensive, high-performance evaluation framework for building medical AI benchmarks with Python 3.12+ based on standard PyTorch models.

## Key Features

* **Fast**: Very high performance evaluation, capable of processing thousands of medical examples efficiently
* **Easy**: Designed to be easy to use and learn. Less time reading docs, more time evaluating models  
* **Comprehensive**: Support for 12+ medical datasets across multiple modalities (text, images, VQA)
* **Model Agnostic**: Works with any model - Qwen, MedGemma, or your custom architecture
* **Smart Caching**: Intelligent result caching with DuckDB/DynamoDB backends for faster re-evaluations
* **Production Ready**: Built-in CLI, progress tracking, and formatted outputs for production workflows
* **Standards-based**: Extensible architecture with registry-based auto-discovery of models and datasets

## Quick Start

Get started with KARMA in minutes:

```bash
# Clone the repository
git clone https://github.com/eka-care/KARMA-OpenMedEvalKit.git
cd KARMA-OpenMedEvalKit

# Install with uv (recommended)
uv install

# Run your first evaluation
karma eval --model qwen --model-path "Qwen/Qwen3-0.6B" --datasets pubmedqa
```

## Supported Models

### Built-in Models
- **qwen** - Qwen3 series models with thinking capabilities
- **medgemma** - Google's medgemma-4b-it model

### Custom Models
KARMA supports custom model integration through its registry system. See our [API Reference](api-reference/models.md) for details on adding new models.

## Architecture Overview

KARMA is built around four core components:

1. **[Models](api-reference/models.md)** - Unified interface for medical AI models
2. **[Datasets](api-reference/datasets.md)** - Standardized medical evaluation datasets
3. **[Metrics](api-reference/metrics.md)** - Comprehensive evaluation metrics
4. **[Registries](api-reference/registries.md)** - Auto-discovery system for components

## What's Next?

- **New to KARMA?** Start with our [Getting Started](getting-started.md) guide
- **Need help with installation?** Check the [Installation Guide](user-guide/installation.md)
- **Want to add custom models?** See the [API Reference](api-reference/models.md)
- **Contributing?** Read our [Contributing Guide](contributing.md)

## License

This project is licensed under the terms of the MIT license.