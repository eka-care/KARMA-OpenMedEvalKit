# KARMA: Knowledge Assessment and Reasoning for Medical Applications

<p align="center">
    <a href="https://github.com/eka-care/KARMA-OpenMedEvalKit" target="_blank">
        <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python Version">
    </a>
    <a href="https://github.com/eka-care/KARMA-OpenMedEvalKit/blob/main/LICENSE" target="_blank">
        <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License">
    </a>
    <a href="https://pepy.tech/project/karma-medeval" target="_blank">
        <img src="https://static.pepy.tech/badge/karma-medeval/month" alt="Downloads">
    </a>
</p>
---

**Documentation**: <https://karma.eka.care>

**Source Code**: <https://github.com/eka-care/KARMA-OpenMedEvalKit>

---

KARMA provides a unified package for evaluating medical AI systems, supporting text, image, and audio-based models. The framework includes support for 12 medical datasets and offers standardized evaluation metrics commonly used in healthcare AI research.

The key features are:

* **Fast**: Very high performance evaluation, capable of processing thousands of medical examples efficiently
* **Easy**: Designed to be easy to use and learn. Less time reading docs, more time evaluating models  
* **Comprehensive**: Support for 12+ medical datasets across multiple modalities (text, images, VQA)
* **Model Agnostic**: Works with any model - Qwen, MedGemma, API providers (OpenAI, AWS Bedrock) or your custom architecture
* **Smart Caching**: Intelligent result caching with DuckDB/DynamoDB backends for faster re-evaluations
* **Standards-based**: Extensible architecture with registry-based auto-discovery of models and datasets

```bash
pip install karma-medeval
```

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Example](#example)
- [Supported Models](#supported-models)
  - [Built-in models](#built-in-models)
  - [Adding Custom Models](#adding-custom-models)
- [Custom Model and Dataset Registration](#custom-model-and-dataset-registration)
  - [Registering a Custom Model](#registering-a-custom-model)
  - [Registering a Custom Dataset](#registering-a-custom-dataset)
  - [Using Your Custom Components](#using-your-custom-components)
  - [Registration Parameters](#registration-parameters)
- [Usage](#usage)
- [Configuration](#configuration)
  - [Caching options](#caching-options)
- [Contributing](#contributing)
  - [Adding New Components](#adding-new-components)
- [License](#license)


## Installation

Install KARMA from PyPI:

```bash
pip install karma-medeval
```

Or install from source:

```bash
# Clone the repository
git clone https://github.com/eka-care/KARMA-OpenMedEvalKit.git
cd KARMA-OpenMedEvalKit

# Install with uv (recommended)
uv sync

# Or install with pip
pip install -e .

# source the environment
source .venv/bin/activate
```

## Example

Evaluate your first medical AI model Using the Example of Qwen3 Model:

```bash
$ karma eval --model "Qwen/Qwen3-0.6B" --datasets openlifescienceai/pubmedqa
```

## Supported Models

KARMA depends on PyTorch and HuggingFace Transformers.

Check supported models through
```bash
$ karma list models
```

### Adding Custom Models

KARMA supports custom model integration through its registry system. See the Contributing section for details on adding new models.

## Custom Model and Dataset Registration

KARMA uses a decorator-based registry system that makes it easy to add your own models and datasets for evaluation.

### Registering a Model

Create a new model by inheriting from `BaseHFModel` and then call the `register_model_meta` method from registry.py with the [`ModelMeta`](https://github.com/eka-care/KARMA-OpenMedEvalKit/blob/8052163b72209aa0ee25d5a6146969213e398cd8/karma/data_models/model_meta.py)

See sample implementation from [qwen.py](https://github.com/eka-care/KARMA-OpenMedEvalKit/blob/8052163b72209aa0ee25d5a6146969213e398cd8/karma/models/qwen.py)
Multiple models from the same family can be imported through this now.

Take any model specific inputs through the `loader_kwargs` in ModelMeta, they have to be set as init parameters to be used.
They are passed as kwargs from the model registry.

```python
from karma.models.base_model_abs import BaseHFModel
from karma.data_models.model_meta import ModelMeta, ModelType, ModalityType
from karma.registries.model_registry import register_model_meta

logger = logging.getLogger(__name__)

class MyCustomModel(BaseHFModel):
    """Custom model implementation."""
    
    def __init__(
        self,
        model_name_or_path: str,
        device: str = "mps",
        max_tokens: int = 32768,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: Optional[int] = None,
        enable_thinking: bool = True,
        **kwargs,
    ):
    super().__init__(
            model_name_or_path=model_name_or_path,
            device=device,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            enable_thinking=enable_thinking,
            **kwargs,
        )
      
    ...
  
my_custom_model = ModelMeta(
    name="Qwen/Qwen3-1.7B",
    description="QWEN model",
    loader_class="karma.models.custom.MyCustomModel",
    loader_kwargs={
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.9,
        "enable_thinking": True,
        "max_tokens": 256,
    },
    revision=None,
    reference=None,
    model_type=ModelType.TEXT_GENERATION,
    modalities=[ModalityType.TEXT],
    n_parameters=None,
    memory_usage_mb=None,
    max_tokens=None,
    embed_dim=None,
    framework=["PyTorch", "Transformers"],
)
register_model_meta(my_custom_model)
```

### Registering a Custom Dataset

Create a new dataset by inheriting from `BaseMultimodalDataset` and using the `@register_dataset` decorator:

```python
from karma.eval_datasets.base_dataset import BaseMultimodalDataset
from karma.registries.dataset_registry import register_dataset

@register_dataset(
    "my_custom_dataset", 
    metrics=["exact_match", "accuracy"], 
    task_type="mcqa",
    required_args=["domain"],
    optional_args=["split", "subset"],
    default_args={"split": "test"}
)
class MyCustomDataset(BaseMultimodalDataset):
    """Custom dataset implementation."""
    
    def __init__(self, domain: str, split: str = "test", subset: str = None, **kwargs):
        self.domain = domain
        self.split = split
        self.subset = subset
        super().__init__(**kwargs)
    
```

### Using Your Custom Components

After defining your custom model and dataset, use them with the CLI:

```bash
# Use your custom model and dataset
karma eval --model my_custom_model --model-path "path/to/model" \
  --datasets "my_custom_dataset" \
  --dataset-args "my_custom_dataset:domain=medical"
  --model-kwargs '{"temperature":0.5}'
```

### Registration Parameters

**Model Registration:**
- `name`: Unique identifier for your model

**Dataset Registration:**
- `name`: Unique identifier for your dataset
- `metrics`: List of applicable metrics (e.g., `["exact_match", "bleu", "accuracy"]`)
- `task_type`: Type of task (`"mcqa"`, `"vqa"`, `"translation"`, `"qa"`)
- `required_args`: Arguments that must be provided when creating the dataset
- `optional_args`: Arguments that can be provided but have defaults
- `default_args`: Default values for arguments

## Usage

List available resources:

```bash
karma list models
karma list datasets
```

Basic evaluation:

```bash
karma eval --model qwen --model-path "Qwen/Qwen3-0.6B"
```

Evaluate specific datasets:

```bash
karma eval --model qwen --model-path "Qwen/Qwen3-0.6B" --datasets "pubmedqa,medmcqa"
```

With dataset-specific arguments:

```bash
karma eval --model qwen --model-path "Qwen/Qwen3-0.6B" --datasets "in22conv" \
  --dataset-args "in22conv:source_language=en,target_language=hi"
```

Advanced options:

```bash
karma eval --model qwen --model-path "Qwen/Qwen3-0.6B" \
  --datasets "pubmedqa" --batch-size 16 --output results.json --no-cache
```

## Configuration

KARMA supports environment-based configuration. Create a `.env` file:

```bash
# Cache configuration  
KARMA_CACHE_TYPE=duckdb
KARMA_CACHE_PATH=./cache.db

# Model configuration
HUGGINGFACE_TOKEN=your_token
LOG_LEVEL=INFO
```

### Caching options

* **DuckDB** (default) - for local development
* **DynamoDB** - for production environments

Enable or disable caching:

```bash
karma eval --cache      # Enable (default)
karma eval --no-cache   # Disable
```

## Contributing

We welcome contributions to KARMA!

### Adding New Components

KARMA uses a registry-based architecture that makes it easy to add:
* **New datasets** - Extend BaseMultimodalDataset and register with @register_dataset
* **New models** - Extend BaseLLM and register with @register_model  
* **New metrics** - Implement custom evaluation metrics
* **New processors** - Add data preprocessing capabilities

See the existing implementations in `karma/eval_datasets/` and `karma/models/` for examples.

## License

This project is licensed under the terms of the MIT license.
