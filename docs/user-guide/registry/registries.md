# Registry System Deep Dive

Registries are the backbone of KARMA's component discovery and management system. They provide a sophisticated, decorator-based mechanism for automatically discovering and utilizing core components including models, datasets, metrics, and processors. This system is designed for high performance with caching, parallel discovery, and thread-safety.

## Architecture Overview

### Core Components

The registry system consists of several key components working together:

1. **Registry Manager** (`karma/registries/registry_manager.py`) - Orchestrates discovery across all registries
2. **Individual Registries** - Specialized registries for each component type
3. **CLI Integration** - Seamless command-line interface integration

## Component Registration

### Models

Models are registered using `ModelMeta` objects that provide comprehensive metadata. The model registry supports multi-modal models and various frameworks.

**Key Features:**

- **ModelMeta System**: Pydantic-based configuration with type validation
- **Multi-modal Support**: Handles text, audio, image, video modalities
- **Type Classification**: Categorizes models by type (text_generation, audio_recognition, etc.)
- **Loader Configuration**: Flexible model loading with parameter overrides

**Registration Example:**
```python
from karma.registries.model_registry import register_model_meta, ModelMeta
from karma.core.model_meta import ModelType, ModalityType

# Define model metadata
QwenModel = ModelMeta(
    name="Qwen/Qwen3-0.6B",
    description="QWEN model for text generation",
    loader_class="karma.models.qwen.QwenThinkingLLM",
    loader_kwargs={
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.9,
        "enable_thinking": True,
        "max_tokens": 32768,
    },
    model_type=ModelType.TEXT_GENERATION,
    modalities=[ModalityType.TEXT],
    framework=["PyTorch", "Transformers"],
)

# Register the model
register_model_meta(QwenModel)
```

### Datasets

Datasets are registered using decorators that specify comprehensive metadata including supported metrics and task types.

**Key Features:**

- **Metric Association**: Links datasets to supported metrics
- **Task Type Classification**: Categorizes by task (mcqa, vqa, translation, etc.)
- **Argument Validation**: Validates required/optional arguments
- **HuggingFace Integration**: Supports commit hashes and splits

**Registration Example:**
```python
from karma.registries.dataset_registry import register_dataset
from karma.datasets.base_multimodal_dataset import BaseMultimodalDataset

@register_dataset(
    "openlifescienceai/medqa",
    commit_hash="153e61cdd129eb79d3c27f82cdf3bc5e018c11b0",
    split="test",
    metrics=["exact_match"],
    task_type="mcqa",
    required_args=["num_choices"],
    optional_args=["language", "subset"],
    default_args={"num_choices": 4, "language": "en"}
)
class MedQADataset(BaseMultimodalDataset):
    """Medical Question Answering dataset."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Dataset-specific initialization
        
    def load_data(self):
        # Implementation for loading dataset
        pass
```
See more at **[Datasets](/user-guide/datasets/datasets_overview/)**

### Metrics

The metrics registry supports both KARMA-native metrics and HuggingFace Evaluate metrics with automatic fallback.

**Key Features:**

- **Dual Support**: Native metrics and HuggingFace Evaluate library fallback
- **Argument Validation**: Validates metric parameters
- **Dynamic Loading**: Lazy loading of HuggingFace metrics

**Registration Example:**
```python
from karma.registries.metrics_registry import register_metric
from karma.metrics.hf_metric import HfMetric

@register_metric(
    "exact_match",
    optional_args=["ignore_case", "normalize_text"],
    default_args={"ignore_case": True, "normalize_text": False}
)
class ExactMatchMetric(HfMetric):
    """Exact match metric with case sensitivity options."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def compute(self, predictions, references):
        # Implementation for exact match computation
        pass
```

### Processors

Processors handle text and data transformation with flexible argument validation.

**Key Features:**
- **Text Processing**: Supports transliteration, normalization, etc.
- **Argument Validation**: Validates processor parameters
- **Modular Design**: Easy to extend with new processors

**Registration Example:**
```python
from karma.registries.processor_registry import register_processor
from karma.processors.base_processor import BaseProcessor

@register_processor(
    "devnagari_transliterator",
    optional_args=["normalize", "fallback_scheme"],
    default_args={"normalize": True, "fallback_scheme": None}
)
class DevanagariTransliterator(BaseProcessor):
    """Transliterator for Devanagari script."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def process(self, text):
        # Implementation for transliteration
        pass
```

## CLI Integration

The registry system seamlessly integrates with the CLI for component discovery and listing.

### Discovery Commands

```bash
# List all models
karma list models

# List datasets with filtering
karma list datasets --task-type mcqa --metric accuracy

# List all metrics
karma list metrics

# List all processors
karma list processors

# List all components
karma list all
```

### Error Handling

The registry system provides robust error handling:

- **Graceful Degradation**: Individual registry failures don't break the system
- **Fallback Mechanisms**: HuggingFace metrics as fallback for missing metrics
- **Validation**: Comprehensive argument validation with helpful error messages
- **Logging**: Detailed logging for debugging and monitoring

## Best Practices

1. **Use Descriptive Names**: Choose clear, descriptive names for your components
2. **Provide Comprehensive Metadata**: Include detailed descriptions and argument specifications
3. **Validate Arguments**: Implement proper argument validation in your components
4. **Follow Naming Conventions**: Use consistent naming patterns across your components
5. **Document Dependencies**: Clearly specify framework and library requirements
6. **Test Registration**: Verify your components are properly registered and discoverable

## File Structure

The registry system is organized across several key files:

```
karma/registries/
├── registry_manager.py      # Central registry coordination
├── model_registry.py        # Model registration and discovery
├── dataset_registry.py      # Dataset registration and discovery
├── metrics_registry.py      # Metrics registration and discovery
├── processor_registry.py    # Processor registration and discovery
└── cache_manager.py         # Caching system implementation
```

This registry system provides a highly scalable, performant, and user-friendly way to manage and discover components in the KARMA framework, with particular emphasis on medical AI evaluation tasks.
