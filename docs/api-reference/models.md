# Models API Reference

This section documents KARMA's model system, including base classes, built-in models, and integration patterns.

## Base Classes

### BaseHFModel

The foundation for all HuggingFace-based models in KARMA.

::: karma.models.base_model_abs.BaseHFModel
    options:
      show_source: false
      show_root_heading: true
      show_category_heading: true
      members_order: source
      filters: ["!^_"]
      merge_init_into_class: true

## Built-in Models

### QwenThinkingLLM

Qwen language model with specialized thinking capabilities.

::: karma.models.qwen.QwenThinkingLLM
    options:
      show_source: false
      show_root_heading: true
      show_category_heading: true
      members_order: source
      filters: ["!^_"]
      merge_init_into_class: true

### MedGemmaModel

Google's MedGemma model for medical applications.

::: karma.models.medgemma.MedGemmaModel
    options:
      show_source: false
      show_root_heading: true
      show_category_heading: true
      members_order: source
      filters: ["!^_"]
      merge_init_into_class: true

### IndicConformerASR

Indic language ASR model using Conformer architecture.

::: karma.models.indic_conformer.IndicConformerASR
    options:
      show_source: false
      show_root_heading: true
      show_category_heading: true
      members_order: source
      filters: ["!^_"]
      merge_init_into_class: true

## Model Configuration

### ModelMeta

Comprehensive model metadata configuration with Pydantic validation.

::: karma.data_models.model_meta.ModelMeta
    options:
      show_source: false
      show_root_heading: true
      show_category_heading: true
      members_order: source
      filters: ["!^_"]
      merge_init_into_class: true

### ModelType

Enumeration of supported model types.

::: karma.data_models.model_meta.ModelType
    options:
      show_source: false
      show_root_heading: true
      show_category_heading: true

### ModalityType

Enumeration of supported modalities.

::: karma.data_models.model_meta.ModalityType
    options:
      show_source: false
      show_root_heading: true
      show_category_heading: true

## Usage Examples

### Basic Model Usage

```python
from karma.models.qwen import QwenThinkingLLM

# Initialize model
model = QwenThinkingLLM(
    model_name_or_path="Qwen/Qwen3-0.6B",
    temperature=0.7,
    max_tokens=512
)

# Load model
model.load_model()

# Generate response
response = model.run("What is the treatment for diabetes?")
print(response)
```

### Custom Model Integration

```python
from karma.models.base_model_abs import BaseHFModel
from karma.data_models.model_meta import ModelMeta, ModelType, ModalityType
from karma.registries.model_registry import register_model_meta

class CustomMedicalModel(BaseHFModel):
    """Custom medical AI model."""
    
    def __init__(self, model_name_or_path: str, **kwargs):
        super().__init__(model_name_or_path=model_name_or_path, **kwargs)
    
    def preprocess(self, prompt: str, **kwargs) -> str:
        return f"Medical Question: {prompt}\nAnswer:"
    
    def postprocess(self, response: str, **kwargs) -> str:
        return response.strip()

# Register custom model
custom_model_meta = ModelMeta(
    name="custom_medical_model",
    description="Custom medical AI model",
    loader_class="path.to.CustomMedicalModel",
    loader_kwargs={"temperature": 0.7},
    model_type=ModelType.TEXT_GENERATION,
    modalities=[ModalityType.TEXT],
    framework=["PyTorch", "Transformers"],
)
register_model_meta(custom_model_meta)
```

### Advanced Configuration

```python
from karma.data_models.model_meta import ModelMeta, ModelType, ModalityType

# Complex model configuration
model_meta = ModelMeta(
    name="advanced_medical_model",
    description="Advanced multimodal medical model",
    loader_class="karma.models.custom.AdvancedMedicalModel",
    loader_kwargs={
        "temperature": 0.7,
        "max_tokens": 1024,
        "top_p": 0.9,
        "enable_thinking": True,
        "use_system_prompt": True,
    },
    model_type=ModelType.MULTIMODAL,
    modalities=[ModalityType.TEXT, ModalityType.IMAGE],
    n_parameters=7_000_000_000,
    memory_usage_mb=14_000,
    max_tokens=8192,
    embed_dim=4096,
    framework=["PyTorch", "Transformers"],
)
```

## Model Types and Capabilities

### Text Generation Models

- **Primary Use**: Medical question answering, text generation
- **Supported Modalities**: Text
- **Examples**: Qwen, MedGemma

### Multimodal Models

- **Primary Use**: Medical image analysis, VQA
- **Supported Modalities**: Text, Image
- **Examples**: Custom vision-language models

### Audio Models

- **Primary Use**: Medical speech recognition, audio processing
- **Supported Modalities**: Audio, Text
- **Examples**: IndicConformer for medical ASR

### Embedding Models

- **Primary Use**: Medical document similarity, retrieval
- **Supported Modalities**: Text
- **Examples**: Medical embedding models

## Best Practices

### Model Initialization

```python
# Always specify explicit parameters
model = QwenThinkingLLM(
    model_name_or_path="Qwen/Qwen3-0.6B",
    device="cuda" if torch.cuda.is_available() else "cpu",
    temperature=0.7,
    max_tokens=512,
    top_p=0.9
)

# Load model explicitly
model.load_model()

# Verify model is loaded
assert model.model is not None
```

### Error Handling

```python
from karma.models.base_model_abs import BaseHFModel
import logging

logger = logging.getLogger(__name__)

try:
    model = QwenThinkingLLM(
        model_name_or_path="Qwen/Qwen3-0.6B"
    )
    model.load_model()
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    # Fallback to smaller model
    model = QwenThinkingLLM(
        model_name_or_path="Qwen/Qwen3-0.1B"
    )
```

### Memory Management

```python
import torch
import gc

# Clear memory after model use
def clear_model_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

# Use in evaluation loops
for model_path in model_paths:
    model = QwenThinkingLLM(model_name_or_path=model_path)
    # ... use model
    del model
    clear_model_memory()
```

## See Also

- [Registries API](registries.md) - Model registry and discovery
- [Datasets API](datasets.md) - Dataset integration
- [Metrics API](metrics.md) - Evaluation metrics
- [CLI Reference](cli.md) - Command-line interface