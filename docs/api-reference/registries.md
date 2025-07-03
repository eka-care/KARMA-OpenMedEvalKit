# Registries API Reference

This section documents KARMA's registry system for automatic discovery and management of models, datasets, metrics, and processors.

## Model Registry

### ModelRegistry

Enhanced model registry supporting ModelMeta configurations with automatic model discovery.

::: karma.registries.model_registry.ModelRegistry
    options:
      show_source: false
      show_root_heading: true
      show_category_heading: true
      members_order: source
      filters: ["!^_"]

### Registration Functions

#### register_model_meta

Register a model using ModelMeta configuration.

::: karma.registries.model_registry.register_model_meta
    options:
      show_source: false
      show_root_heading: true

#### get_model

Retrieve and instantiate a registered model.

::: karma.registries.model_registry.get_model
    options:
      show_source: false
      show_root_heading: true


## Dataset Registry

### DatasetRegistry

Decorator-based registry for datasets with metrics and metadata validation.

::: karma.registries.dataset_registry.DatasetRegistry
    options:
      show_source: false
      show_root_heading: true
      show_category_heading: true
      members_order: source
      filters: ["!^_"]

### Registration Functions

#### register_dataset

Decorator for registering datasets.

::: karma.registries.dataset_registry.register_dataset
    options:
      show_source: false
      show_root_heading: true



## Metric Registry

### MetricRegistry

Decorator-based metric registry for automatic metric discovery.

::: karma.registries.metrics_registry.MetricRegistry
    options:
      show_source: false
      show_root_heading: true
      show_category_heading: true
      members_order: source
      filters: ["!^_"]

### Registration Functions

#### register_metric

Decorator for registering metrics.

::: karma.registries.metrics_registry.register_metric
    options:
      show_source: false
      show_root_heading: true



## Processor Registry

### ProcessorRegistry

Registry for text processors and preprocessing components.

::: karma.registries.processor_registry.ProcessorRegistry
    options:
      show_source: false
      show_root_heading: true
      show_category_heading: true
      members_order: source
      filters: ["!^_"]

### Registration Functions

#### register_processor

Decorator for registering processors.

::: karma.registries.processor_registry.register_processor
    options:
      show_source: false
      show_root_heading: true


## Usage Examples

### Model Registration

```python
from karma.models.base_model_abs import BaseHFModel
from karma.data_models.model_meta import ModelMeta, ModelType, ModalityType
from karma.registries.model_registry import register_model_meta

class CustomMedicalModel(BaseHFModel):
    """Custom medical AI model."""
    
    def __init__(self, model_name_or_path: str, **kwargs):
        super().__init__(model_name_or_path=model_name_or_path, **kwargs)

# Create model metadata
custom_model_meta = ModelMeta(
    name="custom_medical_model",
    description="Custom medical AI model for specialized tasks",
    loader_class="myproject.models.CustomMedicalModel",
    loader_kwargs={
        "temperature": 0.7,
        "max_tokens": 512,
        "top_p": 0.9,
    },
    revision=None,
    reference="https://example.com/model-paper",
    model_type=ModelType.TEXT_GENERATION,
    modalities=[ModalityType.TEXT],
    n_parameters=7_000_000_000,
    memory_usage_mb=14_000,
    max_tokens=8192,
    embed_dim=4096,
    framework=["PyTorch", "Transformers"],
)

# Register the model
register_model_meta(custom_model_meta)

# Use the registered model
from karma.registries.model_registry import get_model

model = get_model(
    model_name="custom_medical_model",
    model_path="path/to/model/weights"
)
```

### Dataset Registration

```python
from karma.eval_datasets.base_dataset import BaseMultimodalDataset
from karma.registries.dataset_registry import register_dataset

@register_dataset(
    name="my_medical_dataset",
    metrics=["exact_match", "accuracy", "bleu"],
    task_type="mcqa",
    required_args=["domain", "split"],
    optional_args=["subset", "language"],
    default_args={"split": "test", "language": "en"}
)
class MyMedicalDataset(BaseMultimodalDataset):
    """Custom medical dataset for evaluation."""
    
    def __init__(self, domain: str, split: str = "test", 
                 subset: str = None, language: str = "en", **kwargs):
        self.domain = domain
        self.split = split
        self.subset = subset
        self.language = language
        super().__init__(**kwargs)
    
    def load_data(self):
        # Custom data loading logic
        return self._load_medical_data()
    
    def format_item(self, item):
        return {
            "prompt": f"Domain: {self.domain}\n{item['question']}",
            "ground_truth": item["answer"],
            "options": item.get("choices", [])
        }

# Use the registered dataset
from karma.registries.dataset_registry import create_dataset

dataset = create_dataset(
    dataset_name="my_medical_dataset",
    domain="cardiology",
    split="validation"
)
```

### Metric Registration

```python
from karma.metrics.base_metric_abs import BaseMetric
from karma.registries.metrics_registry import register_metric

@register_metric("medical_accuracy")
class MedicalAccuracyMetric(BaseMetric):
    """Medical-specific accuracy metric."""
    
    def __init__(self, medical_term_weight=1.5):
        self.medical_term_weight = medical_term_weight
        self.medical_terms = self._load_medical_terms()
    
    def evaluate(self, predictions, references, **kwargs):
        """Evaluate with medical term weighting."""
        total_score = 0
        total_weight = 0
        
        for pred, ref in zip(predictions, references):
            weight = self._calculate_weight(ref)
            score = 1.0 if pred.lower().strip() == ref.lower().strip() else 0.0
            
            total_score += score * weight
            total_weight += weight
        
        accuracy = total_score / total_weight if total_weight > 0 else 0.0
        
        return {
            "medical_accuracy": accuracy,
            "total_weight": total_weight
        }
    
    def _calculate_weight(self, text):
        """Calculate weight based on medical term presence."""
        weight = 1.0
        for term in self.medical_terms:
            if term in text.lower():
                weight = self.medical_term_weight
                break
        return weight
    
    def _load_medical_terms(self):
        """Load medical terminology."""
        return ["diabetes", "hypertension", "surgery", "medication"]

# Use the registered metric
from karma.registries.metrics_registry import get_metric_class

MetricClass = get_metric_class("medical_accuracy")
metric = MetricClass(medical_term_weight=2.0)
```

### Processor Registration

```python
from karma.processors.base import BaseProcessor
from karma.registries.processor_registry import register_processor

@register_processor("medical_text_normalizer")
class MedicalTextNormalizer(BaseProcessor):
    """Processor for normalizing medical text."""
    
    def __init__(self, normalize_units=True, expand_abbreviations=True):
        self.normalize_units = normalize_units
        self.expand_abbreviations = expand_abbreviations
        
        if expand_abbreviations:
            self.abbreviations = {
                "w/": "with",
                "w/o": "without",
                "Dx": "diagnosis",
                "Tx": "treatment",
                "Rx": "prescription"
            }
    
    def process(self, text: str) -> str:
        """Process medical text."""
        processed = text
        
        if self.expand_abbreviations:
            for abbrev, expansion in self.abbreviations.items():
                processed = processed.replace(abbrev, expansion)
        
        if self.normalize_units:
            processed = self._normalize_units(processed)
        
        return processed.strip()
    
    def _normalize_units(self, text):
        """Normalize medical units."""
        import re
        
        # Convert mg to milligrams, etc.
        text = re.sub(r'(\d+)\s*mg\b', r'\1 milligrams', text)
        text = re.sub(r'(\d+)\s*ml\b', r'\1 milliliters', text)
        
        return text

# Use the registered processor
from karma.registries.processor_registry import get_processor

processor = get_processor("medical_text_normalizer", 
                         normalize_units=True, 
                         expand_abbreviations=True)

processed_text = processor.process("Patient needs 500mg w/ meals")
```

### Discovery and Introspection

```python
from karma.registries.model_registry import discover_models
from karma.registries.dataset_registry import discover_datasets
from karma.registries.metrics_registry import discover_metrics

# Discover all registered components
models = discover_models()
datasets = discover_datasets()
metrics = discover_metrics()

print("Available Models:")
for model_name, model_meta in models.items():
    print(f"  {model_name}: {model_meta.description}")

print("\nAvailable Datasets:")
for dataset_name, dataset_info in datasets.items():
    print(f"  {dataset_name}: {dataset_info['task_type']} - {dataset_info['metrics']}")

print("\nAvailable Metrics:")
for metric_name, metric_class in metrics.items():
    print(f"  {metric_name}: {metric_class.__doc__}")
```

### Advanced Registry Usage

```python
from karma.registries.model_registry import ModelRegistry
from karma.registries.dataset_registry import DatasetRegistry

# Access registry instances directly
model_registry = ModelRegistry()
dataset_registry = DatasetRegistry()

# Get detailed information
model_info = model_registry.get_model_info("qwen")
print(f"Model: {model_info['name']}")
print(f"Type: {model_info['model_type']}")
print(f"Modalities: {model_info['modalities']}")

# Validate dataset arguments
valid_args = dataset_registry.validate_dataset_args(
    dataset_name="pubmedqa",
    provided_args={"split": "test"}
)
print(f"Validation result: {valid_args}")

# List models by type
text_models = model_registry.list_models_by_type("TEXT_GENERATION")
print(f"Text generation models: {text_models}")
```

### Runtime Registration

```python
# Register components at runtime
from karma.registries.model_registry import register_model_meta
from karma.data_models.model_meta import ModelMeta, ModelType, ModalityType

def register_runtime_model(name, loader_class, **kwargs):
    """Register a model at runtime."""
    
    model_meta = ModelMeta(
        name=name,
        description=f"Runtime registered model: {name}",
        loader_class=loader_class,
        loader_kwargs=kwargs,
        model_type=ModelType.TEXT_GENERATION,
        modalities=[ModalityType.TEXT],
        framework=["PyTorch"],
    )
    
    register_model_meta(model_meta)
    print(f"Registered model: {name}")

# Usage
register_runtime_model(
    name="my_runtime_model",
    loader_class="myproject.models.RuntimeModel",
    temperature=0.8,
    max_tokens=1024
)
```

## Registry Features

### Auto-Discovery
- Automatic component discovery at import time
- Dynamic loading of classes and modules
- Support for plugin-style architecture

### Validation
- Argument validation for datasets
- Type checking for model metadata
- Consistency checks across registries

### Flexibility
- Runtime registration support
- Decorator-based registration
- Programmatic API access

### Integration
- Seamless CLI integration
- Cross-component dependency handling
- Extensible architecture

## Best Practices

### Registration Patterns

```python
# Use descriptive names
@register_dataset("medical_qa_cardiology", ...)
class CardiologyQADataset(BaseMultimodalDataset):
    pass

# Include comprehensive metadata
model_meta = ModelMeta(
    name="specialized_medical_model",
    description="Specialized model for medical diagnosis",
    # ... complete metadata
)

# Use appropriate task types
@register_dataset(
    "radiology_vqa",
    task_type="vqa",  # Not "qa" for visual tasks
    metrics=["exact_match", "bleu"]
)
```

### Error Handling

```python
from karma.registries.dataset_registry import create_dataset
import logging

logger = logging.getLogger(__name__)

def safe_create_dataset(name, **kwargs):
    """Safely create dataset with error handling."""
    try:
        return create_dataset(name, **kwargs)
    except ValueError as e:
        logger.error(f"Invalid arguments for {name}: {e}")
        return None
    except KeyError as e:
        logger.error(f"Dataset {name} not found: {e}")
        return None
```

## See Also

- [Models API](models.md) - Model implementation details
- [Datasets API](datasets.md) - Dataset implementation details
- [Metrics API](metrics.md) - Metric implementation details
- [CLI Reference](cli.md) - Command-line interface