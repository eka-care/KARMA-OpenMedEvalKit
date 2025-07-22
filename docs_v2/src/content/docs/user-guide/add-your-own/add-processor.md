---
title: Add Your Own Processor
---
Processors are used for tweak the output of the model and then running evaluation on that output.
This is typically required in cases when normalizing text for different languages or dialects.
We have implmemented these for ASR specific datasets but you can use it for any dataset.

### Step 1: Create Processor Class

```python
# karma/processors/my_custom_processor.py
from karma.processors.base import BaseProcessor
from karma.registries.processor_registry import register_processor

@register_processor("medical_text_normalizer")
class MedicalTextNormalizer(BaseProcessor):
    """Processor for normalizing medical text."""

    def __init__(self, normalize_units=True, expand_abbreviations=True):
        self.normalize_units = normalize_units
        self.expand_abbreviations = expand_abbreviations
        self.medical_abbreviations = {
            "bp": "blood pressure",
            "hr": "heart rate",
            "temp": "temperature",
            "mg": "milligrams",
            "ml": "milliliters"
        }

    def process(self, text: str, **kwargs) -> str:
        """Process medical text with normalization."""
        if self.expand_abbreviations:
            text = self._expand_abbreviations(text)

        if self.normalize_units:
            text = self._normalize_units(text)

        return text

    def _expand_abbreviations(self, text: str) -> str:
        """Expand medical abbreviations."""
        for abbrev, expansion in self.medical_abbreviations.items():
            text = text.replace(abbrev, expansion)
        return text

    def _normalize_units(self, text: str) -> str:
        """Normalize medical units."""
        # Add unit normalization logic
        return text
```

### Step 2: Register and Use

```python
# Via CLI
karma eval --model qwen --model-path "Qwen/Qwen3-0.6B" \
  --datasets my_medical_dataset \
  --processor-args "my_medical_dataset.medical_text_normalizer:normalize_units=True"

# Programmatically
from karma.registries.processor_registry import get_processor

processor = get_processor("medical_text_normalizer", normalize_units=True)
```

## Integration Patterns

### Dataset Integration

Processors can be integrated directly with dataset registration:

```python
@register_dataset(
    "my_medical_dataset",
    processors=["general_text_processor", "medical_text_normalizer"],
    processor_configs={
        "general_text_processor": {"lowercase": True},
        "medical_text_normalizer": {"normalize_units": True}
    }
)
class MyMedicalDataset(BaseMultimodalDataset):
    # Dataset implementation
    pass
```

## Advanced Use Cases

### Chain Multiple Processors

```python
# Create processor chain
from karma.registries.processor_registry import get_processor

processors = [
    get_processor("general_text_processor", lowercase=True),
    get_processor("medical_text_normalizer", normalize_units=True),
    get_processor("multilingual_text_processor", target_language="en")
]

# Apply chain to dataset
def process_chain(text: str) -> str:
    for processor in processors:
        text = processor.process(text)
    return text
```

### Language-Specific Processing

```python
# Language-specific processor selection
def get_language_processor(language: str):
    if language in ["hi", "bn", "ta"]:
        return get_processor("devnagari_transliterator")
    else:
        return get_processor("general_text_processor")
```

## Best Practices

1. **Chain Order**: Consider the order of processors in the chain
2. **Language Handling**: Use appropriate processors for different languages
3. **Performance**: Be mindful of processing overhead for large datasets
4. **Testing**: Validate processor output with sample data
5. **Configuration**: Make processors configurable for different use cases
