## Using Custom Processors

### Overview

KARMA-OpenMedEvalKit provides a flexible processor system that allows you to apply custom text transformations to your datasets during evaluation. Processors are particularly useful for normalizing text, handling multilingual content, and preparing data for specific model requirements.

### Architecture

The processor system consists of:

- **Base Processor**: `BaseProcessor` class that all processors inherit from
- **Processor Registry**: Auto-discovery system that finds and registers processors
- **Integration Points**: Processors can be applied at dataset level or via CLI

Processors are defined with the datasets in the decorator.
The processors are by default chained i.e., the output of the previous processor is the input of the next processor.

#### Available Processors

1. **GeneralTextProcessor**
Handles common text normalization. Number to text conversion. Punctuation removal. Case normalization

2. **DevanagariTransliterator**  
Multilingual text processing for indic Devnagri scrips. Script conversion between languages. Handles Devanagari text

3. **GeneralTextProcessor**
Audio transcription normalization. Specialized for STT tasks where numbers need to be normalised

### Using Existing Processors

### Using through CLI
Let's take the example of GeneralTextProcessor, which expects the input language to normalise from.
This is dependent on which dataset is being evaluated, in the example below, we are transcribing from Indicvoices_r dataset.

The language argument is Hindi for dataset as well 
```console
$ karma eval --model "ai4bharat/indic-conformer-600m-multilingual" \
 --model-path "ai4bharat/indic-conformer-600m-multilingual" \
 --datasets "ai4bharat/indicvoices_r" \
 --batch-size 1 \
 --dataset-args "ai4bharat/indicvoices_r:language=Hindi" \
 --processor-args  "ai4bharat/indicvoices_r.general_text_processor:language=Hindi" \
 --max-samples 3
```

The access pattern is as below
```console
--processor-args 
"<dataset>.<processor_name>:argument1=value,<dataset>.<processor_name>:argument2=value"
```

#### Programmatic Usage

```python
from karma.processors.general_text_processor import GeneralTextProcessor
from karma.eval_datasets.medmcqa_dataset import MedMCQADataset

# Create processor with custom configuration
processor = GeneralTextProcessor(
    use_num2text=True,
    use_punc=False,
    use_lowercasing=True
)

# Apply to dataset
dataset = MedMCQADataset(postprocessors=[processor])

# Or apply directly to text
processed_texts = processor.process(["Text with numbers 123 and CAPS."])
# Result: ["text with numbers one hundred twenty three and caps"]
```

### Creating Custom Processors

#### Step 1: Create Processor Class

Create a new file in `karma/processors/` directory:

```python
# karma/processors/my_custom_processor.py
from karma.processors.base import BaseProcessor
from karma.registries.processor_registry import register_processor

@register_processor(name="my_custom_processor")
class MyCustomProcessor(BaseProcessor):
    def __init__(self, normalize_whitespace=True, custom_param=None):
        """
        Initialize your custom processor.
        
        Args:
            normalize_whitespace: Whether to normalize whitespace
            custom_param: Your custom parameter
        """
        self.normalize_whitespace = normalize_whitespace
        self.custom_param = custom_param
    
    def process(self, texts: list[str]) -> list[str]:
        """
        Process a list of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of processed texts
        """
        processed = []
        for text in texts:
            result = text
            
            # Apply your custom transformations
            if self.normalize_whitespace:
                result = ' '.join(result.split())
            
            # Add your custom logic here
            if self.custom_param:
                result = self._apply_custom_logic(result)
            
            processed.append(result)
        
        return processed
    
    def _apply_custom_logic(self, text: str) -> str:
        """Helper method for custom processing logic."""
        # Implement your custom transformation
        return text.upper()  # Example: convert to uppercase
```

#### Step 2: Register and Use

The processor is automatically registered via the `@register_processor` decorator. You can now use it:

```bash
# Via CLI
karma eval --model qwen --datasets "pubmedqa" --processors "my_custom_processor"

# With arguments
karma eval --model qwen --datasets "pubmedqa" \
  --processor-args "pubmedqa.my_custom_processor:normalize_whitespace=true,custom_param=example"
```

```python
# Programmatically
from karma.processors.my_custom_processor import MyCustomProcessor

processor = MyCustomProcessor(normalize_whitespace=True, custom_param="example")
results = processor.process(["Input text with   extra spaces"])
```

### Dataset-Processor Integration

#### Method 1: Dataset Registration

Specify processors when registering datasets:

```python
@register_dataset(
    "my/dataset",
    processors=["general_text_processor", "my_custom_processor"]
)
class MyDataset(BaseMultimodalDataset):
    def format_item(self, sample):
        # Dataset formatting logic
        return DataLoaderIterable(
            input=sample["question"],
            expected_output=sample["answer"]
        )
```

#### Method 2: CLI Integration

Use dataset-specific processor arguments:

```bash
karma eval --model qwen --datasets "dataset1,dataset2" \
  --processor-args "dataset1.general_text_processor:use_punc=false" \
  --processor-args "dataset2.my_custom_processor:custom_param=value"
```

### Advanced Use Cases

#### Chain Multiple Processors

```python
from karma.processors.general_text_processor import GeneralTextProcessor
from karma.processors.my_custom_processor import MyCustomProcessor

# Create processor chain
processors = [
    GeneralTextProcessor(use_lowercasing=True),
    MyCustomProcessor(custom_param="chained")
]

# Apply chain to dataset
dataset = MyDataset(postprocessors=processors)
```

#### Conditional Processing

```python
@register_processor(name="conditional_processor")
class ConditionalProcessor(BaseProcessor):
    def __init__(self, condition_field="language"):
        self.condition_field = condition_field
    
    def process(self, texts: list[str]) -> list[str]:
        processed = []
        for text in texts:
            # Apply different processing based on conditions
            if self._detect_language(text) == "hi":
                result = self._process_hindi(text)
            else:
                result = self._process_english(text)
            processed.append(result)
        return processed
```

#### Language-Specific Processing

```python
@register_processor(name="multilingual_processor")
class MultilingualProcessor(BaseProcessor):
    def __init__(self, source_language="en", target_language="hi"):
        self.source_language = source_language
        self.target_language = target_language
    
    def process(self, texts: list[str]) -> list[str]:
        if self.source_language == "en" and self.target_language == "hi":
            return self._transliterate_to_hindi(texts)
        elif self.source_language == "hi" and self.target_language == "en":
            return self._transliterate_to_english(texts)
        return texts
```

### Best Practices

1. **Keep Processors Simple**: Each processor should have a single responsibility
2. **Use Type Hints**: Provide clear type annotations for parameters and return values
3. **Handle Edge Cases**: Account for empty strings, None values, and special characters
4. **Make Parameters Configurable**: Allow users to customize processor behavior
5. **Document Parameters**: Provide clear docstrings for all parameters
6. **Test Thoroughly**: Create unit tests for your processors

### Examples from the Codebase

#### Medical Text Normalization

```python
# For medical datasets, normalize medical terminology
@register_processor(name="medical_text_processor")
class MedicalTextProcessor(BaseProcessor):
    def __init__(self, normalize_units=True, expand_abbreviations=True):
        self.normalize_units = normalize_units
        self.expand_abbreviations = expand_abbreviations
    
    def process(self, texts: list[str]) -> list[str]:
        processed = []
        for text in texts:
            result = text
            if self.normalize_units:
                result = self._normalize_medical_units(result)
            if self.expand_abbreviations:
                result = self._expand_medical_abbreviations(result)
            processed.append(result)
        return processed
```

#### ASR Post-Processing

```python
# For speech recognition datasets
@register_processor(name="asr_postprocessor")
class ASRPostProcessor(BaseProcessor):
    def __init__(self, remove_disfluencies=True, normalize_numbers=True):
        self.remove_disfluencies = remove_disfluencies
        self.normalize_numbers = normalize_numbers
    
    def process(self, texts: list[str]) -> list[str]:
        processed = []
        for text in texts:
            result = text
            if self.remove_disfluencies:
                result = self._remove_disfluencies(result)
            if self.normalize_numbers:
                result = self._normalize_spoken_numbers(result)
            processed.append(result)
        return processed
```