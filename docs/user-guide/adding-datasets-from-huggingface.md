# Adding Datasets from Hugging Face

This tutorial walks you through the process of adding a new dataset from Hugging Face to the KARMA evaluation framework.

## Overview

KARMA supports automatic integration of datasets from Hugging Face Hub. The process involves:

1. Creating a dataset class that inherits from `BaseMultimodalDataset`
2. Registering the dataset with the appropriate metadata
3. Implementing the required methods for data loading and formatting
4. Adding the dataset to the documentation

## Step 1: Dataset Class Structure

Create a new Python file in the `karma/eval_datasets/` directory for your dataset:

```python
"""
Your dataset implementation with multimodal support.

This module provides the YourDataset class that implements the
multimodal dataset interface for use with the KARMA evaluation framework.
"""

import logging
from typing import Any, Dict, Optional

from karma.data_models.dataloader_iterable import DataLoaderIterable
from karma.eval_datasets.base_dataset import BaseMultimodalDataset
from karma.registries.dataset_registry import register_dataset

logger = logging.getLogger(__name__)

# Dataset configuration
DATASET_NAME = "your_username/your_dataset_name"  # Hugging Face dataset path
SPLIT = "test"  # Default split to use
COMMIT_HASH = "your_commit_hash"  # Optional: Pin to specific commit

@register_dataset(
    DATASET_NAME,
    commit_hash=COMMIT_HASH,
    split=SPLIT,
    metrics=["exact_match", "accuracy"],  # Supported metrics
    task_type="mcqa",  # Task type: "mcqa", "qa", "vqa", "translation", "asr"
    required_args=["required_param"],  # Required parameters
    optional_args=["optional_param"],  # Optional parameters
    default_args={"optional_param": "default_value"},  # Default values
)
class YourDataset(BaseMultimodalDataset):
    """Your dataset for evaluation."""

    def __init__(
        self,
        required_param: str,
        optional_param: str = "default_value",
        **kwargs
    ):
        """
        Initialize the dataset.
        
        Args:
            required_param: Description of required parameter
            optional_param: Description of optional parameter
            **kwargs: Additional arguments passed to parent class
        """
        self.required_param = required_param
        self.optional_param = optional_param
        super().__init__(**kwargs)

    def load_data(self):
        """Load data from Hugging Face."""
        # This method is inherited from BaseMultimodalDataset
        # It automatically loads data from Hugging Face using the dataset_name
        return super().load_data()

    def format_item(self, item: Dict[str, Any]) -> DataLoaderIterable:
        """
        Format a single item from the dataset.
        
        Args:
            item: Raw item from the dataset
            
        Returns:
            DataLoaderIterable: Formatted item
        """
        # Extract and format the data according to your needs
        question = item["question"]
        answer = item["answer"]
        
        # Add any custom formatting logic here
        formatted_question = self._format_question(question)
        
        return DataLoaderIterable(
            text=formatted_question,
            ground_truth=answer,
            metadata={
                "original_question": question,
                "parameter_value": self.required_param,
                # Add any other metadata you need
            }
        )

    def _format_question(self, question: str) -> str:
        """Custom formatting logic for questions."""
        # Add your custom formatting here
        return f"Question: {question}"
```

## Step 2: Understanding Registration Parameters

The `@register_dataset` decorator accepts several important parameters:

### Required Parameters

- **`dataset_name`**: The Hugging Face dataset identifier (e.g., "openlifescienceai/medqa")
- **`metrics`**: List of metrics that can be computed for this dataset
- **`task_type`**: Type of task ("mcqa", "qa", "vqa", "translation", "asr")

### Optional Parameters

- **`commit_hash`**: Pin to a specific commit for reproducibility
- **`split`**: Default split to use (e.g., "test", "validation")
- **`required_args`**: List of required parameters for dataset instantiation
- **`optional_args`**: List of optional parameters
- **`default_args`**: Dictionary of default values for parameters
- **`processors`**: List of processors that can be applied to this dataset

### Task Types and Common Metrics

| Task Type | Description | Common Metrics |
|-----------|-------------|----------------|
| `mcqa` | Multiple Choice Question Answering | `exact_match`, `accuracy` |
| `qa` | Question Answering | `exact_match`, `bleu`, `rouge` |
| `vqa` | Visual Question Answering | `exact_match`, `bleu`, `accuracy` |
| `translation` | Translation | `bleu`, `sacrebleu`, `chrf` |
| `asr` | Automatic Speech Recognition | `wer`, `cer`, `bleu` |

## Step 3: Implementing Data Formatting

The `format_item` method is crucial for converting raw dataset items into the format expected by KARMA:

### For Multiple Choice Questions

```python
def format_item(self, item: Dict[str, Any]) -> DataLoaderIterable:
    """Format multiple choice question item."""
    question = item["question"]
    options = item["options"]  # List of choices
    correct_answer = item["answer"]  # Letter (A, B, C, D) or index
    
    # Format question with options
    formatted_question = f"{question}\n"
    for i, option in enumerate(options):
        formatted_question += f"{chr(65 + i)}. {option}\n"
    
    return DataLoaderIterable(
        text=formatted_question,
        ground_truth=correct_answer,
        metadata={
            "options": options,
            "question_type": "mcqa"
        }
    )
```

### For Visual Question Answering

```python
def format_item(self, item: Dict[str, Any]) -> DataLoaderIterable:
    """Format VQA item."""
    question = item["question"]
    image_path = item["image_path"]  # or image URL
    answer = item["answer"]
    
    return DataLoaderIterable(
        text=question,
        image_path=image_path,  # KARMA handles image loading
        ground_truth=answer,
        metadata={
            "image_id": item.get("image_id"),
            "question_type": "vqa"
        }
    )
```

### For Translation Tasks

```python
def format_item(self, item: Dict[str, Any]) -> DataLoaderIterable:
    """Format translation item."""
    source_text = item[f"text_{self.source_language}"]
    target_text = item[f"text_{self.target_language}"]
    
    return DataLoaderIterable(
        text=source_text,
        ground_truth=target_text,
        metadata={
            "source_language": self.source_language,
            "target_language": self.target_language,
            "domain": item.get("domain", "general")
        }
    )
```

## Step 4: Example Implementation - MedQA Dataset

Here's a complete example based on the MedQA dataset:

```python
"""
MedQA dataset implementation with multimodal support.
"""

import logging
from typing import Any, Dict

from karma.data_models.dataloader_iterable import DataLoaderIterable
from karma.eval_datasets.base_dataset import BaseMultimodalDataset
from karma.registries.dataset_registry import register_dataset

logger = logging.getLogger(__name__)

# Dataset configuration
DATASET_NAME = "openlifescienceai/medqa"
SPLIT = "test"
COMMIT_HASH = "153e61cdd129eb79d3c27f82cdf3bc5e018c11b0"

@register_dataset(
    DATASET_NAME,
    commit_hash=COMMIT_HASH,
    split=SPLIT,
    metrics=["exact_match"],
    task_type="mcqa",
)
class MedQADataset(BaseMultimodalDataset):
    """MedQA dataset for medical question answering."""

    def format_item(self, item: Dict[str, Any]) -> DataLoaderIterable:
        """Format MedQA item."""
        question = item["question"]
        options = item["options"]
        answer = item["answer"]
        
        # Format question with options
        formatted_question = f"{question}\n"
        for option_key, option_text in options.items():
            formatted_question += f"{option_key}. {option_text}\n"
        
        # Add instruction for answer format
        formatted_question += "\nOutput format: 'ANSWER: <option>'"
        
        return DataLoaderIterable(
            text=formatted_question,
            ground_truth=f"ANSWER: {answer}",
            metadata={
                "options": options,
                "raw_answer": answer,
                "question_type": "medical_mcqa"
            }
        )
```

## Step 5: Adding Dataset to Documentation

Add your dataset to the documentation by updating `docs/api-reference/datasets.md`:

```markdown
#### YourDataset (your_username/your_dataset_name)

Description of your dataset.

::: karma.eval_datasets.your_dataset.YourDataset
    options:
      show_source: false
      show_root_heading: true
      show_category_heading: true
      members_order: source
      filters: ["!^_"]
      merge_init_into_class: true
```

## Step 6: Testing Your Dataset

Test your dataset implementation:

```python
# Test dataset loading
from karma.registries.dataset_registry import dataset_registry

# Create dataset instance
dataset = dataset_registry.create_dataset(
    "your_username/your_dataset_name",
    required_param="test_value"
)

# Load data
data = dataset.load_data()
print(f"Dataset loaded: {len(data)} items")

# Test item formatting
first_item = next(iter(dataset))
print(f"Formatted item: {first_item}")
```

## Step 7: Using CLI Commands

Once registered, your dataset can be used with KARMA CLI:

```bash
# List available datasets
karma list datasets

# Get dataset information
karma info dataset your_username/your_dataset_name

# Run evaluation
karma eval --model qwen --model-path "Qwen/Qwen2.5-0.5B-Instruct" \
  --datasets your_username/your_dataset_name \
  --dataset-args "your_username/your_dataset_name:required_param=value"
```

## Best Practices

### 1. Error Handling

```python
def format_item(self, item: Dict[str, Any]) -> DataLoaderIterable:
    """Format item with error handling."""
    try:
        # Validate required fields
        if "question" not in item:
            raise ValueError("Missing required field: question")
        
        # Process item
        return DataLoaderIterable(
            text=item["question"],
            ground_truth=item["answer"]
        )
    except Exception as e:
        logger.error(f"Error formatting item: {e}")
        raise
```

### 2. Parameter Validation

```python
def __init__(self, language: str = "en", **kwargs):
    """Initialize with parameter validation."""
    if language not in ["en", "es", "fr", "de"]:
        raise ValueError(f"Unsupported language: {language}")
    
    self.language = language
    super().__init__(**kwargs)
```

### 3. Flexible Data Handling

```python
def format_item(self, item: Dict[str, Any]) -> DataLoaderIterable:
    """Handle different data formats."""
    # Handle both string and list formats for options
    options = item.get("options", [])
    if isinstance(options, str):
        options = options.split("\n")
    elif isinstance(options, dict):
        options = list(options.values())
    
    return DataLoaderIterable(
        text=self._format_question(item["question"], options),
        ground_truth=item["answer"]
    )
```

### 4. Metadata Preservation

```python
def format_item(self, item: Dict[str, Any]) -> DataLoaderIterable:
    """Preserve important metadata."""
    return DataLoaderIterable(
        text=item["question"],
        ground_truth=item["answer"],
        metadata={
            "dataset_name": self.dataset_name,
            "split": self.split,
            "original_id": item.get("id"),
            "difficulty": item.get("difficulty"),
            "category": item.get("category"),
            "source": item.get("source"),
        }
    )
```

## Advanced Features

### Custom Processors

You can register custom processors for your dataset:

```python
from karma.processors.base import BaseProcessor
from karma.registries.processor_registry import register_processor

@register_processor("medical_text_processor")
class MedicalTextProcessor(BaseProcessor):
    """Custom processor for medical text."""
    
    def process(self, text: str) -> str:
        """Process medical text."""
        # Add your processing logic
        return text.replace("w/", "with").replace("w/o", "without")

# Use in dataset registration
@register_dataset(
    "your_dataset",
    processors=["medical_text_processor"],
    # other parameters...
)
```

### Multi-language Support

```python
@register_dataset(
    "your_multilingual_dataset",
    required_args=["language"],
    optional_args=["region"],
    default_args={"language": "en", "region": "US"},
    # other parameters...
)
class MultilingualDataset(BaseMultimodalDataset):
    """Dataset with multi-language support."""
    
    def __init__(self, language: str = "en", region: str = "US", **kwargs):
        self.language = language
        self.region = region
        super().__init__(**kwargs)
    
    def load_data(self):
        """Load data for specific language/region."""
        # Customize dataset loading based on language
        dataset = super().load_data()
        if self.language != "en":
            dataset = dataset.filter(lambda x: x["language"] == self.language)
        return dataset
```

## Troubleshooting

### Common Issues

1. **Dataset not found**: Ensure the Hugging Face dataset name is correct
2. **Import errors**: Check that all required imports are included
3. **Registration conflicts**: Use unique dataset names
4. **Formatting errors**: Validate that `format_item` returns `DataLoaderIterable`

### Debugging Tips

```python
# Add logging to debug issues
import logging
logger = logging.getLogger(__name__)

def format_item(self, item: Dict[str, Any]) -> DataLoaderIterable:
    """Format item with debugging."""
    logger.debug(f"Processing item: {item.keys()}")
    
    # Your formatting logic
    formatted_item = DataLoaderIterable(...)
    
    logger.debug(f"Formatted item: {formatted_item}")
    return formatted_item
```

## Conclusion

Adding datasets from Hugging Face to KARMA is straightforward with the registration system. The key steps are:

1. Create a dataset class inheriting from `BaseMultimodalDataset`
2. Register it with appropriate metadata using `@register_dataset`
3. Implement `format_item` to convert raw data to `DataLoaderIterable`
4. Add documentation and test your implementation

This approach ensures your dataset integrates seamlessly with KARMA's evaluation pipeline and CLI tools.