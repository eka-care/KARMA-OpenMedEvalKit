# Datasets Guide

This guide covers working with datasets in KARMA, from using built-in datasets to creating your own custom implementations.

## Built-in Datasets

KARMA supports 12+ medical datasets across multiple modalities:

### Text-based Datasets

- **openlifescienceai/pubmedqa** - PubMed Question Answering
- **openlifescienceai/medmcqa** - Medical Multiple Choice QA
- **openlifescienceai/medqa** - Medical Question Answering
- **ChuGyouk/MedXpertQA** - Medical Expert QA

### Vision-Language Datasets

- **mdwiratathya/SLAKE-vqa-english** - Structured Language And Knowledge Extraction
- **flaviagiammarino/vqa-rad** - Visual Question Answering for Radiology

### Audio Datasets

- **ai4bharat/indicvoices_r** - Text to speech dataset that could be used for ASR evaluation as well.
- **ai4bharat/indicvoices** - ASR dataset - Indic Voices Recognition

### Translation Datasets

- **ai4bharat/IN22-Conv** - Indic Language Conversation Translation

## Viewing Available Datasets

```bash
# List all available datasets
karma list datasets

# Get detailed information about a specific dataset
karma info dataset openlifescienceai/pubmedqa
```

## Using Datasets

```bash
# Use specific dataset
karma eval --model qwen --model-path "Qwen/Qwen3-0.6B" \
  --datasets openlifescienceai/pubmedqa

# Use multiple datasets
karma eval --model qwen --model-path "Qwen/Qwen3-0.6B" \
  --datasets "openlifescienceai/pubmedqa,openlifescienceai/medmcqa"
```

## Dataset Configuration

### Dataset-Specific Arguments

Some datasets require additional configuration:

```bash
# Translation datasets with language pairs
karma eval --model qwen --model-path "Qwen/Qwen3-0.6B" \
    --datasets "ai4bharat/IN22-Conv" \
    --dataset-args "ai4bharat/IN22-Conv:source_language=en,target_language=hi"

# Datasets with specific splits
karma eval --model qwen --model-path "Qwen/Qwen3-0.6B" \
  --datasets "openlifescienceai/medmcqa" \
  --dataset-args "openlifescienceai/medmcqa:split=validation"
```

## Custom Dataset Integration

You can create custom datasets by inheriting from `BaseMultimodalDataset`:

```python
from karma.eval_datasets.base_dataset import BaseMultimodalDataset
from karma.registries.dataset_registry import register_dataset

@register_dataset(
    "my_medical_dataset",
    metrics=["exact_match", "accuracy"],
    task_type="mcqa",
    required_args=["split"],
    optional_args=["subset"],
    default_args={"split": "test"}
)
class MyMedicalDataset(BaseMultimodalDataset):
    """Custom medical dataset."""
    
    def __init__(self, split: str = "test", **kwargs):
        self.split = split
        super().__init__(**kwargs)
    
    def load_data(self):
        # Load your dataset
        return your_dataset_loader(split=self.split)
    
    def format_item(self, item):
        # Format each item for evaluation
        return {
            "prompt": item["question"],
            "ground_truth": item["answer"],
            "options": item.get("options", [])
        }
```

## Next Steps

- **Learn about models**: See [Models Guide](../models/overview.md)
- **Understand metrics**: Read [Metrics Guide](../metrics/overview.md)
- **Add custom datasets**: Check [Adding HuggingFace Datasets](../adding-datasets-from-huggingface.md)
- **API reference**: Explore [Datasets API Reference](../../api-reference/datasets.md)