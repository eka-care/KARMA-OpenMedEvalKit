# Datasets API Reference

This section documents KARMA's dataset system, including base classes, built-in datasets, and integration patterns.

## Base Classes

### BaseMultimodalDataset

The foundation for all multimodal evaluation datasets in KARMA.

::: karma.eval_datasets.base_dataset.BaseMultimodalDataset
    options:
      show_source: false
      show_root_heading: true
      show_category_heading: true
      members_order: source
      filters: ["!^_"]
      merge_init_into_class: true

## Built-in Datasets

### Medical Question Answering

#### MedQADataset (openlifescienceai/medqa)

Medical Question Answering dataset.

::: karma.eval_datasets.medqa_dataset.MedQADataset
    options:
      show_source: false
      show_root_heading: true
      show_category_heading: true
      members_order: source
      filters: ["!^_"]
      merge_init_into_class: true

#### PubMedMCQADataset (openlifescienceai/pubmedqa)

PubMed Multiple Choice Question Answering dataset.

::: karma.eval_datasets.pubmedmcqa_dataset.PubMedMCQADataset
    options:
      show_source: false
      show_root_heading: true
      show_category_heading: true
      members_order: source
      filters: ["!^_"]
      merge_init_into_class: true

#### MedMCQADataset (openlifescienceai/medmcqa)

Medical Multiple Choice Question Answering dataset.

::: karma.eval_datasets.medmcqa_dataset.MedMCQADataset
    options:
      show_source: false
      show_root_heading: true
      show_category_heading: true
      members_order: source
      filters: ["!^_"]
      merge_init_into_class: true

#### MedXpertQADataset (ChuGyouk/MedXpertQA)

Medical Expert Question Answering dataset.

::: karma.eval_datasets.medxpertqa_dataset.MedXpertQADataset
    options:
      show_source: false
      show_root_heading: true
      show_category_heading: true
      members_order: source
      filters: ["!^_"]
      merge_init_into_class: true

### Vision-Language Datasets

#### SLAKEDataset (mdwiratathya/SLAKE-vqa-english)

Structured Language And Knowledge Extraction dataset for medical VQA.

::: karma.eval_datasets.slake_dataset.SLAKEDataset
    options:
      show_source: false
      show_root_heading: true
      show_category_heading: true
      members_order: source
      filters: ["!^_"]
      merge_init_into_class: true

#### VQARADDataset (flaviagiammarino/vqa-rad)

Visual Question Answering for Radiology dataset.

::: karma.eval_datasets.vqa_rad_dataset.VQARADDataset
    options:
      show_source: false
      show_root_heading: true
      show_category_heading: true
      members_order: source
      filters: ["!^_"]
      merge_init_into_class: true

### Language and Speech Datasets

#### IN22ConvDataset (ai4bharat/IN22-Conv)

Indic Language Conversation Translation dataset.

::: karma.eval_datasets.in22conv_dataset.IN22ConvDataset
    options:
      show_source: false
      show_root_heading: true
      show_category_heading: true
      members_order: source
      filters: ["!^_"]
      merge_init_into_class: true

#### IndicVoicesRDataset (ai4bharat/indicvoices_r)

Indic Voices Recognition dataset for ASR evaluation.

::: karma.eval_datasets.indicvoices_r_dataset.IndicVoicesRDataset
    options:
      show_source: false
      show_root_heading: true
      show_category_heading: true
      members_order: source
      filters: ["!^_"]
      merge_init_into_class: true

### MMLU Medical Datasets

Medical benchmarks from the MMLU suite.

#### MMLUProfessionalMedicineDataset (openlifescienceai/mmlu_professional_medicine)

MMLU Professional Medicine dataset.

::: karma.eval_datasets.mmlu_medical_datasets.MMLUProfessionalMedicineDataset
    options:
      show_source: false
      show_root_heading: true
      show_category_heading: true
      members_order: source
      filters: ["!^_"]
      merge_init_into_class: true

#### MMLUAnatomyDataset (openlifescienceai/mmlu_anatomy)

MMLU Anatomy dataset.

::: karma.eval_datasets.mmlu_medical_datasets.MMLUAnatomyDataset
    options:
      show_source: false
      show_root_heading: true
      show_category_heading: true
      members_order: source
      filters: ["!^_"]
      merge_init_into_class: true

#### MMLUCollegeBiologyDataset (openlifescienceai/mmlu_college_biology)

MMLU College Biology dataset.

::: karma.eval_datasets.mmlu_medical_datasets.MMLUCollegeBiologyDataset
    options:
      show_source: false
      show_root_heading: true
      show_category_heading: true
      members_order: source
      filters: ["!^_"]
      merge_init_into_class: true

#### MMLUClinicalKnowledgeDataset (openlifescienceai/mmlu_clinical_knowledge)

MMLU Clinical Knowledge dataset.

::: karma.eval_datasets.mmlu_medical_datasets.MMLUClinicalKnowledgeDataset
    options:
      show_source: false
      show_root_heading: true
      show_category_heading: true
      members_order: source
      filters: ["!^_"]
      merge_init_into_class: true

#### MMLUCollegeMedicineDataset (openlifescienceai/mmlu_college_medicine)

MMLU College Medicine dataset.

::: karma.eval_datasets.mmlu_medical_datasets.MMLUCollegeMedicineDataset
    options:
      show_source: false
      show_root_heading: true
      show_category_heading: true
      members_order: source
      filters: ["!^_"]
      merge_init_into_class: true

### Rubric-Based Evaluation Datasets

#### RubricBaseDataset

Base class for rubric-based evaluation datasets that handle medical question answering with rubric-based evaluation.

::: karma.eval_datasets.rubrics.rubric_base_dataset.RubricBaseDataset
    options:
      show_source: false
      show_root_heading: true
      show_category_heading: true
      members_order: source
      filters: ["!^_"]
      merge_init_into_class: true

#### EkaMedicalHistorySummary (ekacare/ekacare_medical_history_summarisation)

EkaCare Medical History Summarization dataset for rubric-based evaluation.

::: karma.eval_datasets.rubrics.eka_medical_history_summary.EkaMedicalHistorySummary
    options:
      show_source: false
      show_root_heading: true
      show_category_heading: true
      members_order: source
      filters: ["!^_"]
      merge_init_into_class: true

#### HealthBenchDataset (Tonic/Health-Bench-Eval-OSS-2025-07)

Health-Bench evaluation dataset for rubric-based medical question answering.

::: karma.eval_datasets.rubrics.healthbench_dataset.HealthBenchDataset
    options:
      show_source: false
      show_root_heading: true
      show_category_heading: true
      members_order: source
      filters: ["!^_"]
      merge_init_into_class: true

## Data Models

### DataLoaderIterable

Pydantic model for multimodal dataset samples.

::: karma.data_models.dataloader_iterable.DataLoaderIterable
    options:
      show_source: false
      show_root_heading: true
      show_category_heading: true
      members_order: source
      filters: ["!^_"]
      merge_init_into_class: true

## Usage Examples

### Basic Dataset Usage

```python
from karma.registries.dataset_registry import dataset_registry

# Initialize dataset using registry
dataset = dataset_registry.create_dataset("openlifescienceai/pubmedmcqa")

# Load data
data = dataset.load_data()

# Iterate through samples
for item in dataset:
    print(f"Question: {item.text}")
    print(f"Options: {item.metadata.get('options', [])}")
    print(f"Answer: {item.ground_truth}")
```

### Custom Dataset Integration

```python
from karma.eval_datasets.base_dataset import BaseMultimodalDataset
from karma.registries.dataset_registry import register_dataset

@register_dataset(
    "my_medical_dataset",
    metrics=["exact_match", "accuracy"],
    task_type="mcqa",
    required_args=["domain"],
    optional_args=["split", "subset"],
    default_args={"split": "test"}
)
class MyMedicalDataset(BaseMultimodalDataset):
    """Custom medical dataset implementation."""
    
    def __init__(self, domain: str, split: str = "test", **kwargs):
        self.domain = domain
        self.split = split
        super().__init__(**kwargs)
    
    def load_data(self):
        # Load your dataset
        return self._load_custom_data()
    
    def format_item(self, item):
        return {
            "prompt": self._format_question(item),
            "ground_truth": item["answer"],
            "options": item.get("choices", [])
        }
    
    def _load_custom_data(self):
        # Custom data loading logic
        pass
    
    def _format_question(self, item):
        # Custom question formatting
        return f"Domain: {self.domain}\nQuestion: {item['question']}"
```

### Multimodal Dataset Example

```python
from karma.eval_datasets.base_dataset import BaseMultimodalDataset
from karma.data_models.dataloader_iterable import DataLoaderIterable

class MedicalVQADataset(BaseMultimodalDataset):
    """Medical Visual Question Answering dataset."""
    
    def format_item(self, item):
        return DataLoaderIterable(
            text=item["question"],
            image_path=item["image_path"],
            ground_truth=item["answer"],
            metadata={
                "image_type": item.get("modality", "unknown"),
                "difficulty": item.get("difficulty", "medium")
            }
        )
    
    def collate_fn(self, batch):
        """Custom collate function for batching."""
        texts = [item.text for item in batch]
        images = [item.image for item in batch if item.image is not None]
        ground_truths = [item.ground_truth for item in batch]
        
        return {
            "texts": texts,
            "images": images,
            "ground_truths": ground_truths
        }
```

### Dataset with Preprocessing

```python
from karma.eval_datasets.base_dataset import BaseMultimodalDataset
import re

class PreprocessedMedicalDataset(BaseMultimodalDataset):
    """Dataset with built-in preprocessing."""
    
    def __init__(self, normalize_text=True, **kwargs):
        self.normalize_text = normalize_text
        super().__init__(**kwargs)
    
    def format_item(self, item):
        question = item["question"]
        answer = item["answer"]
        
        if self.normalize_text:
            question = self._normalize_text(question)
            answer = self._normalize_text(answer)
        
        return {
            "prompt": question,
            "ground_truth": answer,
            "metadata": {
                "original_question": item["question"],
                "original_answer": item["answer"]
            }
        }
    
    def _normalize_text(self, text):
        """Normalize medical text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Standardize medical abbreviations
        text = text.replace('w/', 'with')
        text = text.replace('w/o', 'without')
        
        return text
```

## Dataset Types and Characteristics

### Question Answering Datasets

- **Task Type**: `mcqa`, `qa`
- **Metrics**: `exact_match`, `accuracy`
- **Examples**: PubMedQA, MedMCQA, MedQA

### Visual Question Answering Datasets

- **Task Type**: `vqa`
- **Metrics**: `exact_match`, `bleu`, `accuracy`
- **Examples**: SLAKE, VQA-RAD

### Translation Datasets

- **Task Type**: `translation`
- **Metrics**: `bleu`, `exact_match`
- **Examples**: In22Conv

### Speech Recognition Datasets

- **Task Type**: `asr`
- **Metrics**: `wer`, `cer`
- **Examples**: IndicVoices-R

## Best Practices

### Dataset Initialization

```python
# Always specify required arguments explicitly
dataset = MedMCQADataset(
    split="test",
    subset="dev" if args.debug else None
)

# Verify dataset is properly loaded
assert len(dataset) > 0, "Dataset is empty"
print(f"Loaded {len(dataset)} examples")
```

### Error Handling

```python
from karma.eval_datasets.base_dataset import BaseMultimodalDataset
import logging

logger = logging.getLogger(__name__)

try:
    dataset = PubMedMCQADataset(split="test")
    data = dataset.load_data()
except Exception as e:
    logger.error(f"Failed to load dataset: {e}")
    # Fallback to smaller subset
    dataset = PubMedMCQADataset(split="validation")
```

### Memory Optimization

```python
from torch.utils.data import DataLoader

# Use DataLoader for efficient batching
dataset = dataset_registry.create_dataset("openlifescienceai/medmcqa")
dataloader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=False,
    num_workers=4,
    collate_fn=dataset.collate_fn
)

# Process in batches to manage memory
for batch in dataloader:
    # Process batch
    pass
```

### Custom Validation

```python
class ValidatedMedicalDataset(BaseMultimodalDataset):
    """Dataset with validation checks."""
    
    def format_item(self, item):
        # Validate required fields
        required_fields = ["question", "answer"]
        for field in required_fields:
            if field not in item:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate answer format
        if not item["answer"].strip():
            raise ValueError("Empty answer not allowed")
        
        return {
            "prompt": item["question"],
            "ground_truth": item["answer"],
            "validated": True
        }
```

## See Also

- [Registries API](registries.md) - Dataset registry and discovery
- [Models API](models.md) - Model integration
- [Metrics API](metrics.md) - Evaluation metrics
- [CLI Reference](cli.md) - Command-line interface