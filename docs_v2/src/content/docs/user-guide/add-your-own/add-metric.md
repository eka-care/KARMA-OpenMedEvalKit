---
title: Add metric
---
You can create custom evaluation metrics by inheriting from `BaseMetric`

All you need to do is implement th `evaluate` method in the inherited class. 

```python
from karma.metrics.base_metric_abs import BaseMetric
from karma.registries.metrics_registry import register_metric

@register_metric("medical_accuracy")
class MedicalAccuracyMetric(BaseMetric):
    """Medical-specific accuracy metric with domain weighting."""

    def __init__(self, medical_term_weight=1.5):
        self.medical_term_weight = medical_term_weight
        self.medical_terms = self._load_medical_terms()

    def evaluate(self, predictions, references, **kwargs):
        """Evaluate with medical term weighting."""
        total_score = 0
        total_weight = 0

        for pred, ref in zip(predictions, references):
            # Standard comparison
            is_correct = pred.lower().strip() == ref.lower().strip()

            # Apply weighting for medical terms
            weight = self._get_weight(ref)
            total_weight += weight

            if is_correct:
                total_score += weight

        accuracy = total_score / total_weight if total_weight > 0 else 0.0

        return {
            "medical_accuracy": accuracy,
            "total_examples": len(predictions),
            "total_weight": total_weight
        }

    def _get_weight(self, text):
        """Get weight based on medical content."""
        weight = 1.0
        for term in self.medical_terms:
            if term in text.lower():
                weight = self.medical_term_weight
                break
        return weight

    def _load_medical_terms(self):
        """Load medical terminology."""
        return ["diabetes", "hypertension", "surgery", "medication",
                "diagnosis", "treatment", "symptom", "therapy"]
```

### Using Custom Metrics

Once registered, custom metrics are automatically discovered and need to be specified on the dataset that you want to use.

Let's say you would like to change the openlifescienceai/pubmedqa
Update the @register_dataset in `eval_datasets/pubmedqa.py`
```python
@register_dataset(
    DATASET_NAME,
    commit_hash=COMMIT_HASH,
    split=SPLIT,
    metrics=["exact_match", "medical_accuracy"], # we added the medical accuracy metric to this dataset
    task_type="mcqa",
)
class PubMedMCQADataset(MedQADataset):
...
```

```bash
# The metric will be automatically used if specified in dataset registration
karma eval --model qwen --model-path "Qwen/Qwen3-0.6B" \
  --datasets my_medical_dataset
```
