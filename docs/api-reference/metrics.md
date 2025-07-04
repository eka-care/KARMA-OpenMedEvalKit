# Metrics API Reference

This section documents KARMA's metrics system, including base classes, built-in metrics, and custom metric integration.

## Base Classes

### BaseMetric

Abstract base class for all evaluation metrics.

::: karma.metrics.base_metric_abs.BaseMetric
    options:
      show_source: false
      show_root_heading: true
      show_category_heading: true
      members_order: source
      filters: ["!^_"]

## Common Metrics

### HfMetric

Base wrapper for HuggingFace Evaluate metrics.

::: karma.metrics.common_metrics.HfMetric
    options:
      show_source: false
      show_root_heading: true
      show_category_heading: true
      members_order: source
      filters: ["!^_"]
      merge_init_into_class: true

### ExactMatchMetric

Exact string matching metric for evaluation.

::: karma.metrics.common_metrics.ExactMatchMetric
    options:
      show_source: false
      show_root_heading: true
      show_category_heading: true
      members_order: source
      filters: ["!^_"]
      merge_init_into_class: true

### BleuMetric

BLEU score metric for text generation evaluation.

::: karma.metrics.common_metrics.BleuMetric
    options:
      show_source: false
      show_root_heading: true
      show_category_heading: true
      members_order: source
      filters: ["!^_"]
      merge_init_into_class: true


## Audio/Speech Metrics

### WERMetric

Word Error Rate metric for speech recognition evaluation.

::: karma.metrics.common_metrics.WERMetric
    options:
      show_source: false
      show_root_heading: true
      show_category_heading: true
      members_order: source
      filters: ["!^_"]
      merge_init_into_class: true

### CERMetric

Character Error Rate metric for speech recognition evaluation.

::: karma.metrics.common_metrics.CERMetric
    options:
      show_source: false
      show_root_heading: true
      show_category_heading: true
      members_order: source
      filters: ["!^_"]
      merge_init_into_class: true

## ASR-Specific Metrics

### ASRMetrics

Comprehensive ASR evaluation metrics including WER, CER, and language-specific handling.

::: karma.metrics.asr_metrics.ASRMetrics
    options:
      show_source: false
      show_root_heading: true
      show_category_heading: true
      members_order: source
      filters: ["!^_"]
      merge_init_into_class: true


## Usage Examples

### Basic Metric Usage

```python
from karma.metrics.common_metrics import ExactMatchMetric, BleuMetric

# Initialize metrics
exact_match = ExactMatchMetric()
bleu = BleuMetric()

# Evaluate predictions
predictions = ["The patient has diabetes", "Surgery is recommended"]
references = ["The patient has diabetes mellitus", "Surgical intervention is advised"]

# Compute exact match
em_score = exact_match.evaluate(predictions, references)
print(f"Exact Match: {em_score['exact_match']:.3f}")

# Compute BLEU score
bleu_score = bleu.evaluate(predictions, references)
print(f"BLEU: {bleu_score['bleu']:.3f}")
```

### Custom Metric Implementation

```python
from karma.metrics.base_metric_abs import BaseMetric
from karma.registries.metrics_registry import register_metric
import re

@register_metric("medical_terminology_accuracy")
class MedicalTerminologyAccuracy(BaseMetric):
    """Metric that focuses on medical terminology accuracy."""
    
    def __init__(self, medical_terms_file=None):
        self.medical_terms = self._load_medical_terms(medical_terms_file)
    
    def evaluate(self, predictions, references, **kwargs):
        """Evaluate medical terminology accuracy."""
        correct_terms = 0
        total_terms = 0
        
        for pred, ref in zip(predictions, references):
            pred_terms = self._extract_medical_terms(pred)
            ref_terms = self._extract_medical_terms(ref)
            
            for term in ref_terms:
                total_terms += 1
                if term in pred_terms:
                    correct_terms += 1
        
        accuracy = correct_terms / total_terms if total_terms > 0 else 0.0
        
        return {
            "medical_terminology_accuracy": accuracy,
            "correct_terms": correct_terms,
            "total_terms": total_terms
        }
    
    def _extract_medical_terms(self, text):
        """Extract medical terms from text."""
        words = re.findall(r'\b\w+\b', text.lower())
        return [word for word in words if word in self.medical_terms]
    
    def _load_medical_terms(self, file_path):
        """Load medical terms from file."""
        if file_path:
            with open(file_path, 'r') as f:
                return set(line.strip().lower() for line in f)
        else:
            # Default medical terms
            return {
                'diabetes', 'hypertension', 'pneumonia', 'surgery',
                'medication', 'diagnosis', 'treatment', 'symptom'
            }
```

### ASR Evaluation Example

```python
from karma.metrics.asr_metrics import ASRMetrics

# Initialize ASR metrics
asr_metrics = ASRMetrics(language="en")

# Audio transcription predictions and ground truth
predictions = [
    "the patient has chest pain",
    "blood pressure is one forty over ninety"
]
references = [
    "the patient has chest pain",
    "blood pressure is 140 over 90"
]

# Evaluate ASR performance
results = asr_metrics.evaluate(predictions, references)

print(f"Word Error Rate: {results['wer']:.3f}")
print(f"Character Error Rate: {results['cer']:.3f}")
print(f"Substitutions: {results['substitutions']}")
print(f"Deletions: {results['deletions']}")
print(f"Insertions: {results['insertions']}")
```

### Multi-Language ASR Evaluation

```python
from karma.metrics.asr_metrics import ASRMetrics

# Hindi ASR evaluation
hindi_metrics = ASRMetrics(language="hi")

predictions_hi = ["मरीज़ को सीने में दर्द है"]
references_hi = ["मरीज को सीने में दर्द है"]

results_hi = hindi_metrics.evaluate(predictions_hi, references_hi)
print(f"Hindi WER: {results_hi['wer']:.3f}")

# English ASR evaluation
english_metrics = ASRMetrics(language="en")

predictions_en = ["the patient has chest pain"]
references_en = ["the patient has chest pain"]

results_en = english_metrics.evaluate(predictions_en, references_en)
print(f"English WER: {results_en['wer']:.3f}")
```

### Batch Evaluation

```python
from karma.metrics.common_metrics import ExactMatchMetric, BleuMetric, AccuracyMetric

class BatchEvaluator:
    """Utility class for batch evaluation with multiple metrics."""
    
    def __init__(self):
        self.metrics = {
            'exact_match': ExactMatchMetric(),
            'bleu': BleuMetric(),
            'accuracy': AccuracyMetric()
        }
    
    def evaluate_all(self, predictions, references):
        """Evaluate with all metrics."""
        results = {}
        
        for metric_name, metric in self.metrics.items():
            try:
                metric_results = metric.evaluate(predictions, references)
                results.update(metric_results)
            except Exception as e:
                print(f"Error evaluating {metric_name}: {e}")
                results[metric_name] = None
        
        return results

# Usage
evaluator = BatchEvaluator()
predictions = ["The diagnosis is pneumonia", "Treatment includes antibiotics"]
references = ["The diagnosis is pneumonia", "Treatment includes antibiotics"]

all_results = evaluator.evaluate_all(predictions, references)
for metric, score in all_results.items():
    print(f"{metric}: {score}")
```

### Domain-Specific Evaluation

```python
from karma.metrics.base_metric_abs import BaseMetric
import difflib

class MedicalSemanticSimilarity(BaseMetric):
    """Semantic similarity metric for medical text."""
    
    def __init__(self, similarity_threshold=0.8):
        self.threshold = similarity_threshold
        self.medical_synonyms = {
            'medication': ['drug', 'medicine', 'pharmaceutical'],
            'doctor': ['physician', 'clinician', 'practitioner'],
            'illness': ['disease', 'condition', 'disorder']
        }
    
    def evaluate(self, predictions, references, **kwargs):
        """Evaluate semantic similarity for medical text."""
        similarities = []
        
        for pred, ref in zip(predictions, references):
            similarity = self._compute_medical_similarity(pred, ref)
            similarities.append(similarity)
        
        avg_similarity = sum(similarities) / len(similarities)
        high_similarity_count = sum(1 for s in similarities if s >= self.threshold)
        
        return {
            'medical_semantic_similarity': avg_similarity,
            'high_similarity_ratio': high_similarity_count / len(similarities),
            'similarities': similarities
        }
    
    def _compute_medical_similarity(self, text1, text2):
        """Compute medical-aware semantic similarity."""
        # Normalize medical terms
        text1_norm = self._normalize_medical_terms(text1.lower())
        text2_norm = self._normalize_medical_terms(text2.lower())
        
        # Use difflib for sequence similarity
        similarity = difflib.SequenceMatcher(None, text1_norm, text2_norm).ratio()
        
        return similarity
    
    def _normalize_medical_terms(self, text):
        """Normalize medical terminology."""
        for canonical, synonyms in self.medical_synonyms.items():
            for synonym in synonyms:
                text = text.replace(synonym, canonical)
        return text
```

## Best Practices

### Metric Selection

```python
def get_metrics_for_task(task_type):
    """Get appropriate metrics for different task types."""
    
    metric_mapping = {
        'mcqa': ['exact_match', 'accuracy'],
        'qa': ['exact_match', 'bleu'],
        'vqa': ['exact_match', 'bleu', 'accuracy'],
        'translation': ['bleu', 'exact_match'],
        'asr': ['wer', 'cer'],
        'classification': ['accuracy', 'exact_match']
    }
    
    return metric_mapping.get(task_type, ['exact_match'])

# Usage
task_metrics = get_metrics_for_task('mcqa')
print(f"Metrics for MCQA: {task_metrics}")
```

### Error Handling

```python
from karma.metrics.common_metrics import ExactMatchMetric
import logging

logger = logging.getLogger(__name__)

def safe_evaluate(metric, predictions, references):
    """Safely evaluate with error handling."""
    try:
        return metric.evaluate(predictions, references)
    except Exception as e:
        logger.error(f"Metric evaluation failed: {e}")
        return {"error": str(e)}

# Usage
metric = ExactMatchMetric()
predictions = ["Answer 1", "Answer 2"]
references = ["Answer 1", "Different Answer"]

results = safe_evaluate(metric, predictions, references)
print(results)
```

### Metric Aggregation

```python
class MetricAggregator:
    """Aggregate results from multiple metrics."""
    
    def __init__(self, metrics):
        self.metrics = metrics
    
    def evaluate(self, predictions, references):
        """Evaluate with all metrics and aggregate results."""
        all_results = {}
        individual_results = {}
        
        for metric_name, metric in self.metrics.items():
            try:
                results = metric.evaluate(predictions, references)
                individual_results[metric_name] = results
                all_results.update(results)
            except Exception as e:
                print(f"Failed to evaluate {metric_name}: {e}")
        
        # Compute aggregate statistics
        numeric_metrics = {k: v for k, v in all_results.items() 
                          if isinstance(v, (int, float))}
        
        if numeric_metrics:
            all_results['average_score'] = sum(numeric_metrics.values()) / len(numeric_metrics)
            all_results['max_score'] = max(numeric_metrics.values())
            all_results['min_score'] = min(numeric_metrics.values())
        
        all_results['individual_results'] = individual_results
        
        return all_results
```

## Supported Metric Types

### Text-Based Metrics
- **exact_match**: Exact string matching
- **bleu**: BLEU score for text generation
- **accuracy**: Classification accuracy

### Speech Recognition Metrics
- **wer**: Word Error Rate
- **cer**: Character Error Rate

### Custom Medical Metrics
- **medical_terminology_accuracy**: Medical term precision
- **medical_semantic_similarity**: Semantic similarity for medical text

## See Also

- [Registries API](registries.md) - Metric registry and discovery
- [Models API](models.md) - Model integration
- [Datasets API](datasets.md) - Dataset integration
- [CLI Reference](cli.md) - Command-line interface