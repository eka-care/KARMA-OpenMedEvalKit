# Metrics Guide

This guide covers understanding evaluation metrics in KARMA, interpreting results, and creating custom metrics.

## Understanding Results

KARMA outputs comprehensive evaluation results in JSON format:

```json
{
  "model": "qwen",
  "model_path": "Qwen/Qwen3-0.6B",
  "results": {
    "openlifescienceai/pubmedqa": {
      "metrics": {
        "exact_match": 0.745,
        "accuracy": 0.745
      },
      "num_examples": 1000,
      "runtime_seconds": 45.2,
      "cache_hit_rate": 0.8
    },
    "openlifescienceai/medmcqa": {
      "metrics": {
        "exact_match": 0.623,
        "accuracy": 0.623
      },
      "num_examples": 4183,
      "runtime_seconds": 120.5,
      "cache_hit_rate": 0.2
    }
  },
  "total_runtime": 165.7,
  "timestamp": "2025-01-15T10:30:00Z"
}
```

## Available Metrics

### Text-Based Metrics

- **exact_match**: Percentage of predictions that exactly match the ground truth
- **accuracy**: Overall accuracy (same as exact_match for most datasets)
- **bleu**: BLEU score for text generation tasks

### Speech Recognition Metrics

- **wer**: Word Error Rate for speech recognition tasks
- **cer**: Character Error Rate for speech recognition tasks

### Viewing Available Metrics

```bash
# List all available metrics
karma list metrics

# Check which metrics a dataset uses
karma info dataset openlifescienceai/pubmedqa
```

## Metric Interpretation

### Exact Match

- **Range**: 0.0 to 1.0 (higher is better)
- **Meaning**: Percentage of predictions that exactly match the reference
- **Use case**: Multiple choice questions, classification tasks

### BLEU Score

- **Range**: 0.0 to 1.0 (higher is better)
- **Meaning**: Measures overlap between generated and reference text
- **Use case**: Text generation, translation tasks

### Word Error Rate (WER)

- **Range**: 0.0+ (lower is better)
- **Meaning**: Percentage of words that need to be changed to match reference
- **Use case**: Speech recognition, transcription tasks

### Character Error Rate (CER)

- **Range**: 0.0+ (lower is better)
- **Meaning**: Percentage of characters that need to be changed
- **Use case**: Speech recognition, fine-grained text evaluation

## Custom Metrics

### Creating Custom Metrics

You can create custom evaluation metrics by inheriting from `BaseMetric`:

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

Once registered, custom metrics are automatically discovered and used:

```bash
# The metric will be automatically used if specified in dataset registration
karma eval --model qwen --model-path "Qwen/Qwen3-0.6B" \
  --datasets my_medical_dataset
```

## Metric Selection by Task Type

Different task types typically use different metrics:

- **Multiple Choice QA**: exact_match, accuracy
- **Open-ended QA**: exact_match, bleu
- **Visual QA**: exact_match, bleu, accuracy
- **Translation**: bleu, exact_match
- **Speech Recognition**: wer, cer
- **Classification**: accuracy, exact_match

## Advanced Metric Analysis

### Comparing Models

```bash
# Compare different models on same dataset
karma eval --model qwen --model-path "Qwen/Qwen3-0.6B" \
  --datasets openlifescienceai/pubmedqa --output qwen_0.6b.json

karma eval --model qwen --model-path "Qwen/Qwen3-1.7B" \
  --datasets openlifescienceai/pubmedqa --output qwen_1.7b.json

# Analyze results
python -c "
import json
with open('qwen_0.6b.json') as f: results_0_6b = json.load(f)
with open('qwen_1.7b.json') as f: results_1_7b = json.load(f)

score_0_6b = results_0_6b['openlifescienceai/pubmedqa']['metrics']['exact_match']
score_1_7b = results_1_7b['openlifescienceai/pubmedqa']['metrics']['exact_match']

print(f'Qwen 0.6B: {score_0_6b:.3f}')
print(f'Qwen 1.7B: {score_1_7b:.3f}')
print(f'Improvement: {((score_1_7b - score_0_6b) / score_0_6b * 100):.1f}%')
"
```

### Statistical Significance

For robust evaluation, consider:

1. **Multiple runs**: Run evaluations multiple times with different seeds
2. **Confidence intervals**: Calculate statistical confidence
3. **Sample size**: Ensure adequate test set size
4. **Cross-validation**: Use multiple evaluation splits when available

## Best Practices

### Metric Selection

1. **Match task type**: Use appropriate metrics for your task
2. **Domain relevance**: Consider domain-specific metrics for medical tasks
3. **Multiple metrics**: Use complementary metrics for comprehensive evaluation

### Result Interpretation

1. **Consider context**: Compare against relevant baselines
2. **Look at multiple metrics**: Don't rely on a single metric
3. **Analyze errors**: Examine failure cases for insights
4. **Statistical validity**: Ensure results are statistically significant

### Custom Metric Development

1. **Clear definition**: Define what your metric measures
2. **Validation**: Test metric behavior with known examples
3. **Documentation**: Document metric calculation and interpretation
4. **Edge cases**: Handle edge cases gracefully

## Troubleshooting

### Common Issues

- **Missing metrics**: Check if dataset specifies the metric correctly
- **Unexpected scores**: Verify metric calculation and data format
- **Performance issues**: Optimize metric computation for large datasets

### Debug Tips

```bash
# Check available metrics
karma list metrics

# Verify dataset metrics
karma info dataset openlifescienceai/pubmedqa

# Use verbose output to see metric details
karma eval --model qwen --model-path "Qwen/Qwen3-0.6B" \
  --datasets openlifescienceai/pubmedqa --verbose
```

## Next Steps

- **Learn about datasets**: See [Datasets Guide](../datasets/overview.md)
- **Work with models**: Read [Models Guide](../models/overview.md)
- **Advanced configuration**: Check [Configuration Guide](../configuration/environment-setup.md)
- **API reference**: Explore [Metrics API Reference](../../api-reference/metrics.md)