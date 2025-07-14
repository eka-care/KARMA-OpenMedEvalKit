# Metrics Guide

This guide covers understanding evaluation metrics in KARMA, interpreting results, and creating custom metrics.

## Available Metrics
```bash
# List all available metrics
karma list metrics

# Check which metrics a dataset uses
karma info dataset openlifescienceai/pubmedqa
```

### Text-Based Metrics

- **exact_match**: Percentage of predictions that exactly match the ground truth
- **accuracy**: Overall accuracy (same as exact_match for most datasets)
- **bleu**: BLEU score for text generation tasks

### Speech Recognition Metrics

- **wer**: Word Error Rate for speech recognition tasks
- **cer**: Character Error Rate for speech recognition tasks

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

### Rubric-Based Evaluation Example

Based on healthbench's rubric evaluation criteria, rubrics can be evaluated through rubric_evaluation. 

```bash
# Use rubric evaluation metric with OpenAI
karma eval --model qwen --model-path "Qwen/Qwen3-0.6B" \
  --datasets "Tonic/Health-Bench-Eval-OSS-2025-07" \
  --metric-args "rubric_evaluation:provider_to_use=openai,model_id=gpt-4o-mini,batch_size=5"

# Use with AWS Bedrock
karma eval --model qwen --model-path "Qwen/Qwen3-0.6B" \
  --datasets "ekacare/ekacare_medical_history_summarisation" \
  --metric-args "rubric_evaluation:provider_to_use=bedrock,model_id=anthropic.claude-3-haiku-20240307-v1:0,batch_size=10"
```

The rubric evaluation metric automatically uses the rubric criteria defined in the dataset. For example, Health-Bench datasets contain rubrics like:

```json
{
  "criterion": "States that oral iron supplements can lead to unpleasant gastrointestinal side effects",
  "points": 5,
  "tags": ["level:cluster", "cluster:accuracy", "axis:medical_knowledge"]
}
```

The metric evaluates model responses against these criteria using an LLM evaluator and returns:
- Overall score (0-1 based on achieved points vs total possible points)
- Individual rubric evaluations with explanations
- Tag-based performance breakdowns
- Statistical measures (std dev, bootstrap standard error)

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