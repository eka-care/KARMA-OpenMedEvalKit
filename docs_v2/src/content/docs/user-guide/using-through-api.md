---
title: Using KARMA as a package
---
KARMA provides both a CLI interface and a Python API for programmatic use. This guide walks you through building an evaluation pipeline using the API.

## Overview

The KARMA API centers around the `Benchmark` class, which coordinates models, datasets, metrics, and caching. Here's how to build a complete evaluation pipeline.

Let's work with an example that uses all the core components of KARMA: Models, Datasets, Metrics, and Processors.

Here we are trying to evaluate `IndicVoicesRDataset`, an ASR dataset for evaluating speech recognition models.
We will be using the `IndicConformerASR` model and the `WERMetric` and `CERMetric` metrics.
Before passing to the metrics, the model's output will be passed to the processors, which will perform text normalization and tokenization.

## Essential Imports

Start with the core components:

```python
import sys
import os

# Core KARMA components
from karma.benchmark import Benchmark
from karma.cache.cache_manager import CacheManager

# Model components
from karma.models.indic_conformer import IndicConformerASR, INDIC_CONFORMER_MULTILINGUAL_META

# Dataset components
from karma.eval_datasets.indicvoices_r_dataset import IndicVoicesRDataset

# Metrics components
from karma.metrics.common_metrics import WERMetric, CERMetric

# Processing components
from karma.processors.multilingual_text_processor import MultilingualTextProcessor
```

Here's what each import does:

- `Benchmark`: Orchestrates the entire evaluation process
- `CacheManager`: Caches model predictions to avoid redundant computations
- `IndicConformerASR`: An Indic language speech recognition model
- `INDIC_CONFORMER_MULTILINGUAL_META`: Model metadata for caching
- `IndicVoicesRDataset`: Speech recognition dataset for evaluation
- `WERMetric`/`CERMetric`: Word and character error rate metrics
- `MultilingualTextProcessor`: Normalizes text for consistent comparison

## Complete Example

Here's a working example that evaluates a speech recognition model:

```python
def main():
    # Initialize the model
    print("Initializing model...")
    model = IndicConformerASR(model_name_or_path="ai4bharat/indic-conformer-600m-multilingual")

    # Set up text processing
    processor = MultilingualTextProcessor()

    # Create the dataset
    print("Loading dataset...")
    dataset = IndicVoicesRDataset(
        language="Hindi",
        postprocessors=[processor]
    )

    # Configure metrics
    metrics = [
             WERMetric(metric_name="wer")
             CERMetric(metric_name="cer"),
    ]

    # Set up caching
    cache_manager = CacheManager(
        model_config=INDIC_CONFORMER_MULTILINGUAL_META,
        dataset_name=dataset.dataset_name
    )

    # Create and run benchmark
    benchmark = Benchmark(
        model=model,
        dataset=dataset,
        cache_manager=cache_manager
    )

    print("Running evaluation...")
    results = benchmark.evaluate(
        metrics=metrics,
        batch_size=1
    )

    # Display results
    print(f"Word Error Rate (WER): {results['overall_score']['wer']:.4f}")
    print(f"Character Error Rate (CER): {results['overall_score']['cer']:.4f}")

    return results

if __name__ == "__main__":
    main()
```

## Understanding the Flow

When you run this code, here's what happens:

1. **Model Initialization**: Creates an instance of the speech recognition model and loads pretrained weights
2. **Text Processing**: Sets up text normalization to ensure fair comparison between predictions and ground truth
3. **Dataset Creation**: Loads Hindi speech samples with their transcriptions and applies text processing
4. **Metrics Configuration**: Defines WER (word-level errors) and CER (character-level errors) metrics
5. **Cache Setup**: Creates a cache manager to store predictions and avoid recomputation
6. **Evaluation**: The benchmark iterates through samples, runs inference, and computes metrics

## Advanced Usage

### Batch Processing

```python
# Process multiple samples at once for better performance
results = benchmark.evaluate(
    metrics=metrics,
    batch_size=8,
    max_samples=100
)
```

### Custom Metrics

```python
from karma.metrics.base_metric import BaseMetric

class CustomAccuracyMetric(BaseMetric):
    def __init__(self, metric_name="custom_accuracy"):
        super().__init__(metric_name)

    def evaluate(self, predictions, references, **kwargs):
        correct = sum(1 for p, r in zip(predictions, references) if p.strip() == r.strip())
        return correct / len(predictions)

metrics = [CustomAccuracyMetric()]
```

### Multiple Languages

```python
languages = ["Hindi", "Telugu", "Tamil"]
results_by_language = {}

for language in languages:
    dataset = IndicVoicesRDataset(language=language, postprocessors=[processor])
    benchmark = Benchmark(model=model, dataset=dataset, cache_manager=cache_manager)
    results_by_language[language] = benchmark.evaluate(metrics = metrics)
```

### Multiple Datasets
The user is responsible for creating the multiple dataset objects while using multiple datasets.

```python
# Both these datasets are for ASR
dataset_1 = IndicVoicesRDataset(language=language, postprocessors=[processor])
dataset_2 = IndicVoicesDataset(language=language, postprocessors=[processor])
dataset_results = []
for i in [dataset_1, dataset_2]:
    benchmark = Benchmark(model=model, dataset=dataset, cache_manager=cache_manager)
    dataset_results[i.name] = benchmark.evaluate(metrics=metrics)
```


### Progress Tracking

```python
from rich.progress import Progress

with Progress() as progress:
    benchmark = Benchmark(
        model=model,
        dataset=dataset,
        cache_manager=cache_manager,
        progress=progress
    )
    results = benchmark.evaluate(metrics=metrics, batch_size=1)
```

This API gives you complete control over your evaluation pipeline while maintaining KARMA's performance optimizations and robustness.
