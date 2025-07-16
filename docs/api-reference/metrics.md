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

::: karma.metrics.asr.asr_metrics.ASRMetrics
    options:
      show_source: false
      show_root_heading: true
      show_category_heading: true
      members_order: source
      filters: ["!^_"]
      merge_init_into_class: true

::: karma.metrics.asr.asr_semantic_metrics.ASRSemanticMetrics
    options:
      show_source: false
      show_root_heading: true
      show_category_heading: true
      members_order: source
      filters: ["!^_"]
      merge_init_into_class: true

## Rubric-Based Evaluation Metrics

### RubricMetric

LLM-driven rubric evaluation metric for medical question answering with batch processing support.

::: karma.metrics.rubrics.rubric_evaluation.RubricMetric
    options:
      show_source: false
      show_root_heading: true
      show_category_heading: true
      members_order: source
      filters: ["!^_"]
      merge_init_into_class: true
