# Registries API Reference

This section documents KARMA's registry system for automatic discovery and management of models, datasets, metrics, and processors.

## Model Registry

### ModelRegistry

Enhanced model registry supporting ModelMeta configurations with automatic model discovery.

::: karma.registries.model_registry.ModelRegistry
    options:
      show_source: false
      show_root_heading: true
      show_category_heading: true
      members_order: source
      filters: ["!^_"]

### Registration Functions

#### register_model_meta

Register a model using ModelMeta configuration.

::: karma.registries.model_registry.register_model_meta
    options:
      show_source: false
      show_root_heading: true

#### get_model

Retrieve and instantiate a registered model.

::: karma.registries.model_registry.get_model
    options:
      show_source: false
      show_root_heading: true


## Dataset Registry

### DatasetRegistry

Decorator-based registry for datasets with metrics and metadata validation.

::: karma.registries.dataset_registry.DatasetRegistry
    options:
      show_source: false
      show_root_heading: true
      show_category_heading: true
      members_order: source
      filters: ["!^_"]

### Registration Functions

#### register_dataset

Decorator for registering datasets.

::: karma.registries.dataset_registry.register_dataset
    options:
      show_source: false
      show_root_heading: true



## Metric Registry

### MetricRegistry

Decorator-based metric registry for automatic metric discovery.

::: karma.registries.metrics_registry.MetricRegistry
    options:
      show_source: false
      show_root_heading: true
      show_category_heading: true
      members_order: source
      filters: ["!^_"]

### Registration Functions

#### register_metric

Decorator for registering metrics.

::: karma.registries.metrics_registry.register_metric
    options:
      show_source: false
      show_root_heading: true



## Processor Registry

### ProcessorRegistry

Registry for text processors and preprocessing components.

::: karma.registries.processor_registry.ProcessorRegistry
    options:
      show_source: false
      show_root_heading: true
      show_category_heading: true
      members_order: source
      filters: ["!^_"]

### Registration Functions

#### register_processor

Decorator for registering processors.

::: karma.registries.processor_registry.register_processor
    options:
      show_source: false
      show_root_heading: true