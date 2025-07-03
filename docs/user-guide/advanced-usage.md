# Advanced Usage

This guide covers advanced features and usage patterns for power users and researchers.

## Custom Model Integration

### Creating Custom Models

Extend KARMA with your own models by inheriting from `BaseHFModel`:

```python
from karma.models.base_model_abs import BaseHFModel
from karma.data_models.model_meta import ModelMeta, ModelType, ModalityType
from karma.registries.model_registry import register_model_meta

class MyCustomModel(BaseHFModel):
    """Custom model implementation."""
    
    def __init__(self, model_name_or_path: str, **kwargs):
        super().__init__(model_name_or_path=model_name_or_path, **kwargs)
        # Custom initialization
    
    def preprocess(self, prompt: str, **kwargs) -> str:
        # Custom preprocessing
        return f"[INST] {prompt} [/INST]"
    
    def postprocess(self, response: str, **kwargs) -> str:
        # Custom postprocessing
        return response.strip()

# Register the model
custom_model_meta = ModelMeta(
    name="my_custom_model",
    description="My custom medical AI model",
    loader_class="path.to.MyCustomModel",
    loader_kwargs={"temperature": 0.7},
    model_type=ModelType.TEXT_GENERATION,
    modalities=[ModalityType.TEXT],
    framework=["PyTorch", "Transformers"],
)
register_model_meta(custom_model_meta)
```

### Using Custom Models

```bash
# Use your custom model
karma eval --model my_custom_model --model-path "path/to/model" --datasets pubmedqa
```

## Custom Dataset Integration

### Creating Custom Datasets

Add new datasets by inheriting from `BaseMultimodalDataset`:

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

### Using Custom Datasets

```bash
# Use your custom dataset
karma eval --model qwen --model-path "Qwen/Qwen3-0.6B" \
  --datasets "my_medical_dataset" \
  --dataset-args "my_medical_dataset:split=validation"
```

## Advanced Caching Strategies

### Cache Configuration

KARMA supports multiple cache backends:

#### DuckDB (Local Development)

```bash
# .env file
KARMA_CACHE_TYPE=duckdb
KARMA_CACHE_PATH=./cache.db
```

#### DynamoDB (Production)

```bash
# .env file
KARMA_CACHE_TYPE=dynamodb
AWS_REGION=us-east-1
DYNAMODB_TABLE_NAME=karma-cache
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
```

### Cache Management

```python
from karma.cache.cache_manager import CacheManager

# Initialize cache manager
cache_manager = CacheManager()

# Clear specific cache entries
cache_manager.clear_cache(model_name="qwen", dataset_name="pubmedqa")

# Get cache statistics
stats = cache_manager.get_cache_stats()
print(f"Cache hit rate: {stats['hit_rate']:.2%}")
```

## Batch Processing and Parallel Evaluation

### Multi-GPU Evaluation

```bash
# Specify GPU device
CUDA_VISIBLE_DEVICES=0,1 karma eval --model qwen --model-path "Qwen/Qwen3-0.6B"

# Use specific GPU
karma eval --model qwen --model-path "Qwen/Qwen3-0.6B" \
  --model-kwargs '{"device":"cuda:1"}'
```

### Distributed Evaluation

```python
from karma.cli.orchestrator import MultiDatasetOrchestrator
from karma.registries.model_registry import get_model
from karma.registries.dataset_registry import get_dataset

# Initialize orchestrator
orchestrator = MultiDatasetOrchestrator()

# Run distributed evaluation
results = orchestrator.evaluate_all_datasets(
    model_name="qwen",
    model_path="Qwen/Qwen3-0.6B",
    datasets=["pubmedqa", "medmcqa", "medqa"],
    batch_size=16,
    num_workers=4
)
```

## Custom Metrics

### Creating Custom Metrics

```python
from karma.metrics.base_metric_abs import BaseMetric
from karma.registries.metrics_registry import register_metric

@register_metric("custom_accuracy")
class CustomAccuracyMetric(BaseMetric):
    """Custom accuracy metric with special handling."""
    
    def evaluate(self, predictions, ground_truths, **kwargs):
        # Custom evaluation logic
        correct = 0
        total = len(predictions)
        
        for pred, gt in zip(predictions, ground_truths):
            if self.custom_match(pred, gt):
                correct += 1
        
        return {"custom_accuracy": correct / total}
    
    def custom_match(self, prediction, ground_truth):
        # Custom matching logic
        return prediction.lower().strip() == ground_truth.lower().strip()
```

### Using Custom Metrics

```bash
# The metric will be automatically discovered and used
karma eval --model qwen --model-path "Qwen/Qwen3-0.6B" --datasets pubmedqa
```

## Advanced Configuration

### Environment Variables

```bash
# Model configuration
HUGGINGFACE_TOKEN=your_token
OPENAI_API_KEY=your_openai_key

# Cache configuration
KARMA_CACHE_TYPE=duckdb
KARMA_CACHE_PATH=./cache.db
KARMA_CACHE_TTL=604800  # 7 days in seconds

# Performance tuning
KARMA_DEFAULT_BATCH_SIZE=8
KARMA_MAX_WORKERS=4
KARMA_MEMORY_LIMIT=16GB

# Logging
LOG_LEVEL=INFO
LOG_FILE=karma.log
```

### Configuration Files

Create a `karma.yaml` configuration file:

```yaml
models:
  qwen:
    default_kwargs:
      temperature: 0.7
      max_tokens: 512
      top_p: 0.9
    
datasets:
  pubmedqa:
    default_args:
      split: test
    custom_metrics:
      - exact_match
      - custom_accuracy

cache:
  type: duckdb
  path: ./cache.db
  ttl: 604800

evaluation:
  default_batch_size: 8
  timeout: 300
  max_retries: 3
```

## Monitoring and Logging

### Detailed Logging

```python
import logging
from karma.benchmark import Benchmark

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('karma.log'),
        logging.StreamHandler()
    ]
)

# Run evaluation with detailed logging
benchmark = Benchmark(
    model_name="qwen",
    model_path="Qwen/Qwen3-0.6B",
    use_cache=True,
    verbose=True
)

results = benchmark.evaluate(dataset_name="pubmedqa")
```

### Performance Monitoring

```python
from karma.benchmark import Benchmark
import time

class MonitoredBenchmark(Benchmark):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_time = None
        self.metrics = {}
    
    def evaluate(self, dataset_name, **kwargs):
        self.start_time = time.time()
        
        # Monitor memory usage
        import psutil
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        results = super().evaluate(dataset_name, **kwargs)
        
        # Record metrics
        self.metrics.update({
            "evaluation_time": time.time() - self.start_time,
            "memory_usage_mb": process.memory_info().rss / 1024 / 1024,
            "memory_increase_mb": process.memory_info().rss / 1024 / 1024 - initial_memory
        })
        
        return results
```

## Integration with External Tools

### Weights & Biases Integration

```python
import wandb
from karma.benchmark import Benchmark

# Initialize wandb
wandb.init(project="karma-medical-eval")

# Run evaluation
benchmark = Benchmark(
    model_name="qwen",
    model_path="Qwen/Qwen3-0.6B"
)

results = benchmark.evaluate(dataset_name="pubmedqa")

# Log results
wandb.log({
    "accuracy": results["metrics"]["accuracy"],
    "exact_match": results["metrics"]["exact_match"],
    "runtime": results["runtime_seconds"]
})
```

### MLflow Integration

```python
import mlflow
from karma.benchmark import Benchmark

# Start MLflow run
with mlflow.start_run():
    # Log parameters
    mlflow.log_param("model_name", "qwen")
    mlflow.log_param("model_path", "Qwen/Qwen3-0.6B")
    mlflow.log_param("dataset", "pubmedqa")
    
    # Run evaluation
    benchmark = Benchmark(
        model_name="qwen",
        model_path="Qwen/Qwen3-0.6B"
    )
    
    results = benchmark.evaluate(dataset_name="pubmedqa")
    
    # Log metrics
    mlflow.log_metric("accuracy", results["metrics"]["accuracy"])
    mlflow.log_metric("runtime", results["runtime_seconds"])
```

## Troubleshooting Advanced Issues

### Memory Management

```python
# Clear model cache between evaluations
import gc
import torch

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Use in evaluation loops
for model_path in model_paths:
    # Run evaluation
    clear_memory()
```

### Error Recovery

```python
from karma.benchmark import Benchmark
import time

class RobustBenchmark(Benchmark):
    def __init__(self, *args, max_retries=3, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_retries = max_retries
    
    def evaluate(self, dataset_name, **kwargs):
        for attempt in range(self.max_retries):
            try:
                return super().evaluate(dataset_name, **kwargs)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                print(f"Attempt {attempt + 1} failed: {e}")
                time.sleep(2 ** attempt)  # Exponential backoff
```

## Next Steps

- **API Reference**: Explore the complete [API Reference](../api-reference/models.md)
- **Contributing**: Help improve KARMA by reading our [Contributing Guide](../contributing.md)
- **Examples**: Check out the `examples/` directory for more advanced usage patterns
- **Community**: Join our discussions on [GitHub Discussions](https://github.com/eka-care/KARMA-OpenMedEvalKit/discussions)