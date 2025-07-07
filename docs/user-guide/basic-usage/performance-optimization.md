# Performance Optimization

This guide covers how to optimize KARMA evaluations for better performance and resource usage.

## Batch Processing

Adjust batch size based on your hardware capabilities:

```bash
# Adjust batch size for your hardware
karma eval --model qwen --model-path "Qwen/Qwen3-0.6B" --batch-size 8

# Smaller batch for limited memory
karma eval --model qwen --model-path "Qwen/Qwen3-0.6B" --batch-size 2

# Larger batch for high-end hardware
karma eval --model qwen --model-path "Qwen/Qwen3-0.6B" --batch-size 16
```

### Choosing the Right Batch Size

- **Small batch (1-2)**: Limited memory, slower GPUs
- **Medium batch (4-8)**: Standard setups, balanced performance
- **Large batch (16+)**: High-end GPUs with plenty of VRAM

## Caching

KARMA uses intelligent caching to speed up repeated evaluations:

```bash
# Use cache (default)
karma eval --model qwen --model-path "Qwen/Qwen3-0.6B" --cache

# Force fresh evaluation
karma eval --model qwen --model-path "Qwen/Qwen3-0.6B" --no-cache

# Clear cache before evaluation
karma eval --model qwen --model-path "Qwen/Qwen3-0.6B" --clear-cache
```

### Cache Benefits

- **Faster re-runs**: Avoid recomputing identical evaluations
- **Resume interrupted evaluations**: Pick up where you left off
- **Efficient experimentation**: Test different parameters without re-evaluating unchanged parts

## Memory Management

### Model Memory Optimization

```bash
# Use smaller models for initial testing
karma eval --model qwen --model-path "Qwen/Qwen3-0.6B" --datasets openlifescienceai/pubmedqa

# Reduce context length for memory savings
karma eval --model qwen --model-path "Qwen/Qwen3-0.6B" \
  --model-kwargs '{"max_tokens":256}'
```

### Dataset Optimization

```bash
# Limit number of samples for testing
karma eval --model qwen --model-path "Qwen/Qwen3-0.6B" \
  --datasets openlifescienceai/pubmedqa --max-samples 100

# Process smaller datasets first
karma eval --model qwen --model-path "Qwen/Qwen3-0.6B" \
  --datasets "openlifescienceai/pubmedqa"  # Smaller dataset
```

## Performance Monitoring

### Dry Run

Test your configuration without running actual evaluation:

```bash
karma eval --model qwen --model-path "Qwen/Qwen3-0.6B" \
  --datasets openlifescienceai/pubmedqa --dry-run
```

### Verbose Output

Monitor detailed progress and performance metrics:

```bash
karma eval --model qwen --model-path "Qwen/Qwen3-0.6B" \
  --datasets openlifescienceai/pubmedqa --verbose
```

## Hardware Considerations

### GPU Optimization

- **CUDA Memory**: Monitor GPU memory usage with `nvidia-smi`
- **Batch Size**: Increase gradually until you hit memory limits
- **Model Precision**: Consider using half-precision if supported

### CPU Optimization

- **Worker Processes**: KARMA automatically manages parallel processing
- **Memory Limits**: Set reasonable batch sizes to avoid system memory issues

## Next Steps

- **Advanced caching**: Learn about [Caching Strategies](../configuration/caching-strategies.md)
- **Configuration**: Read [Environment Setup](../configuration/environment-setup.md)