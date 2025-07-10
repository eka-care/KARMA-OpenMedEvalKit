# Caching

KARMA uses intelligent caching to speed up repeated evaluations, the cache is created based on the input configuration.


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
