# Model Configuration

Learn how to configure and customize models for optimal performance in medical AI evaluation.

## Parameter Tuning

### Generation Parameters
Control model behavior with precision:

```bash
# Conservative generation for medical accuracy
karma eval --model qwen --model-path "Qwen/Qwen3-0.6B" \
  --datasets openlifescienceai/pubmedqa \
  --model-kwargs '{
    "temperature": 0.1,
    "top_p": 0.9,
    "top_k": 50,
    "max_tokens": 512,
    "enable_thinking": true,
    "seed": 42
  }'

# Creative generation for medical education
karma eval --model qwen --model-path "Qwen/Qwen3-0.6B" \
  --datasets medical_education_dataset \
  --model-kwargs '{
    "temperature": 0.7,
    "top_p": 0.95,
    "max_tokens": 1024,
    "enable_thinking": false
  }'
```

### Parameter Reference

| Parameter | Range | Description | Medical Use Case |
|-----------|-------|-------------|------------------|
| `temperature` | 0.0-1.0 | Randomness control | 0.1-0.3 for diagnostic accuracy |
| `top_p` | 0.0-1.0 | Nucleus sampling | 0.9 for balanced responses |
| `top_k` | 1-100 | Top-k sampling | 50 for medical terminology |
| `max_tokens` | 1-4096 | Output length | 512 for concise answers |
| `enable_thinking` | boolean | Reasoning mode | true for complex cases |
| `seed` | integer | Reproducibility | Set for consistent results |




## Model-Specific Configuration

### Qwen Models

```bash
# Thinking mode for complex medical reasoning
karma eval --model qwen --model-path "Qwen/Qwen3-0.6B" \
  --datasets openlifescienceai/pubmedqa \
  --model-kwargs '{
    "enable_thinking": true,
    "thinking_depth": 3,
    "temperature": 0.2,
    "max_tokens": 512
  }'

# Fast inference mode
karma eval --model qwen --model-path "Qwen/Qwen3-0.6B" \
  --datasets openlifescienceai/pubmedqa \
  --model-kwargs '{
    "enable_thinking": false,
    "temperature": 0.1,
    "max_tokens": 256,
    "use_cache": true
  }'
```

### MedGemma Models

```bash
# Medical accuracy optimization
karma eval --model medgemma --model-path "google/medgemma-4b-it" \
  --datasets openlifescienceai/medmcqa \
  --model-kwargs '{
    "temperature": 0.05,
    "top_p": 0.8,
    "repetition_penalty": 1.1,
    "max_tokens": 400,
    "medical_mode": true
  }'
```

### Audio Models

```bash
# IndicConformer language-specific configuration
karma eval --model "ai4bharat/indic-conformer-600m-multilingual" \
  --model-path "ai4bharat/indic-conformer-600m-multilingual" \
  --datasets "ai4bharat/indicvoices_r" \
  --model-kwargs '{
    "language": "Hindi",
    "chunk_length": 30,
    "stride": 5,
    "batch_size": 1,
    "use_lm": true
  }'

# Whisper optimization
karma eval --model openai-whisper \
  --datasets medical_audio_dataset \
  --model-kwargs '{
    "model": "whisper-1",
    "language": "en",
    "temperature": 0.0,
    "condition_on_previous_text": true,
    "compression_ratio_threshold": 2.4
  }'
```

## Next Steps

- **Optimize performance**: See [Performance Optimization](performance-optimization.md)
- **Create custom models**: Check [Custom Models](custom-models.md)
- **Run evaluations**: Go to [Running Evaluations](../basic-usage/running-evaluations.md)
- **Explore built-in models**: Return to [Built-in Models](built-in-models.md)