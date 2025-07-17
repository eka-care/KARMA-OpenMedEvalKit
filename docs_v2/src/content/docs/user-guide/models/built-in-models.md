---
title: Built-in Models
---

KARMA includes several pre-configured models optimized for medical AI evaluation across different modalities.

## Available Models Overview

```bash
# List all available models
karma list models

# Expected output:
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┓
┃ Model Name                                  ┃ Status      ┃ Modality           ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━┩
│ Qwen/Qwen3-0.6B                             │ ✓ Available │ Text               │
│ Qwen/Qwen3-1.7B                             │ ✓ Available │ Text               │
│ google/medgemma-4b-it                       │ ✓ Available │ Text               │
│ ai4bharat/indic-conformer-600m-multilingual │ ✓ Available │ Audio              │
│ aws-transcribe                              │ ✓ Available │ Audio              │
│ openai-whisper                              │ ✓ Available │ Audio              │
│ gemini-asr                                  │ ✓ Available │ Audio              │
└─────────────────────────────────────────────┴─────────────┴────────────────────┘
```

## Text Generation Models

### Qwen Models
Alibaba's Qwen models with specialized thinking capabilities for medical reasoning:

```bash
# Get detailed model information
karma info model "Qwen/Qwen3-0.6B"

# Basic usage
karma eval --model "Qwen/Qwen3-0.6B" \
  --datasets openlifescienceai/pubmedqa

# Advanced configuration with thinking mode
karma eval --model "Qwen/Qwen3-0.6B" \
  --datasets openlifescienceai/pubmedqa \
  --model-kwargs '{"enable_thinking": true, "temperature": 0.3}'
```

**Key Features:**
- **Thinking Mode**: Explicit reasoning steps for complex medical problems
- **Multiple Sizes**: 0.6B, 1.7B parameters for different hardware requirements
- **Medical Optimization**: Fine-tuned performance on healthcare tasks
- **Multilingual**: Support for multiple languages including medical terminology

### MedGemma Models
Google's medical-specialized Gemma models:

```bash
# MedGemma for specialized medical tasks
karma eval --model "google/medgemma-4b-it" \
  --datasets openlifescienceai/medmcqa \
  --model-kwargs '{"temperature": 0.1, "max_tokens": 512}'
```

**Key Features:**
- **Medical Specialization**: Pre-trained on medical literature and guidelines
- **Instruction Tuning**: Optimized for following medical instructions
- **Safety Features**: Built-in safety guardrails for medical applications
- **Research Grade**: Suitable for academic and clinical research

## Audio Recognition Models

### IndicConformer ASR
AI4Bharat's Conformer model for Indian languages:

```bash
# Indian language speech recognition
karma eval \
  --model "ai4bharat/indic-conformer-600m-multilingual" \
  --datasets "ai4bharat/indicvoices_r" \
  --batch-size 1 \
  --dataset-args "ai4bharat/indicvoices_r:language=Hindi" \
  --processor-args "ai4bharat/indicvoices_r.general_text_processor:language=Hindi"
```

**Key Features:**
- **22 Indian Languages**: Complete coverage of constitutional languages
- **Medical Audio**: Optimized for healthcare speech recognition
- **Conformer Architecture**: State-of-the-art speech recognition architecture
- **Regional Dialects**: Handles diverse Indian language variations

### Cloud ASR Services
Enterprise-grade speech recognition for production deployments:

```bash
# AWS Transcribe
karma eval --model aws-transcribe \
  --datasets medical_audio_dataset \
  --model-kwargs '{"language_code": "en-US", "medical_vocabulary": true}'

# Google Cloud Speech
karma eval --model gemini-asr \
  --datasets medical_audio_dataset \
  --model-kwargs '{"language": "en-US", "model": "medical_dictation"}'

# OpenAI Whisper
karma eval --model openai-whisper \
  --datasets medical_audio_dataset \
  --model-kwargs '{"model": "whisper-1", "language": "en"}'
```

## Getting Model Information

```bash
# Get detailed information about any model
karma info model "Qwen/Qwen3-0.6B"

# Example output shows:
# - Model capabilities
# - Required parameters
# - Hardware requirements
# - Usage examples
# - Available configurations
```
