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
│ google/medgemma-4b-it                       │ ✓ Available │ Text + Vision      │
│ gpt-4o                                      │ ✓ Available │ Text               │
│ gpt-4o-mini                                 │ ✓ Available │ Text               │
│ gpt-3.5-turbo                               │ ✓ Available │ Text               │
│ us.anthropic.claude-3-5-sonnet-20241022-v2:0│ ✓ Available │ Text               │
│ us.anthropic.claude-sonnet-4-20250514-v1:0  │ ✓ Available │ Text               │
│ ai4bharat/indic-conformer-600m-multilingual │ ✓ Available │ Audio              │
│ aws-transcribe                              │ ✓ Available │ Audio              │
│ gpt-4o-transcribe                           │ ✓ Available │ Audio              │
│ gemini-2.0-flash                            │ ✓ Available │ Audio              │
│ gemini-2.5-flash                            │ ✓ Available │ Audio              │
│ eleven_labs                                 │ ✓ Available │ Audio              │
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

**Available Models:**
- **Qwen/Qwen3-0.6B**: Compact 0.6B parameter model
- **Qwen/Qwen3-1.7B**: Larger 1.7B parameter model

### MedGemma models
Google's medical-specialized Gemma models with vision capabilities:

```bash
# MedGemma for specialized medical tasks
karma eval --model "google/medgemma-4b-it" \
  --datasets openlifescienceai/medmcqa \
  --model-kwargs '{"temperature": 0.1, "max_tokens": 512}'

# MedGemma with image analysis
karma eval --model "google/medgemma-4b-it" \
  --datasets medical_image_dataset \
  --model-kwargs '{"temperature": 0.01, "max_tokens": 1024}'
```


### OpenAI models
OpenAI's GPT models for comprehensive text generation:
When invoking OpenAI models, multiprocessing is leveraged to make multiple calls concurrently.

```bash
# GPT-4o for complex medical reasoning
karma eval --model "gpt-4o" \
  --datasets openlifescienceai/pubmedqa \
  --model-kwargs '{"temperature": 0.7, "max_tokens": 1024}'

# GPT-4o Mini for efficient processing
karma eval --model "gpt-4o-mini" \
  --datasets medical_qa_dataset \
  --model-kwargs '{"temperature": 0.3, "max_tokens": 512}'

# GPT-3.5 Turbo for cost-effective inference
karma eval --model "gpt-3.5-turbo" \
  --datasets simple_medical_tasks \
  --model-kwargs '{"temperature": 0.5, "max_tokens": 1024}'
```

**Available Models:**
- **gpt-4o**: Latest GPT-4 Omni model with advanced reasoning
- **gpt-4o-mini**: Compact version of GPT-4o for efficient processing
- **gpt-3.5-turbo**: Cost-effective model for simpler tasks


### Anthropic models via AWS Bedrock
Anthropic's Claude models via AWS Bedrock:
When invoking Bedrock models, multiprocessing is leveraged to make multiple calls concurrently.

```bash
# Claude 3.5 Sonnet for advanced medical reasoning
karma eval --model "us.anthropic.claude-3-5-sonnet-20241022-v2:0" \
  --datasets complex_medical_cases \
  --model-kwargs '{"temperature": 0.7, "max_tokens": 1024}'

# Claude Sonnet 4 for cutting-edge performance
karma eval --model "us.anthropic.claude-sonnet-4-20250514-v1:0" \
  --datasets advanced_medical_reasoning \
  --model-kwargs '{"temperature": 0.3, "max_tokens": 2048}'
```

**Available Models:**
- **us.anthropic.claude-3-5-sonnet-20241022-v2:0**: Claude 3.5 Sonnet v2
- **us.anthropic.claude-sonnet-4-20250514-v1:0**: Latest Claude Sonnet 4

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
- **Open Source**: MIT licensed with open weights

### Cloud ASR Services
Enterprise-grade speech recognition for production deployments:

#### AWS Transcribe
```bash
# AWS Transcribe with automatic language detection
karma eval --model aws-transcribe \
  --datasets medical_audio_dataset \
  --model-kwargs '{"region_name": "us-east-1", "s3_bucket": "your-bucket"}'
```


#### Google Gemini ASR
```bash
# Gemini 2.0 Flash for audio transcription
karma eval --model gemini-2.0-flash \
  --datasets medical_audio_dataset \
  --model-kwargs '{"thinking_budget": 1000}'

# Gemini 2.5 Flash for enhanced performance
karma eval --model gemini-2.5-flash \
  --datasets medical_audio_dataset \
  --model-kwargs '{"thinking_budget": 2000}'
```

**Available Models:**
- **gemini-2.0-flash**: Fast transcription with thinking capabilities
- **gemini-2.5-flash**: Enhanced model with improved accuracy


#### OpenAI Whisper ASR
```bash
# OpenAI Whisper for high-accuracy transcription
karma eval --model gpt-4o-transcribe \
  --datasets medical_audio_dataset \
  --model-kwargs '{"language": "en"}'
```


#### ElevenLabs ASR
```bash
# ElevenLabs for specialized audio processing
karma eval --model eleven_labs \
  --datasets medical_audio_dataset \
  --model-kwargs '{"diarize": false, "tag_audio_events": false}'
```

## Getting Model Information

```bash
# Get detailed information about any model
$ karma info model "Qwen/Qwen3-0.6B"

Model Information: Qwen/Qwen3-0.6B
──────────────────────────────────────────────────
  Model: Qwen/Qwen3-0.6B
 Name    Qwen/Qwen3-0.6B
 Class   QwenThinkingLLM
 Module  karma.models.qwen

Description:
╭─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ Qwen language model with specialized thinking capabilities.                                                                                         │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

Constructor Signature:
  QwenThinkingLLM(self, model_name_or_path: str, device: str = 'mps', max_tokens: int = 32768, temperature: float = 0.7, top_p: float = 0.9, top_k:
Optional = None, enable_thinking: bool = False, **kwargs)

Usage Examples:

Basic evaluation:
  karma eval --model "Qwen/Qwen3-0.6B" --datasets openlifescienceai/pubmedqa

With multiple datasets:
  karma eval --model "Qwen/Qwen3-0.6B" \
    --datasets openlifescienceai/pubmedqa,openlifescienceai/mmlu_professional_medicine

With custom arguments:
  karma eval --model "Qwen/Qwen3-0.6B" \
    --datasets openlifescienceai/pubmedqa \
    --model-args '{"temperature": 0.8, "top_p": 0.85}'
    --max-samples 100 --batch-size 4

Interactive mode:
  karma eval --model "Qwen/Qwen3-0.6B" --interactive

✓ Model information retrieved successfully
```
