---
title: Datasets Guide
---

This guide covers working with datasets in KARMA, from using built-in datasets to creating your own custom implementations.

## Built-in Datasets

KARMA supports 12+ medical datasets across multiple modalities:

```bash
# List available datasets
karma list datasets

# Get dataset information
karma info dataset openlifescienceai/pubmedqa

# Use a dataset
karma eval --model qwen --model-path "Qwen/Qwen3-0.6B" \
  --datasets openlifescienceai/pubmedqa
```


### Text-based Datasets

- **openlifescienceai/pubmedqa** - PubMed Question Answering
- **openlifescienceai/medmcqa** - Medical Multiple Choice QA
- **openlifescienceai/medqa** - Medical Question Answering
- **ChuGyouk/MedXpertQA** - Medical Expert QA

### Vision-Language Datasets

- **mdwiratathya/SLAKE-vqa-english** - Structured Language And Knowledge Extraction
- **flaviagiammarino/vqa-rad** - Visual Question Answering for Radiology

### Audio Datasets

- **ai4bharat/indicvoices_r** - Text to speech dataset that could be used for ASR evaluation as well.
- **ai4bharat/indicvoices** - ASR dataset - Indic Voices Recognition

### Translation Datasets

- **ai4bharat/IN22-Conv** - Indic Language Conversation Translation

### Rubric-Based Evaluation Datasets

- **ekacare/ekacare_medical_history_summarisation** - Medical History Summarization with rubric evaluation
- **Tonic/Health-Bench-Eval-OSS-2025-07** - Health-Bench evaluation with rubric scoring

These datasets include structured rubric criteria that define evaluation points, scoring weights, and categorization tags. The rubric evaluation is performed by an LLM evaluator (OpenAI or AWS Bedrock) that assesses model responses against multiple criteria simultaneously.

## Viewing Available Datasets

```bash
# List all available datasets
karma list datasets

# Get detailed information about a specific dataset
karma info dataset openlifescienceai/pubmedqa
```

## Using Datasets

```bash
# Use specific dataset
karma eval --model qwen --model-path "Qwen/Qwen3-0.6B" \
  --datasets openlifescienceai/pubmedqa

# Use multiple datasets
karma eval --model qwen --model-path "Qwen/Qwen3-0.6B" \
  --datasets "openlifescienceai/pubmedqa,openlifescienceai/medmcqa"
```

## Dataset Configuration

### Dataset-Specific Arguments

Some datasets require additional configuration:

```bash
# Translation datasets with language pairs
karma eval --model qwen --model-path "Qwen/Qwen3-0.6B" \
    --datasets "ai4bharat/IN22-Conv" \
    --dataset-args "ai4bharat/IN22-Conv:source_language=en,target_language=hi"

# Datasets with specific splits
karma eval --model qwen --model-path "Qwen/Qwen3-0.6B" \
  --datasets "openlifescienceai/medmcqa" \
  --dataset-args "openlifescienceai/medmcqa:split=validation"

# Rubric-based datasets with custom system prompts
karma eval --model qwen --model-path "Qwen/Qwen3-0.6B" \
  --datasets "Tonic/Health-Bench-Eval-OSS-2025-07" \
  --metrics "rubric_evaluation" \
  --dataset-args "Tonic/Health-Bench-Eval-OSS-2025-07:system_prompt=You are a medical expert assistant" \
  --metric-args "rubric_evaluation:provider_to_use=openai,model_id=gpt-4o-mini,batch_size=5"
```
