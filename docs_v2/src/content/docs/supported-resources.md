---
title: Supported Resources
---

> **Note**: This page is auto-generated during the CI/CD pipeline. Last updated: 2025-07-16 19:01:25 UTC

The following resources are currently supported by KARMA:

## Models

Currently supported models (14 total):

| Model Name |
|------------|
| Qwen/Qwen3-0.6B |
| Qwen/Qwen3-1.7B |
| ai4bharat/indic-conformer-600m-multilingual |
| aws-transcribe |
| eleven_labs |
| gemini-2.0-flash |
| gemini-2.5-flash |
| google/medgemma-4b-it |
| gpt-3.5-turbo |
| gpt-4o |
| gpt-4o-mini |
| gpt-4o-transcribe |
| us.anthropic.claude-3-5-sonnet-20241022-v2:0 |
| us.anthropic.claude-sonnet-4-20250514-v1:0 |

Recreate this through
```
karma list models
```
## Datasets

Currently supported datasets (19 total):

| Dataset | Task Type | Metrics | Required Args | Processors | Split |
|---------|-----------|---------|---------------|------------|-------|
| ChuGyouk/MedXpertQA | mcqa | exact_match | — | — | test |
| Tonic/Health-Bench-Eval-OSS-2025-07 | rubric_evaluation | rubric_evaluation | — | — | oss_eval |
| ai4bharat/IN22-Conv | translation | bleu | source_language, target_language | devnagari_transliterator | test |
| ai4bharat/IndicVoices | transcription | wer, cer, asr_semantic_metric | language | multilingual_text_processor | valid |
| ai4bharat/indicvoices_r | transcription | asr_semantic_metric | language | multilingual_text_processor | test |
| ekacare/MedMCQA-Indic | mcqa | exact_match | subset | — | test |
| ekacare/clinical_note_generation_dataset | text_to_json_rubric_evaluation | json_rubric_evaluation | — | — | test |
| ekacare/ekacare_medical_history_summarisation | rubric_evaluation | rubric_evaluation | — | — | test |
| ekacare/medical_records_parsing_validation_set | multimodal_rubric_evaluation | json_rubric_evaluation | — | — | test |
| flaviagiammarino/vqa-rad | vqa | exact_match, tokenised_f1 | — | — | test |
| mdwiratathya/SLAKE-vqa-english | vqa | exact_match, tokenised_f1 | — | — | test |
| openlifescienceai/medmcqa | mcqa | exact_match | — | — | validation |
| openlifescienceai/medqa | mcqa | exact_match | — | — | test |
| openlifescienceai/mmlu_anatomy | mcqa | exact_match | — | — | test |
| openlifescienceai/mmlu_clinical_knowledge | mcqa | exact_match | — | — | test |
| openlifescienceai/mmlu_college_biology | mcqa | exact_match | — | — | test |
| openlifescienceai/mmlu_college_medicine | mcqa | exact_match | — | — | test |
| openlifescienceai/mmlu_professional_medicine | mcqa | exact_match | — | — | test |
| openlifescienceai/pubmedqa | mcqa | exact_match | — | — | test |

Recreate this through
```
karma list datasets
```
## Metrics

Currently supported metrics (10 total):

| Metric Name |
|-------------|
| asr_metric |
| asr_semantic_metric |
| bleu |
| cer |
| exact_match |
| f1 |
| json_rubric_evaluation |
| rubric_evaluation |
| tokenised_f1 |
| wer |

Recreate this through
```
karma list metrics
```

## Quick Reference

Use the following commands to explore available resources:

```bash
# List all models
karma list models

# List all datasets
karma list datasets

# List all metrics
karma list metrics

# List all processors
karma list processors

# Get detailed information about a specific resource
karma info model "Qwen/Qwen3-0.6B"
karma info dataset "openlifescienceai/pubmedqa"
```

## Adding New Resources

To add new models, datasets, or metrics to KARMA:

- See [Adding Models](/user-guide/add-your-own/add-model)
- See [Adding Datasets](/user-guide/add-your-own/add-dataset.mdx)
- See [Metrics Overview](/user-guide/metrics/metrics_overview)

For more detailed information about the registry system, see the [Registry Documentation](/user-guide/registry/registries).
