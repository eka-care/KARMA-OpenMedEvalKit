# KARMA: Knowledge Assessment and Reasoning for Medical Applications

<p align="center">
    <em>Karma is a bench</em>
</p>

KARMA-OpenMedEvalKit is a toolkit to evaluate medical application datasets across multiple modalities.
Currently, KARMA supports over 12 datasets spanning text, image and audio modalities.

## Quick Start

Get started with KARMA in minutes:

```bash
# Clone the repository
git clone https://github.com/eka-care/KARMA-OpenMedEvalKit.git
cd KARMA-OpenMedEvalKit

# Install with uv (recommended)
uv sync

# Run your first evaluation on 3 samples on a MCQA task.
karma eval --model "Qwen/Qwen3-0.6B" --datasets openlifescienceai/pubmedqa --max-samples 3
```

### Explore Available Resources

```console
$ karma --help
Usage: karma [OPTIONS] COMMAND [ARGS]...

  Karma - Healthcare AI Model Evaluation Framework

  A comprehensive toolkit for evaluating healthcare AI models across multiple
  India centric datasets with automatic discovery and rich output formatting.

  Examples:
      karma eval --model "Qwen/Qwen3-0.6B" --datasets pubmedqa
      karma list models
      karma info dataset pubmedqa

Options:
  --version      Show the version and exit.
  -v, --verbose  Enable verbose output
  -q, --quiet    Suppress non-essential output
  --help         Show this message and exit.

Commands:
  eval  Evaluate a model on healthcare datasets.
  info  Get detailed information about models, datasets, and system status.
  list  List available models, datasets, and other resources.
```

### Discover Models and Datasets

```console
$ karma list all
╭─────────────────────────╮
│ Karma Registry Overview │
╰─────────────────────────╯

MODELS
────────────────────
                      Available Models                       
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ Model Name                                  ┃ Status      ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ Qwen/Qwen3-0.6B                             │ ✓ Available │
│ Qwen/Qwen3-1.7B                             │ ✓ Available │
│ ai4bharat/indic-conformer-600m-multilingual │ ✓ Available │
│ google/medgemma-4b-it                       │ ✓ Available │
└─────────────────────────────────────────────┴─────────────┘

✓ Found 4 models

DATASETS
────────────────────
Discovering datasets...
                                                                                         Available Datasets                                                                                          
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Dataset                                      ┃ Task Type     ┃ Metrics        ┃ Processors                                          ┃ Required Args                    ┃ Commit Hash ┃ Split      ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ ChuGyouk/MedXpertQA                          │ mcqa          │ exact_match    │ —                                                   │ —                                │ 7186bd59    │ test       │
│ ai4bharat/IN22-Conv                          │ translation   │ bleu           │ devnagari_transliterator                            │ source_language, target_language │ 18cd4587    │ test       │
│ ai4bharat/indicvoices_r                      │ transcription │ bleu, wer, cer │ general_text_processor, multilingual_text_processor │ language                         │ 5f4495c9    │ test       │
│ flaviagiammarino/vqa-rad                     │ vqa           │ exact_match    │ —                                                   │ —                                │ bcf91e76    │ test       │
│ mdwiratathya/SLAKE-vqa-english               │ vqa           │ exact_match    │ —                                                   │ —                                │ 8d18b4d5    │ test       │
│ openlifescienceai/medmcqa                    │ mcqa          │ exact_match    │ —                                                   │ —                                │ 91c6572c    │ validation │
│ openlifescienceai/medqa                      │ mcqa          │ exact_match    │ —                                                   │ —                                │ 153e61cd    │ test       │
│ openlifescienceai/mmlu_anatomy               │ mcqa          │ exact_match    │ —                                                   │ —                                │ a7a792bd    │ test       │
│ openlifescienceai/mmlu_clinical_knowledge    │ mcqa          │ exact_match    │ —                                                   │ —                                │ e1511676    │ test       │
│ openlifescienceai/mmlu_college_biology       │ mcqa          │ exact_match    │ —                                                   │ —                                │ 94b1278b    │ test       │
│ openlifescienceai/mmlu_college_medicine      │ mcqa          │ exact_match    │ —                                                   │ —                                │ 62ba72a3    │ test       │
│ openlifescienceai/mmlu_professional_medicine │ mcqa          │ exact_match    │ —                                                   │ —                                │ 0f2cda02    │ test       │
│ openlifescienceai/pubmedqa                   │ mcqa          │ exact_match    │ —                                                   │ —                                │ 50fc41dc    │ test       │
└──────────────────────────────────────────────┴───────────────┴────────────────┴─────────────────────────────────────────────────────┴──────────────────────────────────┴─────────────┴────────────┘

✓ Found 13 datasets

METRICS
────────────────────
Discovering metrics...


            Available Metrics             
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Metric Name          ┃ Status          ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ bleu                 │ ✓ Available     │
│ cer                  │ ✓ Available     │
│ exact_match          │ ✓ Available     │
│ wer                  │ ✓ Available     │
└──────────────────────┴─────────────────┘

✓ Found 4 metrics
```

### Preview Your Evaluation

```console
$ karma eval --model "Qwen/Qwen3-0.6B" --datasets "openlifescienceai/pubmedqa" --dry-run
╭────────────────────────────────────────────────────────────────────╮
│ KARMA: Knowledge Assessment and Reasoning for Medical Applications │
╰────────────────────────────────────────────────────────────────────╯

Evaluation Plan
──────────────────────────────────────────────────
Model: Qwen/Qwen3-0.6B
Model Path: Qwen/Qwen3-0.6B
Datasets: 1 datasets
  openlifescienceai/pubmedqa
Batch Size: 8
Cache: Enabled
Output File: results.json
Model Configuration:
  temperature: 0.7
  top_k: 50
  top_p: 0.9
  enable_thinking: True
  max_tokens: 256

Dry run completed. No evaluation performed.
```

The CLI provides rich formatting, auto-discovery of models and datasets, and clear feedback - making it easy to get started with medical AI evaluation.

## Architecture Overview

KARMA is built around four core components:

1. **[Models](api-reference/models.md)** - Unified interface for medical AI models
2. **[Datasets](api-reference/datasets.md)** - Standardized medical evaluation datasets
3. **[Metrics](api-reference/metrics.md)** - Comprehensive evaluation metrics
5. **[Processors]** - A way to post process the output of the model 

## What's Next?

- **New to KARMA?** Start with our [Getting Started](getting-started.md) guide
- **Need help with installation?** Check the [Installation Guide](user-guide/installation.md)
- **Want to add custom models?** See the [API Reference](api-reference/models.md)

## License

This project is licensed under the terms of the MIT license.