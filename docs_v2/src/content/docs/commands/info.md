---
title: karma info
description: Complete reference for the karma info commands
---

The `karma info` command group provides detailed information about models, datasets, and system status.

## Usage

```bash
karma info [COMMAND] [OPTIONS] [ARGUMENTS]
```

## Subcommands

- `karma info model <name>` - Get detailed information about a specific model
- `karma info dataset <name>` - Get detailed information about a specific dataset
- `karma info system` - Get system information and status

---

## karma info model

Get detailed information about a specific model including its class details, module location, and implementation info.

### Usage
```bash
karma info model MODEL_NAME [OPTIONS]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `MODEL_NAME` | Name of the model to get information about **[required]** |

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--show-code` | FLAG | false | Show model class code location and basic info |

### Examples

```bash
# Basic model information
karma info model "Qwen/Qwen3-0.6B"

# Show code location details
karma info model "google/medgemma-4b-it" --show-code

# Check model that might not exist
karma info model "unknown-model"
```

### Output

```bash
$ karma info model "Qwen/Qwen3-0.6B" --show-code 

╭────────────────────────────────────────────────────────────────────╮
│ KARMA: Knowledge Assessment and Reasoning for Medical Applications │
╰────────────────────────────────────────────────────────────────────╯
Model Information: Qwen/Qwen3-0.6B
──────────────────────────────────────────────────
  Model: Qwen/Qwen3-0.6B   
 Name    Qwen/Qwen3-0.6B   
 Class   QwenThinkingLLM   
 Module  karma.models.qwen 

Description:
╭──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ Qwen language model with specialized thinking capabilities.                                                                                                                  │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

Code Location:
  File location not available

Constructor Signature:
  QwenThinkingLLM(self, model_name_or_path: str, device: str = 'mps', max_tokens: int = 32768, temperature: float = 0.7, top_p: float = 0.9, top_k: Optional = None, 
enable_thinking: bool = False, **kwargs)

Usage Examples:

Basic evaluation:
  karma eval --model "Qwen/Qwen3-0.6B" --datasets openlifescienceai/pubmedqa

With multiple datasets:
  karma eval --model "Qwen/Qwen3-0.6B" \
    --datasets openlifescienceai/pubmedqa,openlifescienceai/mmlu_professional_medicine

With custom arguments:
  karma eval --model "Qwen/Qwen3-0.6B" \
    --datasets openlifescienceai/pubmedqa \
    --max-samples 100 --batch-size 4

Interactive mode:
  karma eval --model "Qwen/Qwen3-0.6B" --interactive

✓ Model information retrieved successfully
```


---

## karma info dataset

Get detailed information about a specific dataset including its requirements, supported metrics, and usage examples.

### Usage
```bash
karma info dataset DATASET_NAME [OPTIONS]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `DATASET_NAME` | Name of the dataset to get information about **[required]** |

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--show-examples` | FLAG | false | Show usage examples with arguments |
| `--show-code` | FLAG | false | Show dataset class code location |

### Examples

```bash
# Basic dataset information
karma info dataset openlifescienceai/pubmedqa

# Show usage examples
karma info dataset "ai4bharat/IN22-Conv" --show-examples

# Show code location
karma info dataset "mdwiratathya/SLAKE-vqa-english" --show-code

# Get info for dataset with required args
karma info dataset "ekacare/MedMCQA-Indic" --show-examples
```

### Output
```bash
karma info dataset "ai4bharat/IN22-Conv" --show-examples

╭────────────────────────────────────────────────────────────────────╮
│ KARMA: Knowledge Assessment and Reasoning for Medical Applications │
╰────────────────────────────────────────────────────────────────────╯
[13:13:57] INFO     Imported model module: karma.models.aws_bedrock                                                                                        model_registry.py:235
           INFO     Imported model module: karma.models.aws_transcribe_asr                                                                                 model_registry.py:235
[13:13:58] INFO     Imported model module: karma.models.base_hf_llm                                                                                        model_registry.py:235
           INFO     Imported model module: karma.models.docassist_chat                                                                                     model_registry.py:235
           INFO     Imported model module: karma.models.eleven_labs                                                                                        model_registry.py:235
[13:13:59] INFO     Imported model module: karma.models.gemini_asr                                                                                         model_registry.py:235
           INFO     Imported model module: karma.models.indic_conformer                                                                                    model_registry.py:235
           INFO     Imported model module: karma.models.medgemma                                                                                           model_registry.py:235
           INFO     Imported model module: karma.models.openai_asr                                                                                         model_registry.py:235
           INFO     Imported model module: karma.models.openai_llm                                                                                         model_registry.py:235
           INFO     Imported model module: karma.models.qwen                                                                                               model_registry.py:235
           INFO     Imported model module: karma.models.whisper                                                                                            model_registry.py:235
           INFO     Registry discovery completed: 4/4 successful, 1 cache hits, total time: 1.36s                                                         registry_manager.py:70

Dataset Information: ai4bharat/IN22-Conv
──────────────────────────────────────────────────
                Dataset: ai4bharat/IN22-Conv                 
 Name           ai4bharat/IN22-Conv                          
 Class          IN22ConvDataset                              
 Module         karma.eval_datasets.in22conv_dataset         
 Task Type      translation                                  
 Metrics        bleu                                         
 Processors     devnagari_transliterator                     
 Required Args  source_language, target_language             
 Optional Args  domain, processors, confinement_instructions 
 Default Args   source_language=en, domain=conversational    

Description:
╭──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ IN22Conv PyTorch Dataset implementing the new multimodal interface.                                                                                                          │
│ Translates from English to specified Indian language.                                                                                                                        │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

Usage Examples:

With required arguments:
  karma eval --model "Qwen/Qwen3-0.6B"  \
    --datasets ai4bharat/IN22-Conv \
    --dataset-args "ai4bharat/IN22-Conv:source_language=en,target_language=hi"

With optional arguments:
  karma eval --model "Qwen/Qwen3-0.6B" \
    --datasets ai4bharat/IN22-Conv \
    --dataset-args "ai4bharat/IN22-Conv:source_language=en,target_language=hi,domain=conversational,processors=<optional_value>,confinement_instructions=<optional_value>"

Interactive mode (prompts for arguments):
  karma eval --model "Qwen/Qwen3-0.6B" \
    --datasets ai4bharat/IN22-Conv --interactive

✓ Dataset information retrieved successfully
```

## karma info system

Get system information and status including available resources, cache status, and environment details.

### Usage
```bash
karma info system [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--cache-path TEXT` | TEXT | ./cache.db | Path to cache database to check |

### Examples

```bash
# Basic system information
karma info system

# Check specific cache location
karma info system --cache-path /path/to/cache.db

# Check system status
karma info system --cache-path ~/.karma/cache.db
```

### Output

```bash
karma info system
╭────────────────────────────────────────────────────────────────────╮
│ KARMA: Knowledge Assessment and Reasoning for Medical Applications │
╰────────────────────────────────────────────────────────────────────╯
Discovering system resources...
[13:14:43] INFO     Imported model module: karma.models.aws_bedrock                                                                                        model_registry.py:235
           INFO     Imported model module: karma.models.aws_transcribe_asr                                                                                 model_registry.py:235
           INFO     Imported model module: karma.models.base_hf_llm                                                                                        model_registry.py:235
           INFO     Imported model module: karma.models.docassist_chat                                                                                     model_registry.py:235
           INFO     Imported model module: karma.models.eleven_labs                                                                                        model_registry.py:235
[13:14:44] INFO     Imported model module: karma.models.gemini_asr                                                                                         model_registry.py:235
           INFO     Imported model module: karma.models.indic_conformer                                                                                    model_registry.py:235
           INFO     Imported model module: karma.models.medgemma                                                                                           model_registry.py:235
           INFO     Imported model module: karma.models.openai_asr                                                                                         model_registry.py:235
           INFO     Imported model module: karma.models.openai_llm                                                                                         model_registry.py:235
           INFO     Imported model module: karma.models.qwen                                                                                               model_registry.py:235
           INFO     Imported model module: karma.models.whisper                                                                                            model_registry.py:235
           INFO     Registry discovery completed: 4/4 successful, 1 cache hits, total time: 1.24s                                                         registry_manager.py:70

System Information
──────────────────────────────────────────────────
            System Information            
 Available Models    21                   
 Available Datasets  21                   
 Cache Database      ✓ Available (5.0 MB) 
 Cache Path          cache.db             

Environment:
  Python: 3.10.15
  Platform: macOS-15.5-arm64-arm-64bit
  Architecture: arm64
  Karma CLI: development

Dependencies:
  ✓ PyTorch: 2.7.1
  ✓ Transformers: 4.53.0
  ✓ HuggingFace Datasets: 3.6.0
  ✓ Rich: unknown
  ✓ Click: 8.2.1
  ✓ Weave: 0.51.54
  ✓ DuckDB: 1.3.1

Usage Examples:

List available resources:
  karma list models
  karma list datasets

Get detailed information:
  karma info model "Qwen/Qwen3-0.6B"
  karma info dataset openlifescienceai/pubmedqa

Run evaluation:
  karma eval --model "Qwen/Qwen3-0.6B" --datasets openlifescienceai/pubmedqa

Check cache status:
  karma info system --cache-path ./cache.db

✓ System information retrieved successfully
```

## Common Usage Patterns

### Model Discovery and Validation
```bash
# 1. List available models
karma list models

# 2. Get detailed info about a specific model
karma info model "Qwen/Qwen3-0.6B"

# 3. Check model implementation
karma info model "Qwen/Qwen3-0.6B" --show-code
```

### Dataset Analysis
```bash
# 1. Find datasets for a task
karma list datasets --task-type mcqa

# 2. Get detailed dataset info
karma info dataset "openlifescienceai/medmcqa"

# 3. See usage examples with arguments
karma info dataset "ai4bharat/IN22-Conv" --show-examples
```

### System Debugging
```bash
# Check overall system status
karma info system

# Verify dependencies
karma info system --cache-path ~/.karma/cache.db

# Check cache status
karma info system --cache-path ./evaluation_cache.db
```

### Development Workflow
```bash
# Quick resource check
karma info model "new-model-name"
karma info dataset "new-dataset-name" --show-code

# System health check
karma info system
```

## Error Handling

### Model Not Found
```bash
$ karma info model "nonexistent-model"
Error: Model 'nonexistent-model' not found in registry
Available models: Qwen/Qwen3-0.6B, google/medgemma-4b-it, ...
```

### Dataset Not Found
```bash
$ karma info dataset "nonexistent-dataset"  
Error: Dataset 'nonexistent-dataset' not found in registry
Available datasets: openlifescienceai/pubmedqa, openlifescienceai/medmcqa, ...
```

### Invalid Cache Path
```bash
$ karma info system --cache-path /invalid/path/cache.db
Cache Status: Path not accessible
```