---
title: How CLI Arguments Flow to Model Classes
description: Understanding how initialization arguments pass from the CLI through the registry system to model constructors
---

This guide explains the internal mechanics of how CLI arguments flow through KARMA's architecture to initialize model classes. Understanding this flow is essential for debugging model configuration issues and extending the framework.

## Overview

Arguments flow through four main layers with a clear hierarchy:

```
CLI Command (eval.py)
    ↓
Argument Processing (_prepare_model_overrides)
    ↓  
Model Registry (model_registry.py)
    ↓
Model Class (__init__)
```

## Parameter Precedence Hierarchy

KARMA uses a layered configuration system where each layer can override the previous one:

1. **Model Metadata Defaults** (lowest priority)
2. **CLI Model Path** (if provided)
3. **Config File Parameters** (if provided)
4. **CLI Arguments** (highest priority)

## Detailed Flow

### 1. CLI Layer (`karma/cli/commands/eval.py`)

The evaluation command accepts multiple ways to configure models:

```bash
# Basic usage with model metadata defaults
karma eval --model "gpt-4o"

# Override with CLI arguments
karma eval --model "gpt-4o" --model-kwargs '{"temperature": 0.7, "max_tokens": 1024}'

# Use config file
karma eval --model "gpt-4o" --model-config config.json

# Override model path
karma eval --model "gpt-4o" --model-path "path/to/custom/model"
```

**Key CLI Options:**
- `--model`: Model name from registry (required)
- `--model-path`: Override model path 
- `--model-config`: JSON/YAML config file path
- `--model-args`: JSON string of parameter overrides

**Code Reference:** `karma/cli/commands/eval.py:36-106`

### 2. Argument Processing (`_prepare_model_overrides`)

The `_prepare_model_overrides()` function merges configuration from all sources:

```python
def _prepare_model_overrides(
    model_name: str,
    model_path: str, 
    model_config: str,
    model_kwargs: str,
    console: Console,
) -> dict:
```

**Processing Steps:**

1. **Load Model Metadata Defaults**
   ```python
   model_meta = model_registry.get_model_meta(model_name)
   final_config.update(model_meta.loader_kwargs)
   ```

2. **Apply CLI Model Path**
   ```python
   if model_path:
       final_config["model_name_or_path"] = model_path
   ```

3. **Load Config File**
   ```python
   if model_config:
       config_data = _load_config_file(model_config)
       final_config.update(config_data)
   ```

4. **Apply CLI Overrides**
   ```python
   if model_kwargs:
       cli_overrides = json.loads(model_kwargs)
       final_config.update(cli_overrides)
   ```

**Code Reference:** `karma/cli/commands/eval.py:702-775`

### 3. Model Registry (`karma/registries/model_registry.py`)

The registry handles model instantiation through `_get_model_from_meta()`:

```python
def _get_model_from_meta(self, name: str, **override_kwargs) -> BaseModel:
    model_meta = self.model_metas[name]
    model_class = model_meta.get_loader_class()
    
    # Merge kwargs: defaults < model_meta < overrides  
    final_kwargs = model_meta.merge_kwargs(override_kwargs)
    
    # Ensure model path is set
    final_kwargs["model_name_or_path"] = (
        model_meta.name if model_meta.model_path is None else model_meta.model_path
    )
    
    return model_class(**final_kwargs)
```

**Key Functions:**
- Retrieves model metadata and loader class
- Merges default kwargs with overrides
- Ensures `model_name_or_path` is properly set
- Instantiates the model class with final parameters

**Code Reference:** `karma/registries/model_registry.py:117-139`

### 4. Model Class Instantiation

The model class receives the merged parameters in its `__init__` method:

```python
class OpenAILLM(BaseModel):
    def __init__(
        self,
        model_name_or_path: str = "gpt-4o",
        api_key: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        max_workers: int = 4,
        **kwargs,
    ):
        super().__init__(model_name_or_path=model_name_or_path, **kwargs)
        
        # Set instance variables from parameters
        self.model_id = model_name_or_path
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.max_tokens = max_tokens
        self.temperature = temperature
        # ... other parameters
```

**Code Reference:** `karma/models/openai_llm.py:21-67`

## ModelMeta Configuration

Models define their default parameters using ModelMeta objects:

```python
GPT4o_LLM = ModelMeta(
    name="gpt-4o",
    description="OpenAI GPT-4o language model",
    loader_class="karma.models.openai_llm.OpenAILLM",
    loader_kwargs={
        "model_name_or_path": "gpt-4o",
        "max_tokens": 4096,
        "temperature": 0.0,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
    },
    model_type=ModelType.TEXT_GENERATION,
    modalities=[ModalityType.TEXT],
    # ... other metadata
)
```

These defaults serve as the base configuration layer that can be overridden through the CLI.

**Code Reference:** `karma/models/openai_llm.py:228-247`

## Practical Examples

### Example 1: Using Defaults

```bash
karma eval --model "gpt-4o" --datasets "pubmedqa"
```

**Flow:**
1. CLI passes `model="gpt-4o"`
2. Registry loads GPT4o_LLM metadata
3. Uses default `loader_kwargs`: `temperature=0.0`, `max_tokens=4096`
4. Instantiates `OpenAILLM(model_name_or_path="gpt-4o", temperature=0.0, ...)`

### Example 2: CLI Override

```bash
karma eval --model "gpt-4o" --model-kwargs '{"temperature": 0.7, "max_tokens": 1024}'
```

**Flow:**
1. CLI passes overrides: `temperature=0.7`, `max_tokens=1024`
2. `_prepare_model_overrides()` merges: defaults + CLI overrides
3. Final config: `temperature=0.7`, `max_tokens=1024`, other defaults unchanged
4. Instantiates `OpenAILLM(temperature=0.7, max_tokens=1024, ...)`

### Example 3: Config File + CLI Override

**config.json:**
```json
{
    "temperature": 0.5,
    "max_tokens": 2048,
    "top_p": 0.9
}
```

**CLI:**
```bash
karma eval --model "gpt-4o" --model-config config.json --model-kwargs '{"temperature": 0.7}'
```

**Flow:**
1. Loads defaults from metadata
2. Applies config file: `temperature=0.5`, `max_tokens=2048`, `top_p=0.9`
3. Applies CLI override: `temperature=0.7` (overrides config file)
4. Final: `temperature=0.7`, `max_tokens=2048`, `top_p=0.9`

## Orchestrator Integration

The MultiDatasetOrchestrator receives the final configuration:

```python
orchestrator = MultiDatasetOrchestrator(
    model_name=model,
    model_path=final_model_path,
    model_kwargs=model_overrides,  # The merged configuration
    console=console,
)
```

**Code Reference:** `karma/cli/commands/eval.py:299-304`

## Debugging Tips

### 1. Check Parameter Precedence
If your model isn't using expected parameters, verify the precedence:
- CLI args override everything
- Config file overrides metadata defaults
- Metadata provides base defaults

### 2. Validate JSON Format
CLI model arguments must be valid JSON:
```bash
# ✅ Correct
--model-kwargs '{"temperature": 0.7, "max_tokens": 1024}'

# ❌ Incorrect (single quotes inside)
--model-kwargs '{"temperature": 0.7, "max_tokens": '1024'}'
```

### 3. Model Path Resolution
The `model_name_or_path` parameter is set in this order:
1. CLI `--model-path` (if provided)
2. Config file `model_name_or_path` (if in config)
3. ModelMeta `name` field (fallback)

### 4. Environment Variables
Some models (like OpenAI) use environment variables:
```python
self.api_key = api_key or os.getenv("OPENAI_API_KEY")
```

Make sure required environment variables are set when using models that depend on them.

## Summary

The argument flow system provides flexible model configuration while maintaining clear precedence rules. Understanding this flow helps with:
- Debugging configuration issues
- Creating custom model implementations
- Building configuration management tools
- Extending the framework with new parameter sources

The key insight is that configuration flows through multiple layers, with each layer able to override the previous one, giving users maximum flexibility while providing sensible defaults.