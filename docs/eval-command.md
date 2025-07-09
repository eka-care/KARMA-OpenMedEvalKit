# Eval command reference

Eval command is one of the central pillars of karma. 
Through this command you can specify which model, the datasets and the arguments to run

See all the options
```
karma eval --help
```

| Option | Type | Description |
|:-------|:-----|:------------|
| <code style="white-space: nowrap">--model</code> | TEXT | Model name from registry (e.g., 'qwen', 'medgemma') [required] |
| <code style="white-space: nowrap">--model-path</code> | TEXT | Model path (local path or HuggingFace model ID). If not provided, uses path from model metadata |
| <code style="white-space: nowrap">--datasets</code> | TEXT | Comma-separated dataset names (default: evaluate on all datasets) |
| <code style="white-space: nowrap">--dataset-args</code> | TEXT | Dataset arguments in format 'dataset:key=val,key2=val2;dataset2:key=val' |
| <code style="white-space: nowrap">--processor-args</code> | TEXT | Processor arguments in format 'dataset.processor:key=val,key2=val2;dataset2.processor:key=val' |
| <code style="white-space: nowrap">--batch-size</code> | INTEGER RANGE | Batch size for evaluation [1<=x<=128] |
| <code style="white-space: nowrap">--cache / --no-cache</code> | FLAG | Enable or disable caching for evaluation |
| <code style="white-space: nowrap">--output</code> | TEXT | Output file path |
| <code style="white-space: nowrap">--format</code> | [table\|json] | Results display format |
| <code style="white-space: nowrap">--save-format</code> | [json\|yaml\|csv] | Results save format |
| <code style="white-space: nowrap">--progress / --no-progress</code> | FLAG | Show progress bars during evaluation |
| <code style="white-space: nowrap">--interactive</code> | FLAG | Interactively prompt for missing dataset arguments |
| <code style="white-space: nowrap">--dry-run</code> | FLAG | Validate arguments and show what would be evaluated without running |
| <code style="white-space: nowrap">--model-config</code> | TEXT | Path to model configuration file (JSON/YAML) with model-specific parameters |
| <code style="white-space: nowrap">--model-kwargs</code> | TEXT | Model parameter overrides as JSON string (e.g., '{"temperature": 0.7, "top_p": 0.9}') |
| <code style="white-space: nowrap">--max-samples</code> | TEXT | Maximum number of samples to use for evaluation |
| <code style="white-space: nowrap">--verbose</code> | TEXT | Pass this argument to have a verbose output |

**Examples:**

#### Evaluate specific datasets
```bash
karma eval --model qwen --model-path "path/to/model" --datasets "pubmedqa,medmcqa"
```

#### With dataset and processor arguments
```bash
karma eval --model qwen --model-path "path" --datasets "in22conv" \
  --dataset-args "in22conv:source_language=en,target_language=hi" \
  --processor-args "in22conv.devnagari_transliterator:source_script=en,target_script=hi"
```