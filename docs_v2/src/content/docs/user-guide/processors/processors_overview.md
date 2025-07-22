---
title: Processors Guide
---
Processors run on the output of the model and used to perform some normalisation or similar operations before computing the metrics.
They are registered in the dataset along with the metrics.
Processors output is piped from the previous processor to the next.

## Quick Start

```bash
# Use processor with evaluation
karma eval --model "ai4bharat/indic-conformer-600m-multilingual" \
  --datasets "ai4bharat/IN22-Conv" \
  --processor-args "ai4bharat/IN22-Conv.devnagari_transliterator:source_script=en,target_script=hi"
```


## Architecture

The processor system consists of:

- **Base Processor**: `BaseProcessor` class that all processors inherit from
- **Processor Registry**: Auto-discovery system that finds and registers processors
- **Integration Points**: Processors can be applied at dataset level or via CLI

Processors are defined with the datasets in the decorator.
The processors are by default chained i.e., the output of the previous processor is the input of the next processor.

## Available Processors

**GeneralTextProcessor**

- Handles common text normalization
- Number to text conversion
- Punctuation removal
- Case normalization

**DevanagariTransliterator**

- Multilingual text processing for indic Devanagri scripts
- Script conversion between languages
- Handles Devanagari text

**MultilingualTextProcessor**

- Audio transcription normalization
- Specialized for STT tasks where numbers need to be normalized
