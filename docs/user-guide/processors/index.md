# Processors

Learn how to work with text processors in KARMA, from using built-in processors to creating your own custom implementations.

## Quick Start

```bash
# List available processors
karma list processors

# Use processor with evaluation
karma eval --model qwen --model-path "Qwen/Qwen3-0.6B" \
  --datasets "ai4bharat/IN22-Conv" \
  --processor-args "ai4bharat/IN22-Conv.devnagari_transliterator:source_script=en,target_script=hi"
```

## Guide

**[Processors Overview](overview.md)** - Complete guide to working with processors, including built-in processors, integration patterns, and creating custom processors.

## Key Topics Covered

- **Available Processors** - Built-in text processing capabilities
- **Using Existing Processors** - How to apply processors via CLI and programmatically
- **Creating Custom Processors** - Building domain-specific text processors
- **Integration Patterns** - Dataset integration and CLI usage
- **Advanced Use Cases** - Chaining processors and conditional processing

## Built-in Processors

- **GeneralTextProcessor** - Common text normalization
- **DevanagariTransliterator** - Indic script conversion
- **MultilingualTextProcessor** - Audio transcription normalization

## Next Steps

Once you understand processors, explore these related topics:

- **[Datasets](../datasets/overview.md)** - Apply processors to datasets
- **[Models](../models/overview.md)** - Understand model text requirements
- **[Running Evaluations](../basic-usage/running-evaluations.md)** - Use processors in evaluations