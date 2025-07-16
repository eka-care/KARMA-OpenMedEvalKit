# CLI API Reference

This section documents KARMA's command-line interface, including main commands, orchestration, and utility functions.

## Main CLI Entry Point

### main

Main CLI entry point using Click for subcommand organization.

::: karma.cli.main.main
    options:
      show_source: false
      show_root_heading: true

### karma

Root command group for KARMA CLI.

::: karma.cli.main.karma
    options:
      show_source: false
      show_root_heading: true

## Commands

### Evaluation Command

#### eval_cmd

Main evaluation command for running model evaluations.

::: karma.cli.commands.eval.eval_cmd
    options:
      show_source: false
      show_root_heading: true

### List Commands

#### list_models

List all available models in the registry.

::: karma.cli.commands.list
    options:
      show_source: false
      show_root_heading: true

### Info Commands

#### info_commands

Get detailed information about models and datasets.

::: karma.cli.commands.info
    options:
      show_source: false
      show_root_heading: true

## Orchestration

### MultiDatasetOrchestrator

Enhanced orchestrator for multi-dataset evaluation with CLI support.

::: karma.cli.orchestrator.MultiDatasetOrchestrator
    options:
      show_source: false
      show_root_heading: true
      show_category_heading: true
      members_order: source
      filters: ["!^_"]
      merge_init_into_class: true

## Output and Formatting

#### ResultsFormatter

Rich table formatter for evaluation results.

::: karma.cli.formatters.table.ResultsFormatter
    options:
      show_source: false
      show_root_heading: true
      show_category_heading: true
      members_order: source
      filters: ["!^_"]

#### ModelFormatter

Rich table formatter for model information.

::: karma.cli.formatters.table.ModelFormatter
    options:
      show_source: false
      show_root_heading: true
      show_category_heading: true
      members_order: source
      filters: ["!^_"]

#### DatasetFormatter

Rich table formatter for dataset information.

::: karma.cli.formatters.table.DatasetFormatter
    options:
      show_source: false
      show_root_heading: true
      show_category_heading: true
      members_order: source
      filters: ["!^_"]

## Utilities

### CLI Utilities

Utility functions for CLI operations.

::: karma.cli.utils
    options:
      show_source: false
      show_root_heading: true
