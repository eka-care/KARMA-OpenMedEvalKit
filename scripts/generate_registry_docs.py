#!/usr/bin/env python3
"""
Generate documentation for KARMA registry (models, datasets, metrics).
This script runs 'karma list' commands with CSV format for better data processing.
"""

import subprocess
import sys
import csv
import io
from pathlib import Path
from datetime import datetime
import os


def run_karma_list_csv(resource_type):
    """Run 'karma list' command with CSV format and capture output."""
    try:
        # Ensure we're in the right directory
        os.chdir(Path(__file__).parent.parent)

        # Run karma list command with CSV format
        result = subprocess.run(
            ["karma", "list", resource_type, "--format", "csv"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running karma list {resource_type}: {e}")
        print(f"Stderr: {e.stderr}")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: karma command not found. Make sure it's installed and in PATH.")
        sys.exit(1)


def parse_csv_data(csv_text):
    """Parse CSV text into list of dictionaries."""
    # Find the actual CSV content by looking for the header line
    lines = csv_text.strip().split("\n")
    csv_start = -1

    # Look for the first line that starts with common CSV headers
    for i, line in enumerate(lines):
        if line.startswith("name,") or line.startswith("name"):
            csv_start = i
            break

    if csv_start == -1:
        print("Warning: Could not find CSV header in output")
        return []

    # Extract only the CSV portion
    csv_content = "\n".join(lines[csv_start:])

    reader = csv.DictReader(io.StringIO(csv_content))
    return list(reader)


def format_models_table(models_data):
    """Format models data as markdown table."""
    if not models_data:
        return "No models found."

    table = ["| Model Name |", "|------------|"]
    for model in models_data:
        table.append(f"| {model['name']} |")

    return "\n".join(table)


def format_datasets_table(datasets_data):
    """Format datasets data as markdown table."""
    if not datasets_data:
        return "No datasets found."

    table = [
        "| Dataset | Task Type | Metrics | Required Args | Processors | Split |",
        "|---------|-----------|---------|---------------|------------|-------|",
    ]

    for dataset in datasets_data:
        # Parse JSON fields for better display
        import json

        try:
            metrics = json.loads(dataset["metrics"])
            required_args = json.loads(dataset["required_args"])
            processors = (
                json.loads(dataset["processors"])
                if dataset["processors"] != "null"
                else []
            )
        except (json.JSONDecodeError, KeyError):
            metrics = []
            required_args = []
            processors = []

        metrics_str = ", ".join(metrics) if metrics else "—"
        required_args_str = ", ".join(required_args) if required_args else "—"
        processors_str = ", ".join(processors) if processors else "—"

        table.append(
            f"| {dataset['name']} | {dataset['task_type']} | {metrics_str} | {required_args_str} | {processors_str} | {dataset['split']} |"
        )

    return "\n".join(table)


def format_metrics_table(metrics_data):
    """Format metrics data as markdown table."""
    if not metrics_data:
        return "No metrics found."

    table = ["| Metric Name |", "|-------------|"]
    for metric in metrics_data:
        table.append(f"| {metric['name']} |")

    return "\n".join(table)


def format_as_markdown(models_data, datasets_data, metrics_data):
    """Format the registry data as markdown."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

    models_table = format_models_table(models_data)
    datasets_table = format_datasets_table(datasets_data)
    metrics_table = format_metrics_table(metrics_data)

    markdown_content = f"""# Supported Resources

> **Note**: This page is auto-generated during the CI/CD pipeline. Last updated: {timestamp}

The following resources are currently supported by KARMA:

## Models

Currently supported models ({len(models_data)} total):

{models_table}

Recreate this through
```
karma list models
```
## Datasets

Currently supported datasets ({len(datasets_data)} total):

{datasets_table}

Recreate this through
```
karma list datasets
```
## Metrics

Currently supported metrics ({len(metrics_data)} total):

{metrics_table}

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

- **Models**: See [Adding Models](user-guide/add-your-own/add-model.md)
- **Datasets**: See [Adding Datasets](user-guide/add-your-own/add-dataset.md)
- **Metrics**: See [Metrics Overview](user-guide/metrics/metrics_overview.md)

For more detailed information about the registry system, see the [Registry Documentation](user-guide/registry/registries.md).
"""

    return markdown_content


def main():
    """Main function to generate registry documentation."""
    print("Generating registry documentation...")

    # Get CSV data for each resource type
    print("Fetching models data...")
    models_csv = run_karma_list_csv("models")
    models_data = parse_csv_data(models_csv)

    print("Fetching datasets data...")
    datasets_csv = run_karma_list_csv("datasets")
    datasets_data = parse_csv_data(datasets_csv)

    print("Fetching metrics data...")
    metrics_csv = run_karma_list_csv("metrics")
    metrics_data = parse_csv_data(metrics_csv)

    # Format as markdown
    markdown_content = format_as_markdown(models_data, datasets_data, metrics_data)

    # Write to docs directory
    docs_dir = Path(__file__).parent.parent / "docs"
    output_file = docs_dir / "supported-resources.md"

    # Create docs directory if it doesn't exist
    docs_dir.mkdir(exist_ok=True)

    # Write the markdown file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    print(f"Successfully generated documentation at {output_file}")


if __name__ == "__main__":
    main()
