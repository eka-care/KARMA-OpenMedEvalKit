# Contributing to KARMA

We welcome contributions to KARMA! This guide will help you get started with contributing to the project.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Adding New Components](#adding-new-components)
- [Testing](#testing)
- [Documentation](#documentation)
- [Code Review Process](#code-review-process)
- [Community](#community)

## Getting Started

### Prerequisites

- Python 3.12 or higher
- Git
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- Basic knowledge of PyTorch and HuggingFace Transformers

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:

```bash
git clone https://github.com/your-username/KARMA-OpenMedEvalKit.git
cd KARMA-OpenMedEvalKit
```

3. Add the upstream repository:

```bash
git remote add upstream https://github.com/eka-care/KARMA-OpenMedEvalKit.git
```

## Development Setup

### Install Development Dependencies

```bash
# Install with all development dependencies
uv install --group dev --group docs --group audio

# Or with pip
pip install -e ".[audio]"
pip install mkdocs mkdocs-material mkdocstrings[python] mkdocs-llmstxt
```

### Pre-commit Hooks

We use pre-commit hooks to ensure code quality:

```bash
# Install pre-commit
pip install pre-commit

# Install the hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

### Environment Setup

Create a `.env` file for development:

```bash
# Development configuration
KARMA_CACHE_TYPE=duckdb
KARMA_CACHE_PATH=./dev_cache.db
LOG_LEVEL=DEBUG

# Add your tokens
HUGGINGFACE_TOKEN=your_token_here
OPENAI_API_KEY=your_openai_key  # Optional
```

## Contributing Guidelines

### Code Style

We use `ruff` for code formatting and linting:

```bash
# Format code
ruff format .

# Check for linting issues
ruff check .

# Fix auto-fixable issues
ruff check . --fix
```

### Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
feat: add support for new medical dataset
fix: resolve memory leak in model loading
docs: update installation instructions
test: add tests for metric calculations
refactor: improve registry architecture
```

### Branch Naming

Use descriptive branch names:

```
feature/add-medical-llama-support
fix/memory-leak-in-caching
docs/improve-api-documentation
test/add-integration-tests
```

## Adding New Components

### Adding a New Model

1. Create your model class by inheriting from `BaseModel`:

```python
# karma/models/my_model.py
from karma.models.base_model_abs import BaseModel
from karma.data_models.model_meta import ModelMeta, ModelType, ModalityType
from karma.registries.model_registry import register_model_meta

class MyMedicalModel(BaseModel):
    """My custom medical model."""
    
    def __init__(self, model_name_or_path: str, **kwargs):
        super().__init__(model_name_or_path=model_name_or_path, **kwargs)
    
    def preprocess(self, prompt: str, **kwargs) -> str:
        # Custom preprocessing
        return f"Medical Query: {prompt}\nResponse:"
    
    def postprocess(self, response: str, **kwargs) -> str:
        # Custom postprocessing
        return response.strip()

# Register the model
my_model_meta = ModelMeta(
    name="my_medical_model",
    description="Custom medical model for specialized tasks",
    loader_class="karma.models.my_model.MyMedicalModel",
    loader_kwargs={"temperature": 0.7},
    model_type=ModelType.TEXT_GENERATION,
    modalities=[ModalityType.TEXT],
    framework=["PyTorch", "Transformers"],
)
register_model_meta(my_model_meta)
```

2. Add tests:

```python
# tests/test_my_model.py
import pytest
from karma.models.my_model import MyMedicalModel
from karma.registries.model_registry import get_model

def test_my_model_initialization():
    model = MyMedicalModel("test-model-path")
    assert model.model_name_or_path == "test-model-path"

def test_my_model_preprocessing():
    model = MyMedicalModel("test-model-path")
    prompt = "What is diabetes?"
    processed = model.preprocess(prompt)
    assert "Medical Query:" in processed

def test_model_registry_integration():
    model = get_model("my_medical_model", "test-model-path")
    assert isinstance(model, MyMedicalModel)
```

### Adding a New Dataset

1. Create your dataset class:

```python
# karma/eval_datasets/my_dataset.py
from karma.eval_datasets.base_dataset import BaseMultimodalDataset
from karma.registries.dataset_registry import register_dataset

@register_dataset(
    "my_medical_dataset",
    metrics=["exact_match", "accuracy"],
    task_type="mcqa",
    required_args=["domain"],
    optional_args=["split", "subset"],
    default_args={"split": "test"}
)
class MyMedicalDataset(BaseMultimodalDataset):
    """Custom medical dataset."""
    
    def __init__(self, domain: str, split: str = "test", **kwargs):
        self.domain = domain
        self.split = split
        super().__init__(**kwargs)
    
    def load_data(self):
        # Load your dataset
        return self._load_custom_data()
    
    def format_item(self, item):
        return {
            "prompt": f"Domain: {self.domain}\n{item['question']}",
            "ground_truth": item["answer"],
            "options": item.get("choices", [])
        }
```

2. Add tests:

```python
# tests/test_my_dataset.py
import pytest
from karma.eval_datasets.my_dataset import MyMedicalDataset
from karma.registries.dataset_registry import create_dataset

def test_dataset_initialization():
    dataset = MyMedicalDataset(domain="cardiology")
    assert dataset.domain == "cardiology"
    assert dataset.split == "test"

def test_dataset_registry_integration():
    dataset = create_dataset("my_medical_dataset", domain="cardiology")
    assert isinstance(dataset, MyMedicalDataset)
```

### Adding a New Metric

1. Create your metric class:

```python
# karma/metrics/my_metric.py
from karma.metrics.base_metric_abs import BaseMetric
from karma.registries.metrics_registry import register_metric

@register_metric("medical_accuracy")
class MedicalAccuracyMetric(BaseMetric):
    """Medical-specific accuracy metric."""
    
    def evaluate(self, predictions, references, **kwargs):
        # Custom evaluation logic
        correct = 0
        total = len(predictions)
        
        for pred, ref in zip(predictions, references):
            if self._medical_match(pred, ref):
                correct += 1
        
        return {"medical_accuracy": correct / total}
    
    def _medical_match(self, prediction, reference):
        # Custom matching logic for medical text
        return prediction.lower().strip() == reference.lower().strip()
```

2. Add tests:

```python
# tests/test_my_metric.py
import pytest
from karma.metrics.my_metric import MedicalAccuracyMetric

def test_medical_accuracy_perfect_match():
    metric = MedicalAccuracyMetric()
    predictions = ["diabetes", "hypertension"]
    references = ["diabetes", "hypertension"]
    
    result = metric.evaluate(predictions, references)
    assert result["medical_accuracy"] == 1.0

def test_medical_accuracy_no_match():
    metric = MedicalAccuracyMetric()
    predictions = ["diabetes", "hypertension"]
    references = ["cancer", "pneumonia"]
    
    result = metric.evaluate(predictions, references)
    assert result["medical_accuracy"] == 0.0
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_my_model.py

# Run with coverage
pytest --cov=karma --cov-report=html

# Run integration tests
pytest tests/integration/
```

### Test Structure

```
tests/
├── unit/                    # Unit tests
│   ├── test_models.py
│   ├── test_datasets.py
│   └── test_metrics.py
├── integration/            # Integration tests
│   ├── test_benchmark.py
│   └── test_cli.py
├── fixtures/               # Test fixtures
│   ├── sample_data.json
│   └── mock_models.py
└── conftest.py            # Pytest configuration
```

### Writing Good Tests

```python
import pytest
from unittest.mock import Mock, patch

class TestMyComponent:
    """Test class for MyComponent."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.component = MyComponent()
    
    def test_basic_functionality(self):
        """Test basic functionality."""
        result = self.component.process("input")
        assert result == "expected_output"
    
    @pytest.mark.parametrize("input,expected", [
        ("input1", "output1"),
        ("input2", "output2"),
    ])
    def test_parametrized(self, input, expected):
        """Test with multiple parameters."""
        result = self.component.process(input)
        assert result == expected
    
    @patch('karma.models.base_model_abs.AutoTokenizer')
    def test_with_mock(self, mock_tokenizer):
        """Test with mocked dependencies."""
        mock_tokenizer.from_pretrained.return_value = Mock()
        # Test code that uses the mocked tokenizer
```

## Documentation

### Building Documentation Locally

```bash
# Install documentation dependencies
uv install --group docs

# Serve documentation locally
mkdocs serve

# Build documentation
mkdocs build
```

### Documentation Style

- Use clear, concise language
- Include code examples
- Add docstrings to all public functions and classes
- Follow Google-style docstrings:

```python
def my_function(param1: str, param2: int = 10) -> bool:
    """Brief description of the function.
    
    Longer description if needed, explaining the purpose,
    behavior, and any important details.
    
    Args:
        param1: Description of param1.
        param2: Description of param2. Defaults to 10.
    
    Returns:
        Description of the return value.
    
    Raises:
        ValueError: Description of when this exception is raised.
    
    Example:
        ```python
        result = my_function("hello", 20)
        print(result)  # True
        ```
    """
    return True
```

### API Documentation

API documentation is automatically generated from docstrings using mkdocstrings. To add your component to the API docs:

1. Ensure your class has comprehensive docstrings
2. Add it to the appropriate API reference page in `docs/api-reference/`

## Code Review Process

### Before Submitting a PR

1. Ensure all tests pass:
   ```bash
   pytest
   ```

2. Run code formatting and linting:
   ```bash
   ruff format .
   ruff check .
   ```

3. Update documentation if needed

4. Add/update tests for new functionality

### Submitting a Pull Request

1. Create a descriptive PR title and description
2. Reference any related issues
3. Include screenshots or examples if applicable
4. Ensure CI checks pass

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests added/updated
- [ ] All tests pass locally
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or marked as such)
```

## Community

### Getting Help

- **Documentation**: Check the [official documentation](https://karma-docs.example.com)
- **Issues**: Report bugs or request features on [GitHub Issues](https://github.com/eka-care/KARMA-OpenMedEvalKit/issues)
- **Discussions**: Join conversations on [GitHub Discussions](https://github.com/eka-care/KARMA-OpenMedEvalKit/discussions)

### Communication

- Be respectful and inclusive
- Provide context and details when asking questions
- Search existing issues before creating new ones
- Use clear, descriptive titles for issues and PRs

### Recognition

Contributors are recognized in:
- The project README
- Release notes for significant contributions
- Annual contributor acknowledgments

## Release Process

### Versioning

We use [Semantic Versioning](https://semver.org/):
- MAJOR: Breaking changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes (backward compatible)

### Release Checklist

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Create release PR
4. Tag release after merge
5. Publish to PyPI
6. Update documentation

Thank you for contributing to KARMA! 🎉