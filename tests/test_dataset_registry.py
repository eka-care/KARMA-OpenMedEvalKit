"""
Comprehensive tests for the Dataset Registry system.

This module tests the decorator-based dataset registration, discovery,
and management functionality with metrics and task type metadata.
"""

import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any

from karma.registries.dataset_registry import (
    DatasetRegistry,
    dataset_registry,
    register_dataset,
)
from karma.eval_datasets.base_dataset import BaseMultimodalDataset


class MockBaseMultimodalDataset(BaseMultimodalDataset):
    """Mock implementation of BaseMultimodalDataset for testing."""

    def __init__(
        self, dataset_name: str = "mock_dataset", split: str = "test", **kwargs
    ):
        # Skip the parent init to avoid loading actual datasets
        self.dataset_name = dataset_name
        self.split = split
        self.kwargs = kwargs
        self.dataset = Mock()

    def format_item(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Mock format_item method."""
        return {
            "prompt": sample.get("question", "Mock question"),
            "expected_output": sample.get("answer", "Mock answer"),
            "images": None,
            "audios": None,
        }


class TestDatasetRegistry:
    """Test cases for DatasetRegistry class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.registry = DatasetRegistry()

    def test_registry_initialization(self):
        """Test that registry initializes with empty datasets dict."""
        assert self.registry.datasets == {}
        assert not self.registry._discovered

    def test_register_dataset_decorator_basic(self):
        """Test the register_dataset decorator with basic parameters."""

        @self.registry.register_dataset("test_dataset", metrics=["accuracy"])
        class TestDataset(MockBaseMultimodalDataset):
            pass

        assert "test_dataset" in self.registry.datasets
        dataset_info = self.registry.datasets["test_dataset"]
        assert dataset_info["class"] == TestDataset
        assert dataset_info["metrics"] == ["accuracy"]
        assert dataset_info["task_type"] == "mcqa"  # default

    def test_register_dataset_decorator_full_params(self):
        """Test the register_dataset decorator with all parameters."""

        @self.registry.register_dataset(
            "full_dataset", metrics=["accuracy", "bleu", "rouge"], task_type="vqa"
        )
        class FullDataset(MockBaseMultimodalDataset):
            pass

        assert "full_dataset" in self.registry.datasets
        dataset_info = self.registry.datasets["full_dataset"]
        assert dataset_info["class"] == FullDataset
        assert dataset_info["metrics"] == ["accuracy", "bleu", "rouge"]
        assert dataset_info["task_type"] == "vqa"
        assert dataset_info["module"] == FullDataset.__module__
        assert dataset_info["class_name"] == "FullDataset"
        # Test that argument fields are initialized to empty/default values
        assert dataset_info["required_args"] == []
        assert dataset_info["optional_args"] == []
        assert dataset_info["default_args"] == {}

    def test_register_dataset_with_arguments(self):
        """Test the register_dataset decorator with argument specifications."""

        @self.registry.register_dataset(
            "arg_dataset",
            metrics=["bleu", "accuracy"],
            task_type="translation",
            required_args=["source_language", "target_language"],
            optional_args=["domain", "variant"],
            default_args={"domain": "general", "max_length": 512},
        )
        class ArgDataset(MockBaseMultimodalDataset):
            pass

        assert "arg_dataset" in self.registry.datasets
        dataset_info = self.registry.datasets["arg_dataset"]
        assert dataset_info["class"] == ArgDataset
        assert dataset_info["metrics"] == ["bleu", "accuracy"]
        assert dataset_info["task_type"] == "translation"
        assert dataset_info["required_args"] == ["source_language", "target_language"]
        assert dataset_info["optional_args"] == ["domain", "variant"]
        assert dataset_info["default_args"] == {"domain": "general", "max_length": 512}

    def test_register_dataset_single_metric_string(self):
        """Test register_dataset with single metric as string."""

        @self.registry.register_dataset("single_metric", metrics="accuracy")
        class SingleMetricDataset(MockBaseMultimodalDataset):
            pass

        dataset_info = self.registry.datasets["single_metric"]
        assert dataset_info["metrics"] == ["accuracy"]

    def test_register_dataset_invalid_class(self):
        """Test that registering invalid class raises ValueError."""

        with pytest.raises(ValueError, match="must inherit from BaseMultimodalDataset"):

            @self.registry.register_dataset("invalid_dataset", metrics=["accuracy"])
            class InvalidDataset:
                pass

    def test_register_dataset_override_warning(self, caplog):
        """Test that registering duplicate dataset name logs warning."""

        @self.registry.register_dataset("duplicate_dataset", metrics=["accuracy"])
        class FirstDataset(MockBaseMultimodalDataset):
            pass

        @self.registry.register_dataset("duplicate_dataset", metrics=["bleu"])
        class SecondDataset(MockBaseMultimodalDataset):
            pass

        assert "is already registered. Overriding" in caplog.text
        assert self.registry.datasets["duplicate_dataset"]["class"] == SecondDataset
        assert self.registry.datasets["duplicate_dataset"]["metrics"] == ["bleu"]

    def test_get_dataset_info_success(self):
        """Test successful dataset info retrieval by name."""

        @self.registry.register_dataset("info_dataset", metrics=["accuracy", "f1"])
        class InfoDataset(MockBaseMultimodalDataset):
            pass

        info = self.registry.get_dataset_info("info_dataset")
        assert info["class"] == InfoDataset
        assert info["metrics"] == ["accuracy", "f1"]
        assert info["task_type"] == "mcqa"

        # Test that returned info is a copy
        info["metrics"].append("new_metric")
        original_info = self.registry.get_dataset_info("info_dataset")
        assert "new_metric" not in original_info["metrics"]

    def test_get_dataset_info_not_found(self):
        """Test that getting non-existent dataset info raises ValueError."""

        with pytest.raises(ValueError, match="Dataset 'nonexistent' not found"):
            self.registry.get_dataset_info("nonexistent")

    def test_get_dataset_class_success(self):
        """Test successful dataset class retrieval by name."""

        @self.registry.register_dataset("class_dataset", metrics=["accuracy"])
        class ClassDataset(MockBaseMultimodalDataset):
            pass

        dataset_class = self.registry.get_dataset_class("class_dataset")
        assert dataset_class == ClassDataset

    def test_list_datasets_empty(self):
        """Test listing datasets when registry is empty."""
        assert self.registry.list_datasets() == []

    def test_list_datasets_with_registered_datasets(self):
        """Test listing datasets when datasets are registered."""

        @self.registry.register_dataset("dataset_a", metrics=["accuracy"])
        class DatasetA(MockBaseMultimodalDataset):
            pass

        @self.registry.register_dataset("dataset_b", metrics=["bleu"])
        class DatasetB(MockBaseMultimodalDataset):
            pass

        datasets = self.registry.list_datasets()
        assert set(datasets) == {"dataset_a", "dataset_b"}

    def test_create_dataset_success(self):
        """Test successful dataset instance creation."""

        @self.registry.register_dataset("create_dataset", metrics=["accuracy"])
        class CreateDataset(MockBaseMultimodalDataset):
            def __init__(self, custom_param="default", **kwargs):
                super().__init__(**kwargs)
                self.custom_param = custom_param

        dataset_instance = self.registry.create_dataset(
            "create_dataset", custom_param="test_value", dataset_name="test_name"
        )

        assert isinstance(dataset_instance, CreateDataset)
        assert dataset_instance.custom_param == "test_value"
        assert dataset_instance.dataset_name == "test_name"

    def test_get_dataset_metrics(self):
        """Test getting dataset metrics."""

        @self.registry.register_dataset(
            "metrics_dataset", metrics=["accuracy", "precision", "recall"]
        )
        class MetricsDataset(MockBaseMultimodalDataset):
            pass

        metrics = self.registry.get_dataset_metrics("metrics_dataset")
        assert metrics == ["accuracy", "precision", "recall"]

        # Test that returned metrics is a copy
        metrics.append("new_metric")
        original_metrics = self.registry.get_dataset_metrics("metrics_dataset")
        assert "new_metric" not in original_metrics

    def test_get_dataset_task_type(self):
        """Test getting dataset task type."""

        @self.registry.register_dataset(
            "task_dataset", metrics=["accuracy"], task_type="vqa"
        )
        class TaskDataset(MockBaseMultimodalDataset):
            pass

        task_type = self.registry.get_dataset_task_type("task_dataset")
        assert task_type == "vqa"

    def test_is_registered_true(self):
        """Test is_registered returns True for registered dataset."""

        @self.registry.register_dataset("registered_dataset", metrics=["accuracy"])
        class RegisteredDataset(MockBaseMultimodalDataset):
            pass

        assert self.registry.is_registered("registered_dataset")

    def test_is_registered_false(self):
        """Test is_registered returns False for unregistered dataset."""
        assert not self.registry.is_registered("unregistered_dataset")

    def test_unregister_dataset_success(self):
        """Test successful dataset unregistration."""

        @self.registry.register_dataset("unregister_me", metrics=["accuracy"])
        class UnregisterMe(MockBaseMultimodalDataset):
            pass

        assert self.registry.is_registered("unregister_me")
        result = self.registry.unregister_dataset("unregister_me")
        assert result is True
        assert not self.registry.is_registered("unregister_me")

    def test_unregister_dataset_not_found(self):
        """Test unregistering non-existent dataset returns False."""
        result = self.registry.unregister_dataset("nonexistent")
        assert result is False

    def test_clear_registry(self):
        """Test clearing the registry."""

        @self.registry.register_dataset("clear_me", metrics=["accuracy"])
        class ClearMe(MockBaseMultimodalDataset):
            pass

        assert len(self.registry.datasets) > 0
        assert self.registry._discovered is False

        self.registry.clear_registry()
        assert self.registry.datasets == {}
        assert self.registry._discovered is False

    def test_list_datasets_by_task_type(self):
        """Test filtering datasets by task type."""

        @self.registry.register_dataset(
            "mcqa_dataset", metrics=["accuracy"], task_type="mcqa"
        )
        class MCQADataset(MockBaseMultimodalDataset):
            pass

        @self.registry.register_dataset(
            "vqa_dataset", metrics=["bleu"], task_type="vqa"
        )
        class VQADataset(MockBaseMultimodalDataset):
            pass

        @self.registry.register_dataset("qa_dataset", metrics=["rouge"], task_type="qa")
        class QADataset(MockBaseMultimodalDataset):
            pass

        mcqa_datasets = self.registry.list_datasets_by_task_type("mcqa")
        vqa_datasets = self.registry.list_datasets_by_task_type("vqa")
        nonexistent_datasets = self.registry.list_datasets_by_task_type("nonexistent")

        assert mcqa_datasets == ["mcqa_dataset"]
        assert vqa_datasets == ["vqa_dataset"]
        assert nonexistent_datasets == []

    def test_list_datasets_by_metric(self):
        """Test filtering datasets by metric."""

        @self.registry.register_dataset("accuracy_dataset", metrics=["accuracy", "f1"])
        class AccuracyDataset(MockBaseMultimodalDataset):
            pass

        @self.registry.register_dataset("bleu_dataset", metrics=["bleu", "rouge"])
        class BleuDataset(MockBaseMultimodalDataset):
            pass

        @self.registry.register_dataset("mixed_dataset", metrics=["accuracy", "bleu"])
        class MixedDataset(MockBaseMultimodalDataset):
            pass

        accuracy_datasets = self.registry.list_datasets_by_metric("accuracy")
        bleu_datasets = self.registry.list_datasets_by_metric("bleu")
        f1_datasets = self.registry.list_datasets_by_metric("f1")
        nonexistent_datasets = self.registry.list_datasets_by_metric("nonexistent")

        assert set(accuracy_datasets) == {"accuracy_dataset", "mixed_dataset"}
        assert set(bleu_datasets) == {"bleu_dataset", "mixed_dataset"}
        assert f1_datasets == ["accuracy_dataset"]
        assert nonexistent_datasets == []

    def test_get_dataset_required_args(self):
        """Test getting required arguments for a dataset."""

        @self.registry.register_dataset(
            "req_args_dataset", metrics=["accuracy"], required_args=["lang", "domain"]
        )
        class ReqArgsDataset(MockBaseMultimodalDataset):
            pass

        required_args = self.registry.get_dataset_required_args("req_args_dataset")
        assert required_args == ["lang", "domain"]

        # Test that returned list is a copy
        required_args.append("new_arg")
        original_args = self.registry.get_dataset_required_args("req_args_dataset")
        assert "new_arg" not in original_args

    def test_get_dataset_optional_args(self):
        """Test getting optional arguments for a dataset."""

        @self.registry.register_dataset(
            "opt_args_dataset",
            metrics=["accuracy"],
            optional_args=["variant", "max_length"],
        )
        class OptArgsDataset(MockBaseMultimodalDataset):
            pass

        optional_args = self.registry.get_dataset_optional_args("opt_args_dataset")
        assert optional_args == ["variant", "max_length"]

        # Test that returned list is a copy
        optional_args.append("new_arg")
        original_args = self.registry.get_dataset_optional_args("opt_args_dataset")
        assert "new_arg" not in original_args

    def test_get_dataset_default_args(self):
        """Test getting default arguments for a dataset."""

        @self.registry.register_dataset(
            "default_args_dataset",
            metrics=["accuracy"],
            default_args={"batch_size": 32, "max_length": 512},
        )
        class DefaultArgsDataset(MockBaseMultimodalDataset):
            pass

        default_args = self.registry.get_dataset_default_args("default_args_dataset")
        assert default_args == {"batch_size": 32, "max_length": 512}

        # Test that returned dict is a copy
        default_args["new_arg"] = "new_value"
        original_args = self.registry.get_dataset_default_args("default_args_dataset")
        assert "new_arg" not in original_args

    def test_get_dataset_all_args(self):
        """Test getting all argument information for a dataset."""

        @self.registry.register_dataset(
            "all_args_dataset",
            metrics=["accuracy"],
            required_args=["source_lang"],
            optional_args=["target_lang"],
            default_args={"max_length": 1024},
        )
        class AllArgsDataset(MockBaseMultimodalDataset):
            pass

        all_args = self.registry.get_dataset_all_args("all_args_dataset")
        expected = {
            "required": ["source_lang"],
            "optional": ["target_lang"],
            "defaults": {"max_length": 1024},
        }
        assert all_args == expected

    def test_validate_dataset_args_success(self):
        """Test successful argument validation."""

        @self.registry.register_dataset(
            "validation_dataset",
            metrics=["accuracy"],
            required_args=["source_language", "target_language"],
            optional_args=["domain"],
            default_args={"max_length": 512, "domain": "general"},
        )
        class ValidationDataset(MockBaseMultimodalDataset):
            pass

        # Test with all required args
        provided_args = {"source_language": "en", "target_language": "hi"}
        validated_args = self.registry.validate_dataset_args(
            "validation_dataset", provided_args
        )
        expected = {
            "source_language": "en",
            "target_language": "hi",
            "max_length": 512,
            "domain": "general",
        }
        assert validated_args == expected

        # Test with optional args overriding defaults
        provided_args = {
            "source_language": "en",
            "target_language": "hi",
            "domain": "medical",
        }
        validated_args = self.registry.validate_dataset_args(
            "validation_dataset", provided_args
        )
        expected = {
            "source_language": "en",
            "target_language": "hi",
            "max_length": 512,
            "domain": "medical",
        }
        assert validated_args == expected

    def test_validate_dataset_args_missing_required(self):
        """Test validation failure with missing required arguments."""

        @self.registry.register_dataset(
            "missing_req_dataset",
            metrics=["accuracy"],
            required_args=["source_language", "target_language"],
        )
        class MissingReqDataset(MockBaseMultimodalDataset):
            pass

        # Missing one required argument
        provided_args = {"source_language": "en"}

        with pytest.raises(ValueError, match="Missing required arguments"):
            self.registry.validate_dataset_args("missing_req_dataset", provided_args)

    def test_validate_dataset_args_unexpected_arguments(self, caplog):
        """Test validation with unexpected arguments (should warn but not fail)."""

        @self.registry.register_dataset(
            "unexpected_args_dataset",
            metrics=["accuracy"],
            required_args=["source_language"],
        )
        class UnexpectedArgsDataset(MockBaseMultimodalDataset):
            pass

        provided_args = {"source_language": "en", "unexpected_arg": "value"}

        validated_args = self.registry.validate_dataset_args(
            "unexpected_args_dataset", provided_args
        )

        # Should include the unexpected arg but log warning
        assert "source_language" in validated_args
        assert "unexpected_arg" in validated_args
        assert "Unexpected arguments" in caplog.text

    def test_create_dataset_with_validation(self):
        """Test dataset creation with argument validation."""

        @self.registry.register_dataset(
            "validated_creation_dataset",
            metrics=["accuracy"],
            required_args=["source_language"],
            default_args={"max_length": 256},
        )
        class ValidatedCreationDataset(MockBaseMultimodalDataset):
            def __init__(self, source_language, max_length=128, **kwargs):
                super().__init__(**kwargs)
                self.source_language = source_language
                self.max_length = max_length

        # Test successful creation with validation
        dataset = self.registry.create_dataset(
            "validated_creation_dataset", source_language="en", dataset_name="test"
        )

        assert isinstance(dataset, ValidatedCreationDataset)
        assert dataset.source_language == "en"
        assert dataset.max_length == 256  # From default_args
        assert dataset.dataset_name == "test"

    def test_create_dataset_without_validation(self):
        """Test dataset creation without argument validation (backwards compatibility)."""

        @self.registry.register_dataset(
            "no_validation_dataset",
            metrics=["accuracy"],
            required_args=["source_language"],
        )
        class NoValidationDataset(MockBaseMultimodalDataset):
            def __init__(self, custom_arg="default", **kwargs):
                super().__init__(**kwargs)
                self.custom_arg = custom_arg

        # Test creation without validation - should work even without required args
        dataset = self.registry.create_dataset(
            "no_validation_dataset", validate_args=False, custom_arg="test_value"
        )

        assert isinstance(dataset, NoValidationDataset)
        assert dataset.custom_arg == "test_value"

    def test_create_dataset_validation_failure(self):
        """Test dataset creation failure due to validation."""

        @self.registry.register_dataset(
            "validation_failure_dataset",
            metrics=["accuracy"],
            required_args=["source_language", "target_language"],
        )
        class ValidationFailureDataset(MockBaseMultimodalDataset):
            pass

        # Missing required arguments should cause validation to fail
        with pytest.raises(ValueError, match="Missing required arguments"):
            self.registry.create_dataset(
                "validation_failure_dataset",
                source_language="en",  # missing target_language
            )

    @patch("karma.dataset_registry.pkgutil.iter_modules")
    @patch("karma.dataset_registry.importlib.import_module")
    def test_discover_datasets_success(self, mock_import, mock_iter):
        """Test successful dataset discovery."""
        # Mock the module iteration
        mock_iter.return_value = [
            (None, "karma.eval_datasets.pubmedmcqa_dataset", False),
            (None, "karma.eval_datasets.slake_dataset", False),
            (None, "karma.eval_datasets.base_dataset", False),  # Should be skipped
        ]

        # Mock successful imports
        mock_import.side_effect = [None, None]  # Two successful imports

        self.registry.discover_datasets()

        # Verify import_module was called for non-base modules
        expected_calls = [
            pytest.call("karma.eval_datasets.pubmedmcqa_dataset"),
            pytest.call("karma.eval_datasets.slake_dataset"),
        ]
        mock_import.assert_has_calls(expected_calls)
        assert self.registry._discovered is True

    @patch("karma.dataset_registry.pkgutil.iter_modules")
    @patch("karma.dataset_registry.importlib.import_module")
    def test_discover_datasets_import_error(self, mock_import, mock_iter, caplog):
        """Test dataset discovery with import errors."""
        mock_iter.return_value = [
            (None, "karma.eval_datasets.failing_dataset", False),
        ]

        # Mock import failure
        mock_import.side_effect = ImportError("Module not found")

        self.registry.discover_datasets()

        assert "Could not import dataset module" in caplog.text
        assert self.registry._discovered is True

    @patch("karma.dataset_registry.importlib.import_module")
    def test_discover_datasets_package_import_error(self, mock_import, caplog):
        """Test dataset discovery when karma.eval_datasets package import fails."""
        # Make the karma.eval_datasets import fail
        mock_import.side_effect = ImportError("karma.eval_datasets not found")

        self.registry.discover_datasets()

        assert "Could not import karma.eval_datasets package" in caplog.text
        assert self.registry._discovered is True

    def test_discover_datasets_called_once(self):
        """Test that discover_datasets is only called once."""
        with patch.object(
            self.registry, "discover_datasets", wraps=self.registry.discover_datasets
        ) as mock_discover:
            # First call to list_datasets should call discover_datasets
            self.registry.list_datasets()
            assert mock_discover.call_count == 1

            # Second call should not call discover_datasets again
            self.registry.list_datasets()
            assert mock_discover.call_count == 1


class TestGlobalDatasetRegistry:
    """Test cases for the global dataset registry instance."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Clear the global registry before each test
        dataset_registry.clear_registry()

    def test_global_registry_is_singleton(self):
        """Test that there's only one global registry instance."""
        from karma.registries.dataset_registry import dataset_registry as registry1
        from karma.registries.dataset_registry import dataset_registry as registry2

        assert registry1 is registry2

    def test_register_dataset_convenience_function(self):
        """Test the convenience register_dataset function."""

        @register_dataset("convenience_dataset", metrics=["accuracy"])
        class ConvenienceDataset(MockBaseMultimodalDataset):
            pass

        assert dataset_registry.is_registered("convenience_dataset")
        assert (
            dataset_registry.get_dataset_class("convenience_dataset")
            == ConvenienceDataset
        )
        assert dataset_registry.get_dataset_metrics("convenience_dataset") == [
            "accuracy"
        ]

    def teardown_method(self):
        """Clean up after each test method."""
        dataset_registry.clear_registry()


class TestDatasetRegistryIntegration:
    """Integration tests for the dataset registry system."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.registry = DatasetRegistry()

    def test_full_dataset_lifecycle(self):
        """Test complete dataset registration, retrieval, and usage lifecycle."""

        @self.registry.register_dataset(
            "lifecycle_dataset", metrics=["accuracy", "f1"], task_type="qa"
        )
        class LifecycleDataset(MockBaseMultimodalDataset):
            def custom_method(self):
                return "custom_result"

        # Test registration
        assert self.registry.is_registered("lifecycle_dataset")
        assert "lifecycle_dataset" in self.registry.list_datasets()

        # Test metadata retrieval
        assert self.registry.get_dataset_metrics("lifecycle_dataset") == [
            "accuracy",
            "f1",
        ]
        assert self.registry.get_dataset_task_type("lifecycle_dataset") == "qa"

        # Test class retrieval
        dataset_class = self.registry.get_dataset_class("lifecycle_dataset")
        assert dataset_class == LifecycleDataset

        # Test instance creation
        dataset_instance = self.registry.create_dataset(
            "lifecycle_dataset", dataset_name="test"
        )
        assert dataset_instance.dataset_name == "test"
        assert dataset_instance.custom_method() == "custom_result"

        # Test filtering
        qa_datasets = self.registry.list_datasets_by_task_type("qa")
        assert "lifecycle_dataset" in qa_datasets

        accuracy_datasets = self.registry.list_datasets_by_metric("accuracy")
        assert "lifecycle_dataset" in accuracy_datasets

        # Test unregistration
        self.registry.unregister_dataset("lifecycle_dataset")
        assert not self.registry.is_registered("lifecycle_dataset")

    def test_multiple_datasets_with_different_configs(self):
        """Test registering multiple datasets with different configurations."""

        @self.registry.register_dataset(
            "mcqa_1", metrics=["accuracy"], task_type="mcqa"
        )
        class MCQA1(MockBaseMultimodalDataset):
            pass

        @self.registry.register_dataset(
            "mcqa_2", metrics=["accuracy", "f1"], task_type="mcqa"
        )
        class MCQA2(MockBaseMultimodalDataset):
            pass

        @self.registry.register_dataset(
            "vqa_1", metrics=["bleu", "rouge"], task_type="vqa"
        )
        class VQA1(MockBaseMultimodalDataset):
            pass

        # Test all datasets are registered
        datasets = self.registry.list_datasets()
        assert set(datasets) == {"mcqa_1", "mcqa_2", "vqa_1"}

        # Test filtering by task type
        mcqa_datasets = self.registry.list_datasets_by_task_type("mcqa")
        vqa_datasets = self.registry.list_datasets_by_task_type("vqa")
        assert set(mcqa_datasets) == {"mcqa_1", "mcqa_2"}
        assert vqa_datasets == ["vqa_1"]

        # Test filtering by metric
        accuracy_datasets = self.registry.list_datasets_by_metric("accuracy")
        bleu_datasets = self.registry.list_datasets_by_metric("bleu")
        f1_datasets = self.registry.list_datasets_by_metric("f1")

        assert set(accuracy_datasets) == {"mcqa_1", "mcqa_2"}
        assert bleu_datasets == ["vqa_1"]
        assert f1_datasets == ["mcqa_2"]

        # Test that each dataset can be retrieved and instantiated correctly
        for dataset_name in datasets:
            dataset_class = self.registry.get_dataset_class(dataset_name)
            dataset_instance = self.registry.create_dataset(dataset_name)
            assert isinstance(dataset_instance, dataset_class)


if __name__ == "__main__":
    pytest.main([__file__])
