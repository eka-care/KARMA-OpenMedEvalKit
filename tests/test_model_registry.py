"""
Comprehensive tests for the Model Registry system.

This module tests the decorator-based model registration, discovery,
and management functionality.
"""

import pytest
from unittest.mock import Mock, patch
from typing import List, Optional, Dict, Any
from PIL import Image
from IPython.display import Audio

from karma.registries.model_registry import (
    ModelRegistry,
    model_registry,
    register_model,
)
from karma.models.base import BaseLLM


class MockBaseLLM(BaseLLM):
    """Mock implementation of BaseLLM for testing."""

    def __init__(self, model_path: str, **kwargs):
        # Skip the parent init to avoid loading actual models
        self.model_path = model_path
        self.kwargs = kwargs

    def load_model(self):
        """Mock load_model method."""
        self.model = Mock()
        self.processor = Mock()

    def format_inputs(
        self,
        prompts: List[str],
        images: Optional[List[List[Image.Image]]] = None,
        audios: Optional[List[Audio]] = None,
    ) -> Dict[str, Any]:
        """Mock format_inputs method."""
        return {"input_ids": Mock(), "attention_mask": Mock()}

    def format_outputs(self, outputs: List[str]) -> List[str]:
        """Mock format_outputs method."""
        return [output.strip() for output in outputs]


class TestModelRegistry:
    """Test cases for ModelRegistry class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.registry = ModelRegistry()

    def test_registry_initialization(self):
        """Test that registry initializes with empty models dict."""
        assert self.registry.models == {}
        assert not self.registry._discovered

    def test_register_model_decorator(self):
        """Test the register_model decorator functionality."""

        @self.registry.register_model("test_model")
        class TestModel(MockBaseLLM):
            pass

        assert "test_model" in self.registry.models
        assert self.registry.models["test_model"] == TestModel

    def test_register_model_invalid_class(self):
        """Test that registering invalid class raises ValueError."""

        with pytest.raises(ValueError, match="must inherit from BaseLLM"):

            @self.registry.register_model("invalid_model")
            class InvalidModel:
                pass

    def test_register_model_override_warning(self, caplog):
        """Test that registering duplicate model name logs warning."""

        @self.registry.register_model("duplicate_model")
        class FirstModel(MockBaseLLM):
            pass

        @self.registry.register_model("duplicate_model")
        class SecondModel(MockBaseLLM):
            pass

        assert "is already registered. Overriding" in caplog.text
        assert self.registry.models["duplicate_model"] == SecondModel

    def test_get_model_success(self):
        """Test successful model retrieval by name."""

        @self.registry.register_model("retrievable_model")
        class RetrievableModel(MockBaseLLM):
            pass

        retrieved_class = self.registry.get_model("retrievable_model")
        assert retrieved_class == RetrievableModel

    def test_get_model_not_found(self):
        """Test that getting non-existent model raises ValueError."""

        with pytest.raises(ValueError, match="Model 'nonexistent' not found"):
            self.registry.get_model("nonexistent")

    def test_list_models_empty(self):
        """Test listing models when registry is empty."""
        assert self.registry.list_models() == []

    def test_list_models_with_registered_models(self):
        """Test listing models when models are registered."""

        @self.registry.register_model("model_a")
        class ModelA(MockBaseLLM):
            pass

        @self.registry.register_model("model_b")
        class ModelB(MockBaseLLM):
            pass

        models = self.registry.list_models()
        print(models)
        assert set(models) == {"model_a", "model_b"}

    def test_is_registered_true(self):
        """Test is_registered returns True for registered model."""

        @self.registry.register_model("registered_model")
        class RegisteredModel(MockBaseLLM):
            pass

        assert self.registry.is_registered("registered_model")

    def test_is_registered_false(self):
        """Test is_registered returns False for unregistered model."""
        assert not self.registry.is_registered("unregistered_model")

    def test_unregister_model_success(self):
        """Test successful model unregistration."""

        @self.registry.register_model("unregister_me")
        class UnregisterMe(MockBaseLLM):
            pass

        assert self.registry.is_registered("unregister_me")
        result = self.registry.unregister_model("unregister_me")
        assert result is True
        assert not self.registry.is_registered("unregister_me")

    def test_unregister_model_not_found(self):
        """Test unregistering non-existent model returns False."""
        result = self.registry.unregister_model("nonexistent")
        assert result is False

    def test_clear_registry(self):
        """Test clearing the registry."""

        @self.registry.register_model("clear_me")
        class ClearMe(MockBaseLLM):
            pass

        assert len(self.registry.models) > 0
        assert self.registry._discovered is False

        self.registry.clear_registry()
        assert self.registry.models == {}
        assert self.registry._discovered is False

    @patch("karma.registries.model_registry.pkgutil.iter_modules")
    @patch("karma.registries.model_registry.importlib.import_module")
    def test_discover_models_success(self, mock_import, mock_iter):
        """Test successful model discovery."""
        # Mock the module iteration
        mock_iter.return_value = [
            (None, "karma.models.qwen", False),
            (None, "karma.models.medgemma", False),
            (None, "karma.models.base", False),  # Should be skipped
        ]

        # Mock successful imports
        mock_import.side_effect = [None, None]  # Two successful imports

        self.registry.discover_models()

        # Verify import_module was called for non-base modules
        expected_calls = [
            pytest.call("karma.models.qwen"),
            pytest.call("karma.models.medgemma"),
        ]
        mock_import.assert_has_calls(expected_calls)
        assert self.registry._discovered is True

    @patch("karma.registries.model_registry.pkgutil.iter_modules")
    @patch("karma.registries.model_registry.importlib.import_module")
    def test_discover_models_import_error(self, mock_import, mock_iter, caplog):
        """Test model discovery with import errors."""
        mock_iter.return_value = [
            (None, "karma.models.failing_model", False),
        ]

        # Mock import failure
        mock_import.side_effect = ImportError("Module not found")

        self.registry.discover_models()

        assert "Could not import model module" in caplog.text
        assert self.registry._discovered is True

    @patch("karma.registries.model_registry.importlib.import_module")
    def test_discover_models_package_import_error(self, mock_import, caplog):
        """Test model discovery when karma.models package import fails."""
        # Make the karma.models import fail
        mock_import.side_effect = ImportError("karma.models not found")

        self.registry.discover_models()

        assert "Could not import karma.models" in caplog.text
        assert self.registry._discovered is True

    def test_discover_models_called_once(self):
        """Test that discover_models is only called once."""
        with patch.object(
            self.registry, "discover_models", wraps=self.registry.discover_models
        ) as mock_discover:
            # First call to list_models should call discover_models
            self.registry.list_models()
            assert mock_discover.call_count == 1

            # Second call should not call discover_models again
            self.registry.list_models()
            assert mock_discover.call_count == 1


class TestGlobalModelRegistry:
    """Test cases for the global model registry instance."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Clear the global registry before each test
        model_registry.clear_registry()

    def test_global_registry_is_singleton(self):
        """Test that there's only one global registry instance."""
        from karma.registries.model_registry import model_registry as registry1
        from karma.registries.model_registry import model_registry as registry2

        assert registry1 is registry2

    def test_register_model_convenience_function(self):
        """Test the convenience register_model function."""

        @register_model("convenience_model")
        class ConvenienceModel(MockBaseLLM):
            pass

        assert model_registry.is_registered("convenience_model")
        assert model_registry.get_model("convenience_model") == ConvenienceModel

    def teardown_method(self):
        """Clean up after each test method."""
        model_registry.clear_registry()


class TestModelRegistryIntegration:
    """Integration tests for the model registry system."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.registry = ModelRegistry()

    def test_full_model_lifecycle(self):
        """Test complete model registration, retrieval, and usage lifecycle."""

        @self.registry.register_model("lifecycle_model")
        class LifecycleModel(MockBaseLLM):
            def custom_method(self):
                return "custom_result"

        # Test registration
        assert self.registry.is_registered("lifecycle_model")
        assert "lifecycle_model" in self.registry.list_models()

        # Test retrieval
        model_class = self.registry.get_model("lifecycle_model")
        assert model_class == LifecycleModel

        # Test instantiation
        model_instance = model_class("test/path")
        assert model_instance.model_path == "test/path"
        assert model_instance.custom_method() == "custom_result"

        # Test unregistration
        self.registry.unregister_model("lifecycle_model")
        assert not self.registry.is_registered("lifecycle_model")

    def test_multiple_models_registration(self):
        """Test registering multiple models and their independence."""

        @self.registry.register_model("model_1")
        class Model1(MockBaseLLM):
            model_type = "type_1"

        @self.registry.register_model("model_2")
        class Model2(MockBaseLLM):
            model_type = "type_2"

        @self.registry.register_model("model_3")
        class Model3(MockBaseLLM):
            model_type = "type_3"

        # Test all models are registered
        models = self.registry.list_models()
        assert set(models) == {"model_1", "model_2", "model_3"}

        # Test each model can be retrieved correctly
        assert self.registry.get_model("model_1").model_type == "type_1"
        assert self.registry.get_model("model_2").model_type == "type_2"
        assert self.registry.get_model("model_3").model_type == "type_3"

        # Test models are independent
        assert self.registry.get_model("model_1") != self.registry.get_model("model_2")


if __name__ == "__main__":
    pytest.main([__file__])
