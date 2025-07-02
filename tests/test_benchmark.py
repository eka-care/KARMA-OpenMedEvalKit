import pytest
from unittest.mock import Mock, MagicMock

import logging
from datasets import IterableDataset
from karma.benchmark import Benchmark
from karma.eval_datasets.pubmedmcqa_dataset import PubMedMCQADataset
from karma.metrics import ExactMatchMetric

logger = logging.getLogger(__name__)


class MockModel:
    def __init__(self):
        self.model_config = MagicMock()
        self.model_config.model_id = "mock_model"

    def batch_generate(self, prompts, **kwargs):
        return [("dummy thinking", f"{idx}") for idx, prompt in enumerate(prompts)]


@pytest.fixture
def mock_benchmark():
    model = MockModel()
    dataset = PubMedMCQADataset()
    return Benchmark(logger=logger, model=model, dataset=dataset)


def test_dataloader(mock_benchmark, monkeypatch):
    metric_config = {"metric": ExactMatchMetric(), "processors": []}

    # Run evaluation
    results = mock_benchmark.evaluate(
        metric_config=metric_config, batch_size=2, dry_run=False
    )

    # Check if results contain expected keys
    assert "overall_score" in results
    assert "predictions" in results
    assert "summary" in results

    # Check if predictions were made for all samples
    assert len(results["predictions"]) == 3  # Number of samples in mock dataset

    # Check prediction structure
    for pred in results["predictions"]:
        assert "prediction" in pred
        assert "expected_output" in pred
        assert "sample" in pred
        assert "from_cache" in pred
        assert "success" in pred
