from karma.benchmark import Benchmark
from karma.eval_datasets.pubmedmcqa_dataset import PubMedMCQADataset
from karma.metrics import HfMetric
from karma.models.qwen import QwenThinkingLLM
import logging
from dotenv import load_dotenv
from karma.cache.cache_manager import CacheManager
load_dotenv()

logger = logging.getLogger(__name__)
qwen_model = QwenThinkingLLM(
    model_path="Qwen/Qwen3-0.6B",  # MedGemma model
    device="mps",
    max_tokens=256,  # Sufficient for translation outputs
    temperature=0.01,  # Lower temperature for more consistent translations
    top_p=0.9,  # Nucleus sampling
    top_k=50,
    enable_thinking=False,
)

dataset = PubMedMCQADataset()
cache_manager = CacheManager(dataset.dataset_name, qwen_model.model_config)
benchmark = Benchmark(
    logger=logger,
    model=qwen_model,
    dataset=dataset,
    verbose_mode=True,
    enable_cache=False,
    use_weave=False,
    cache_manager=cache_manager,
)
results = benchmark.evaluate(
    {"metric": HfMetric("exact_match"), "processors": []}, batch_size=1, dry_run=False
)
print(results)
