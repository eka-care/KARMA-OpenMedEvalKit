from karma.benchmark import Benchmark
from karma.eval_datasets.pubmedmcqa_dataset import PubMedMCQADataset
from karma.metrics.common_metrics import ExactMatchMetric
from karma.models.qwen import QwenThinkingLLM
import logging
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)
qwen_model = QwenThinkingLLM(
    model_name_or_path="Qwen/Qwen3-0.6B",  # MedGemma model
    device="mps",
    max_tokens=256,  # Sufficient for translation outputs
    temperature=0.01,  # Lower temperature for more consistent translations
    top_p=0.9,  # Nucleus sampling
    top_k=50,
    enable_thinking=False,
)

dataset = PubMedMCQADataset()
benchmark = Benchmark(
    model=qwen_model,
    dataset=dataset,
    verbose_mode=True,
    use_weave=False,
    cache_manager=None,
)
results = benchmark.evaluate(
    [ExactMatchMetric()], batch_size=1
)
