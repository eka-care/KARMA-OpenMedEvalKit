from dataclasses import dataclass
from typing import Dict, List, Optional
from jiwer import wer, cer

from karma.metrics.base_metric_abs import BaseMetric
from karma.registries.metrics_registry import register_metric


@dataclass
class EvalResult:
    wer: float
    cer: float
    entity_wer: Optional[float] = None
    num_sentences: int = 0
    additional_info: Optional[Dict] = None


@register_metric(
    name="asr_metric",
    optional_args=["use_entity_wer", "entity_extraction_method"],
    default_args={"use_entity_wer": False, "entity_extraction_method": "simple"}
)
class ASRMetrics(BaseMetric):
    def __init__(self, metric_name: str = "asr_metric", **kwargs):
        super().__init__(metric_name, **kwargs)

    def evaluate(self, predictions, references, **kwargs) -> EvalResult:
        assert len(predictions) == len(references), "Mismatch in ref/hyp count"

        total_chars = 0
        total_distance = 0
        total_words = 0
        total_word_dist = 0

        for ref, hyp in zip(references, predictions):
            total_chars += len(ref)
            cer_score = cer(ref, hyp)
            if isinstance(cer_score, dict):
                cer_score = cer_score.get("cer", 0.0)
            total_distance += cer_score * len(ref)
            ref_words = ref.split()
            total_words += len(ref_words)
            wer_score = wer(ref, hyp)
            if isinstance(wer_score, dict):
                wer_score = wer_score.get("wer", 0.0)
            total_word_dist += wer_score * len(ref_words)

        overall_cer = total_distance / total_chars if total_chars > 0 else 0
        overall_wer = total_word_dist / total_words if total_words > 0 else 0

        return EvalResult(
            wer=overall_wer, cer=overall_cer, num_sentences=len(references)
        )
