from dataclasses import dataclass
from typing import Dict, List, Optional
from jiwer import wer, cer

@dataclass
class EvalResult:
    wer: float
    cer: float
    entity_wer: Optional[float] = None
    num_sentences: int = 0
    additional_info: Optional[Dict] = None
        
class Metrics:
    def __init__(self, ref_list: List[str], hyp_list: List[str]):
        self.ref_list = ref_list
        self.hyp_list = hyp_list
        
    def evaluate(
        self,
    ) -> EvalResult:
        assert len(self.ref_list) == len(self.hyp_list), "Mismatch in ref/hyp count"
        
        references = self.ref_list
        hypotheses = self.hyp_list

        total_chars = 0
        total_distance = 0
        total_words = 0
        total_word_dist = 0

        for ref, hyp in zip(references, hypotheses):
            total_chars += len(ref)
            cer_score = cer(ref, hyp)
            if isinstance(cer_score, dict):
                cer_score = cer_score.get('cer', 0.0)
            total_distance += cer_score * len(ref)
            ref_words = ref.split()
            total_words += len(ref_words)
            wer_score = wer(ref, hyp)
            if isinstance(wer_score, dict):
                wer_score = wer_score.get('wer', 0.0)
            total_word_dist += wer_score * len(ref_words)

        overall_cer = total_distance / total_chars if total_chars > 0 else 0
        overall_wer = total_word_dist / total_words if total_words > 0 else 0
        
        return EvalResult(
            wer=overall_wer,
            cer=overall_cer,
            num_sentences=len(references)
        )

