"""
Analysis Metrics for Medical ASR

This module provides metrics that return detailed error analysis including
substitution rate, deletion rate, and insertion rate (all normalized by total words).
These metrics use the same normalization pipeline as semantic_wer_metric.
"""

import threading
from typing import List, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor

from karma.metrics.base_metric_abs import BaseMetric
from karma.registries.metrics_registry import register_metric
from karma.metrics.wer_metrics import (
    optimize_wer,
    load_semantic_resources,
)

from karma.metrics.language_support import resolve_indic_normalizer
from karma.metrics.wer_utils import (
    count_edit_operations_difflib,
)


# ============================================================================
# ERROR COUNTING FUNCTIONS
# ============================================================================





def _rates_with_support(
    subs: int, dels: int, ins: int, ref_words: int, references: List[str]
) -> Dict[str, Any]:
    """Error rates + per-field support (scored = samples with a non-empty reference)."""
    if ref_words == 0:
        rates = {
            "substitution_rate": 0.0,
            "deletion_rate": 0.0,
            "insertion_rate": 0.0,
        }
    else:
        rates = {
            "substitution_rate": subs / ref_words,
            "deletion_rate": dels / ref_words,
            "insertion_rate": ins / ref_words,
        }
    scored = sum(1 for ref in references if ref)
    rates["_field_support"] = {
        field: {"value": value, "scored": scored, "total": len(references)}
        for field, value in rates.items()
    }
    return rates


# ============================================================================
# METRIC CLASSES
# ============================================================================

@register_metric("semantic_error_analysis", required_args=["language"])
class SemanticErrorAnalysisMetric(BaseMetric):
    """
    Error Analysis metric that applies the same normalizations as semantic_wer_metric
    but returns detailed error breakdown instead of WER.
    
    Returns:
        Dict with:
            - substitution_rate: substitutions / total_reference_words
            - deletion_rate: deletions / total_reference_words
            - insertion_rate: insertions / total_reference_words
    """
    
    def __init__(self, metric_name: str = "semantic_error_analysis", **kwargs):
        self.language, self.normalizer = resolve_indic_normalizer(
            kwargs.get("language"), do_normalize_chandras=True, remove_nuktas=True
        )
        super().__init__(metric_name, **kwargs)

        resources = load_semantic_resources(self.language)
        self.equivalents = resources.equivalents
        self.units_dict = resources.units_dict
        self.glm_arrays = resources.glm_arrays
        self.ortho_map = resources.ortho_map

    def evaluate(self, predictions: List[str], references: List[str], **kwargs) -> Dict[str, float]:
        """
        Evaluate multiple samples and return overall error rates.
        
        Returns:
            Dict with substitution_rate, deletion_rate, insertion_rate
        """
        total_substitutions = 0
        total_deletions = 0
        total_insertions = 0
        total_ref_words = 0
        
        results_lock = threading.Lock()
        
        def process_pair(ref, hyp):
            nonlocal total_substitutions, total_deletions, total_insertions, total_ref_words
            
            if not ref:
                return
            
            if len(hyp.split()) > 2 * len(ref.split()):
                hyp = ' '.join(hyp.split()[:2 * len(ref.split())])
            
            if self.language != "en":
                ref = self.normalizer.normalize(ref)
                hyp = self.normalizer.normalize(hyp)
            
            # Apply the same normalization as semantic_wer_metric
            _, normalized_ref, normalized_hyp = optimize_wer(
                ref, hyp, self.equivalents, 
                language=self.language, 
                normalizer=self.normalizer,
                glm_arrays=self.glm_arrays,
                pure_units_dict=self.units_dict,
                ortho_map=self.ortho_map,
            )
            
            # Count edit operations
            subs, dels, ins, ref_words = count_edit_operations_difflib(
                normalized_ref, normalized_hyp
            )
            
            with results_lock:
                total_substitutions += subs
                total_deletions += dels
                total_insertions += ins
                total_ref_words += ref_words
        
        # Use ThreadPoolExecutor for multithreading
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [
                executor.submit(process_pair, ref, hyp) 
                for ref, hyp in zip(references, predictions)
            ]
            for future in futures:
                future.result()
        
        return _rates_with_support(
            total_substitutions,
            total_deletions,
            total_insertions,
            total_ref_words,
            references,
        )
    
    def evaluate_single(
        self, 
        prediction: str, 
        reference: str
    ) -> Dict:
        """
        Evaluate a single sample and return detailed results.
        
        Returns:
            Dictionary with error counts and rates
        """
        if not reference:
            return {
                "substitutions": 0,
                "deletions": 0,
                "insertions": 0,
                "total_ref_words": 0,
                "substitution_rate": 0.0,
                "deletion_rate": 0.0,
                "insertion_rate": 0.0,
                "normalized_ref": "",
                "normalized_hyp": ""
            }
        
        if len(prediction.split()) > 2 * len(reference.split()):
            prediction = ' '.join(prediction.split()[:2 * len(reference.split())])
        
        ref = reference
        hyp = prediction
        
        if self.language != "en":
            ref = self.normalizer.normalize(ref)
            hyp = self.normalizer.normalize(hyp)
        
        # Apply the same normalization as semantic_wer_metric
        _, normalized_ref, normalized_hyp = optimize_wer(
            ref, hyp, self.equivalents, 
            language=self.language, 
            normalizer=self.normalizer,
            glm_arrays=self.glm_arrays,
            pure_units_dict=self.units_dict,
            ortho_map=self.ortho_map,
        )
        
        # Count edit operations
        subs, dels, ins, ref_words = count_edit_operations_difflib(
            normalized_ref, normalized_hyp
        )
        
        return {
            "substitutions": subs,
            "deletions": dels,
            "insertions": ins,
            "total_ref_words": ref_words,
            "substitution_rate": subs / ref_words if ref_words > 0 else 0.0,
            "deletion_rate": dels / ref_words if ref_words > 0 else 0.0,
            "insertion_rate": ins / ref_words if ref_words > 0 else 0.0,
            "normalized_ref": normalized_ref,
            "normalized_hyp": normalized_hyp
        }


@register_metric("error_analysis", required_args=["language"])
class ErrorAnalysisMetric(BaseMetric):
    """
    Basic Error Analysis metric that returns substitution, deletion, and insertion rates
    without the semantic normalizations. Uses basic preprocessing only.
    
    Returns:
        Dict with:
            - substitution_rate: substitutions / total_reference_words
            - deletion_rate: deletions / total_reference_words
            - insertion_rate: insertions / total_reference_words
    """
    
    def __init__(self, metric_name: str = "error_analysis", **kwargs):
        self.language, self.normalizer = resolve_indic_normalizer(
            kwargs.get("language"), do_normalize_chandras=True, remove_nuktas=True
        )
        super().__init__(metric_name, **kwargs)

    def evaluate(self, predictions: List[str], references: List[str], **kwargs) -> Dict[str, float]:
        """
        Evaluate multiple samples and return overall error rates.
        
        Returns:
            Dict with substitution_rate, deletion_rate, insertion_rate
        """
        total_substitutions = 0
        total_deletions = 0
        total_insertions = 0
        total_ref_words = 0
        
        for ref, hyp in zip(references, predictions):
            if not ref:
                continue
            
            # Basic normalization
            ref_normalized = ref.lower().strip()
            hyp_normalized = hyp.lower().strip()
            
            if self.language != "en" and self.normalizer:
                ref_normalized = self.normalizer.normalize(ref_normalized)
                hyp_normalized = self.normalizer.normalize(hyp_normalized)
            
            # Count edit operations
            subs, dels, ins, ref_words = count_edit_operations_difflib(
                ref_normalized, hyp_normalized
            )
            
            total_substitutions += subs
            total_deletions += dels
            total_insertions += ins
            total_ref_words += ref_words
        
        return _rates_with_support(
            total_substitutions,
            total_deletions,
            total_insertions,
            total_ref_words,
            references,
        )
    
    def evaluate_single(
        self, 
        prediction: str, 
        reference: str
    ) -> Dict:
        """
        Evaluate a single sample and return detailed results.
        
        Returns:
            Dictionary with error counts and rates
        """
        if not reference:
            return {
                "substitutions": 0,
                "deletions": 0,
                "insertions": 0,
                "total_ref_words": 0,
                "substitution_rate": 0.0,
                "deletion_rate": 0.0,
                "insertion_rate": 0.0
            }
        
        # Basic normalization
        ref_normalized = reference.lower().strip()
        hyp_normalized = prediction.lower().strip()
        
        if self.language != "en" and self.normalizer:
            ref_normalized = self.normalizer.normalize(ref_normalized)
            hyp_normalized = self.normalizer.normalize(hyp_normalized)
        
        # Count edit operations
        subs, dels, ins, ref_words = count_edit_operations_difflib(
            ref_normalized, hyp_normalized
        )
        
        return {
            "substitutions": subs,
            "deletions": dels,
            "insertions": ins,
            "total_ref_words": ref_words,
            "substitution_rate": subs / ref_words if ref_words > 0 else 0.0,
            "deletion_rate": dels / ref_words if ref_words > 0 else 0.0,
            "insertion_rate": ins / ref_words if ref_words > 0 else 0.0
        }


@register_metric("semantic_wer_by_quality", required_args=["language"])
class SemanticWERByQualityMetric(BaseMetric):
    """
    Semantic WER metric that computes WER grouped by quality values.
    
    Returns:
        Dict with WER for each quality value and overall statistics:
            - wer_by_quality: {quality_value: {"wer": float, "sample_count": int, "total_words": int}}
            - overall_wer: float
            - quality_distribution: {quality_value: count}
    """
    
    def __init__(self, metric_name: str = "semantic_wer_by_quality", **kwargs):
        self.language, self.normalizer = resolve_indic_normalizer(
            kwargs.get("language"), do_normalize_chandras=True, remove_nuktas=True
        )
        super().__init__(metric_name, **kwargs)
        
        # Get the directory of the current file
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        resources = load_semantic_resources(self.language)
        self.equivalents = resources.equivalents
        self.units_dict = resources.units_dict
        self.glm_arrays = resources.glm_arrays
        self.ortho_map = resources.ortho_map

    def _compute_wer_for_pair(self, ref: str, hyp: str) -> Tuple[int, int]:
        """
        Compute edit distance and reference word count for a single pair.
        
        Returns:
            Tuple of (edit_distance, reference_word_count)
        """
        if not ref:
            return 0, 0
        
        if len(hyp.split()) > 2 * len(ref.split()):
            hyp = ' '.join(hyp.split()[:2 * len(ref.split())])
        
        if self.language != "en":
            ref = self.normalizer.normalize(ref)
            hyp = self.normalizer.normalize(hyp)
        
        # Apply the same normalization as semantic_wer_metric
        wer_score, normalized_ref, normalized_hyp = optimize_wer(
            ref, hyp, self.equivalents, 
            language=self.language, 
            normalizer=self.normalizer,
            glm_arrays=self.glm_arrays,
            pure_units_dict=self.units_dict,
            ortho_map=self.ortho_map,
        )
        
        ref_words = normalized_ref.split()
        hyp_words = normalized_hyp.split()
        
        # Count edit operations to get total errors
        subs, dels, ins, ref_word_count = count_edit_operations_difflib(
            normalized_ref, normalized_hyp
        )
        
        total_errors = subs + dels + ins
        return total_errors, ref_word_count

    def evaluate(self, predictions: List[str], references: List[str], **kwargs) -> Dict:
        """
        Evaluate multiple samples and return WER grouped by quality.
        
        Args:
            predictions: List of hypothesis strings
            references: List of reference strings
            **kwargs: Must include 'qualities' - list of quality values for each sample
        
        Returns:
            Dict with WER statistics grouped by quality
        """
        qualities = kwargs.get("qualities", [])
        
        # If no qualities provided, return overall WER only
        if not qualities or len(qualities) != len(predictions):
            qualities = [None] * len(predictions)
        
        # Group samples by quality
        quality_groups: Dict[Any, Dict] = {}
        self._skipped_no_quality = 0

        # Process pairs with threading for performance
        results_lock = threading.Lock()
        
        def process_pair(idx: int, ref: str, hyp: str, quality):
            # No quality rating means this sample cannot belong to any bucket, so
            # it is skipped here rather than lumped into an "unknown" group that
            # would sit alongside the real 1-4 ratings and distort the breakdown.
            # The sample still counts in every other metric. NaN is checked too:
            # it arrives from pandas for a missing value, is not None, and would
            # otherwise become a nan dict key that never compares equal to itself.
            if quality is None or (isinstance(quality, float) and quality != quality):
                with results_lock:
                    self._skipped_no_quality += 1
                return

            errors, ref_words = self._compute_wer_for_pair(ref, hyp)

            with results_lock:
                quality_key = quality
                if quality_key not in quality_groups:
                    quality_groups[quality_key] = {
                        "total_errors": 0,
                        "total_ref_words": 0,
                        "sample_count": 0,
                        "scored_count": 0,
                    }
                quality_groups[quality_key]["total_errors"] += errors
                quality_groups[quality_key]["total_ref_words"] += ref_words
                quality_groups[quality_key]["sample_count"] += 1
                if ref:
                    quality_groups[quality_key]["scored_count"] += 1
        
        # Use ThreadPoolExecutor for multithreading
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [
                executor.submit(process_pair, idx, ref, hyp, qual) 
                for idx, (ref, hyp, qual) in enumerate(zip(references, predictions, qualities))
            ]
            for future in futures:
                future.result()
        
        # Calculate WER for each quality group
        wer_by_quality = {}
        field_support = {}
        overall_errors = 0
        overall_ref_words = 0
        quality_distribution = {}

        for quality_key, stats in quality_groups.items():
            total_errors = stats["total_errors"]
            total_ref_words = stats["total_ref_words"]
            sample_count = stats["sample_count"]

            wer = total_errors / total_ref_words if total_ref_words > 0 else 0.0

            field = f"Quality_{quality_key}_Semantic_WER"
            wer_by_quality[field] = round(wer, 4)
            field_support[field] = {
                "value": round(wer, 4),
                "scored": stats["scored_count"],
                "total": sample_count,
            }

            quality_distribution[quality_key] = sample_count
            overall_errors += total_errors
            overall_ref_words += total_ref_words

        overall_wer = overall_errors / overall_ref_words if overall_ref_words > 0 else 0.0

        if self._skipped_no_quality:
            print(
                f"semantic_wer_by_quality: skipped {self._skipped_no_quality} of "
                f"{len(predictions)} samples with no quality rating"
            )

        return {
            "semantic_wer_by_quality": wer_by_quality,
            "overall_wer": round(overall_wer, 4),
            "quality_distribution": quality_distribution,
            "total_samples": len(predictions),
            "skipped_no_quality": self._skipped_no_quality,
            "_field_support": field_support,
            "_overall_value": round(overall_wer, 4),
        }


class _EntityMetricBase(BaseMetric):
    """Shared base class for entity accuracy metrics. Not registered."""

    def __init__(self, metric_name: str, **kwargs):
        self.language, self.normalizer = resolve_indic_normalizer(
            kwargs.get("language"), do_normalize_chandras=True, remove_nuktas=True
        )

        super().__init__(metric_name, **kwargs)

        resources = load_semantic_resources(self.language)
        self.equivalents = resources.equivalents
        self.units_dict = resources.units_dict
        self.glm_arrays = resources.glm_arrays
        self.ortho_map = resources.ortho_map

    def _process_samples(
        self,
        predictions: List[str],
        references: List[str],
        entities: List[Dict],
        qualities: List[Any] = None,
    ) -> List[Dict]:
        """
        Process all samples and return a list of per-keyword row dicts.

        Each row contains:
            sample_idx, ref_keyword, hyp_keyword, total_words_in_ref,
            matched_words_in_hyp, is_exact_match, quality, keyword_type
        """
        from karma.metrics.wer_metrics import (
            normalize_text_for_keyword_extraction,
            normalize_keyword,
            extract_keywords_with_alignment,
            ENTITY_TYPES,
        )

        entities = [item if item else {} for item in entities]
        if qualities is None or len(qualities) != len(predictions):
            qualities = [None] * len(predictions)

        rows: List[Dict] = []
        results_lock = threading.Lock()

        def process_sample(sample_idx: int, ref: str, hyp: str, sample_entities: Dict, quality):
            if len(hyp.split()) > 2 * len(ref.split()):
                hyp = " ".join(hyp.split()[: 2 * len(ref.split())])

            normalized_ref = normalize_text_for_keyword_extraction(
                ref, self.equivalents, self.language, self.normalizer
            )
            # counterpart=ref so a keyword the model spelled out ("h b a 1 c")
            # still lines up with the reference's "hba1c" instead of counting as
            # a missed keyword
            normalized_hyp = normalize_text_for_keyword_extraction(
                hyp, self.equivalents, self.language, self.normalizer, counterpart=ref
            )

            local_rows = []
            for etype in ENTITY_TYPES:
                keywords = sample_entities.get(etype, [])
                if not keywords:
                    continue

                normalized_kws = [
                    normalize_keyword(kw, self.language, self.normalizer, self.equivalents)
                    for kw in keywords
                ]

                ref_kws, hyp_kws = extract_keywords_with_alignment(
                    normalized_ref, normalized_hyp, normalized_kws, self.glm_arrays
                )

                for ref_kw, hyp_kw in zip(ref_kws, hyp_kws):
                    ref_words_list = ref_kw.split()
                    hyp_words_list = hyp_kw.split() if hyp_kw else []
                    matched_words = sum(1 for w in ref_words_list if w in hyp_words_list)
                    local_rows.append(
                        {
                            "sample_idx": sample_idx,
                            "ref_keyword": ref_kw,
                            "hyp_keyword": hyp_kw,
                            "total_words_in_ref": len(ref_words_list),
                            "matched_words_in_hyp": matched_words,
                            "is_exact_match": ref_kw == hyp_kw,
                            "quality": quality,
                            "keyword_type": etype,
                        }
                    )

            with results_lock:
                rows.extend(local_rows)

        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [
                executor.submit(process_sample, idx, ref, hyp, ent, qual)
                for idx, (ref, hyp, ent, qual) in enumerate(
                    zip(references, predictions, entities, qualities)
                )
            ]
            for future in futures:
                future.result()

        return rows


@register_metric("keyword_accuracy_analysis", required_args=["language"])
class KeywordAccuracyAnalysisMetric(_EntityMetricBase):
    """
    Keyword Accuracy Analysis metric that returns a detailed DataFrame with:
    - Aligned reference keyword
    - Aligned hypothesis keyword  
    - Total words in reference keyword
    - Matched words in hypothesis
    - Quality value
    - Keyword type (entity type)
    
    This metric processes keywords by entity type (drugs, conditions, symptoms, etc.)
    and returns row-level analysis for each keyword occurrence.
    
    Args:
        language: Language code (e.g., 'en', 'hi', 'te')
    """
    
    def __init__(self, metric_name: str = "keyword_accuracy_analysis", **kwargs):
        super().__init__(metric_name, **kwargs)

    def evaluate(
        self,
        predictions: List[str],
        references: List[str],
        entities: List[Dict] = None,
        qualities: List[Any] = None,
        **kwargs,
    ):
        """
        Evaluate multiple samples and return a detailed DataFrame.

        Returns:
            {"keyword_accuracy_analysis": pd.DataFrame} with columns:
                sample_idx, ref_keyword, hyp_keyword, total_words_in_ref,
                matched_words_in_hyp, is_exact_match, quality, keyword_type
        """
        import pandas as pd

        if entities is None:
            entities = kwargs.get("entities", [])
        if qualities is None:
            qualities = kwargs.get("qualities", [None] * len(predictions))

        if not entities or len(entities) != len(predictions):
            raise ValueError(
                f"entities must be a list of dicts with same length as predictions. "
                f"Got {len(entities) if entities else 0} entity dicts for {len(predictions)} predictions."
            )

        if len(qualities) != len(predictions):
            qualities = [None] * len(predictions)

        rows = self._process_samples(predictions, references, entities, qualities)
        return {"keyword_accuracy_analysis": pd.DataFrame(rows)}
    
    def evaluate_single(
        self, 
        prediction: str, 
        reference: str, 
        entities: Dict[str, List[str]],
        quality: Any = None
    ):
        """
        Evaluate a single sample and return detailed keyword analysis.
        
        Args:
            prediction: Prediction string
            reference: Reference string
            entities: Dict with entity type keys and list of keywords as values
            quality: Quality value for this sample
            
        Returns:
            DataFrame with keyword analysis for this single sample
        """
        import pandas as pd
        from karma.metrics.wer_metrics import (
            normalize_text_for_keyword_extraction,
            normalize_keyword,
            extract_keywords_with_alignment,
            ENTITY_TYPES
        )
        
        # Truncate overly long predictions
        if len(prediction.split()) > 2 * len(reference.split()):
            prediction = ' '.join(prediction.split()[:2 * len(reference.split())])
        
        # Normalize text
        normalized_ref = normalize_text_for_keyword_extraction(
            reference, self.equivalents, self.language, self.normalizer
        )
        normalized_hyp = normalize_text_for_keyword_extraction(
            prediction, self.equivalents, self.language, self.normalizer
        )
        
        analysis_rows = []
        
        # Process each entity type
        for etype in ENTITY_TYPES:
            keywords = entities.get(etype, [])
            if not keywords:
                continue
            
            # Normalize keywords
            normalized_kws = [
                normalize_keyword(kw, self.language, self.normalizer, self.equivalents) 
                for kw in keywords
            ]
            
            # Extract keywords with alignment
            ref_kws, hyp_kws = extract_keywords_with_alignment(
                normalized_ref, normalized_hyp, normalized_kws, self.glm_arrays
            )
            
            # Create a row for each keyword
            for ref_kw, hyp_kw in zip(ref_kws, hyp_kws):
                ref_words = ref_kw.split()
                hyp_words = hyp_kw.split() if hyp_kw else []
                
                # Count matched words (words that appear in both ref and hyp keyword)
                matched_words = sum(1 for word in ref_words if word in hyp_words)
                
                analysis_rows.append({
                    "ref_keyword": ref_kw,
                    "hyp_keyword": hyp_kw,
                    "total_words_in_ref": len(ref_words),
                    "matched_words_in_hyp": matched_words,
                    "quality": quality,
                    "keyword_type": etype
                })
        
        # Create DataFrame
        df = pd.DataFrame(analysis_rows)

        return df


# ============================================================================
# ENTITY ACCURACY METRICS
# ============================================================================

@register_metric("entity_word_level_accuracy", required_args=["language"])
class EntityWordLevelAccuracyMetric(_EntityMetricBase):
    """
    Entity Word Level Accuracy (fuzzy match).

    For each entity keyword, counts what fraction of the reference keyword's
    words appear in the aligned hypothesis keyword.

        entity_word_level_accuracy = sum(matched_words_in_hyp) / sum(total_words_in_ref)

    Returns a single float between 0 and 1.
    """

    def __init__(self, metric_name: str = "entity_word_level_accuracy", **kwargs):
        super().__init__(metric_name, **kwargs)

    def evaluate(
        self,
        predictions: List[str],
        references: List[str],
        entities: List[Dict] = None,
        **kwargs,
    ) -> float:
        if entities is None:
            entities = kwargs.get("entities", [])
        if not entities or len(entities) != len(predictions):
            raise ValueError(
                f"entities must be a list of dicts with same length as predictions. "
                f"Got {len(entities) if entities else 0} entity dicts for {len(predictions)} predictions."
            )
        rows = self._process_samples(predictions, references, entities)
        total_ref_words = sum(r["total_words_in_ref"] for r in rows)
        total_matched = sum(r["matched_words_in_hyp"] for r in rows)
        return total_matched / total_ref_words if total_ref_words > 0 else 0.0


@register_metric("entity_word_level_accuracy_by_type", required_args=["language"])
class EntityWordLevelAccuracyByTypeMetric(_EntityMetricBase):
    """
    Entity Word Level Accuracy broken down by entity type (fuzzy match).

    Same computation as entity_word_level_accuracy but returns a per-type dict.

    Returns:
        {"entity_word_level_accuracy_by_type": {entity_type: float|None, ...}}
        None means no keywords of that type were present.
    """

    def __init__(self, metric_name: str = "entity_word_level_accuracy_by_type", **kwargs):
        super().__init__(metric_name, **kwargs)

    def evaluate(
        self,
        predictions: List[str],
        references: List[str],
        entities: List[Dict] = None,
        **kwargs,
    ) -> Dict:
        from karma.metrics.wer_metrics import ENTITY_TYPES

        if entities is None:
            entities = kwargs.get("entities", [])
        if not entities or len(entities) != len(predictions):
            raise ValueError(
                f"entities must be a list of dicts with same length as predictions. "
                f"Got {len(entities) if entities else 0} entity dicts for {len(predictions)} predictions."
            )
        rows = self._process_samples(predictions, references, entities)

        result = {}
        field_support = {}
        for etype in ENTITY_TYPES:
            etype_rows = [r for r in rows if r["keyword_type"] == etype]
            if etype_rows:
                total_ref = sum(r["total_words_in_ref"] for r in etype_rows)
                total_matched = sum(r["matched_words_in_hyp"] for r in etype_rows)
                value = total_matched / total_ref if total_ref > 0 else None
                result[f"{etype}_ewa"] = value
                if value is not None:
                    covered = {r["sample_idx"] for r in etype_rows}
                    field_support[f"{etype}_ewa"] = {
                        "value": value,
                        "scored": len(covered),
                        "total": len(covered),
                    }
            else:
                result[f"{etype}_ewa"] = None
        return {
            "entity_word_level_accuracy_by_type": result,
            "_field_support": field_support,
        }


@register_metric("entity_accuracy", required_args=["language"])
class EntityAccuracyMetric(_EntityMetricBase):
    """
    Entity Accuracy (hard / exact match).

    An entity is counted as correct only when its entire text matches exactly
    between the aligned reference and hypothesis keywords.

        entity_accuracy = count(exact_matches) / total_entities

    Returns a single float between 0 and 1.
    """

    def __init__(self, metric_name: str = "entity_accuracy", **kwargs):
        super().__init__(metric_name, **kwargs)

    def evaluate(
        self,
        predictions: List[str],
        references: List[str],
        entities: List[Dict] = None,
        **kwargs,
    ) -> float:
        if entities is None:
            entities = kwargs.get("entities", [])
        if not entities or len(entities) != len(predictions):
            raise ValueError(
                f"entities must be a list of dicts with same length as predictions. "
                f"Got {len(entities) if entities else 0} entity dicts for {len(predictions)} predictions."
            )
        rows = self._process_samples(predictions, references, entities)
        if not rows:
            return 0.0
        exact_matches = sum(1 for r in rows if r["is_exact_match"])
        return exact_matches / len(rows)


@register_metric("entity_accuracy_by_type", required_args=["language"])
class EntityAccuracyByTypeMetric(_EntityMetricBase):
    """
    Entity Accuracy broken down by entity type (hard / exact match).

    Same computation as entity_accuracy but returns a per-type dict.

    Returns:
        {"entity_accuracy_by_type": {entity_type: float|None, ...}}
        None means no keywords of that type were present.
    """

    def __init__(self, metric_name: str = "entity_accuracy_by_type", **kwargs):
        super().__init__(metric_name, **kwargs)

    def evaluate(
        self,
        predictions: List[str],
        references: List[str],
        entities: List[Dict] = None,
        **kwargs,
    ) -> Dict:
        from karma.metrics.wer_metrics import ENTITY_TYPES

        if entities is None:
            entities = kwargs.get("entities", [])
        if not entities or len(entities) != len(predictions):
            raise ValueError(
                f"entities must be a list of dicts with same length as predictions. "
                f"Got {len(entities) if entities else 0} entity dicts for {len(predictions)} predictions."
            )
        rows = self._process_samples(predictions, references, entities)

        result = {}
        field_support = {}
        for etype in ENTITY_TYPES:
            etype_rows = [r for r in rows if r["keyword_type"] == etype]
            if etype_rows:
                exact_matches = sum(1 for r in etype_rows if r["is_exact_match"])
                value = exact_matches / len(etype_rows)
                result[f"{etype}_ea"] = value
                covered = {r["sample_idx"] for r in etype_rows}
                field_support[f"{etype}_ea"] = {
                    "value": value,
                    "scored": len(covered),
                    "total": len(covered),
                }
            else:
                result[f"{etype}_ea"] = None
        return {
            "entity_accuracy_by_type": result,
            "_field_support": field_support,
        }