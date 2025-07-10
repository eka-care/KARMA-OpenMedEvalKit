#!/usr/bin/env python3
"""
CER-based word alignment across languages.
ASR Metrics class for evaluating speech recognition performance.
"""

import sys
import os
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from karma.metrics.base_metric_abs import BaseMetric
from karma.registries.metrics_registry import register_metric
# Import language-specific aligners
from karma.metrics.asr.base_aligner import BaseCERAligner
from karma.metrics.asr.lang.english_aligner import EnglishCERAligner
from karma.metrics.asr.lang.hindi_aligner import HindiCERAligner

@dataclass
class EvalResult:
    wer: float
    cer: float
    entity_wer: Optional[float] = None
    num_sentences: int = 0
    total_ref_words: int = 0
    additional_info: Optional[Dict] = None

@register_metric(
    name = "asr_semantic_metric",
    required_args = ["language"]
)
class ASRSemanticMetrics(BaseMetric):
    def __init__(self, metric_name: str, language = "en", **kwargs):
        super().__init__(metric_name, **kwargs)
        self.language = language

    @staticmethod
    def get_aligner(language: str, cer_threshold: float = 0.4) -> BaseCERAligner:
        """Factory function to get language-specific aligner."""
        aligners = {
            'english': EnglishCERAligner,
            'hindi': HindiCERAligner,
            'en': EnglishCERAligner,
            'hi': HindiCERAligner,
        }
        
        if language.lower() not in aligners:
            available = ', '.join(aligners.keys())
            raise ValueError(f"Unsupported language: {language}. Available: {available}")
        
        return aligners[language.lower()](cer_threshold)

    @staticmethod
    def process_files(aligner: BaseCERAligner, predictions: List[str], references: List[str]) -> EvalResult:
        """Process reference and hypothesis files with utterance ID alignment."""
        if len(predictions) != len(references):
            raise ValueError(f"Mismatch in ref/hyp count: {len(references)} refs vs {len(predictions)} predictions")
        
        if len(references) == 0:
            print("No data to process!")
            return EvalResult(
                wer=0.0, cer=0.0, num_sentences=0, total_ref_words=0
            )

        # Initialize accumulators
        total_ref_words = 0
        total_correct = 0
        total_substitutions = 0
        total_deletions = 0
        total_insertions = 0
        total_ref_chars = 0
        total_edit_distance = 0
        processed_count = 0
            
        # Process each utterance pair
        for idx, (ref, hyp) in enumerate(zip(references, predictions)):  # FIXED: was zip(references, references)
            utt_id = f"utterance_{idx}"  # FIXED: Added utt_id definition
            
            try:
                alignments = aligner.align_words_dp(ref, hyp)
                
                # Print results
                aligner.print_alignment_visual(alignments)
                
                # Calculate statistics
                stats = aligner.calculate_error_rates(alignments)
                
                # Accumulate dataset-wide statistics
                total_ref_words += stats['total_ref_words']
                total_correct += stats['word_correct']
                total_substitutions += stats['word_substitutions']
                total_deletions += stats['word_deletions']
                total_insertions += stats['word_insertions']
                
                # Accumulate character-level statistics for weighted CER
                for alignment in alignments:
                    if alignment.ref_words and alignment.hyp_words:
                        ref_text = ' '.join(alignment.ref_words)
                        ref_chars = len(ref_text.replace(' ', ''))
                        total_ref_chars += ref_chars
                        total_edit_distance += alignment.character_error_rate * ref_chars
                
                processed_count += 1
                
                # Optional: Add file-specific stats to output (using utterance ID instead of index)
                # file_stats_lines.append(
                #     f"{utt_id}\t{stats['total_ref_words']}\t{stats['word_correct']}\t"
                #     f"{stats['word_substitutions']}\t{stats['word_deletions']}\t{stats['word_insertions']}\t"
                #     f"{stats['wer']:.4f}\t{stats['weighted_word_cer']:.4f}"
                # )
                
            except Exception as e:
                print(f"Error processing utterance {utt_id}: {e}")
                print("Skipping this utterance...")
                continue
                
        # Calculate dataset-wide statistics
        dataset_wer = (total_substitutions + total_deletions + total_insertions) / max(total_ref_words, 1)
        dataset_weighted_cer = total_edit_distance / total_ref_chars if total_ref_chars > 0 else 0.0
    
        # Optional: Write output files
        # with open(file_stats_output, 'w', encoding='utf-8') as f:
        #     f.write('\n'.join(file_stats_lines))
        # print(f"\nFile-specific statistics written to: {file_stats_output}")
        
        return EvalResult(
            wer=dataset_wer, 
            cer=dataset_weighted_cer, 
            num_sentences=processed_count
        )

    def evaluate(self, predictions: List[str], references: List[str], **kwargs) -> EvalResult:
        """
        Evaluate ASR predictions against references.
        
        Args:
            predictions: List of hypothesis transcriptions
            references: List of reference transcriptions
            **kwargs: Additional parameters including:
                - language: Language code (e.g., 'english', 'hindi', 'en', 'hi') - REQUIRED
                - cer_threshold: CER threshold for alignment (default: 0.4)
                
        Returns:
            Dictionary containing evaluation results
        """
        # Extract language from cli
        language = self.language
        if not language:
            raise ValueError("Language parameter is required. Pass it via cli: 'asr_metric:language=hi'")
        
        try:
            print(f"Creating {language} aligner...")
            # Get language-specific aligner
            cer_threshold = kwargs.get('cer_threshold')
            aligner = self.get_aligner(language)
            print(f"Aligner created successfully")
            
            # Validate inputs
            if not predictions or not references:
                raise ValueError("Both predictions and references must be non-empty lists")
            
            # Process files
            print(f"Processing {len(references)} utterances...")
            results = self.process_files(aligner, predictions, references)  # FIXED: Added self.
            
            return EvalResult(
                wer=results.wer,
                cer=results.cer,
                num_sentences=results.num_sentences
            )
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            raise  # Re-raise instead of sys.exit for better error handling