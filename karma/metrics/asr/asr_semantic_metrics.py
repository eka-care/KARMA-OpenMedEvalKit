#!/usr/bin/env python3
"""
CER-based word alignment across languages.
ASR Metrics class for evaluating speech recognition performance.
"""

import logging
from collections import Counter
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count

from karma.metrics.base_metric_abs import BaseMetric
from karma.registries.metrics_registry import register_metric

# Import language-specific aligners
from karma.metrics.asr.base_aligner import BaseCERAligner
from karma.metrics.asr.lang.english_aligner import EnglishCERAligner
from karma.metrics.asr.lang.hindi_aligner import HindiCERAligner
logger = logging.getLogger(__name__)

@dataclass
class EvalResult:
    semantic_wer: float
    semantic_cer: float
    entity_wer: Optional[float] = None
    num_sentences: int = 0
    total_ref_words: int = 0
    additional_info: Optional[Dict] = None

@dataclass
class AlignmentResult:
    """Result from processing a single utterance alignment."""
    success: bool
    utt_id: str
    total_ref_words: int = 0
    word_correct: int = 0
    word_substitutions: int = 0
    word_deletions: int = 0
    word_insertions: int = 0
    ref_chars: int = 0
    edit_distance: float = 0.0
    error: Optional[str] = None

@dataclass
class DatasetMetrics:
    """Aggregated metrics for the entire dataset."""
    total_ref_words: int
    total_correct: int
    total_substitutions: int
    total_deletions: int
    total_insertions: int
    total_ref_chars: int
    total_edit_distance: float
    processed_count: int
    
    @property
    def wer(self) -> float:
        """Calculate Word Error Rate."""
        total_errors = self.total_substitutions + self.total_deletions + self.total_insertions
        return total_errors / max(self.total_ref_words, 1)
    
    @property
    def cer(self) -> float:
        """Calculate Character Error Rate."""
        return self.total_edit_distance / self.total_ref_chars if self.total_ref_chars > 0 else 0.0

@register_metric(
    name = "asr_semantic_metric",
#  required_args = ["language"]     #obtained from the dataset argument
)
class ASRSemanticMetrics(BaseMetric):
    def __init__(self, metric_name: str, language = "en", **kwargs):
        super().__init__(metric_name, **kwargs)
        self.language = language
        self.cer_threshold = 0.4

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
        
        aligner = aligners[language.lower()](cer_threshold)
            # Add keyword WER method to the instance
        def calculate_keyword_wer(reference_text: str, reference_annotations: str, hypothesis_text: str):
            """Calculate keyword-specific WER using existing error rate calculation."""
            alignments = aligner.process_keywords_for_wer(
                reference_text, reference_annotations, hypothesis_text, aligner
            )
            results = aligner.calculate_error_rates(alignments)
            aligner.print_alignment_visual(alignments) #enable for debugging
            return results
        
        # Bind the method to the aligner instance
        aligner.calculate_keyword_wer = calculate_keyword_wer
        
        return aligner

    @staticmethod
    def _process_utterance_regular(args: Tuple) -> AlignmentResult:
        """
        Process a single utterance for regular WER calculation.
        
        Args:
            args: Tuple of (idx, reference, hypothesis, language, cer_threshold)
            
        Returns:
            AlignmentResult with alignment statistics
        """
        idx, ref, hyp, language, cer_threshold = args
        utt_id = f"utterance_{idx}"
        
        try:
            # Create aligner in the worker process to avoid pickling issues
            aligner = ASRSemanticMetrics.get_aligner(language, cer_threshold)
            alignments = aligner.align_words_dp(ref, hyp)

            aligner.print_alignment_visual(alignments) #to be used for debugging

            # Calculate statistics
            stats = aligner.calculate_error_rates(alignments)
            
            # Calculate character-level statistics for weighted CER
            ref_chars = 0
            edit_distance = 0
            for alignment in alignments:
                if alignment.ref_words and alignment.hyp_words:
                    ref_text = ' '.join(alignment.ref_words)
                    chars = len(ref_text.replace(' ', ''))
                    ref_chars += chars
                    edit_distance += alignment.character_error_rate * chars
            
            return AlignmentResult(
                success=True,
                utt_id=utt_id,
                total_ref_words=stats['total_ref_words'],
                word_correct=stats['word_correct'],
                word_substitutions=stats['word_substitutions'],
                word_deletions=stats['word_deletions'],
                word_insertions=stats['word_insertions'],
                ref_chars=ref_chars,
                edit_distance=edit_distance
            )
            
        except Exception as e:
            logger.error(f"Error processing utterance {utt_id}: {e}")
            return AlignmentResult(
                success=False,
                utt_id=utt_id,
                error=str(e)
            )

    @staticmethod
    def _process_utterance_keyword(args: Tuple) -> AlignmentResult:
        """
        Process a single utterance for keyword WER calculation.
        
        Args:
            args: Tuple of (idx, reference, hypothesis, annotations, language, cer_threshold)
            
        Returns:
            AlignmentResult with alignment statistics
        """
        idx, ref, hyp, annotations, language, cer_threshold = args
        utt_id = f"utterance_{idx}"

        try:
            aligner = ASRSemanticMetrics.get_aligner(language, cer_threshold)
            try:
                annotation_list = eval(annotations)  # Use ast.literal_eval for safety in production
            except:
                raise ValueError("Could not parse annotations")
                # Extract keywords
            reference_keywords = aligner.extract_keywords_from_text(ref, annotation_list)

            alignments = aligner.align_keywords(reference_keywords, hyp)
            # Print results
            aligner.print_alignment_visual(alignments) #to be used for debugging
            # Calculate statistics
            stats = aligner.calculate_error_rates(alignments)
            # Accumulate character-level statistics for weighted CER
            total_ref_chars = 0
            ref_chars = 0
            total_edit_distance = 0
            for alignment in alignments:
                if alignment.ref_words and alignment.hyp_words:
                    ref_text = ' '.join(alignment.ref_words)
                    ref_chars = len(ref_text.replace(' ', ''))
                    total_ref_chars += ref_chars
                    total_edit_distance += alignment.character_error_rate * ref_chars
                
            return AlignmentResult(
                success=True,
                utt_id=utt_id,
                total_ref_words=stats['total_ref_words'],
                word_correct=stats['word_correct'],
                word_substitutions=stats['word_substitutions'],
                word_deletions=stats['word_deletions'],
                word_insertions=stats['word_insertions'],
                ref_chars=ref_chars,
                edit_distance=total_edit_distance
            )                
                # Optional: Add file-specific stats to output (using utterance ID instead of index)
                # file_stats_lines.append(
                #     f"{utt_id}\t{stats['total_ref_words']}\t{stats['word_correct']}\t"
                #     f"{stats['word_substitutions']}\t{stats['word_deletions']}\t{stats['word_insertions']}\t"
                #     f"{stats['wer']:.4f}\t{stats['weighted_word_cer']:.4f}"
                # )
        except Exception as e:
            return AlignmentResult(
                success=False,
                utt_id=utt_id,
                error=str(e)
            )

    @staticmethod
    def process_for_wer(aligner: BaseCERAligner, predictions: List[str], 
                       references: List[str]) -> EvalResult:
        """
        Process reference and hypothesis pairs for regular WER calculation.
        
        Args:
            aligner: Language-specific aligner instance
            predictions: List of hypothesis texts
            references: List of reference texts
            
        Returns:
            EvalResult with semantic WER and CER metrics
        """
        logger.info(f"Processing {len(references)} utterances for regular WER...")
        
        if not references:
            logger.info("No data to process!")
            return EvalResult(
                semantic_wer=0.0, semantic_cer=0.0, num_sentences=0, total_ref_words=0
            )

        # Prepare for multiprocessing
        args_list, num_processes = ASRSemanticMetrics._prepare_multiprocessing(
            aligner, predictions, references
        )        
        # Process utterances in parallel
        with Pool(processes=num_processes) as pool:
            results = pool.map(ASRSemanticMetrics._process_utterance_regular, args_list)

        # Aggregate results
        metrics = ASRSemanticMetrics._aggregate_results(results)
        
        logger.info(f"Processed {metrics.processed_count}/{len(references)} utterances successfully")
     
        # Optional: Write output files
        # with open(file_stats_output, 'w', encoding='utf-8') as f:
        #     f.write('\n'.join(file_stats_lines))
        # print(f"\nFile-specific statistics written to: {file_stats_output}")
        
        return EvalResult(
            semantic_wer=round(metrics.wer,3), 
            semantic_cer=round(metrics.cer,3), 
            num_sentences=metrics.processed_count,
            total_ref_words=metrics.total_ref_words,
        )
    
    @staticmethod
    def process_keywords_for_wer(aligner: BaseCERAligner, predictions: List[str], 
                                references: List[str], entities: List[str]) -> float:
        """
        Process reference, hypothesis and entity annotations for keyword WER calculation.
        
        Args:
            aligner: Language-specific aligner instance
            predictions: List of hypothesis texts
            references: List of reference texts
            entities: List of entity annotations
            
        Returns:
            Keyword WER as a float
        """
        logger.info(f"Processing {len(references)} utterances for keyword WER...")
        
        if not references:
            logger.warning("No data to process!")
            return 0.0
        
        # Prepare for multiprocessing
        args_list, num_processes = ASRSemanticMetrics._prepare_multiprocessing(
            aligner, predictions, references, entities
        )
        
        # Process utterances in parallel
        with Pool(processes=num_processes) as pool:
            results = pool.map(ASRSemanticMetrics._process_utterance_keyword, args_list)
        
        # Aggregate results
        metrics = ASRSemanticMetrics._aggregate_results(results)
        
        logger.info(f"Processed {metrics.processed_count}/{len(references)} utterances successfully")
        logger.info(f"Total keywords: {metrics.total_ref_words}, Keyword WER: {metrics.wer:.3f}")
        
        return round(metrics.wer, 3)

    @staticmethod
    def _aggregate_results(results: List[AlignmentResult]) -> DatasetMetrics:
        """
        Aggregate individual utterance results into dataset metrics.
        
        Args:
            results: List of AlignmentResult objects
            
        Returns:
            DatasetMetrics with aggregated statistics
        """
        metrics = DatasetMetrics(
            total_ref_words=0,
            total_correct=0,
            total_substitutions=0,
            total_deletions=0,
            total_insertions=0,
            total_ref_chars=0,
            total_edit_distance=0.0,
            processed_count=0
        )
        
        for result in results:
            if result.success:
                metrics.total_ref_words += result.total_ref_words
                metrics.total_correct += result.word_correct
                metrics.total_substitutions += result.word_substitutions
                metrics.total_deletions += result.word_deletions
                metrics.total_insertions += result.word_insertions
                metrics.total_ref_chars += result.ref_chars
                metrics.total_edit_distance += result.edit_distance
                metrics.processed_count += 1
            else:
                logger.warning(f"Skipping {result.utt_id} due to error: {result.error}")
        
        return metrics
    
    @staticmethod
    def _prepare_multiprocessing(aligner: BaseCERAligner, predictions: List[str], 
                                references: List[str], entities: Optional[List[str]] = None) -> Tuple[List[Tuple], int]:
        """
        Prepare arguments for multiprocessing.
        
        Args:
            aligner: Language-specific aligner instance
            predictions: List of hypothesis texts
            references: List of reference texts
            entities: Optional list of entity annotations
            
        Returns:
            Tuple of (args_list, num_processes)
        """
        # Validate inputs
        if len(predictions) != len(references):
            raise ValueError(f"Mismatch: {len(references)} references vs {len(predictions)} predictions")
        
        if entities and len(entities) != len(references):
            raise ValueError(f"Mismatch: {len(references)} references vs {len(entities)} entities")
        
        # Get aligner parameters
        language = aligner.__class__.__name__.replace('CERAligner', '').lower()
        if language == 'english':
            language = 'en'
        elif language == 'hindi':
            language = 'hi'
        cer_threshold = aligner.cer_threshold
        
        # Determine number of processes
        num_processes = max(1, int(cpu_count() * 0.75))
        
        # Create argument tuples
        if entities:
            args_list = [
                (idx, ref, hyp, ent, language, cer_threshold) 
                for idx, (ref, hyp, ent) in enumerate(zip(references, predictions, entities))
            ]
        else:
            args_list = [
                (idx, ref, hyp, language, cer_threshold) 
                for idx, (ref, hyp) in enumerate(zip(references, predictions))
            ]
        
        return args_list, num_processes

    def evaluate(self, predictions: List[str], references: List[str], **kwargs) -> EvalResult:
        """
        Evaluate ASR predictions against references.
        
        Args:
            predictions: List of hypothesis transcriptions
            references: List of reference transcriptions
            **kwargs: Additional parameters including:
                - language: Language code (e.g., 'english', 'hindi', 'en', 'hi') - REQUIRED
                
        Returns:
            Dictionary containing evaluation results
        """
        # Extract language from cli
        language = kwargs.get("language")
        entities = kwargs.get("entities", [])
        
        
        # Ensure language is a string
        if not isinstance(language, str):
            raise ValueError(f"Language parameter must be a string, got {type(language)}")
        # Validate inputs
        if not predictions or not references:
            raise ValueError("Both predictions and references must be non-empty lists")
            
        try:
            logger.info(f"Creating {language} aligner...")
            # Get language-specific aligner
            aligner = self.get_aligner(language)
            logger.info(f"Aligner created successfully")
            
            # Validate inputs
            if not predictions or not references:
                raise ValueError("Both predictions and references must be non-empty lists")
            
            # Process files
            logger.info(f"Processing {len(references)} utterances...")
            logger.info("Using default aligner to obtain semWER")
            results = self.process_for_wer(aligner, predictions, references) 
            
            if entities:
                logger.info("Entities found, using keyword aligner")
                entity_wer = self.process_keywords_for_wer(aligner, predictions, references, entities) 
                results.entity_wer = entity_wer
            return results
            
        except Exception as e:
            logger.error(f"Error: {e}")
            raise  # Re-raise instead of sys.exit for better error handling