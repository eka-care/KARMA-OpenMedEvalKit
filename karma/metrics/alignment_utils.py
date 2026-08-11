"""
Keyword Alignment Analysis Utilities

This module provides functions for aligning and analyzing keywords between
reference and hypothesis texts, with support for various normalizations.
"""

from typing import List, Tuple, Dict
from karma.metrics.wer_utils import (
    get_word_alignment,
    find_keyword_positions,
)


# ============================================================================
# EDIT DISTANCE AND SIMILARITY
# ============================================================================











# ============================================================================
# HIGH-LEVEL ALIGNMENT FUNCTIONS
# ============================================================================

def get_keyword_alignments(
    predictions: List[str],
    references: List[str],
    entities: List[Dict],
    language: str = 'en'
) -> List[List[Tuple]]:
    """
    Standalone function to get detailed keyword alignments with all normalization steps.
    
    This is a convenience function that doesn't require instantiating a metric class directly.
    
    Args:
        predictions: List of prediction strings
        references: List of reference strings
        entities: List of dicts with entity type keys, each containing list of keywords
            Example: [
                {"drugs": ["paracetamol 500mg"], "conditions": ["fever"]},
                {"symptoms": ["headache"]}
            ]
        language: Language code (e.g., 'en', 'hi', 'te')
        
    Returns:
        List of lists, where each inner list contains tuples for each keyword:
        [
            [  # Sample 1
                (original_keyword, normalized_keyword, ref_found, hyp_aligned, match_status),
                (original_keyword, normalized_keyword, ref_found, hyp_aligned, match_status),
                ...
            ],
            [  # Sample 2
                ...
            ],
            ...
        ]
        
        Match status can be: "match", "substituted", or "deleted"
        
    Example:
        >>> from karma.metrics.alignment_utils import get_keyword_alignments
        >>> 
        >>> alignments = get_keyword_alignments(
        ...     predictions=["थायरोइड चेक हुआ हंड्रेड एन सी जी दवा"],
        ...     references=["थाइरोइड चेक हुआ एक सौ एमसीजी दवा"],
        ...     entities=[{"conditions": ["थाइरोइड"], "drugs": ["एक सौ एमसीजी दवा"]}],
        ...     language="hi"
        ... )
        >>> 
        >>> # Print alignments for each sample
        >>> for sample_idx, sample_alignments in enumerate(alignments):
        ...     print(f"Sample {sample_idx + 1}:")
        ...     for orig, norm, ref, hyp, status in sample_alignments:
        ...         print(f"  Original: {orig}")
        ...         print(f"    Normalized: {norm}")
        ...         print(f"    Found in ref: {ref}")
        ...         print(f"    Aligned in hyp: {hyp}")
        ...         print(f"    Status: {status}")
        ...         print()
    """
    # Import here to avoid circular imports
    from karma.metrics.analysis_metrics import KeywordAccuracyAnalysisMetric

    # Create metric instance
    metric = KeywordAccuracyAnalysisMetric(language=language)
    
    # Use the class method
    return metric.get_detailed_alignments(predictions, references, entities)


def get_keyword_alignments_df(
    predictions: List[str],
    references: List[str],
    entities: List[Dict],
    language: str = 'en',
    max_workers: int = 16
):
    """
    Get detailed keyword alignments as a pandas DataFrame with multithreading.
    
    Returns a DataFrame with columns:
    - sample_idx: Index of the sample
    - original_keyword: The keyword as provided in entities
    - normalized_keyword: After all normalizations applied
    - ref_keyword: Normalized keyword found in reference
    - hyp_keyword: Normalized keyword aligned in hypothesis
    - match_status: "match", "substituted", or "deleted"
    
    Args:
        predictions: List of prediction strings
        references: List of reference strings
        entities: List of dicts with entity type keys, each containing list of keywords
        language: Language code (e.g., 'en', 'hi', 'te')
        max_workers: Maximum number of threads for parallel processing (default: 16)
        
    Returns:
        pandas DataFrame with alignment details
        
    Example:
        >>> import pandas as pd
        >>> from karma.metrics.alignment_utils import get_keyword_alignments_df
        >>> 
        >>> df = get_keyword_alignments_df(
        ...     predictions=["थायरोइड चेक हुआ हंड्रेड एन सी जी दवा"],
        ...     references=["थाइरोइड चेक हुआ एक सौ एमसीजी दवा"],
        ...     entities=[{"conditions": ["थाइरोइड"], "drugs": ["एक सौ एमसीजी दवा"]}],
        ...     language="hi"
        ... )
        >>> 
        >>> # View the DataFrame
        >>> print(df)
        >>> 
        >>> # Filter mismatches
        >>> mismatches = df[df['match_status'] != 'match']
        >>> print(mismatches)
        >>> 
        >>> # Group by status
        >>> print(df.groupby('match_status').size())
    """
    import pandas as pd
    from concurrent.futures import ThreadPoolExecutor
    
    # Import here to avoid circular imports
    from karma.metrics.analysis_metrics import KeywordAccuracyAnalysisMetric
    from karma.metrics.wer_metrics import (
        normalize_keyword,
        normalize_text_for_keyword_extraction,
        # the wer_metrics copy is the one that GLM-normalizes mismatched pairs
        extract_keywords_with_alignment,
    )

    # Create metric instance (shared across threads)
    metric = KeywordAccuracyAnalysisMetric(language=language)

    def process_sample(sample_idx, ref, hyp, sample_entities):
        """Process a single sample and return list of row dicts."""
        sample_keywords = sum(sample_entities.values(), []) if sample_entities else []
        
        if not sample_keywords:
            return []
        
        # Normalize keywords for this sample
        normalized_kws = [
            normalize_keyword(kw, metric.language, metric.normalizer, metric.equivalents) 
            for kw in sample_keywords
        ]
        
        # Truncate overly long predictions
        if len(hyp.split()) > 2 * len(ref.split()):
            hyp = ' '.join(hyp.split()[:2 * len(ref.split())])
        
        # Normalize text
        normalized_ref = normalize_text_for_keyword_extraction(
            ref, metric.equivalents, metric.language, metric.normalizer
        )
        normalized_hyp = normalize_text_for_keyword_extraction(
            hyp, metric.equivalents, metric.language, metric.normalizer, counterpart=ref
        )

        # Extract keywords with alignment-based matching
        ref_kws_in_text_order, hyp_kws_in_text_order = extract_keywords_with_alignment(
            normalized_ref, normalized_hyp, normalized_kws
        )
        
        # Build a mapping from normalized keyword to its (ref, hyp) pair
        keyword_map = {}
        for ref_kw, hyp_kw in zip(ref_kws_in_text_order, hyp_kws_in_text_order):
            if ref_kw in normalized_kws and ref_kw not in keyword_map:
                keyword_map[ref_kw] = (ref_kw, hyp_kw)
        
        # Build row dicts for each keyword IN REQUEST ORDER
        rows = []
        for orig_kw, norm_kw in zip(sample_keywords, normalized_kws):
            # Get the ref and hyp for this specific keyword
            if norm_kw in keyword_map:
                ref_kw, hyp_kw = keyword_map[norm_kw]
            else:
                ref_kw, hyp_kw = '', ''
            
            # Determine match status
            if ref_kw == hyp_kw and ref_kw != '':
                status = "match"
            elif hyp_kw == '' or ref_kw == '':
                status = "deleted"
            else:
                status = "substituted"
            
            rows.append({
                'sample_idx': sample_idx,
                'original_keyword': orig_kw,
                'normalized_keyword': norm_kw,
                'ref_keyword': ref_kw,
                'hyp_keyword': hyp_kw,
                'match_status': status
            })
        
        return rows
    
    # Process all samples in parallel
    all_rows = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_sample, idx, ref, hyp, ent)
            for idx, (ref, hyp, ent) in enumerate(zip(references, predictions, entities))
        ]
        
        # Collect results in order
        for future in futures:
            rows = future.result()
            all_rows.extend(rows)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_rows)
    
    return df


def get_word_error_counts(
    predictions: List[str],
    references: List[str],
    word: str,
    language: str = 'en',
    normalize: bool = True,
    max_workers: int = 16
) -> Dict[str, int]:
    """
    Count substitutions and deletions for a specific word across all prediction/reference pairs.

    Uses word-level alignment to determine how each occurrence of the word in the reference
    was handled in the prediction.

    Args:
        predictions: List of prediction strings
        references: List of reference strings
        word: The word to track
        language: Language code (e.g., 'en', 'hi', 'te')
        normalize: If True (default), apply full text normalization before matching —
            suitable for regular medical terms. Set to False for literal/special tokens
            like annotation tags (e.g., '<is>', '<filler>') that normalization would
            corrupt by stripping angle brackets, causing false matches with regular words.
        max_workers: Maximum number of threads for parallel processing

    Returns:
        Dict with keys:
        - 'substitutions': occurrences where the word was replaced with a different word
        - 'deletions': occurrences where the word was missing from the hypothesis
        - 'matches': occurrences where the word was correctly transcribed
        - 'total': total occurrences of the word in references

    Examples:
        # Regular medical term — use normalization (default)
        >>> counts = get_word_error_counts(
        ...     predictions=["patient has headache"],
        ...     references=["patient has fever"],
        ...     word="fever"
        ... )
        >>> # {'substitutions': 1, 'deletions': 0, 'matches': 0, 'total': 1}

        # Annotation tag — disable normalization to avoid '<is>' -> 'is' stripping
        >>> counts = get_word_error_counts(
        ...     predictions=d.model_output.tolist(),
        ...     references=d.raw_text.tolist(),
        ...     word="<is>",
        ...     language="hi",
        ...     normalize=False
        ... )
    """
    from concurrent.futures import ThreadPoolExecutor
    from karma.metrics.analysis_metrics import KeywordAccuracyAnalysisMetric
    from karma.metrics.wer_metrics import normalize_keyword, normalize_text_for_keyword_extraction

    if normalize:
        metric = KeywordAccuracyAnalysisMetric(language=language)
        search_word = normalize_keyword(word, metric.language, metric.normalizer, metric.equivalents)
    else:
        metric = None
        search_word = word.lower()

    def _prepare_texts(ref, hyp):
        """Return (search_word, ref_text, hyp_text) ready for alignment."""
        if len(hyp.split()) > 2 * len(ref.split()):
            hyp = ' '.join(hyp.split()[:2 * len(ref.split())])

        if normalize:
            ref_text = normalize_text_for_keyword_extraction(
                ref, metric.equivalents, metric.language, metric.normalizer
            )
            hyp_text = normalize_text_for_keyword_extraction(
                hyp, metric.equivalents, metric.language, metric.normalizer, counterpart=ref
            )
        else:
            # Lowercase only — preserve all punctuation/special characters literally
            ref_text = ref.lower()
            hyp_text = hyp.lower()

        return ref_text, hyp_text

    def process_sample(ref, hyp):
        ref_text, hyp_text = _prepare_texts(ref, hyp)

        positions = find_keyword_positions(ref_text, [search_word])
        if not positions:
            return {"substitutions": 0, "deletions": 0, "matches": 0}

        ref_words = ref_text.split()
        hyp_words = hyp_text.split()
        alignment = get_word_alignment(ref_words, hyp_words)

        sample_counts = {"substitutions": 0, "deletions": 0, "matches": 0}
        for _kw, start_idx, end_idx in positions:
            aligned_hyp_indices = set()
            for ref_idx in range(start_idx, end_idx):
                if ref_idx in alignment:
                    aligned_hyp_indices.update(alignment[ref_idx])

            if aligned_hyp_indices:
                sorted_indices = sorted(aligned_hyp_indices)
                hyp_segment = ' '.join(
                    hyp_words[idx] for idx in sorted_indices if idx < len(hyp_words)
                )
                if hyp_segment == search_word:
                    sample_counts["matches"] += 1
                else:
                    sample_counts["substitutions"] += 1
            else:
                sample_counts["deletions"] += 1

        return sample_counts

    counts = {"substitutions": 0, "deletions": 0, "matches": 0}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_sample, ref, hyp)
            for ref, hyp in zip(references, predictions)
        ]
        for future in futures:
            result = future.result()
            for key in ("substitutions", "deletions", "matches"):
                counts[key] += result[key]

    counts["total"] = counts["substitutions"] + counts["deletions"] + counts["matches"]
    return counts


def replace_aligned_text(
    predictions: List[str],
    references: List[str],
    string: str,
    language: str = 'en',
    normalize: bool = True,
    exclude_tokens: List[str] = None,
    max_workers: int = 16,
) -> List[str]:
    """
    Replace occurrences of `string` in each reference with the aligned text from prediction.

    Alignment is performed on normalized/transliterated text, but the replacement
    uses the original (unnormalized) prediction words.

    Args:
        predictions: List of prediction strings
        references: List of reference strings
        string: The string to find in each reference and replace with aligned prediction text
        language: Language code for normalization (e.g., 'en', 'hi', 'te')
        normalize: If True, apply full normalization/transliteration for alignment.
            Set to False to disable all normalization (use for fully literal matching).
        exclude_tokens: List of tokens to preserve literally and skip normalization for,
            even when normalize=True. Useful for special tokens like '<is>' that
            normalization would corrupt (e.g., stripping angle brackets).
        max_workers: Maximum number of threads for parallel processing

    Returns:
        List of reference strings with `string` replaced by its aligned counterpart
        from the prediction. References where `string` is not found are returned unchanged.
        If the string was deleted in the prediction, those words are removed.

    Example:
        >>> from karma.metrics.alignment_utils import replace_aligned_text
        >>>
        >>> # Replace a Hindi keyword using normalization
        >>> result = replace_aligned_text(
        ...     predictions=["थायरोइड चेक हुआ"],
        ...     references=["थाइरोइड चेक हुआ"],
        ...     string="थाइरोइड",
        ...     language="hi"
        ... )
        >>> print(result)  # ["थायरोइड चेक हुआ"]
        >>>
        >>> # Replace <is> tag, normalizing everything else for better alignment
        >>> result = replace_aligned_text(
        ...     predictions=["patient <is> बीमार"],
        ...     references=["patient <is> sick"],
        ...     string="<is>",
        ...     language="hi",
        ...     exclude_tokens=["<is>"]
        ... )
        >>> print(result)  # ["patient <is> बीमार"]
    """
    from concurrent.futures import ThreadPoolExecutor
    from karma.metrics.analysis_metrics import KeywordAccuracyAnalysisMetric
    from karma.metrics.wer_metrics import normalize_keyword, normalize_text_for_keyword_extraction

    excluded = set(exclude_tokens) if exclude_tokens else set()

    if normalize:
        metric = KeywordAccuracyAnalysisMetric(language=language)
        if string in excluded:
            search_string = string.lower()
        else:
            search_string = normalize_keyword(string, metric.language, metric.normalizer, metric.equivalents)
    else:
        metric = None
        search_string = string.lower()

    def _normalize_word(word: str) -> str:
        """Normalize a single word, bypassing normalization for excluded tokens."""
        if word in excluded:
            return word.lower()
        if normalize:
            return normalize_text_for_keyword_extraction(
                word, metric.equivalents, metric.language, metric.normalizer
            )
        return word.lower()

    def process_sample(ref, hyp):
        if len(hyp.split()) > 2 * len(ref.split()):
            hyp = ' '.join(hyp.split()[:2 * len(ref.split())])

        ref_original_words = ref.split()
        hyp_original_words = hyp.split()

        if normalize:
            ref_normalized_words = []
            ref_word_map = {}  # norm_idx -> orig_idx
            norm_idx = 0
            for orig_idx, orig_word in enumerate(ref_original_words):
                for part in _normalize_word(orig_word).split():
                    ref_normalized_words.append(part)
                    ref_word_map[norm_idx] = orig_idx
                    norm_idx += 1

            hyp_normalized_words = []
            hyp_word_map = {}  # norm_idx -> orig_idx
            norm_idx = 0
            for orig_idx, orig_word in enumerate(hyp_original_words):
                for part in _normalize_word(orig_word).split():
                    hyp_normalized_words.append(part)
                    hyp_word_map[norm_idx] = orig_idx
                    norm_idx += 1

            ref_text = ' '.join(ref_normalized_words)
            hyp_text = ' '.join(hyp_normalized_words)
        else:
            ref_text = ref.lower()
            hyp_text = hyp.lower()
            ref_normalized_words = ref_text.split()
            hyp_normalized_words = hyp_text.split()
            ref_word_map = {i: i for i in range(len(ref_normalized_words))}
            hyp_word_map = {i: i for i in range(len(hyp_normalized_words))}

        positions = find_keyword_positions(ref_text, [search_string])
        if not positions:
            return ref

        alignment = get_word_alignment(ref_normalized_words, hyp_normalized_words)

        result_words = list(ref_original_words)
        offset = 0  # tracks word count delta from previous replacements

        for _kw, start_idx, end_idx in positions:
            # Map normalized ref range to original ref word range
            orig_start = ref_word_map.get(start_idx, start_idx)
            orig_end = ref_word_map.get(end_idx - 1, end_idx - 1) + 1  # exclusive

            # Find all aligned hyp normalized indices
            aligned_hyp_norm_indices = set()
            for ref_idx in range(start_idx, end_idx):
                if ref_idx in alignment:
                    aligned_hyp_norm_indices.update(alignment[ref_idx])

            # Map to original hyp word indices (preserve order, deduplicate)
            seen: set = set()
            aligned_hyp_orig_indices = []
            for idx in sorted(aligned_hyp_norm_indices):
                if idx < len(hyp_normalized_words):
                    orig_idx = hyp_word_map.get(idx, idx)
                    if orig_idx not in seen:
                        seen.add(orig_idx)
                        aligned_hyp_orig_indices.append(orig_idx)

            replacement_words = [
                hyp_original_words[i] for i in aligned_hyp_orig_indices
                if i < len(hyp_original_words)
            ]

            # Replace in result, accounting for word count changes from prior replacements
            adjusted_start = orig_start + offset
            adjusted_end = orig_end + offset
            result_words[adjusted_start:adjusted_end] = replacement_words
            offset += len(replacement_words) - (orig_end - orig_start)

        return ' '.join(result_words)

    results: List[str] = [None] * len(references)  # type: ignore[list-item]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_sample, ref, hyp): idx
            for idx, (ref, hyp) in enumerate(zip(references, predictions))
        }
        for future in futures:
            idx = futures[future]
            results[idx] = future.result()

    return results


def get_full_word_alignments_df(
    predictions: List[str],
    references: List[str],
    entities: List[Dict] = None,
    language: str = 'en',
    max_workers: int = 16
):
    """
    Get word-level alignments for ALL words in reference and hypothesis as a DataFrame.
    
    Similar to get_keyword_alignments_df, but returns alignments for every word,
    not just keywords. Includes a flag to indicate which words are keywords.
    
    Returns a DataFrame with columns:
    - sample_idx: Index of the sample
    - word_idx: Position of word in reference
    - ref_word_original: Original word from reference (before normalization)
    - ref_word: Word from reference (after normalization)
    - hyp_word_original: Original word from hypothesis (before normalization)
    - hyp_word: Aligned word from hypothesis (after normalization, may be empty for deletions)
    - is_keyword: Boolean indicating if this word is part of a keyword
    - keyword_name: Original keyword name if is_keyword=True, else None
    - match_status: "match", "substituted", or "deleted"
    
    Note: Since normalization can change word counts (e.g., "100" -> "एक सौ"),
    the mapping from normalized to original words is approximate (proportional).
    
    Args:
        predictions: List of prediction strings
        references: List of reference strings
        entities: List of dicts with entity type keys (optional, for keyword marking)
        language: Language code (e.g., 'en', 'hi', 'te')
        max_workers: Maximum number of threads for parallel processing (default: 16)
        
    Returns:
        pandas DataFrame with word-level alignment details
        
    Example:
        >>> from karma.metrics.alignment_utils import get_full_word_alignments_df
        >>> 
        >>> df = get_full_word_alignments_df(
        ...     predictions=["थायरोइड 100 mcg"],
        ...     references=["थाइरोइड एक सौ एमसीजी"],
        ...     entities=[{"conditions": ["थाइरोइड"]}],
        ...     language="hi"
        ... )
        >>> 
        >>> # See word-level original vs normalized
        >>> print(df[['ref_word_original', 'ref_word', 'hyp_word_original', 'hyp_word']])
        >>> 
        >>> # Compare normalization effects
        >>> print(df[df['ref_word_original'] != df['ref_word']])
    """
    import pandas as pd
    from concurrent.futures import ThreadPoolExecutor
    
    # Import here to avoid circular imports
    from karma.metrics.analysis_metrics import KeywordAccuracyAnalysisMetric
    from karma.metrics.wer_metrics import (
        normalize_keyword,
        normalize_text_for_keyword_extraction,
    )

    # Handle case where no entities provided
    if entities is None:
        entities = [{}] * len(predictions)

    # Create metric instance (shared across threads)
    metric = KeywordAccuracyAnalysisMetric(language=language)
    
    def process_sample(sample_idx, ref, hyp, sample_entities):
        """Process a single sample and return list of row dicts for all words."""
        rows = []
        
        # Store original words before normalization
        ref_original_words = ref.split()
        hyp_original_words = hyp.split()
        
        # Get keywords for this sample
        sample_keywords = sum(sample_entities.values(), []) if sample_entities else []
        
        # Normalize keywords
        normalized_kws = []
        keyword_map = {}  # Map normalized keyword -> original keyword
        if sample_keywords:
            for orig_kw in sample_keywords:
                norm_kw = normalize_keyword(orig_kw, metric.language, metric.normalizer, metric.equivalents)
                normalized_kws.append(norm_kw)
                keyword_map[norm_kw] = orig_kw
        
        # Truncate overly long predictions
        if len(hyp.split()) > 2 * len(ref.split()):
            hyp = ' '.join(hyp.split()[:2 * len(ref.split())])
            hyp_original_words = hyp.split()
        
        # Normalize each word individually and track mapping
        ref_normalized_words = []
        ref_word_map = {}  # Maps normalized word index -> original word index
        norm_idx = 0
        for orig_idx, orig_word in enumerate(ref_original_words):
            # Normalize the word
            norm_word = normalize_text_for_keyword_extraction(
                orig_word, metric.equivalents, metric.language, metric.normalizer
            )
            # A single original word might become multiple normalized words
            norm_parts = norm_word.split()
            for part in norm_parts:
                ref_normalized_words.append(part)
                ref_word_map[norm_idx] = orig_idx
                norm_idx += 1
        
        hyp_normalized_words = []
        hyp_word_map = {}  # Maps normalized word index -> original word index
        norm_idx = 0
        for orig_idx, orig_word in enumerate(hyp_original_words):
            # Normalize the word
            norm_word = normalize_text_for_keyword_extraction(
                orig_word, metric.equivalents, metric.language, metric.normalizer
            )
            # A single original word might become multiple normalized words
            norm_parts = norm_word.split()
            for part in norm_parts:
                hyp_normalized_words.append(part)
                hyp_word_map[norm_idx] = orig_idx
                norm_idx += 1
        
        # Get word-level alignment using normalized words
        alignment = get_word_alignment(ref_normalized_words, hyp_normalized_words)
        
        # Find keyword positions in normalized reference
        normalized_ref_text = ' '.join(ref_normalized_words)
        keyword_positions = find_keyword_positions(normalized_ref_text, normalized_kws)
        
        # Build a set of word indices that are part of keywords
        keyword_word_indices = set()
        keyword_name_by_position = {}
        for keyword, start_idx, end_idx in keyword_positions:
            for pos in range(start_idx, end_idx):
                keyword_word_indices.add(pos)
                # Get original keyword name
                orig_keyword = keyword_map.get(keyword, keyword)
                keyword_name_by_position[pos] = orig_keyword
        
        # Build row for each normalized reference word
        for ref_idx, ref_word in enumerate(ref_normalized_words):
            # Check if this word is part of a keyword
            is_keyword = ref_idx in keyword_word_indices
            keyword_name = keyword_name_by_position.get(ref_idx, None)
            
            # Get original reference word
            orig_ref_idx = ref_word_map.get(ref_idx, 0)
            ref_word_original = ref_original_words[orig_ref_idx] if orig_ref_idx < len(ref_original_words) else ''
            
            # Get aligned hypothesis word(s)
            if ref_idx in alignment:
                hyp_indices = alignment[ref_idx]
                if hyp_indices:
                    hyp_word = ' '.join([hyp_normalized_words[idx] for idx in hyp_indices if idx < len(hyp_normalized_words)])
                    # Get original hypothesis word (use first aligned index)
                    first_hyp_idx = hyp_indices[0] if hyp_indices else 0
                    orig_hyp_idx = hyp_word_map.get(first_hyp_idx, 0)
                    hyp_word_original = hyp_original_words[orig_hyp_idx] if orig_hyp_idx < len(hyp_original_words) else ''
                else:
                    hyp_word = ''
                    hyp_word_original = ''
            else:
                hyp_word = ''
                hyp_word_original = ''
            
            # Determine match status
            if ref_word == hyp_word and hyp_word != '':
                status = "match"
            elif hyp_word == '':
                status = "deleted"
            else:
                status = "substituted"
            
            rows.append({
                'sample_idx': sample_idx,
                'word_idx': ref_idx,
                'ref_word_original': ref_word_original,
                'ref_word': ref_word,
                'hyp_word_original': hyp_word_original,
                'hyp_word': hyp_word,
                'is_keyword': is_keyword,
                'keyword_name': keyword_name,
                'match_status': status
            })
        
        return rows
    
    # Process all samples in parallel
    all_rows = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_sample, idx, ref, hyp, ent)
            for idx, (ref, hyp, ent) in enumerate(zip(references, predictions, entities))
        ]
        
        # Collect results in order
        for future in futures:
            rows = future.result()
            all_rows.extend(rows)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_rows)
    
    return df