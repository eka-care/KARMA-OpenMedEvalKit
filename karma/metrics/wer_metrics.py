"""
Normalized WER Metric for Medical ASR

This module provides a WER metric that normalizes medical units, numbers, and punctuation
to improve accuracy in medical transcription evaluation.
"""

import re
import difflib
import string
import os
import json
import threading
from typing import List, Tuple, Dict
# from num2words import num2words
import evaluate

from karma.metrics.base_metric_abs import BaseMetric
from karma.registries.metrics_registry import register_metric
from karma.metrics.common_metrics import WERMetric, CERMetric
from indic_numtowords import num2words
from google.transliteration import transliterate_word
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

# Punctuation marks for normalization
PUNCTUATION_MARKS = ['.', ',', '-', '!', '?', ';', ':', '"', "'"]


# ============================================================================
# METRICS
# ============================================================================

import threading

# Thread-local storage for metrics to avoid issues with HuggingFace evaluate library
_thread_local = threading.local()


# ============================================================================
# TRANSLITERATION CACHE (DynamoDB-backed)
# ============================================================================

from karma.cache.dynamodb_transliteration_cache import DynamoDBTransliterationCache

# Get cache file path (for fallback)
_CACHE_FILE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    "lang_resources", 
    "transliteration_cache.json"
)

# Initialize DynamoDB-backed cache with file fallback
print("="*60)
print("Initializing transliteration cache for WER metrics...")
_TRANSLITERATION_CACHE = DynamoDBTransliterationCache(
    table_name=os.getenv("DYNAMODB_TRANSLITERATION_TABLE", "transliteration-cache"),
    region_name=os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    enable_fallback=True,
    fallback_cache_path=_CACHE_FILE_PATH,
    create_table_if_not_exists=True
)

# Log cache statistics
_cache_stats = _TRANSLITERATION_CACHE.get_stats()
if "dynamodb_entries" in _cache_stats and _cache_stats["dynamodb_entries"] > 0:
    print(f"✓ Using DynamoDB cache with {_cache_stats['dynamodb_entries']} entries")
    print(f"  Table: {os.getenv('DYNAMODB_TRANSLITERATION_TABLE', 'transliteration-cache')}")
    print(f"  Region: {os.getenv('AWS_REGION') or os.getenv('AWS_DEFAULT_REGION', 'us-east-1')}")
elif "fallback_entries" in _cache_stats and _cache_stats["fallback_entries"] > 0:
    print(f"⚠ Using fallback file cache with {_cache_stats['fallback_entries']} entries")
    print(f"  Cache file: {_CACHE_FILE_PATH}")
else:
    print("⚠ Cache initialized (empty)")
print("="*60)


def _get_wer_metric():
    """Get thread-local WER metric instance."""
    if not hasattr(_thread_local, 'wer_metric'):
        _thread_local.wer_metric = WERMetric()
    return _thread_local.wer_metric


def _get_cer_metric():
    """Get thread-local CER metric instance."""
    if not hasattr(_thread_local, 'cer_metric'):
        _thread_local.cer_metric = CERMetric()
    return _thread_local.cer_metric


def calculate_wer(ref: str, hyp: str) -> float:
    """Calculate Word Error Rate between reference and hypothesis."""
    wer_metric = _get_wer_metric()
    return wer_metric.evaluate(predictions=[hyp], references=[ref])


def calculate_cer(ref: str, hyp: str) -> float:
    """Calculate Character Error Rate between reference and hypothesis."""
    cer_metric = _get_cer_metric()
    return cer_metric.evaluate(predictions=[hyp], references=[ref])


# ============================================================================
# TEXT PREPROCESSING
# ============================================================================

def add_spaces_around_symbols(text: str, symbols: List[str] = ['/']) -> str:
    """
    Ensure all given symbols in text have spaces on both sides.
    
    Example:
        add_spaces_around_symbols("a+b=c", ["+", "="]) -> "a + b = c"
    """
    text = text.replace("।", " ")
    for sym in symbols:
        text = text.replace(sym, f" {sym} ")
    return " ".join(text.split()).strip().replace("  ", " ")


def add_spaces_around_hyphens_in_ranges(text: str) -> str:
    """
    Add spaces around hyphens ONLY when they appear between numbers (ranges).
    Preserves hyphens in compound words like drug names.
    
    Examples:
        >>> add_spaces_around_hyphens_in_ranges("3-4 times")
        "3 - 4 times"
        >>> add_spaces_around_hyphens_in_ranges("pantop-dsr")
        "pantop-dsr"
        >>> add_spaces_around_hyphens_in_ranges("10-15mg")
        "10 - 15mg"
    """
    # Pattern: digit, hyphen, digit -> add spaces around hyphen
    # Uses lookahead and lookbehind to check for digits
    text = re.sub(r'(\d)-(\d)', r'\1 - \2', text)
    return text
        

def add_space_before_units(input_string: str, units: List[str]) -> str:
    """
    Add space between numbers and unit strings (including symbols like %).
    
    Args:
        input_string: The input string to process
        units: List of unit strings to search for (default: UNITS_DICT.keys() + '%')
        
    Returns:
        Modified string with spaces added between numbers and units
        
    Examples:
        >>> add_space_before_units('20mg', ['mg', 'ml', 'kg'])
        '20 mg'
        >>> add_space_before_units('2% solution', ['%'])
        '2 % solution'
    """
    
    result = input_string.lower()
    
    # Sort by length descending to match longer strings first
    sorted_units = sorted(units, key=len, reverse=True)
    
    for unit in sorted_units:
        # Pattern: digit followed immediately by unit
        pattern = r'(\d)(' + re.escape(unit) + r')'
        result = re.sub(pattern, r'\1 \2', result)
    
    return result

def get_transliterated_variations(text: str, language: str = 'en') -> List[str]:
    """
    Get transliterated variations with DynamoDB-backed persistent caching.
    
    Flow:
    1. Check DynamoDB cache (shared across all processes and machines)
    2. If not found, call API and immediately save to DynamoDB
    3. Other processes see the new entry immediately
    
    Falls back to file-based cache if DynamoDB unavailable.
    
    Returns a list of transliterated variations, or empty list if transliteration not available.
    Retries transliteration up to 5 times if it fails, sleeping 1s, 2s, ..., 5s between attempts.
    """
    import time
    
    # Step 1: Check DynamoDB cache (atomic, no locks needed)
    cached_result = _TRANSLITERATION_CACHE.get(text, language)
    if cached_result is not None:
        return cached_result
    
    # Step 2: Not in cache - call API with retry logic
    max_retries = 5
    for attempt in range(1, max_retries + 1):
        try:
            result = transliterate_word(text, lang_code=language)
            
            # Immediately save to DynamoDB cache (atomic operation)
            _TRANSLITERATION_CACHE.set(text, language, result)
            
            return result
            
        except Exception as e:
            print(f"Error transliterating word {text}: {e}, attempt {attempt} of {max_retries}")
            if attempt == max_retries:
                # Cache empty result to avoid retrying failed words
                _TRANSLITERATION_CACHE.set(text, language, [])
                return []
            time.sleep(attempt)


def remove_punctuation_except_slash(text: str) -> str:
    """
    Replace all punctuation (except '/', '-', '%') with spaces, then normalize spaces.
    Preserves: '/' for rates, '-' for compound words, '%' for percentages.
    
    Examples:
        >>> remove_punctuation_except_slash("hello, world.")
        "hello world"
        >>> remove_punctuation_except_slash("zero-dol")
        "zero-dol"
        >>> remove_punctuation_except_slash("2% solution")
        "2% solution"
        >>> remove_punctuation_except_slash("mg/dl test.")
        "mg/dl test"
    """
    # Exclude '/', '-', and '%' from punctuation removal
    punctuation_to_replace = string.punctuation.replace('/', '').replace('-', '').replace('%', '')
    trans_table = str.maketrans(punctuation_to_replace, ' ' * len(punctuation_to_replace))
    text = text.translate(trans_table)
    return ' '.join(text.split()).strip()


# ============================================================================
# TEXT NORMALIZATION
# ============================================================================

def convert_numbers_to_words(text: str, language: str = 'en') -> str:
    """Convert all digit numbers in text to word form."""
    words = text.split()
    result = []
    for word in words:
        if re.fullmatch(r'\d+', word):
            try:
                # print(f"Converting number {word} to words")
                num_word = num2words(int(word), lang=language, variations=True)
                if isinstance(num_word, list):
                    num_word = num_word[0]
                result.append(num_word)
            except Exception as e:
                print(f"Error converting number {word} to words: {e}")
                result.append(word)
        else:
            result.append(word)
    return ' '.join(result)


def expand_contractions(text: str) -> str:
    """
    Expand common English contractions to their full forms.
    Handles contractions like 'm, 's, 'll, 've, 'd, 're, n't, etc.
    
    Examples:
        >>> expand_contractions("I'm fine")
        "I am fine"
        >>> expand_contractions("He's sick")
        "He is sick"
        >>> expand_contractions("doesn't hurt")
        "does not hurt"
    """
    # Common contractions mapping
    # Order matters - handle specific cases before general patterns
    contractions = {
        r"\bwon't\b": "will not",
        r"\bcan't\b": "cannot",
        r"\bshan't\b": "shall not",
        r"\blet's\b": "let us",
        r"\b(\w+)n't\b": r"\1 not",  # General negation (doesn't -> does not, isn't -> is not)
        r"\b(\w+)'ll\b": r"\1 will",
        r"\b(\w+)'ve\b": r"\1 have",
        r"\b(\w+)'re\b": r"\1 are",
        r"\b(\w+)'d\b": r"\1 would",  # Could also be "had" but "would" is more common
        r"\b(\w+)'m\b": r"\1 am",
        r"\b(\w+)'s\b": r"\1 is",  # Could also be "has" but "is" is more common in medical context
    }
    
    result = text
    for pattern, replacement in contractions.items():
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    
    # Clean up extra spaces
    result = ' '.join(result.split())
    return result


def replace_whole_word(text: str, old_word: str, new_word: str) -> str:
    """
    Replace whole word occurrences only, not substrings within words.
    Handles both regular words and symbols (like '/').
    Case-insensitive for words, exact match for symbols.
    """
    if not old_word:
        return text
    
    # For symbols/punctuation, use space-bounded replacement
    # \w matches word characters (letters, digits, underscore)
    if not re.match(r'\w+$', old_word):
        # It's a symbol - replace as standalone token (surrounded by spaces or at boundaries)
        # Handle start/end of string and space-separated tokens
        text = re.sub(r'(^|\s)' + re.escape(old_word) + r'(\s|$)', 
                     r'\1' + new_word + r'\2', text)
        return text
    
    # For regular words, use word boundaries
    pattern = r'\b' + re.escape(old_word) + r'\b'
    return re.sub(pattern, new_word, text, flags=re.IGNORECASE)


def find_matching_units(text: str, units_dict: Dict) -> List[Tuple[str, str]]:
    """
    Find all unit keys that have variations present in the text as whole words.
    Checks both:
    1. If the unit key itself is in the text (e.g., '/' in "/ min")
    2. If any variation is in the text (e.g., 'per' in "per minute")
    
    Returns a list of (unit_key, found_variation) tuples.
    """
    matches = []
    words_in_text = text.lower().split()
    
    for unit_key, variations in units_dict.items():
        # First check if the unit key itself is in the text
        if unit_key.lower() in words_in_text:
            matches.append((unit_key, unit_key))
            continue
        
        # Then check if any variation is in the text
        for var in variations:
            if var.lower() in words_in_text:
                matches.append((unit_key, var))
                break
    return matches


def apply_punctuation_normalizations(ref: str, hyp: str, best_cer: float) -> Tuple[str, str, float]:
    """
    Try various punctuation normalizations and return the best result.
    Handles hyphens specially to avoid breaking compound words like 'zero-dol'.
    Returns (best_ref, best_hyp, best_cer)
    """
    best_ref, best_hyp = ref, hyp
    
    # Try removing hyphens completely (zero-dol -> zerodol)
    temp_ref = ref.replace('-', '')
    temp_hyp = hyp.replace('-', '')
    temp_ref = ' '.join(temp_ref.split())
    temp_hyp = ' '.join(temp_hyp.split())
    
    if temp_ref and temp_hyp:
        cer = calculate_cer(temp_ref, temp_hyp)
        if cer < best_cer:
            best_cer = cer
            best_ref, best_hyp = temp_ref, temp_hyp
    
    # Try removing all punctuation at once (excluding hyphens initially)
    temp_ref = ref
    temp_hyp = hyp
    for punct in ['.', ',', '!', '?', ';', ':', '"', "'"]:
        temp_ref = temp_ref.replace(punct, '')
        temp_hyp = temp_hyp.replace(punct, '')
    temp_ref = ' '.join(temp_ref.split())
    temp_hyp = ' '.join(temp_hyp.split())
    
    if temp_ref and temp_hyp:
        cer = calculate_cer(temp_ref, temp_hyp)
        if cer < best_cer:
            best_cer = cer
            best_ref, best_hyp = temp_ref, temp_hyp
    
    # Try individual punctuation transformations
    transformations = [
        ('.', ''),   # Remove dots
        (',', ''),   # Remove commas
        ('.', ','),  # Dot to comma
        (',', '.'),  # Comma to dot
        ('-', ' '),  # Hyphen to space (zero-dol -> zero dol)
    ]
    
    for old_char, new_char in transformations:
        temp_ref = ref.replace(old_char, new_char)
        temp_hyp = hyp.replace(old_char, new_char)
        temp_ref = ' '.join(temp_ref.split())
        temp_hyp = ' '.join(temp_hyp.split())
        
        if temp_ref and temp_hyp:
            cer = calculate_cer(temp_ref, temp_hyp)
            if cer < best_cer:
                best_cer = cer
                best_ref, best_hyp = temp_ref, temp_hyp
    
    return best_ref, best_hyp, best_cer


def is_english_word(word: str) -> bool:
    """
    Check if a word is likely English (contains only ASCII letters).
    Excludes numbers.
    """
    # Remove common punctuation
    word = word.strip('.,!?;:\'"')
    # Check if it's a number
    if re.fullmatch(r'\d+', word):
        return False
    # Check if contains only ASCII letters (English)
    return bool(re.match(r'^[a-zA-Z]+$', word))

def transliterate(text: str, language: str = 'en', units_dict: Dict = None, normalizer = None) -> str:
    words = []
    for word in text.split():
        if is_english_word(word) and word not in units_dict.keys() and not re.fullmatch(r'\d+', word):
            try:
                transliterated_words= get_transliterated_variations(word, language=language)
            except Exception as e:
                transliterated_words = []
                print(f"Error transliterating word {word}: {e}")
                continue
            if transliterated_words:
                transliterated_word = transliterated_words[0]

            else:
                transliterated_word = word

            if normalizer:
                transliterated_word = normalizer.normalize(transliterated_word)
            words.append(transliterated_word)
        else:
            if not is_english_word(word) and normalizer:
                word = normalizer.normalize(word)
            words.append(word)
    return ' '.join(words)
            

def apply_word_mappings(text: str, word_mappings: Dict[str, str]) -> str:
    """
    Replace words in text with their mapped values if they exist in word_mappings.
    Performs case-insensitive matching and preserves word boundaries.
    
    Args:
        text: Input text
        word_mappings: Dictionary mapping source words to target words
        
    Returns:
        Text with words replaced according to the mapping
        
    Examples:
        >>> apply_word_mappings("Take Dolo twice", {"dolo": "paracetamol", "twice": "two times"})
        "Take paracetamol two times"
    """
    if not word_mappings:
        return text
    
    words = text.split()
    result_words = []
    
    for word in words:
        # Check if word (case-insensitive) exists in mapping
        word_lower = word.lower()
        if word_lower in word_mappings:
            result_words.append(word_mappings[word_lower])
        else:
            result_words.append(word)
    
    return ' '.join(result_words)


def normalize_pair(ref_part: str, hyp_part: str, units_dict: Dict, language: str = 'en', glm_arrays: Dict[str, str] = None) -> Tuple[str, str]:
    """
    Find the best normalization for a pair of unmatched strings.
    
    Steps:
    1. Expand contractions FIRST (I'm -> I am, he's -> he is) - before removing punctuation
    2. Remove punctuation (except '/' and '-')
    3. Try transliteration variations for non-English languages (if English words present)
    4. Try number-to-words conversion if improves CER
    5. Try unit variations if improves CER
    6. Try punctuation normalizations if improves CER
    7. Apply GLM arrays as FINAL step (only on substitutions after all normalizations)
    
    Returns: (best_ref_part, best_hyp_part)
    """
    # Step 0: Expand contractions BEFORE removing punctuation (apostrophes needed!)
    if "'" in ref_part or "'" in hyp_part:
        ref_part = expand_contractions(ref_part)
        hyp_part = expand_contractions(hyp_part)
    
    # Step 1: Remove punctuation (except '/' and '-')
    ref_part = remove_punctuation_except_slash(ref_part)
    hyp_part = remove_punctuation_except_slash(hyp_part)
    
    best_ref = ref_part
    best_hyp = hyp_part
    best_cer = calculate_cer(ref_part, hyp_part) if ref_part and hyp_part else float('inf')
    
    
    # Step 2: Try number-to-words conversion
    if re.search(r'\d+', ref_part) or re.search(r'\d+', hyp_part):
        ref_with_words = convert_numbers_to_words(ref_part, language=language) if ref_part else ''
        hyp_with_words = convert_numbers_to_words(hyp_part, language=language) if hyp_part else ''
        
        if ref_with_words and hyp_with_words:
            cer = calculate_cer(ref_with_words, hyp_with_words)
            if cer < best_cer:
                best_cer = cer
                best_ref = ref_with_words
                best_hyp = hyp_with_words
    
   
    # Step 3: Try unit variations
    ref_matches = find_matching_units(ref_part, units_dict)
    hyp_matches = find_matching_units(hyp_part, units_dict)
    all_unit_keys = set([key for key, _ in ref_matches] + [key for key, _ in hyp_matches])
    
    for unit_key in all_unit_keys:
        variations = units_dict[unit_key]
        
        for target_var in variations:
            # Start from best normalized version so far to combine multiple normalizations
            temp_ref = best_ref
            temp_hyp = best_hyp
            
            # Replace unit key AND all variations with target variation
            # This handles both the key (e.g., 'min', '/') and its variations (e.g., 'minute', 'per')
            for var in [unit_key] + variations:
                if var == '':
                    continue
                if target_var == '':
                    temp_ref = replace_whole_word(temp_ref, var, '')
                    temp_hyp = replace_whole_word(temp_hyp, var, '')
                else:
                    temp_ref = replace_whole_word(temp_ref, var, target_var)
                    temp_hyp = replace_whole_word(temp_hyp, var, target_var)
            
            temp_ref = ' '.join(temp_ref.split())
            temp_hyp = ' '.join(temp_hyp.split())
            
            if not temp_ref or not temp_hyp:
                continue
            
            cer = calculate_cer(temp_ref, temp_hyp)
            
            # Also try with transliteration for non-English languages
            
            # Also try with number conversion
            if re.search(r'\d+', temp_ref) or re.search(r'\d+', temp_hyp):
                temp_ref_nums = convert_numbers_to_words(temp_ref, language=language) if temp_ref else ''
                temp_hyp_nums = convert_numbers_to_words(temp_hyp, language=language) if temp_hyp else ''
                
                if temp_ref_nums and temp_hyp_nums:
                    cer_nums = calculate_cer(temp_ref_nums, temp_hyp_nums)
                    if cer_nums < cer:
                        temp_ref = temp_ref_nums
                        temp_hyp = temp_hyp_nums
                        cer = cer_nums
            
            if cer < best_cer:
                best_cer = cer
                best_ref = temp_ref
                best_hyp = temp_hyp
    
    # Step 4: Try punctuation normalizations
    best_ref, best_hyp, best_cer = apply_punctuation_normalizations(
        best_ref, best_hyp, best_cer
    )
    
    # Step 5: Apply GLM arrays as FINAL step (only on substitutions)
    # This happens AFTER all other normalizations to preserve alignment
    if glm_arrays:
        # Try applying GLM arrays and see if it improves CER
        temp_ref = apply_word_mappings(best_ref, glm_arrays)
        temp_hyp = apply_word_mappings(best_hyp, glm_arrays)
        
        if temp_ref and temp_hyp:
            cer_glm = calculate_cer(temp_ref, temp_hyp)
            if cer_glm < best_cer:
                best_ref = temp_ref
                best_hyp = temp_hyp
                best_cer = cer_glm
    
    return best_ref, best_hyp


# ============================================================================
# ALIGNMENT AND DIFF DETECTION
# ============================================================================

def find_unmatched_pairs(ref: str, hyp: str) -> Tuple[List, List[str], List[str]]:
    """
    Find pairs of unmatched substrings between reference and hypothesis.
    
    Returns:
        diffs: List of (tag, i1, i2, j1, j2, diff_ref, diff_hyp) tuples
        words_ref: Reference words
        words_hyp: Hypothesis words
    """
    words_ref = ref.split()
    words_hyp = hyp.split()
    
    matcher = difflib.SequenceMatcher(None, words_ref, words_hyp)
    opcodes = matcher.get_opcodes()
    
    diffs = []
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == 'replace':
            diff_ref = ' '.join(words_ref[i1:i2])
            diff_hyp = ' '.join(words_hyp[j1:j2])
            diffs.append((tag, i1, i2, j1, j2, diff_ref, diff_hyp))
        elif tag == 'delete':
            diff_ref = ' '.join(words_ref[i1:i2])
            diffs.append((tag, i1, i2, j1, j2, diff_ref, ''))
        elif tag == 'insert':
            diff_hyp = ' '.join(words_hyp[j1:j2])
            diffs.append((tag, i1, i2, j1, j2, '', diff_hyp))
    
    return diffs, words_ref, words_hyp


def print_alignment(ref: str, hyp: str, separator: str = '|') -> None:
    """
    Print word-level alignment showing substitutions (~), deletions (D), 
    insertions (I), and matches (=).
    
    Example output:
        REF: Patient | has | fever | and | headache
                =    |  =  |   ~   |  =  |    =
        HYP: Patient | has | Fever | and | headache
    """
    ref_words = ref.split()
    hyp_words = hyp.split()
    
    matcher = difflib.SequenceMatcher(None, ref_words, hyp_words)
    opcodes = matcher.get_opcodes()
    
    ref_line, op_line, hyp_line = [], [], []
    
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == 'equal':
            for ref_word, hyp_word in zip(ref_words[i1:i2], hyp_words[j1:j2]):
                ref_line.append(ref_word)
                op_line.append('=')
                hyp_line.append(hyp_word)
        
        elif tag == 'replace':
            ref_segment = ref_words[i1:i2]
            hyp_segment = hyp_words[j1:j2]
            max_len = max(len(ref_segment), len(hyp_segment))
            
            for idx in range(max_len):
                ref_line.append(ref_segment[idx] if idx < len(ref_segment) else '')
                hyp_line.append(hyp_segment[idx] if idx < len(hyp_segment) else '')
                
                if idx < len(ref_segment) and idx < len(hyp_segment):
                    op_line.append('~')  # Substitution
                elif idx < len(ref_segment):
                    op_line.append('D')  # Deletion
                else:
                    op_line.append('I')  # Insertion
        
        elif tag == 'delete':
            for ref_word in ref_words[i1:i2]:
                ref_line.append(ref_word)
                op_line.append('D')
                hyp_line.append('')
        
        elif tag == 'insert':
            for hyp_word in hyp_words[j1:j2]:
                ref_line.append('')
                op_line.append('I')
                hyp_line.append(hyp_word)
    
    # Calculate column widths and print
    col_widths = [max(len(r), len(h), 1) for r, h in zip(ref_line, hyp_line)]
    sep = f' {separator} '
    
    ref_formatted = sep.join(w.center(width) for w, width in zip(ref_line, col_widths))
    op_formatted = sep.join(op.center(width) for op, width in zip(op_line, col_widths))
    hyp_formatted = sep.join(w.center(width) for w, width in zip(hyp_line, col_widths))
    
    print(f"REF: {ref_formatted}")
    print(f"     {op_formatted}")
    print(f"HYP: {hyp_formatted}")
    print('-' * (len(ref_formatted) + 5))


# ============================================================================
# WER OPTIMIZATION
# ============================================================================

def optimize_wer(ref: str, hyp: str, units_dict: Dict, language: str = 'en', normalizer = None, glm_arrays: Dict[str, str] = None) -> Tuple[float, str, str]:
    """
    Optimize WER by normalizing unmatched pairs.
    Only applies normalization if it reduces WER.
    
    Args:
        ref: Reference string
        hyp: Hypothesis string
        units_dict: Dict where keys are unit symbols, values are lists of variations
        language: Language code for normalization
        normalizer: Optional normalizer for non-English languages
        glm_arrays: Optional GLM arrays dict (applied only on substitutions)
    
    Returns:
        (wer, normalized_ref, normalized_hyp)
    """
    if not ref:
        return 0.0, "", ""
    
    
    # Calculate baseline WER
    original_wer = calculate_wer(ref.lower(), hyp.lower())

    # Preprocess: add spaces around hyphens in ranges (3-4 -> 3 - 4)
    ref = add_spaces_around_hyphens_in_ranges(ref)
    hyp = add_spaces_around_hyphens_in_ranges(hyp)
    
    # Preprocess: add spaces around symbols and before units
    ref = add_space_before_units(add_spaces_around_symbols(ref), units=units_dict.keys())
    hyp = add_space_before_units(add_spaces_around_symbols(hyp), units=units_dict.keys())
    
    # Preprocess: transliterate
    if language != "en":
        ref = transliterate(ref, language=language, units_dict=units_dict, normalizer=normalizer)
        hyp = transliterate(hyp, language=language, units_dict=units_dict, normalizer=normalizer)
    
    # Find unmatched pairs
    diffs, words_ref, words_hyp = find_unmatched_pairs(ref, hyp)
    
    if not diffs:
        return 0.0, ref, hyp
    
    # Normalize each unmatched pair
    normalized_ref_parts = {}
    normalized_hyp_parts = {}
    
    for tag, i1, i2, j1, j2, ref_part, hyp_part in diffs:
        best_ref_part, best_hyp_part = normalize_pair(ref_part, hyp_part, units_dict, language=language, glm_arrays=glm_arrays)
        normalized_ref_parts[(i1, i2)] = best_ref_part.split() if best_ref_part else []
        normalized_hyp_parts[(j1, j2)] = best_hyp_part.split() if best_hyp_part else []
    
    # Reconstruct full strings with normalized parts
    new_ref_words, new_hyp_words = [], []
    
    matcher = difflib.SequenceMatcher(None, words_ref, words_hyp)
    opcodes = matcher.get_opcodes()
    
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == 'equal':
            new_ref_words.extend(words_ref[i1:i2])
            new_hyp_words.extend(words_hyp[j1:j2])
        elif tag == 'replace':
            new_ref_words.extend(normalized_ref_parts[(i1, i2)])
            new_hyp_words.extend(normalized_hyp_parts[(j1, j2)])
        elif tag == 'delete':
            new_ref_words.extend(normalized_ref_parts[(i1, i2)])
        elif tag == 'insert':
            new_hyp_words.extend(normalized_hyp_parts[(j1, j2)])
    
    optimized_ref = ' '.join(new_ref_words).strip()
    optimized_hyp = ' '.join(new_hyp_words).strip()
    optimized_wer = calculate_wer(optimized_ref, optimized_hyp)
    
    # Debug output
    # print("Actual Ref:", ref)
    # print("Actual Hyp:", hyp)
    # print("Original WER:", original_wer)
    # print("Optimized WER:", optimized_wer)
    # print_alignment(optimized_ref, optimized_hyp)

    
    # Only return normalized version if it improves WER
    if optimized_wer <= original_wer:
        return optimized_wer, optimized_ref, optimized_hyp
    else:
        return original_wer, ref, hyp


# ============================================================================
# METRIC CLASS
# ============================================================================

@register_metric("semantic_wer_metric", required_args=["language"])
class SemanticWERMetric(BaseMetric):
    """
    Normalized WER metric that applies medical unit, number, and punctuation 
    normalization before computing WER.
    """
    
    def __init__(self, metric_name: str = "semantic_wer_metric", **kwargs):
        self.language = kwargs.get("language")
        self.normalizer = None
        if self.language != "en":
            self.normalizer = IndicNormalizerFactory().get_normalizer(self.language, do_normalize_chandras=True, remove_nuktas=True)
        super().__init__(metric_name, **kwargs)
        
        # Get the directory of the current file
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Load units and symbols from JSON files
        import json
        units_path = os.path.join(current_dir, "lang_resources", f"{self.language}_units.json")
        symbols_path = os.path.join(current_dir, "lang_resources", f"{self.language}_symbols.json")
        global units
        # Load units dictionary
        if os.path.exists(units_path):
            with open(units_path, 'r', encoding='utf-8') as f:
                units_dict = json.load(f)
                units = units_dict.keys()
        else:
            units, units_dict = [], {}
        
        # Load symbols dictionary
        if os.path.exists(symbols_path):
            with open(symbols_path, 'r', encoding='utf-8') as f:
                symbols_dict = json.load(f)
        else:
            symbols_dict = {}
        
        # Load GLM pairs (Grapheme-to-Lexeme Mapping) - global for all languages
        # Applied ONLY on substitutions after all other normalizations
        glm_pairs_path = os.path.join(current_dir, "lang_resources", "glm_pairs.json")
        if os.path.exists(glm_pairs_path):
            with open(glm_pairs_path, 'r', encoding='utf-8') as f:
                self.glm_arrays = json.load(f)
            print(f"Loaded global GLM pairs with {len(self.glm_arrays)} entries")
        else:
            self.glm_arrays = None
        
        # Combine units and symbols into equivalents dictionary
        self.equivalents = {**units_dict, **symbols_dict}

    def evaluate(self, predictions: List[str], references: List[str], **kwargs) -> float:
        """Evaluate multiple samples and return overall WER"""
        refs, hys = [], []
        # data = {"ref": references, "hyp": predictions}
        # import json
        # with open("data_te.json", "w") as f:
        #     json.dump(data, f)
        # return 0.0
        
        from concurrent.futures import ThreadPoolExecutor
        
        
        def process_pair(ref, hyp):
            if len(hyp.split()) > 2*len(ref.split()):
                hyp = ' '.join(hyp.split()[:2*len(ref.split())])
            if self.language != "en":
                ref = self.normalizer.normalize(ref)
                hyp = self.normalizer.normalize(hyp)
            else:
                ref = ref
                hyp = hyp
            # normalized_ref_text = self.normalizer.normalize(ref)
            # normalized_hyp_text = self.normalizer.normalize(hyp)
            _, normalized_ref, normalized_hyp = optimize_wer(ref, hyp, self.equivalents, language=self.language, normalizer=self.normalizer, glm_arrays=self.glm_arrays)
            
            # with refs_lock:
            refs.append(normalized_ref)
            # with hys_lock:
            hys.append(normalized_hyp)

            return None
            
        
        # Use ThreadPoolExecutor for multithreading
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(process_pair, ref, hyp) for ref, hyp in zip(references, predictions)]
            # Wait for all threads to complete
            for future in futures:
                future.result()
        # data = {"ref": refs, "hyp": hys}
        # import json
        # with open("data.json", "w") as f:
        #     json.dump(data, f)
        wer_metric = _get_wer_metric()
        return wer_metric.evaluate(predictions=hys, references=refs)


# ============================================================================
# KEYWORD ERROR RATE METRIC
# ============================================================================

def normalize_keyword(keyword: str, language: str = 'en', normalizer=None, units_dict: Dict = None) -> str:
    """
    Normalize a single keyword using the same normalizations as semantic_wer_metric.
    Note: GLM pairs are NOT applied here - they are only applied to substitutions in semantic_wer.
    
    Args:
        keyword: The keyword to normalize
        language: Language code for normalization
        normalizer: Optional Indic normalizer for non-English languages
        units_dict: Dictionary of units for adding spaces before units
        
    Returns:
        Normalized keyword string
    """
    if units_dict is None:
        units_dict = {}
    
    # Apply Indic normalization for non-English languages
    if language != 'en' and normalizer:
        keyword = normalizer.normalize(keyword)
    
    # Lowercase
    keyword = keyword.lower()
    
    # Expand contractions
    if "'" in keyword:
        keyword = expand_contractions(keyword)
    
    # Remove punctuation except slash and hyphen
    keyword = remove_punctuation_except_slash(keyword)
    
    # Add spaces around hyphens in ranges
    keyword = add_spaces_around_hyphens_in_ranges(keyword)
    
    # Add spaces around symbols and before units
    keyword = add_space_before_units(add_spaces_around_symbols(keyword), units=units_dict.keys())
    
    # Convert numbers to words (same as text normalization)
    if re.search(r'\d+', keyword):
        keyword = convert_numbers_to_words(keyword, language=language)
    
    # Transliterate for non-English languages (e.g., "surgery" -> "सर्जरी" for Hindi)
    if language != "en":
        keyword = transliterate(keyword, language=language, units_dict=units_dict, normalizer=normalizer)
    
    return ' '.join(keyword.split()).strip()


def normalize_text_for_keyword_extraction(text: str, units_dict: Dict, language: str = 'en', normalizer=None) -> str:
    """
    Normalize text using the same normalizations as semantic_wer_metric for keyword extraction.
    Note: GLM pairs are NOT applied here - they are only applied to substitutions in semantic_wer.
    
    Args:
        text: Text to normalize
        units_dict: Dictionary of units and their variations
        language: Language code
        normalizer: Optional Indic normalizer for non-English languages
        
    Returns:
        Normalized text string
    """
    # Apply Indic normalization for non-English languages
    if language != 'en' and normalizer:
        text = normalizer.normalize(text)
    
    # Lowercase
    text = text.lower()
    
    # Expand contractions
    if "'" in text:
        text = expand_contractions(text)
    
    # Remove punctuation except slash and hyphen
    text = remove_punctuation_except_slash(text)
    
    # Add spaces around hyphens in ranges
    text = add_spaces_around_hyphens_in_ranges(text)
    
    # Add spaces around symbols and before units
    text = add_space_before_units(add_spaces_around_symbols(text), units=units_dict.keys())
    
    # Convert numbers to words
    if re.search(r'\d+', text):
        text = convert_numbers_to_words(text, language=language)
    
    # Transliterate for non-English languages
    if language != "en":
        text = transliterate(text, language=language, units_dict=units_dict, normalizer=normalizer)
    
    return ' '.join(text.split()).strip()


def extract_keywords_from_text(text: str, keywords: List[str], normalize: bool = True) -> List[str]:
    """
    Extract keywords from text in order of appearance.
    
    Args:
        text: The text to search for keywords
        keywords: List of keywords to look for (should be normalized if normalize=True was used on text)
        normalize: Whether keywords are already normalized
        
    Returns:
        List of found keywords in order of appearance
    """
    found_keywords = []
    words = text.split()
    
    # Sort keywords by length (descending) to match longer phrases first
    sorted_keywords = sorted(keywords, key=lambda k: len(k.split()), reverse=True)
    
    i = 0
    while i < len(words):
        matched = False
        for keyword in sorted_keywords:
            keyword_words = keyword.split()
            keyword_len = len(keyword_words)
            
            if i + keyword_len <= len(words):
                # Check if this keyword matches at current position
                text_segment = ' '.join(words[i:i + keyword_len])
                if text_segment == keyword:
                    found_keywords.append(keyword)
                    i += keyword_len
                    matched = True
                    break
        
        if not matched:
            i += 1
    
    return found_keywords


def find_keyword_positions(text: str, keywords: List[str]) -> List[Tuple[str, int, int]]:
    """
    Find keyword positions (start_idx, end_idx) in text.
    
    Args:
        text: The text to search
        keywords: List of normalized keywords
        
    Returns:
        List of (keyword, start_word_idx, end_word_idx) tuples in order of appearance
    """
    found = []
    words = text.split()
    sorted_keywords = sorted(keywords, key=lambda k: len(k.split()), reverse=True)
    
    used_positions = set()
    
    i = 0
    while i < len(words):
        matched = False
        for keyword in sorted_keywords:
            keyword_words = keyword.split()
            keyword_len = len(keyword_words)
            
            if i + keyword_len <= len(words):
                text_segment = ' '.join(words[i:i + keyword_len])
                if text_segment == keyword:
                    found.append((keyword, i, i + keyword_len))
                    for pos in range(i, i + keyword_len):
                        used_positions.add(pos)
                    i += keyword_len
                    matched = True
                    break
        
        if not matched:
            i += 1
    
    return found


def get_word_alignment(ref_words: List[str], hyp_words: List[str]) -> Dict[int, List[int]]:
    """
    Get word-level alignment mapping from reference word indices to hypothesis word indices.
    
    Uses difflib SequenceMatcher to find alignment.
    
    Args:
        ref_words: List of reference words
        hyp_words: List of hypothesis words
        
    Returns:
        Dict mapping ref_word_idx -> list of hyp_word_indices it aligns to
        (empty list means deletion, multiple indices mean the ref word aligns to multiple hyp words)
    """
    matcher = difflib.SequenceMatcher(None, ref_words, hyp_words)
    opcodes = matcher.get_opcodes()
    
    alignment = {}
    
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == 'equal':
            # One-to-one mapping for equal segments
            for offset, ref_idx in enumerate(range(i1, i2)):
                alignment[ref_idx] = [j1 + offset]
        
        elif tag == 'replace':
            # Distribute hypothesis words among reference words
            ref_len = i2 - i1
            hyp_len = j2 - j1
            
            if ref_len == hyp_len:
                # One-to-one substitution
                for offset, ref_idx in enumerate(range(i1, i2)):
                    alignment[ref_idx] = [j1 + offset]
            elif hyp_len == 0:
                # All reference words deleted
                for ref_idx in range(i1, i2):
                    alignment[ref_idx] = []
            else:
                # Map all ref words to the hyp range
                hyp_indices = list(range(j1, j2))
                for ref_idx in range(i1, i2):
                    alignment[ref_idx] = hyp_indices
        
        elif tag == 'delete':
            # Reference words have no corresponding hypothesis words
            for ref_idx in range(i1, i2):
                alignment[ref_idx] = []
        
        elif tag == 'insert':
            # Hypothesis words have no corresponding reference words
            # No ref indices to map
            pass
    
    return alignment


def extract_keywords_with_alignment(
    ref_text: str, 
    hyp_text: str, 
    keywords: List[str],
    glm_arrays: Dict[str, str] = None
) -> Tuple[List[str], List[str]]:
    """
    Extract keywords from reference and find their aligned counterparts in hypothesis.
    
    This function:
    1. Finds keywords in the reference text
    2. Uses word-level alignment to find what those keywords map to in hypothesis
    3. Applies GLM normalization to mismatched pairs (if glm_arrays provided)
    4. Returns both ref keywords and their aligned hyp counterparts
    
    Args:
        ref_text: Normalized reference text
        hyp_text: Normalized hypothesis text
        keywords: List of normalized keywords to look for
        glm_arrays: Optional GLM pairs dict for normalizing mismatched keyword pairs
        
    Returns:
        Tuple of (ref_keywords, hyp_keywords) where hyp_keywords are the aligned
        counterparts (may be different text, partial match, or empty for deletions)
    """
    ref_words = ref_text.split()
    hyp_words = hyp_text.split()
    
    # Find keyword positions in reference
    ref_keyword_positions = find_keyword_positions(ref_text, keywords)
    
    if not ref_keyword_positions:
        return [], []
    
    # Get word-level alignment
    alignment = get_word_alignment(ref_words, hyp_words)
    
    ref_keywords = []
    hyp_keywords = []
    
    for keyword, start_idx, end_idx in ref_keyword_positions:
        ref_keywords.append(keyword)
        
        # Find aligned hypothesis words for this keyword's positions
        aligned_hyp_indices = set()
        for ref_idx in range(start_idx, end_idx):
            if ref_idx in alignment:
                aligned_hyp_indices.update(alignment[ref_idx])
        
        if aligned_hyp_indices:
            # Sort indices and extract the corresponding hypothesis words
            sorted_indices = sorted(aligned_hyp_indices)
            aligned_hyp_words = [hyp_words[idx] for idx in sorted_indices if idx < len(hyp_words)]
            hyp_keyword = ' '.join(aligned_hyp_words)
            
            # Apply GLM normalization if keyword pair is mismatched
            if glm_arrays and hyp_keyword and keyword != hyp_keyword:
                # Try applying GLM to both ref and hyp keywords
                normalized_ref_kw = apply_word_mappings(keyword, glm_arrays)
                normalized_hyp_kw = apply_word_mappings(hyp_keyword, glm_arrays)
                
                # Only use GLM-normalized versions if they improve the match
                if normalized_ref_kw == normalized_hyp_kw:
                    # Perfect match after GLM - update both
                    ref_keywords[-1] = normalized_ref_kw
                    hyp_keyword = normalized_hyp_kw
                elif calculate_cer(normalized_ref_kw, normalized_hyp_kw) < calculate_cer(keyword, hyp_keyword):
                    # GLM improves CER - use normalized versions
                    ref_keywords[-1] = normalized_ref_kw
                    hyp_keyword = normalized_hyp_kw
            
            hyp_keywords.append(hyp_keyword)
        else:
            # Deletion - no aligned words in hypothesis
            hyp_keywords.append('')
    
    return ref_keywords, hyp_keywords


# Entity types for entity accuracy metrics
ENTITY_TYPES = [
    "drugs",
    "conditions", 
    "symptoms",
    "diagnostic_entities",
    "procedure",
    "examination",
    "other_medical_named_entities"
]

