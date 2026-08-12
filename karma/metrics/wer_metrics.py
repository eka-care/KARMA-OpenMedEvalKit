"""
Normalized WER Metric for Medical ASR

This module provides a WER metric that normalizes medical units, numbers, and punctuation
to improve accuracy in medical transcription evaluation.
"""

import re
import difflib
import logging
import os
from typing import List, Tuple, Dict
# from num2words import num2words

from karma.metrics.base_metric_abs import BaseMetric
from karma.registries.metrics_registry import register_metric
from indic_numtowords import num2words
from google.transliteration import transliterate_word
from karma.metrics.language_support import resolve_indic_normalizer
# Re-exported so existing callers (notebooks, examples/, analysis_metrics) keep
# importing these from wer_metrics after the move to wer_utils.
from karma.metrics.wer_utils import (  # noqa: F401
    PUNCTUATION_MARKS, SPECIAL_TOKENS, ENTITY_TYPES,
    calculate_wer, calculate_cer,
    strip_special_tokens, add_spaces_around_symbols, add_spaces_around_hyphens_in_ranges,
    add_space_before_units, remove_punctuation_except_slash, expand_contractions,
    replace_whole_word, is_english_word, apply_word_mappings, find_matching_units,
    align_tokenization, _join_letter_runs,
    canonicalize_orthography, canonicalize_units, sanitize_glm_pairs,
    SemanticResources, load_semantic_resources,
    find_keyword_positions, extract_keywords_from_text, get_word_alignment,
    count_edit_operations, count_edit_operations_difflib,
    _get_wer_metric, _get_cer_metric,
)

logger = logging.getLogger(__name__)


# ============================================================================
# TRANSLITERATION CACHE (file-backed)
# ============================================================================

import json
import threading

_CACHE_FILE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "lang_resources",
    "transliteration_cache.json"
)

_CACHE_LOCK = threading.Lock()


def _load_transliteration_cache() -> Dict[str, List[str]]:
    try:
        with open(_CACHE_FILE_PATH, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {}


_TRANSLITERATION_CACHE: Dict[str, List[str]] = _load_transliteration_cache()


def _save_transliteration_cache() -> None:
    try:
        with open(_CACHE_FILE_PATH, "w", encoding="utf-8") as handle:
            json.dump(_TRANSLITERATION_CACHE, handle, ensure_ascii=False, indent=2)
    except OSError as e:
        logger.warning(f"Could not persist transliteration cache: {e}")


def get_transliterated_variations(text: str, language: str = 'en') -> List[str]:
    """
    Get transliterated variations with file-backed persistent caching.

    Flow:
    1. Check the local cache (loaded from lang_resources/transliteration_cache.json)
    2. If not found, call the transliteration API and save the result

    Returns a list of transliterated variations, or empty list if transliteration not available.
    Retries transliteration up to 5 times if it fails, sleeping 1s, 2s, ..., 5s between attempts.
    """
    import time

    # Step 1: Check the cache
    cache_key = f"{language}:{text}"
    with _CACHE_LOCK:
        cached_result = _TRANSLITERATION_CACHE.get(cache_key)
    if cached_result is not None:
        return cached_result

    # Step 2: Not in cache - call API with retry logic
    max_retries = 5
    for attempt in range(1, max_retries + 1):
        try:
            result = transliterate_word(text, lang_code=language)

            with _CACHE_LOCK:
                _TRANSLITERATION_CACHE[cache_key] = result
                _save_transliteration_cache()

            return result

        except Exception as e:
            print(f"Error transliterating word {text}: {e}, attempt {attempt} of {max_retries}")
            if attempt == max_retries:
                # Deliberately NOT cached. Persisting [] here would record a
                # transient network failure as a permanent fact: the entry is
                # written to the shared cache file, never expires, and silently
                # downgrades that word to "leave it in Latin script" for every
                # future run. An outage once clobbered 756 real entries this way.
                return []
            time.sleep(attempt)


# ============================================================================
# TEXT NORMALIZATION
# ============================================================================

def _num_to_words(value: int, language: str, normalizer=None) -> List[str]:
    """Every word form indic_numtowords knows for an integer, in this language
    and in English, normalized to match the rest of the pipeline."""
    forms = []
    for lang in (language, 'en'):
        try:
            produced = num2words(int(value), lang=lang, variations=True)
        except Exception:
            continue
        if not isinstance(produced, list):
            produced = [produced]
        for form in produced:
            # num2words emits nukta/chandrabindu spellings (पाँच, हज़ार) that the
            # Indic normalizer strips everywhere else in the pipeline; without
            # this the digit-derived word can never match the same word in text.
            if normalizer:
                form = normalizer.normalize(form)
            if form and form not in forms:
                forms.append(form)
    return forms


def number_candidates(token: str, language: str = 'en', normalizer=None) -> List[str]:
    """All the ways a numeric token might have been spoken.

    Covers what the single num2words(...)[0] call missed: the alternate readings
    ('150' is एक सौ पचास or डेढ़ सौ), digit-by-digit ('09' as "zero nine"), year
    readings ('2024' as "twenty twenty four"), decimals, and comma-grouped
    thousands — which previously became a separate '000' token and converted to
    "शून्य", silently rewriting 60,000 as "sixty zero".
    """
    token = token.strip()
    candidates: List[str] = []

    def add(value):
        if value and value not in candidates:
            candidates.append(value)

    grouped = re.fullmatch(r'\d{1,3}(?:,\d{2,3})+', token)
    plain = token.replace(',', '') if grouped else token

    if re.fullmatch(r'\d+', plain):
        number = int(plain)
        for form in _num_to_words(number, language, normalizer):
            add(form)

        # Digit by digit, the way identifiers and dates are read aloud
        for lang in (language, 'en'):
            digits = []
            for digit in plain:
                forms = _num_to_words(int(digit), lang, normalizer)
                if not forms:
                    digits = []
                    break
                digits.append(forms[0])
            if digits:
                add(' '.join(digits))

        # Years: 2024 as "twenty twenty four" / "बीस चौबीस"
        if len(plain) == 4 and plain[0] != '0':
            for lang in (language, 'en'):
                left = _num_to_words(int(plain[:2]), lang, normalizer)
                right = _num_to_words(int(plain[2:]), lang, normalizer)
                if left and right:
                    add(f"{left[0]} {right[0]}")
        add(plain)
        return candidates

    decimal = re.fullmatch(r'(\d+)\.(\d+)', token)
    if decimal:
        whole, frac = decimal.group(1), decimal.group(2)
        point = {'hi': 'दशमलव', 'mr': 'दशांश', 'te': 'దశాంశ', 'ta': 'புள்ளி', 'kn': 'ದಶಮಾಂಶ'}.get(language)
        whole_forms = _num_to_words(int(whole), language, normalizer)
        frac_digits = [_num_to_words(int(d), language, normalizer) for d in frac]
        if whole_forms and all(frac_digits):
            spoken = ' '.join(f[0] for f in frac_digits)
            if point:
                add(f"{whole_forms[0]} {point} {spoken}")
            add(f"{whole_forms[0]} point {spoken}")
        # Hindi has dedicated words for the common halves and quarters
        if language == 'hi':
            colloquial = {'0.5': 'आधा', '1.5': 'डेढ़', '2.5': 'ढाई', '1.25': 'सवा', '0.25': 'पाव'}.get(token)
            if colloquial:
                add(normalizer.normalize(colloquial) if normalizer else colloquial)
        add(token)
    return candidates


def convert_numbers_to_words(text: str, language: str = 'en', normalizer=None,
                             counterpart: str = None) -> str:
    """Convert digit tokens to word form.

    When `counterpart` is given, a token is rewritten only if one of its readings
    actually appears on the other side, and that matching reading is the one
    used. Converting unconditionally would inflate the reference word count (one
    "500" becoming two words) and so quietly dilute the error rate even when the
    two sides already agreed.
    """
    # Empty string, not None: an insertion or deletion segment has an empty
    # counterpart, and `"..." in None` raises. That exception was swallowed by
    # the per-pair fallback in evaluate(), quietly scoring every such utterance
    # un-normalized.
    other = f" {counterpart} " if counterpart else ""
    result = []

    for word in text.split():
        candidates = number_candidates(word, language, normalizer) if counterpart is not None \
            else None

        if candidates:
            chosen = None
            for candidate in candidates:
                if candidate != word and f" {candidate} " in other:
                    chosen = candidate
                    break
            result.append(chosen if chosen else word)
            continue

        if re.fullmatch(r'\d+', word):
            try:
                num_word = num2words(int(word), lang=language, variations=True)
                if isinstance(num_word, list):
                    num_word = num_word[0]
                if normalizer:
                    num_word = normalizer.normalize(num_word)
                result.append(num_word)
            except Exception as e:
                print(f"Error converting number {word} to words: {e}")
                result.append(word)
        else:
            result.append(word)

    return ' '.join(result)


def apply_punctuation_normalizations(ref: str, hyp: str, best_cer: float) -> Tuple[str, str, float]:
    """
    Try various punctuation normalizations and return the best result.
    Handles hyphens specially to avoid breaking compound words like 'zero-dol'.
    Returns (best_ref, best_hyp, best_cer)
    """
    best_ref, best_hyp = ref, hyp
    
    has_punct = any(p in ref or p in hyp for p in ['-', '.', ',', '!', '?', ';', ':', '"', "'"])
    if not has_punct:
        return best_ref, best_hyp, best_cer

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
        if old_char not in ref and old_char not in hyp:
            # Nothing to transform — the candidate would be byte-identical to the
            # input. Over 99% of these transforms are no-ops on real text, and
            # this check is what keeps them from dominating the CER budget.
            continue

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


def transliterate(
    text: str,
    language: str = 'en',
    units_dict: Dict = None,
    normalizer=None,
    candidate_map: Dict[str, set] = None,
) -> str:
    """Transliterate Latin tokens into the target script.

    The emitted token is candidate[0], as before. When `candidate_map` is passed
    it is filled in place with {emitted_token -> every candidate for that token},
    so a later pass can revisit the choice once it can see what the token had to
    align against — see repair_candidate_substitutions().
    """
    words = []
    for word in text.split():
        if is_english_word(word) and word not in units_dict.keys() and not re.fullmatch(r'\d+', word):
            try:
                transliterated_words= get_transliterated_variations(word, language=language)
            except Exception as e:
                # Fall through rather than `continue`: skipping the append would
                # DELETE the word from the text. On the reference side that
                # shrinks the WER denominator, so a transliteration outage would
                # quietly change the yardstick instead of failing loudly.
                transliterated_words = []
                print(f"Error transliterating word {word}: {e}")
            if transliterated_words:
                normalized = []
                for cand in transliterated_words:
                    cand = normalizer.normalize(cand) if normalizer else cand
                    if cand not in normalized:
                        normalized.append(cand)
                transliterated_word = normalized[0]
                if candidate_map is not None:
                    entry = candidate_map.setdefault(transliterated_word, set())
                    entry.update(normalized)
                    entry.add(word)
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


def repair_candidate_substitutions(
    ref: str,
    hyp: str,
    cmap_ref: Dict[str, set],
    cmap_hyp: Dict[str, set],
) -> Tuple[str, str, int]:
    """Revisit transliteration choices that the alignment shows were wrong.

    transliterate() has to commit to candidate[0] before it knows what the token
    will align against, and for romanized Hindi that guess is often the wrong
    spelling of the right word ("ki" -> की where the reference says कि). Here the
    alignment is known, so within each mismatched segment a token may switch to
    any of its own candidates that matches its counterpart.

    Only substitutions are touched, each token is repaired at most once, and the
    move is symmetric, so this can turn a mismatch into a match but can never
    invent a match between two tokens that had no candidate in common.
    """
    ref_words, hyp_words = ref.split(), hyp.split()
    repairs = 0

    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ref_words, hyp_words).get_opcodes():
        if tag != 'replace':
            continue

        ref_seg, hyp_seg = ref_words[i1:i2], hyp_words[j1:j2]
        used_ref, used_hyp = set(), set()

        # A hypothesis token owns a candidate equal to some reference token
        for j, hyp_word in enumerate(hyp_seg):
            candidates = cmap_hyp.get(hyp_word)
            if not candidates:
                continue
            for i, ref_word in enumerate(ref_seg):
                if i in used_ref or ref_word == hyp_word:
                    continue
                if ref_word in candidates:
                    hyp_seg[j] = ref_word
                    used_ref.add(i); used_hyp.add(j); repairs += 1
                    break

        # Mirror image: a reference token owns a candidate equal to some hypothesis token
        for i, ref_word in enumerate(ref_seg):
            if i in used_ref:
                continue
            candidates = cmap_ref.get(ref_word)
            if not candidates:
                continue
            for j, hyp_word in enumerate(hyp_seg):
                if j in used_hyp or hyp_word == ref_word:
                    continue
                if hyp_word in candidates:
                    ref_seg[i] = hyp_word
                    used_ref.add(i); used_hyp.add(j); repairs += 1
                    break

        # Both sides were Latin: settle on a spelling they agree on
        for i, ref_word in enumerate(ref_seg):
            if i in used_ref:
                continue
            ref_candidates = cmap_ref.get(ref_word)
            if not ref_candidates:
                continue
            for j, hyp_word in enumerate(hyp_seg):
                if j in used_hyp:
                    continue
                hyp_candidates = cmap_hyp.get(hyp_word)
                if not hyp_candidates:
                    continue
                shared = ref_candidates & hyp_candidates
                if shared:
                    agreed = sorted(shared)[0]
                    ref_seg[i] = hyp_seg[j] = agreed
                    used_ref.add(i); used_hyp.add(j); repairs += 1
                    break

        ref_words[i1:i2], hyp_words[j1:j2] = ref_seg, hyp_seg

    return ' '.join(ref_words), ' '.join(hyp_words), repairs


def _canonicalize_candidate_map(cmap: Dict[str, set], ortho_map: Dict[str, str]) -> Dict[str, set]:
    """Apply the orthography folding to both sides of a candidate map."""
    folded: Dict[str, set] = {}
    for token, candidates in cmap.items():
        key = ortho_map.get(token, token)
        folded.setdefault(key, set()).update(
            ortho_map.get(c, c) for c in candidates
        )
    return folded


            


# Longest run of tokens that may be joined into one. Covers the spelled-out
# acronyms that actually occur ("h b a 1 c", "b u n", "e c g"); beyond this the
# match is more likely to be coincidence than tokenization.


def normalize_pair(ref_part: str, hyp_part: str, units_dict: Dict, language: str = 'en', glm_arrays: Dict[str, str] = None, normalizer=None) -> Tuple[str, str]:
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

    # Step 0b: settle digit-vs-spoken mismatches while the numbers are still
    # whole. After punctuation removal "2.5" is two tokens and "1,000" has a
    # bogus "000" that converts to "zero".
    if re.search(r'\d', ref_part) or re.search(r'\d', hyp_part):
        ref_part = re.sub(r'(?<=\d),(?=\d)', '', ref_part)
        hyp_part = re.sub(r'(?<=\d),(?=\d)', '', hyp_part)
        resolved_ref = convert_numbers_to_words(
            ref_part, language=language, normalizer=normalizer, counterpart=hyp_part
        )
        resolved_hyp = convert_numbers_to_words(
            hyp_part, language=language, normalizer=normalizer, counterpart=ref_part
        )
        ref_part, hyp_part = resolved_ref, resolved_hyp

    # Step 1: Remove punctuation (except '/' and '-')
    ref_part = remove_punctuation_except_slash(ref_part)
    hyp_part = remove_punctuation_except_slash(hyp_part)

    # Step 1b: an acronym dictated letter by letter belongs to one word even when
    # a letter came back wrong. align_tokenization already folded the cases that
    # match the reference exactly; here "h b a 1 d" against "hba1c" is still four
    # insertions on top of the one substitution it deserves. Kept only when it
    # lowers CER, so a genuine run of one-letter words is left alone.
    if ref_part and hyp_part:
        collapsed = _join_letter_runs(hyp_part)
        if collapsed != hyp_part and \
                calculate_cer(ref_part, collapsed) < calculate_cer(ref_part, hyp_part):
            hyp_part = collapsed

    best_ref = ref_part
    best_hyp = hyp_part
    best_cer = calculate_cer(ref_part, hyp_part) if ref_part and hyp_part else float('inf')
    
    
    # Step 2: Try number-to-words conversion
    if re.search(r'\d+', ref_part) or re.search(r'\d+', hyp_part):
        ref_with_words = convert_numbers_to_words(ref_part, language=language, normalizer=normalizer) if ref_part else ''
        hyp_with_words = convert_numbers_to_words(hyp_part, language=language, normalizer=normalizer) if hyp_part else ''
        
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
                temp_ref_nums = convert_numbers_to_words(temp_ref, language=language, normalizer=normalizer) if temp_ref else ''
                temp_hyp_nums = convert_numbers_to_words(temp_hyp, language=language, normalizer=normalizer) if temp_hyp else ''
                
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

def optimize_wer(ref: str, hyp: str, units_dict: Dict, language: str = 'en', normalizer = None, glm_arrays: Dict[str, str] = None, pure_units_dict: Dict = None, ortho_map: Dict[str, str] = None) -> Tuple[float, str, str]:
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
    # Strip ASR special/artifact tokens (e.g. "<is>") before any scoring
    ref = strip_special_tokens(ref)
    hyp = strip_special_tokens(hyp)

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

    # Preprocess: adopt the reference's word boundaries where the two sides only
    # disagree about spacing. Must run before transliteration, which sends each
    # token to the API on its own and would scatter the letters of "h b a 1 c"
    # into five unrelated Devanagari fragments the acronym can never come back
    # from.
    hyp = align_tokenization(ref, hyp)

    # Preprocess: fold every spelling of a unit onto its canonical key, before
    # transliteration puts the Latin variants out of reach
    if pure_units_dict:
        ref = canonicalize_units(ref, pure_units_dict)
        hyp = canonicalize_units(hyp, pure_units_dict)

    # Preprocess: transliterate, remembering the candidates not chosen
    cmap_ref, cmap_hyp = {}, {}
    if language != "en":
        ref = transliterate(ref, language=language, units_dict=units_dict,
                            normalizer=normalizer, candidate_map=cmap_ref)
        hyp = transliterate(hyp, language=language, units_dict=units_dict,
                            normalizer=normalizer, candidate_map=cmap_hyp)

    # Fold spelling variants once both sides are in the same script. The
    # candidate maps are folded with them, so repair still keys off what is
    # actually in the text.
    if ortho_map:
        ref = canonicalize_orthography(ref, ortho_map)
        hyp = canonicalize_orthography(hyp, ortho_map)
        cmap_ref = _canonicalize_candidate_map(cmap_ref, ortho_map)
        cmap_hyp = _canonicalize_candidate_map(cmap_hyp, ortho_map)
    
    # Find unmatched pairs
    diffs, words_ref, words_hyp = find_unmatched_pairs(ref, hyp)
    
    if not diffs:
        return 0.0, ref, hyp
    
    # Normalize each unmatched pair
    normalized_ref_parts = {}
    normalized_hyp_parts = {}
    
    for tag, i1, i2, j1, j2, ref_part, hyp_part in diffs:
        best_ref_part, best_hyp_part = normalize_pair(ref_part, hyp_part, units_dict, language=language, glm_arrays=glm_arrays, normalizer=normalizer)
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
        best_ref, best_hyp = optimized_ref, optimized_hyp
    else:
        best_ref, best_hyp = ref, hyp

    # Now that the alignment is settled, let mis-guessed transliterations switch
    # to a candidate that actually matches their counterpart
    if cmap_ref or cmap_hyp:
        repaired_ref, repaired_hyp, repairs = repair_candidate_substitutions(
            best_ref, best_hyp, cmap_ref, cmap_hyp
        )
        if repairs:
            return calculate_wer(repaired_ref, repaired_hyp), repaired_ref, repaired_hyp

    return calculate_wer(best_ref, best_hyp), best_ref, best_hyp


# ============================================================================
# SHARED RESOURCE LOADING
# ============================================================================


# ============================================================================
# METRIC CLASS
# ============================================================================

@register_metric("semantic_wer_metric", optional_args=["language"], default_args={"language": "en"})
class SemanticWERMetric(BaseMetric):
    """
    Normalized WER metric that applies medical unit, number, and punctuation 
    normalization before computing WER.
    """
    
    def __init__(self, metric_name: str = "semantic_wer_metric", **kwargs):
        self.language, self.normalizer = resolve_indic_normalizer(
            kwargs.get("language"), do_normalize_chandras=True, remove_nuktas=True
        )
        super().__init__(metric_name, **kwargs)

        resources = load_semantic_resources(self.language)
        self.equivalents = resources.equivalents
        # Units only. Symbol variations ('से', 'to', 'per', 'by') are ordinary
        # words and must never be canonicalized corpus-wide the way units are.
        self.units_dict = resources.units_dict
        self.glm_arrays = resources.glm_arrays
        self.ortho_map = resources.ortho_map

    def evaluate(self, predictions: List[str], references: List[str], **kwargs) -> float:
        """Evaluate multiple samples and return overall WER"""
        from concurrent.futures import ThreadPoolExecutor

        def process_pair(ref, hyp):
            # Strip ASR special/artifact tokens (e.g. "<is>") before the length
            # guard so they don't inflate the hypothesis word count
            original_ref, original_hyp = ref, hyp
            try:
                ref = strip_special_tokens(ref)
                hyp = strip_special_tokens(hyp)
                if len(hyp.split()) > 2*len(ref.split()):
                    hyp = ' '.join(hyp.split()[:2*len(ref.split())])
                if self.language != "en":
                    ref = self.normalizer.normalize(ref)
                    hyp = self.normalizer.normalize(hyp)
                _, normalized_ref, normalized_hyp = optimize_wer(
                    ref, hyp, self.equivalents, language=self.language,
                    normalizer=self.normalizer, glm_arrays=self.glm_arrays,
                    pure_units_dict=self.units_dict, ortho_map=self.ortho_map,
                )
                return normalized_ref, normalized_hyp
            except Exception:
                # Score this pair un-normalized rather than losing the whole run
                # to one pathological utterance.
                logger.exception("Normalization failed for one pair; scoring it raw")
                return original_ref, original_hyp

        # executor.map yields results in input order, so ref[i] always stays
        # paired with hyp[i]. Appending from the workers instead — as this did
        # previously, with two separate unlocked appends — lets an interleaving
        # mis-pair a reference with someone else's hypothesis, which silently
        # corrupts the corpus WER.
        with ThreadPoolExecutor(max_workers=16) as executor:
            results = list(executor.map(process_pair, references, predictions))

        refs = [ref for ref, _ in results]
        hys = [hyp for _, hyp in results]

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

    # Strip ASR special/artifact tokens (e.g. "<is>") first
    keyword = strip_special_tokens(keyword)

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


def _normalize_before_transliteration(text: str, units_dict: Dict, language: str = 'en',
                                      normalizer=None, convert_numbers: bool = True) -> str:
    """Everything normalize_text_for_keyword_extraction does up to transliteration.

    Split out so tokenization can be settled while the text is still in its
    original script — once transliterate() has sent "h", "b", "a" and "c" to the
    API one token at a time, the acronym cannot be put back together.
    """
    # Strip ASR special/artifact tokens (e.g. "<is>") first
    text = strip_special_tokens(text)

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
    if convert_numbers and re.search(r'\d+', text):
        text = convert_numbers_to_words(text, language=language)

    return text


def normalize_text_for_keyword_extraction(text: str, units_dict: Dict, language: str = 'en',
                                          normalizer=None, counterpart: str = None) -> str:
    """
    Normalize text using the same normalizations as semantic_wer_metric for keyword extraction.
    Note: GLM pairs are NOT applied here - they are only applied to substitutions in semantic_wer.

    Args:
        text: Text to normalize
        units_dict: Dictionary of units and their variations
        language: Language code
        normalizer: Optional Indic normalizer for non-English languages
        counterpart: The other side of the pair, when there is one. Pass the raw
            reference when normalizing a hypothesis so the two end up with the
            same word boundaries: a keyword like "hba1c" is otherwise missed
            outright when the model spelled it "h b a 1 c", counting as a full
            keyword error for a correct transcription.

    Returns:
        Normalized text string
    """
    if counterpart:
        # Settle the word boundaries while the digits are still digits: number
        # conversion would turn "hba1c" into "hbaonec" and the acronym could
        # never be matched against the reference again.
        text = _normalize_before_transliteration(
            text, units_dict, language, normalizer, convert_numbers=False
        )
        reference = _normalize_before_transliteration(
            counterpart, units_dict, language, normalizer, convert_numbers=False
        )
        text = align_tokenization(reference, text)
        if re.search(r'\d+', text):
            text = convert_numbers_to_words(text, language=language)
    else:
        text = _normalize_before_transliteration(text, units_dict, language, normalizer)

    # Transliterate for non-English languages
    if language != "en":
        text = transliterate(text, language=language, units_dict=units_dict, normalizer=normalizer)

    return ' '.join(text.split()).strip()


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