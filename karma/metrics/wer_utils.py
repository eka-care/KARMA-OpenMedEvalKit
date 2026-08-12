"""
Shared primitives for the WER / keyword / entity metrics.

Everything here is pure and stateless enough to be used by any of the metric
modules. It exists because these helpers had been copy-pasted between
wer_metrics, analysis_metrics and alignment_utils and had begun to diverge:
get_word_alignment had two implementations with different behaviour on
unequal-length substitutions, find_keyword_positions was byte-identical in two
files, and each metric class carried its own copy of the resource loading.

Where two versions existed, the better one was kept:
  * get_word_alignment  - the similarity-based version from alignment_utils,
    which avoids aligning every reference word to the whole hypothesis span.
  * extract_keywords_with_alignment stays in wer_metrics, since that copy is
    the one that applies GLM normalization.
"""

import os
import re
import json
import string
import difflib
import threading
from typing import List, Tuple, Dict, NamedTuple

from karma.metrics.common_metrics import WERMetric, CERMetric



# ==========================================================================
# CONSTANTS
# ==========================================================================

PUNCTUATION_MARKS = ['.', ',', '-', '!', '?', ';', ':', '"', "'"]


SPECIAL_TOKENS = ['<is>']


_SPECIAL_TOKENS_RE = re.compile('|'.join(re.escape(tok) for tok in SPECIAL_TOKENS))


ENTITY_TYPES = [
    "drugs",
    "conditions", 
    "symptoms",
    "diagnostic_entities",
    "procedure",
    "examination",
    "other_medical_named_entities"
]



# ==========================================================================
# SCORING PRIMITIVES
# ==========================================================================

_thread_local = threading.local()


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
    """Calculate Character Error Rate between reference and hypothesis.

    Memoized: the hill-climb in normalize_pair re-scores the same pair over and
    over while trying unit variations, so ~86% of calls repeat an argument pair
    already computed (one pair was seen 504 times in a 100-utterance run).
    """
    key = (ref, hyp)
    cache = _thread_local.__dict__.setdefault('cer_cache', {})
    cached = cache.get(key)
    if cached is not None:
        return cached

    cer_metric = _get_cer_metric()
    value = cer_metric.evaluate(predictions=[hyp], references=[ref])
    if len(cache) < 100_000:
        cache[key] = value
    return value



# ==========================================================================
# TEXT PREPROCESSING
# ==========================================================================

def strip_special_tokens(text: str) -> str:
    """
    Remove ASR special/artifact tokens (e.g. "<is>") and normalize whitespace.

    Applied as the first preprocessing step for every WER / keyword / entity
    metric in this module so these tokens never influence the score.

    Examples:
        >>> strip_special_tokens("एक <is> दो")
        "एक दो"
        >>> strip_special_tokens("<is>")
        ""
    """
    if not text:
        return text
    text = _SPECIAL_TOKENS_RE.sub(' ', text)
    return ' '.join(text.split())


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



# ==========================================================================
# TOKENIZATION ALIGNMENT
# ==========================================================================

_MERGE_KEY_RE = re.compile(r'[^\w]+', re.UNICODE)


_MAX_MERGE_RUN = 8


def _join_letter_runs(text: str) -> str:
    """Glue runs of two or more single-character tokens into one word.

    "h b a 1 d" -> "hba1d". Used where the reference has no exactly matching
    token, so the caller must check that this actually helps before keeping it.
    """
    tokens = text.split()
    joined, index = [], 0
    while index < len(tokens):
        end = index
        while end < len(tokens) and len(tokens[end]) == 1 and tokens[end].isalnum():
            end += 1
        if end - index >= 2:
            joined.append(''.join(tokens[index:end]))
            index = end
        else:
            joined.append(tokens[index])
            index += 1
    return ' '.join(joined)


def _merge_key(token: str) -> str:
    """Comparison form for tokenization matching: letters and digits only.

    Lets "h. b. a. 1. c." reach the same key as "hba1c" — the dictated form
    often arrives with the letters punctuated.
    """
    return _MERGE_KEY_RE.sub('', token)


def align_tokenization(ref: str, hyp: str) -> str:
    """Give the hypothesis the reference's word boundaries.

    ASR disagrees about where spaces belong far more often than about what was
    said. An acronym dictated letter by letter comes back as "h b a 1 c" where
    the reference writes "hba1c"; compounds arrive glued together or split
    apart. Word-level WER charges full price for that: "hba1c" against
    "h b a 1 c" is one substitution plus four insertions against a one-word
    reference — a 500% error rate for a correct transcription.

    A hypothesis run is re-tokenized only when its characters concatenate to
    exactly what the reference has, so this can never turn a misheard word into
    a match; it can only move spaces. Only the hypothesis is rewritten — the
    reference keeps its own word count, so the WER denominator is untouched and
    the yardstick cannot drift.
    """
    ref_tokens = ref.split()
    hyp_tokens = hyp.split()
    if not ref_tokens or not hyp_tokens:
        return hyp

    # What a reference token looks like with its punctuation and spaces gone,
    # both for single tokens and for runs of adjacent ones.
    single: Dict[str, str] = {}
    runs: Dict[str, List[str]] = {}
    for start in range(len(ref_tokens)):
        joined = ''
        for end in range(start, min(start + _MAX_MERGE_RUN, len(ref_tokens))):
            joined += _merge_key(ref_tokens[end])
            if not joined:
                continue
            if end == start:
                single.setdefault(joined, ref_tokens[start])
            else:
                runs.setdefault(joined, ref_tokens[start:end + 1])

    result: List[str] = []
    index, total = 0, len(hyp_tokens)
    while index < total:
        key = _merge_key(hyp_tokens[index])

        # Several hypothesis tokens spell one reference token. Longest run wins,
        # so "h b a 1 c" collapses whole rather than stopping at "hb".
        merged = None
        joined = key
        for end in range(index + 1, min(index + _MAX_MERGE_RUN, total)):
            joined += _merge_key(hyp_tokens[end])
            if joined and joined in single:
                merged = (single[joined], end - index + 1)
        if merged:
            result.append(merged[0])
            index += merged[1]
            continue

        # One hypothesis token spells several reference tokens
        if key and key not in single and key in runs:
            result.extend(runs[key])
            index += 1
            continue

        result.append(hyp_tokens[index])
        index += 1

    rewritten = ' '.join(result)
    if rewritten == hyp:
        return hyp
    # Belt and braces: a re-tokenization must never make the score worse
    if calculate_wer(ref, rewritten) <= calculate_wer(ref, hyp):
        return rewritten
    return hyp



# ==========================================================================
# CANONICALIZATION
# ==========================================================================

def canonicalize_orthography(text: str, ortho_map: Dict[str, str]) -> str:
    """Fold Hindi spelling variants of the same word onto one form.

    Devanagari admits several spellings per word (तो/तोह, ये/यह, कि/की, जाएगा/
    जायेगा, nukta and anusvara variants). Annotators and models pick different
    ones, and each disagreement scores as a substitution even though the words
    are identical. Applied to both sides, so it can only remove spelling noise.
    """
    if not ortho_map:
        return text
    return ' '.join(ortho_map.get(word, word) for word in text.split())


def canonicalize_units(text: str, units_dict: Dict) -> str:
    """Rewrite every spelling of a unit to its canonical key.

    Without this the unit tables are nearly unreachable in Indic runs: by the
    time the per-segment unit step runs, transliteration has already turned
    "milligram" into Devanagari, and find_matching_units only ever compared
    whole tokens, so multi-word variations like "घन मिलीमीटर" could never match
    at all. Applied to reference and hypothesis alike, so it cannot bias either.
    """
    if not units_dict:
        return text

    replacements = []
    for key, variations in units_dict.items():
        for variation in list(variations) + [key]:
            if variation:
                replacements.append((variation, key))
    # Longest first so "cubic millimeter" wins over "millimeter"
    replacements.sort(key=lambda kv: (-len(kv[0].split()), -len(kv[0])))

    # Only fold a unit that follows a quantity. Several entries are ordinary
    # words rather than units — hi_units maps 'ok' to 'ठीक है', which is a
    # conversational filler, not a measurement — and rewriting those wherever
    # they appear corrupts far more text than the unit tables ever fix.
    for variation, key in replacements:
        text = re.sub(
            r'(?<!\w)(\d[\d.,:/-]*\s+)' + re.escape(variation) + r'(?!\w)',
            lambda mo: mo.group(1) + key,
            text,
            flags=re.IGNORECASE,
        )
    return text


_DIGITS_ONLY_RE = re.compile(r'^[\d\s:,./+-]+$')


def sanitize_glm_pairs(glm_arrays: Dict[str, str]) -> Tuple[Dict[str, str], Dict[str, int]]:
    """Drop GLM rules that rewrite quantities or fire on punctuation.

    glm_pairs.json is mined, not hand-written, and contains rules whose target is
    a bare number: '-' -> '010', '0' -> '200000', 'एंड' -> '188000'. Those inject
    invented digits into the scored text and can only ever be wrong. Numbers must
    be canonicalized by value, never by a lexical lookup table.

    Returns the filtered mapping and a count of what was dropped by reason.
    """
    if not glm_arrays:
        return glm_arrays, {}

    kept, dropped = {}, {"digit_target": 0, "digit_source": 0, "value_changing": 0, "punct_key": 0}
    for key, value in glm_arrays.items():
        key_s, value_s = str(key), str(value)

        if _DIGITS_ONLY_RE.match(value_s):
            dropped["digit_target"] += 1
            continue
        if _DIGITS_ONLY_RE.match(key_s):
            dropped["digit_source"] += 1
            continue
        # A mapping must not change the quantity it mentions ('₹24' -> '24000')
        if re.sub(r'\D', '', key_s) != re.sub(r'\D', '', value_s):
            dropped["value_changing"] += 1
            continue
        # Single punctuation marks are not words; they match the whole corpus
        if not key_s.strip() or all(not c.isalnum() for c in key_s):
            dropped["punct_key"] += 1
            continue
        kept[key] = value

    return kept, {k: v for k, v in dropped.items() if v}



# ==========================================================================
# RESOURCES
# ==========================================================================

class SemanticResources(NamedTuple):
    """The tables every semantic metric normalizes against."""
    equivalents: Dict      # units + symbols, for spacing and the per-segment hill climb
    units_dict: Dict       # units only, safe to canonicalize corpus-wide
    glm_arrays: Dict       # sanitized word mappings
    ortho_map: Dict        # spelling variants for this language


_RESOURCE_CACHE: Dict[str, SemanticResources] = {}


_RESOURCE_LOCK = threading.Lock()


def load_semantic_resources(language: str) -> SemanticResources:
    """Load every table the semantic normalization needs, once per language.

    Each metric used to inline its own copy of this loading code, and the copies
    drifted badly. The error-analysis and by-quality metrics read
    "glm_arrays_<lang>.json", a file that has never existed in this repo, so
    they silently ran with no GLM at all. The entity metrics read glm_pairs.json
    without sanitizing it, keeping the mined rules that inject invented digits.
    None of them loaded the orthography table or passed the units table through
    to optimize_wer. All of them describe themselves as applying "the same
    normalization as semantic_wer_metric"; going through here makes that true.
    """
    with _RESOURCE_LOCK:
        cached = _RESOURCE_CACHE.get(language)
        if cached is not None:
            return cached

    current_dir = os.path.dirname(os.path.abspath(__file__))

    def read(filename):
        path = os.path.join(current_dir, "lang_resources", filename)
        if not os.path.exists(path):
            return {}
        with open(path, 'r', encoding='utf-8') as handle:
            return json.load(handle)

    units_dict = read(f"{language}_units.json")
    symbols_dict = read(f"{language}_symbols.json")

    glm_arrays, dropped = sanitize_glm_pairs(read("glm_pairs.json"))
    if glm_arrays:
        note = f" ({sum(dropped.values())} unsafe rules dropped: {dropped})" if dropped else ""
        print(f"Loaded global GLM pairs with {len(glm_arrays)} entries{note}")

    ortho_map = {k: v for k, v in read(f"{language}_ortho.json").items()
                 if not k.startswith('_')}
    if ortho_map:
        print(f"Loaded {language} orthographic variants: {len(ortho_map)} entries")

    resources = SemanticResources(
        equivalents={**units_dict, **symbols_dict},
        units_dict=units_dict,
        glm_arrays=glm_arrays or None,
        ortho_map=ortho_map,
    )
    with _RESOURCE_LOCK:
        _RESOURCE_CACHE.setdefault(language, resources)
    return resources



# ==========================================================================
# EDIT COUNTING
# ==========================================================================

def count_edit_operations(ref: str, hyp: str) -> Tuple[int, int, int, int]:
    """
    Count the number of substitutions, deletions, and insertions between
    reference and hypothesis at the word level.
    
    Uses the Levenshtein distance algorithm to compute edit operations.
    
    Args:
        ref: Reference string
        hyp: Hypothesis string
        
    Returns:
        Tuple of (substitutions, deletions, insertions, total_ref_words)
    """
    ref_words = ref.split()
    hyp_words = hyp.split()
    
    n = len(ref_words)
    m = len(hyp_words)
    
    if n == 0:
        return 0, 0, m, 0
    if m == 0:
        return 0, n, 0, n
    
    # Create DP table for edit distance
    # dp[i][j] = (distance, substitutions, deletions, insertions)
    dp = [[(0, 0, 0, 0) for _ in range(m + 1)] for _ in range(n + 1)]
    
    # Initialize base cases
    for i in range(1, n + 1):
        dp[i][0] = (i, 0, i, 0)  # All deletions
    for j in range(1, m + 1):
        dp[0][j] = (j, 0, 0, j)  # All insertions
    
    # Fill DP table
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                # Match - no operation needed
                dp[i][j] = dp[i - 1][j - 1]
            else:
                # Consider substitution, deletion, insertion
                sub_dist, sub_s, sub_d, sub_i = dp[i - 1][j - 1]
                del_dist, del_s, del_d, del_i = dp[i - 1][j]
                ins_dist, ins_s, ins_d, ins_i = dp[i][j - 1]
                
                sub_cost = sub_dist + 1
                del_cost = del_dist + 1
                ins_cost = ins_dist + 1
                
                if sub_cost <= del_cost and sub_cost <= ins_cost:
                    dp[i][j] = (sub_cost, sub_s + 1, sub_d, sub_i)
                elif del_cost <= ins_cost:
                    dp[i][j] = (del_cost, del_s, del_d + 1, del_i)
                else:
                    dp[i][j] = (ins_cost, ins_s, ins_d, ins_i + 1)
    
    _, substitutions, deletions, insertions = dp[n][m]
    return substitutions, deletions, insertions, n


def count_edit_operations_difflib(ref: str, hyp: str) -> Tuple[int, int, int, int]:
    """
    Count edit operations, using the same alignment the WER metric uses.

    Kept under its historical name because callers import it, but it no longer
    uses difflib. difflib.SequenceMatcher finds the longest matching blocks; it
    does not minimise edit distance, so on 28% of real pairs it disagreed with
    jiwer — both inflating the total and reporting deletions plus insertions
    where the optimal alignment has substitutions. Since semantic_wer_metric
    scores with jiwer, the error rates could never sum to the WER: measured
    across 300 Marathi utterances the gap was +4.97%.

    Counting from jiwer's alignment makes substitution_rate + deletion_rate +
    insertion_rate equal the reported WER by construction.

    Args:
        ref: Reference string
        hyp: Hypothesis string

    Returns:
        Tuple of (substitutions, deletions, insertions, total_ref_words)
    """
    import jiwer

    ref_words = ref.split()
    hyp_words = hyp.split()

    n = len(ref_words)
    if n == 0:
        return 0, 0, len(hyp_words), 0
    if len(hyp_words) == 0:
        return 0, n, 0, n

    m = jiwer.process_words(ref, hyp)
    # S + D + H is jiwer's reference-length denominator; use it so the rates
    # divide by exactly what the WER divides by.
    return m.substitutions, m.deletions, m.insertions, (
        m.substitutions + m.deletions + m.hits)



# ==========================================================================
# ALIGNMENT
# ==========================================================================

def compute_edit_distance(s1: str, s2: str) -> int:
    """
    Compute Levenshtein edit distance between two strings.
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        Edit distance (number of edits needed to transform s1 to s2)
    """
    if len(s1) < len(s2):
        return compute_edit_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Cost of insertions, deletions, or substitutions
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def align_mismatched_segment(
    ref_words: List[str], 
    hyp_words: List[str], 
    ref_offset: int, 
    hyp_offset: int
) -> Dict[int, List[int]]:
    """
    Align words within a mismatched (replace) segment using similarity-based matching.
    
    This function prevents duplicate alignments by ensuring each hypothesis word
    is assigned to at most one reference word based on edit distance similarity.
    
    Args:
        ref_words: Reference words in this segment
        hyp_words: Hypothesis words in this segment
        ref_offset: Starting index of ref segment in full reference
        hyp_offset: Starting index of hyp segment in full hypothesis
        
    Returns:
        Dict mapping ref_word_idx -> list of hyp_word_indices
    """
    ref_len = len(ref_words)
    hyp_len = len(hyp_words)
    alignment = {}
    
    if hyp_len == 0:
        # All deletions
        for i in range(ref_len):
            alignment[ref_offset + i] = []
        return alignment
    
    if ref_len == 0:
        # All insertions - no ref indices to map
        return alignment
    
    # Compute similarity matrix (lower edit distance = more similar)
    similarity_matrix = []
    for i, ref_word in enumerate(ref_words):
        row = []
        for j, hyp_word in enumerate(hyp_words):
            edit_dist = compute_edit_distance(ref_word.lower(), hyp_word.lower())
            # Normalize by max length to get similarity score (0 = identical, 1 = completely different)
            max_len = max(len(ref_word), len(hyp_word))
            normalized_dist = edit_dist / max_len if max_len > 0 else 0
            row.append(normalized_dist)
        similarity_matrix.append(row)
    
    # Track which hypothesis words have been assigned
    used_hyp_indices = set()
    
    # Greedy assignment: for each ref word, find best available hyp word
    # Process in order of best overall match first
    ref_best_matches = []
    for i in range(ref_len):
        if similarity_matrix[i]:
            best_score = min(similarity_matrix[i])
            ref_best_matches.append((best_score, i))
        else:
            ref_best_matches.append((float('inf'), i))
    
    # Sort by best match quality (lower score = better match)
    ref_best_matches.sort()
    
    for _, ref_idx in ref_best_matches:
        ref_global_idx = ref_offset + ref_idx
        
        # Find best available hypothesis word for this reference word
        best_hyp_idx = None
        best_score = float('inf')
        
        for hyp_idx in range(hyp_len):
            if hyp_idx not in used_hyp_indices:
                score = similarity_matrix[ref_idx][hyp_idx]
                if score < best_score:
                    best_score = score
                    best_hyp_idx = hyp_idx
        
        if best_hyp_idx is not None:
            # Assign this hypothesis word to this reference word
            alignment[ref_global_idx] = [hyp_offset + best_hyp_idx]
            used_hyp_indices.add(best_hyp_idx)
        else:
            # No hypothesis words left - mark as deletion
            alignment[ref_global_idx] = []
    
    return alignment


def get_word_alignment(ref_words: List[str], hyp_words: List[str]) -> Dict[int, List[int]]:
    """
    Get word-level alignment mapping from reference word indices to hypothesis word indices.
    
    Uses difflib SequenceMatcher to find alignment, with improved handling of replace
    operations to prevent duplicate alignments.
    
    Args:
        ref_words: List of reference words
        hyp_words: List of hypothesis words
        
    Returns:
        Dict mapping ref_word_idx -> list of hyp_word_indices it aligns to
        (empty list means deletion, single index for most cases, multiple indices rare)
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
            # Use similarity-based alignment for mismatched segments
            ref_segment = ref_words[i1:i2]
            hyp_segment = hyp_words[j1:j2]
            
            # Special case: equal length - use one-to-one mapping
            if len(ref_segment) == len(hyp_segment):
                for offset, ref_idx in enumerate(range(i1, i2)):
                    alignment[ref_idx] = [j1 + offset]
            else:
                # Use similarity-based matching to prevent duplicates
                segment_alignment = align_mismatched_segment(
                    ref_segment, hyp_segment, i1, j1
                )
                alignment.update(segment_alignment)
        
        elif tag == 'delete':
            # Reference words have no corresponding hypothesis words
            for ref_idx in range(i1, i2):
                alignment[ref_idx] = []
        
        elif tag == 'insert':
            # Hypothesis words have no corresponding reference words
            # No ref indices to map
            pass
    
    return alignment


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

