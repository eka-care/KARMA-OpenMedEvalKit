#!/usr/bin/env python3
"""
Complete EnglishCERAligner with all required methods including those from BaseCERAligner.
Clean version with proper indentation and enhanced functionality.
"""

import re
import sys
from typing import List, Tuple, Dict
from itertools import product
from dataclasses import dataclass
from enum import Enum

sys.setrecursionlimit(10000)

class AlignmentType(Enum):
    MATCH = "match"
    SUBSTITUTION = "substitution"
    INSERTION = "insertion"
    DELETION = "deletion"

@dataclass
class WordAlignment:
    ref_words: List[str]
    hyp_words: List[str]
    alignment_type: AlignmentType
    character_error_rate: float
    ref_positions: List[Tuple[int, int]]
    hyp_positions: List[Tuple[int, int]]

class EnglishCERAligner:
    """English-specific CER-based word aligner - complete standalone version."""
    
    def __init__(self, cer_threshold: float = 0.4):
        """Initialize the aligner with language-specific mappings."""
        self.cer_threshold = cer_threshold
        self._initialize_language_specific_mappings()
    
    def _initialize_language_specific_mappings(self):
        """Initialize English-specific mappings."""
        
        # English number words
        self.extended_number_words = {
            'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
            'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
            'ten': '10', 'eleven': '11', 'twelve': '12', 'thirteen': '13',
            'fourteen': '14', 'fifteen': '15', 'sixteen': '16', 'seventeen': '17',
            'eighteen': '18', 'nineteen': '19', 'twenty': '20', 'thirty': '30',
            'forty': '40', 'fifty': '50', 'sixty': '60', 'seventy': '70',
            'eighty': '80', 'ninety': '90', 'hundred': '100', 'thousand': '1000',
            'million': '1000000', 'billion': '1000000000'
        }
        
        # English symbol words
        self.symbol_words = {
            '.': ['dot', 'point', 'period', 'full stop'],
            '/': ['slash', 'over', 'by', 'or', 'per', 'divided by'],
            '-': ['dash', 'hyphen', 'minus', 'negative'],
            '+': ['plus', 'add', 'positive', 'and'],
            '%': ['percent', 'percentage', 'per cent'],
            '*': ['star', 'asterisk', 'times', 'multiply'],
            '=': ['equals', 'equal', 'is', 'equal to'],
            '&': ['and', 'ampersand'],
            '@': ['at', 'at sign'],
            '#': ['hash', 'pound', 'number', 'hashtag'],
        }
        
        # English unit abbreviations
        self.unit_words = {
            'mm': ['millimeter', 'millimeters', 'millimetre', 'millimetres'],
            'cm': ['centimeter', 'centimeters', 'centimetre', 'centimetres'],
            'm': ['meter', 'meters', 'metre', 'metres'],
            'km': ['kilometer', 'kilometers', 'kilometre', 'kilometres'],
            'g': ['gram', 'grams', 'gramme', 'grammes'],
            'kg': ['kilogram', 'kilograms', 'kilogramme', 'kilogrammes'],
            'mg': ['milligram', 'milligrams', 'mg', 'MG'],
            'ml': ['milliliter', 'milliliters', 'millilitre', 'millilitres'],
            'l': ['liter', 'liters', 'litre', 'litres'],
            'sec': ['second', 'seconds'],
            'min': ['minute', 'minutes'],
            'hr': ['hour', 'hours'],
        }
        
        # Build reverse mappings
        self.word_to_number = {}
        self.word_to_symbol = {}
        self.word_to_unit = {}
        
        for word, digit in self.extended_number_words.items():
            self.word_to_number[word.lower()] = digit
            
        for symbol, words in self.symbol_words.items():
            for word in words:
                self.word_to_symbol[word.lower()] = symbol
                
        for unit, words in self.unit_words.items():
            for word in words:
                self.word_to_unit[word.lower()] = unit

    # CORE METHODS FROM BASE ALIGNER
    
    def tokenize_with_positions(self, text: str) -> List[Tuple[str, int, int]]:
        """Tokenize text and return (word, start_pos, end_pos)."""
        tokens = []
        for match in re.finditer(r'\S+', text):
            tokens.append((match.group(), match.start(), match.end()))
        return tokens
    
    def _edit_distance_cer(self, ref_text: str, hyp_text: str) -> float:
        """Calculate edit distance based CER."""
        m, n = len(ref_text), len(hyp_text)
        if m == 0:
            return 1.0 if n > 0 else 0.0
            
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_text[i-1] == hyp_text[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i-1][j],      # deletion
                        dp[i][j-1],      # insertion
                        dp[i-1][j-1]     # substitution
                    )
        
        edit_distance = dp[m][n]
        cer = edit_distance / m
        return cer
    
    def calculate_character_error_rate(self, ref_text: str, hyp_text: str) -> float:
        """Calculate CER using direct and semantic comparison with improved substring handling."""
        ref_direct = ref_text.lower().replace(' ', '')
        hyp_direct = hyp_text.lower().replace(' ', '')
        cer_direct = self._edit_distance_cer(ref_direct, hyp_direct)
        
        ref_semantic = self.normalize_text_semantically(ref_text)
        hyp_semantic = self.normalize_text_semantically(hyp_text)
        cer_semantic = self._edit_distance_cer(ref_semantic, hyp_semantic)
        
        # Add substring-aware CER calculation
        cer_substring = self._calculate_substring_aware_cer(ref_text.lower(), hyp_text.lower())
        
        # Return the best (lowest) CER among all methods
        final_cer = min(cer_direct, cer_semantic, cer_substring)
        return final_cer
    
    def _calculate_substring_aware_cer(self, ref_text: str, hyp_text: str) -> float:
        """Calculate CER that's aware of substring relationships."""
        ref_clean = ref_text.replace(' ', '').replace('.', '').replace('-', '')
        hyp_clean = hyp_text.replace(' ', '').replace('.', '').replace('-', '')
        
        # If one is a substring of the other, calculate based on that
        if ref_clean in hyp_clean:
            # ref is substring of hyp
            extra_chars = len(hyp_clean) - len(ref_clean)
            return extra_chars / max(len(ref_clean), 1)
        elif hyp_clean in ref_clean:
            # hyp is substring of ref  
            missing_chars = len(ref_clean) - len(hyp_clean)
            return missing_chars / max(len(ref_clean), 1)
        
        # Check for acronym-like relationships (e.g., MDCTA contains TA)
        ref_upper = ref_clean.upper()
        hyp_upper = hyp_clean.upper()
        
        if hyp_upper in ref_upper:
            missing_ratio = (len(ref_upper) - len(hyp_upper)) / max(len(ref_upper), 1)
            return min(missing_ratio, 0.8)  # Cap at 0.8 to show it's still a decent match
        elif ref_upper in hyp_upper:
            extra_ratio = (len(hyp_upper) - len(ref_upper)) / max(len(ref_upper), 1) 
            return min(extra_ratio, 0.8)
        
        # Check for common prefix/suffix
        common_prefix = 0
        for i in range(min(len(ref_upper), len(hyp_upper))):
            if ref_upper[i] == hyp_upper[i]:
                common_prefix += 1
            else:
                break
        
        if common_prefix >= 3:  # At least 3 characters in common at start
            max_len = max(len(ref_upper), len(hyp_upper))
            similarity = common_prefix / max_len
            return 1.0 - similarity
        
        # Fall back to regular edit distance
        return self._edit_distance_cer(ref_clean, hyp_clean)

    def align_words_dp(self, reference: str, hypothesis: str) -> List[WordAlignment]:
        """Use dynamic programming for optimal alignment with correct position tracking."""
        ref_tokens = self.tokenize_with_positions(reference)
        hyp_tokens = self.tokenize_with_positions(hypothesis)
        
        m, n = len(ref_tokens), len(hyp_tokens)
        
        # DP table: dp[i][j] = (score, alignments)
        dp = {}
        
        def solve(ref_start: int, hyp_start: int) -> Tuple[float, List[WordAlignment]]:
            if (ref_start, hyp_start) in dp:
                return dp[(ref_start, hyp_start)]
            
            if ref_start >= m and hyp_start >= n:
                result = (0.0, [])
            elif ref_start >= m:
                # Only insertions remaining
                prev_score, prev_alignments = solve(ref_start, hyp_start + 1)
                alignment = WordAlignment(
                    ref_words=[],
                    hyp_words=[hyp_tokens[hyp_start][0]],
                    alignment_type=AlignmentType.INSERTION,
                    character_error_rate=1.0,
                    ref_positions=[],
                    hyp_positions=[(hyp_tokens[hyp_start][1], hyp_tokens[hyp_start][2])]
                )
                result = (prev_score - 1.0, [alignment] + prev_alignments)
            elif hyp_start >= n:
                # Only deletions remaining
                prev_score, prev_alignments = solve(ref_start + 1, hyp_start)
                alignment = WordAlignment(
                    ref_words=[ref_tokens[ref_start][0]],
                    hyp_words=[],
                    alignment_type=AlignmentType.DELETION,
                    character_error_rate=1.0,
                    ref_positions=[(ref_tokens[ref_start][1], ref_tokens[ref_start][2])],
                    hyp_positions=[]
                )
                result = (prev_score - 1.0, [alignment] + prev_alignments)
            else:
                candidates = []
                
                # Try different alignment combinations
                max_ref_span = min(3, m - ref_start)
                max_hyp_span = min(8, n - hyp_start)
                
                # Check if current ref token has semantic expansion
                if ref_start < m:
                    ref_token = ref_tokens[ref_start][0]
                    expanded_ref = self.expand_ref_token_semantically(ref_token)
                    
                    if expanded_ref != ref_token:
                        expanded_tokens = expanded_ref.split()
                        target_span = len(expanded_tokens)
                        
                        if hyp_start + target_span <= n:
                            hyp_seq = [hyp_tokens[hyp_start + k][0] for k in range(target_span)]
                            
                            if [w.lower() for w in expanded_tokens] == [w.lower() for w in hyp_seq]:
                                alignment = WordAlignment(
                                    ref_words=[ref_token],
                                    hyp_words=hyp_seq,
                                    alignment_type=AlignmentType.MATCH,
                                    character_error_rate=0.0,
                                    ref_positions=[(ref_tokens[ref_start][1], ref_tokens[ref_start][2])],
                                    hyp_positions=[(hyp_tokens[hyp_start + k][1], hyp_tokens[hyp_start + k][2]) for k in range(target_span)]
                                )
                                
                                prev_score, prev_alignments = solve(ref_start + 1, hyp_start + target_span)
                                total_score = prev_score + 20.0
                                
                                result = (total_score, [alignment] + prev_alignments)
                                dp[(ref_start, hyp_start)] = result
                                return result
                    
                    # Try alternative expansions using our enhanced method
                    all_expansions = self.get_all_possible_expansions_with_units(ref_token)
                    for expanded_ref in all_expansions:
                        if expanded_ref != ref_token:
                            expanded_tokens = expanded_ref.split()
                            target_span = len(expanded_tokens)
                            
                            if hyp_start + target_span <= n:
                                hyp_seq = [hyp_tokens[hyp_start + k][0] for k in range(target_span)]
                                
                                if [w.lower() for w in expanded_tokens] == [w.lower() for w in hyp_seq]:
                                    alignment = WordAlignment(
                                        ref_words=[ref_token],
                                        hyp_words=hyp_seq,
                                        alignment_type=AlignmentType.MATCH,
                                        character_error_rate=0.0,
                                        ref_positions=[(ref_tokens[ref_start][1], ref_tokens[ref_start][2])],
                                        hyp_positions=[(hyp_tokens[hyp_start + k][1], hyp_tokens[hyp_start + k][2]) for k in range(target_span)]
                                    )
                                    
                                    prev_score, prev_alignments = solve(ref_start + 1, hyp_start + target_span)
                                    total_score = prev_score + 20.0
                                    
                                    result = (total_score, [alignment] + prev_alignments)
                                    dp[(ref_start, hyp_start)] = result
                                    return result
                for ref_span in range(1, max_ref_span + 1):
                    for hyp_span in range(1, max_hyp_span + 1):
                        ref_seq = [ref_tokens[ref_start + k][0] for k in range(ref_span)]
                        hyp_seq = [hyp_tokens[hyp_start + k][0] for k in range(hyp_span)]
                        
                        # Calculate similarity with heavy penalties for poor multi-word alignments
                        if ref_span == 1:
                            ref_token = ref_seq[0]
                            expanded_ref = self.expand_ref_token_semantically(ref_token)
                            
                            if expanded_ref != ref_token:
                                expanded_tokens = expanded_ref.split()
                                
                                if len(expanded_tokens) == hyp_span and \
                                   [w.lower() for w in expanded_tokens] == [w.lower() for w in hyp_seq]:
                                    similarity = 1.0
                                    cer = 0.0
                                    score_bonus = 15.0
                                else:
                                    ref_text = ref_token
                                    hyp_text = ' '.join(hyp_seq)
                                    cer = self.calculate_character_error_rate(ref_text, hyp_text)
                                    similarity = 1.0 - cer
                                    
                                    if similarity >= (1.0 - self.cer_threshold):
                                        score_bonus = similarity + 0.5
                                    else:
                                        score_bonus = similarity - 0.5
                            else:
                                ref_text = ref_token
                                hyp_text = ' '.join(hyp_seq)
                                cer = self.calculate_character_error_rate(ref_text, hyp_text)
                                similarity = 1.0 - cer
                                
                                # Heavily favor perfect/near-perfect matches
                                if similarity >= 0.99:
                                    score_bonus = similarity + 5.0  # Big bonus for perfect matches
                                elif similarity >= 0.95:
                                    score_bonus = similarity + 3.0  # Good bonus for near-perfect
                                elif hyp_span == 1:
                                    score_bonus = similarity + 1.0
                                else:
                                    # HEAVILY penalize single ref token to many hyp tokens with poor similarity
                                    if similarity < 0.3:  # Very poor similarity
                                        span_penalty = (hyp_span - 1) * 2.0  # Heavy penalty
                                        score_bonus = similarity - 2.0 - span_penalty
                                    else:
                                        span_penalty = (hyp_span - 1) * 0.5
                                        score_bonus = similarity - 0.3 - span_penalty
                        else:
                            ref_text = ' '.join(ref_seq)
                            hyp_text = ' '.join(hyp_seq)
                            cer = self.calculate_character_error_rate(ref_text, hyp_text)
                            similarity = 1.0 - cer
                            
                            # Penalize multi-ref-word spans unless they're very good matches
                            if similarity >= 0.95:
                                score_bonus = similarity + 1.0
                            else:
                                span_penalty = (ref_span - 1) * 0.3 + (hyp_span - 1) * 0.3
                                score_bonus = similarity - 0.8 - span_penalty
                        
                        # Determine alignment type
                        if similarity >= 0.95:
                            alignment_type = AlignmentType.MATCH
                        elif similarity >= (1.0 - self.cer_threshold):
                            alignment_type = AlignmentType.SUBSTITUTION
                        else:
                            alignment_type = AlignmentType.SUBSTITUTION
                        
                        prev_score, prev_alignments = solve(ref_start + ref_span, hyp_start + hyp_span)
                        
                        alignment = WordAlignment(
                            ref_words=ref_seq,
                            hyp_words=hyp_seq,
                            alignment_type=alignment_type,
                            character_error_rate=cer,
                            ref_positions=[(ref_tokens[ref_start + k][1], ref_tokens[ref_start + k][2]) for k in range(ref_span)],
                            hyp_positions=[(hyp_tokens[hyp_start + k][1], hyp_tokens[hyp_start + k][2]) for k in range(hyp_span)]
                        )
                        
                        total_score = prev_score + score_bonus
                        candidates.append((total_score, [alignment] + prev_alignments))
                
                result = max(candidates, key=lambda x: x[0])
            
            dp[(ref_start, hyp_start)] = result
            return result
        
        _, alignments = solve(0, 0)
        return alignments
    
    def calculate_error_rates(self, alignments: List[WordAlignment]) -> Dict[str, float]:
        """Calculate error rates."""
        total_ref_words = sum(len(a.ref_words) for a in alignments)
        
        correct = sum(1 for a in alignments if a.alignment_type == AlignmentType.MATCH)
        substitutions = sum(1 for a in alignments if a.alignment_type == AlignmentType.SUBSTITUTION)
        deletions = sum(1 for a in alignments if a.alignment_type == AlignmentType.DELETION)
        insertions = sum(1 for a in alignments if a.alignment_type == AlignmentType.INSERTION)
        
        wer = (substitutions + deletions + insertions) / max(total_ref_words, 1)

        cer_scores = []
        total_ref_chars = 0
        total_edit_distance = 0
        
        for alignment in alignments:
            if alignment.ref_words and alignment.hyp_words:
                cer_scores.append(alignment.character_error_rate)
                ref_text = ' '.join(alignment.ref_words)
                ref_chars = len(ref_text.replace(' ', ''))
                total_ref_chars += ref_chars
                total_edit_distance += alignment.character_error_rate * ref_chars
            elif alignment.ref_words:  # Deletion case
                ref_text = ' '.join(alignment.ref_words)
                ref_chars = len(ref_text.replace(' ', ''))
                total_ref_chars += ref_chars
                total_edit_distance += ref_chars
        
        avg_word_cer = sum(cer_scores) / len(cer_scores) if cer_scores else 0.0
        weighted_word_cer = total_edit_distance / total_ref_chars if total_ref_chars > 0 else 0.0
         
        return {
            'total_ref_words': total_ref_words,
            'word_correct': correct,
            'word_substitutions': substitutions,
            'word_deletions': deletions,
            'word_insertions': insertions,
            'wer': wer,
            'alignments_with_cer': len(cer_scores),
            'avg_word_cer': avg_word_cer,
            'weighted_word_cer': weighted_word_cer
        }
    
    def print_alignment_visual(self, alignments: List[WordAlignment]):
        """Print visual alignment."""
        print("Visual Alignment:")
        print("-" * 100)
        
        ref_parts = []
        hyp_parts = []
        symbols = []
        
        for alignment in alignments:
            ref_text = ' '.join(alignment.ref_words) if alignment.ref_words else "∅"
            hyp_text = ' '.join(alignment.hyp_words) if alignment.hyp_words else "∅"
            
            if alignment.alignment_type == AlignmentType.MATCH:
                symbol = "="
            elif alignment.alignment_type == AlignmentType.SUBSTITUTION:
                symbol = "~"
            elif alignment.alignment_type == AlignmentType.DELETION:
                symbol = "D"
            elif alignment.alignment_type == AlignmentType.INSERTION:
                symbol = "I"
            
            max_len = max(len(ref_text), len(hyp_text), 3)
            ref_parts.append(ref_text.center(max_len))
            hyp_parts.append(hyp_text.center(max_len))
            symbols.append(symbol.center(max_len))
        
        print("REF: " + " | ".join(ref_parts))
        print("     " + " | ".join(symbols))
        print("HYP: " + " | ".join(hyp_parts))
        print()

    # TEXT PROCESSING METHODS
    
    def normalize_hyphenated_words(self, text: str) -> str:
        """Remove hyphens from hyphenated words."""
        return re.sub(r'\b([a-zA-Z]+)-([a-zA-Z]+)\b', r'\1\2', text)

    def normalize_text_semantically(self, text: str) -> str:
        """Normalize text for semantic comparison - enhanced to handle punctuation better."""
        text = text.lower().strip()
        text = self.normalize_hyphenated_words(text)
        
        # Remove common punctuation that shouldn't affect semantic meaning
        text = re.sub(r'[.\-_]', ' ', text)  # Replace punctuation with spaces
        text = re.sub(r'\s+', ' ', text)      # Normalize multiple spaces
        
        # Keep alphabetic chars only for word sequences
        words = text.split()
        normalized_parts = []
        
        for word in words:
            clean_word = ''.join(c for c in word if c.isalpha())
            if clean_word:
                normalized_parts.append(clean_word)
        
        return ''.join(normalized_parts)

    def expand_ref_token_semantically(self, token: str) -> str:
        """Expand a reference token to its semantic word form."""
        token = self.normalize_hyphenated_words(token)
        
        # Handle tokens with digits
        if re.search(r'\d', token):
            result_parts = []
            current_number = ""
            current_alpha = ""
            
            for char in token:
                if char.isdigit():
                    if current_alpha:
                        if current_alpha.lower() in self.unit_words:
                            result_parts.append(self.unit_words[current_alpha.lower()][0])
                        else:
                            result_parts.append(current_alpha.lower())
                        current_alpha = ""
                    current_number += char
                elif char.isalpha():
                    if current_number:
                        number_words = self.convert_number_to_words(current_number)
                        result_parts.extend(number_words)
                        current_number = ""
                    current_alpha += char
                else:
                    if current_number:
                        number_words = self.convert_number_to_words(current_number)
                        result_parts.extend(number_words)
                        current_number = ""
                    if current_alpha:
                        if current_alpha.lower() in self.unit_words:
                            result_parts.append(self.unit_words[current_alpha.lower()][0])
                        else:
                            result_parts.append(current_alpha.lower())
                        current_alpha = ""
                    
                    if char in self.symbol_words:
                        result_parts.append(self.symbol_words[char][0])
            
            # Process remaining
            if current_number:
                number_words = self.convert_number_to_words(current_number)
                result_parts.extend(number_words)
            if current_alpha:
                if current_alpha.lower() in self.unit_words:
                    result_parts.append(self.unit_words[current_alpha.lower()][0])
                else:
                    result_parts.append(current_alpha.lower())
            
            if result_parts:
                return ' '.join(result_parts)
        
        # Check if token is a unit abbreviation
        elif token.lower() in self.unit_words:
            return self.unit_words[token.lower()][0]
        
        # Check if token contains symbols
        elif re.search(r'[./+%*=&@#-]', token):
            expanded_parts = []
            for char in token:
                if char in self.symbol_words:
                    expanded_parts.append(self.symbol_words[char][0])
                elif char.isalpha():
                    expanded_parts.append(char.lower())
            
            if expanded_parts:
                return ' '.join(expanded_parts)
        
        return token

    def convert_number_to_words(self, number_str: str) -> List[str]:
        """Convert a number string to English words."""
        if not number_str:
            return ["zero"]
        
        try:
            num = int(number_str)
        except ValueError:
            # Fall back to digit-by-digit
            digit_words = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
            words = []
            for digit in number_str:
                if digit.isdigit():
                    words.append(digit_words[int(digit)])
            return words
        
        if num == 0:
            return ["zero"]
        
        if num < 0:
            return ["negative"] + self.convert_number_to_words(str(-num))
        
        # Handle large numbers with fallback to digit-by-digit
        if num > 999999999999:  # Above 999 billion
            words = []
            digit_words = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
            for digit in str(num):
                if digit.isdigit():
                    words.append(digit_words[int(digit)])
            return words
        
        words = []
        remaining = num
        
        # Billions
        if remaining >= 1000000000:
            billion_part = remaining // 1000000000
            words.extend(self._convert_below_1000(billion_part))
            words.append("billion")
            remaining %= 1000000000
        
        # Millions
        if remaining >= 1000000:
            million_part = remaining // 1000000
            words.extend(self._convert_below_1000(million_part))
            words.append("million")
            remaining %= 1000000
        
        # Thousands
        if remaining >= 1000:
            thousand_part = remaining // 1000
            words.extend(self._convert_below_1000(thousand_part))
            words.append("thousand")
            remaining %= 1000
        
        # Hundreds, tens, and ones
        if remaining > 0:
            words.extend(self._convert_below_1000(remaining))
        
        return words

    def convert_number_to_words_with_and(self, number_str: str) -> List[str]:
        """Convert a number string to English words with British 'and' style."""
        if not number_str:
            return ["zero"]
        
        try:
            num = int(number_str)
        except ValueError:
            # Fall back to digit-by-digit
            digit_words = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
            words = []
            for digit in number_str:
                if digit.isdigit():
                    words.append(digit_words[int(digit)])
            return words
        
        if num == 0:
            return ["zero"]
        
        if num < 0:
            return ["negative"] + self.convert_number_to_words_with_and(str(-num))
        
        # Handle large numbers with fallback to digit-by-digit
        if num > 999999999999:  # Above 999 billion
            words = []
            digit_words = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
            for digit in str(num):
                if digit.isdigit():
                    words.append(digit_words[int(digit)])
            return words
        
        words = []
        remaining = num
        
        # Billions
        if remaining >= 1000000000:
            billion_part = remaining // 1000000000
            words.extend(self._convert_below_1000_with_and(billion_part))
            words.append("billion")
            remaining %= 1000000000
        
        # Millions
        if remaining >= 1000000:
            million_part = remaining // 1000000
            words.extend(self._convert_below_1000_with_and(million_part))
            words.append("million")
            remaining %= 1000000
        
        # Thousands
        if remaining >= 1000:
            thousand_part = remaining // 1000
            words.extend(self._convert_below_1000_with_and(thousand_part))
            words.append("thousand")
            remaining %= 1000
        
        # Hundreds, tens, and ones
        if remaining > 0:
            words.extend(self._convert_below_1000_with_and(remaining))
        
        return words

    def _convert_below_1000_with_and(self, num: int) -> List[str]:
        """Convert numbers below 1000 to English words with 'and'."""
        if num == 0:
            return []
        
        words = []
        
        # Hundreds
        if num >= 100:
            hundreds_digit = num // 100
            ones_words = ['', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
            words.append(ones_words[hundreds_digit])
            words.append("hundred")
            num %= 100
            # Add "and" for British style numbers like "seven hundred and fifty"
            if num > 0:
                words.append("and")
        
        # Tens and ones
        if num >= 20:
            tens_digit = num // 10
            tens_words = {2: "twenty", 3: "thirty", 4: "forty", 5: "fifty",
                         6: "sixty", 7: "seventy", 8: "eighty", 9: "ninety"}
            words.append(tens_words[tens_digit])
            num %= 10
        elif num >= 10:
            # Teens
            teen_words = {10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen",
                         14: "fourteen", 15: "fifteen", 16: "sixteen", 17: "seventeen",
                         18: "eighteen", 19: "nineteen"}
            words.append(teen_words[num])
            num = 0
        
        # Ones
        if num > 0:
            ones_words = ['', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
            words.append(ones_words[num])
        
        return words

    def _convert_below_1000(self, num: int) -> List[str]:
        """Convert numbers below 1000 to English words."""
        if num == 0:
            return []
        
        words = []
        
        # Hundreds
        if num >= 100:
            hundreds_digit = num // 100
            ones_words = ['', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
            words.append(ones_words[hundreds_digit])
            words.append("hundred")
            num %= 100
            # Don't add "and" - use American style to match hypothesis
            # if num > 0:
            #     words.append("and")
        
        # Tens and ones
        if num >= 20:
            tens_digit = num // 10
            tens_words = {2: "twenty", 3: "thirty", 4: "forty", 5: "fifty",
                         6: "sixty", 7: "seventy", 8: "eighty", 9: "ninety"}
            words.append(tens_words[tens_digit])
            num %= 10
        elif num >= 10:
            # Teens
            teen_words = {10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen",
                         14: "fourteen", 15: "fifteen", 16: "sixteen", 17: "seventeen",
                         18: "eighteen", 19: "nineteen"}
            words.append(teen_words[num])
            num = 0
        
        # Ones
        if num > 0:
            ones_words = ['', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
            words.append(ones_words[num])
        
        return words

    # ENHANCED EXPANSION METHODS
    
    def get_all_possible_expansions(self, token: str) -> List[str]:
        """Get all possible semantic expansions for a token - generic approach."""
        expansions = set()  # Use set to avoid duplicates
        
        # Handle tokens with digits and/or symbols
        if re.search(r'[\d\./\-+%*=&@#$€£¥°™©®]', token):
            # Find all symbols in the token and their positions
            symbols_info = []
            for i, char in enumerate(token):
                if char in self.symbol_words:
                    symbols_info.append((i, char, self.symbol_words[char]))
            
            if symbols_info:
                # Generate all combinations of symbol word choices
                symbol_combinations = self._generate_symbol_combinations(symbols_info)
                
                for symbol_combo in symbol_combinations:
                    expansion = self._build_expansion_with_symbol_combo_generic(token, symbol_combo)
                    if expansion and expansion != token:
                        expansions.add(expansion)
            else:
                # No symbols, but has digits - try regular expansion
                regular_expansion = self.expand_ref_token_semantically(token)
                if regular_expansion and regular_expansion != token:
                    expansions.add(regular_expansion)
                
                # Handle number+unit combinations (like 750mg)
                if re.search(r'\d', token) and re.search(r'[a-zA-Z]', token):
                    potential_expansions = set()
                    
                    # Pattern 1: Number followed by letters (750mg)
                    match = re.match(r'(\d+)([a-zA-Z]+)$', token)
                    if match:
                        number_part, unit_part = match.groups()
                        number_words = self.convert_number_to_words(number_part)
                        
                        if unit_part.lower() in self.unit_words:
                            for unit_variant in self.unit_words[unit_part.lower()]:
                                expansion = ' '.join(number_words + [unit_variant])
                                potential_expansions.add(expansion)
                    
                    # Pattern 2: Letters followed by number (mg750, pH7)
                    match = re.match(r'^([a-zA-Z]+)(\d+)$', token)
                    if match:
                        unit_part, number_part = match.groups()
                        number_words = self.convert_number_to_words(number_part)
                        
                        if unit_part.lower() in self.unit_words:
                            for unit_variant in self.unit_words[unit_part.lower()]:
                                expansion = ' '.join([unit_variant] + number_words)
                                potential_expansions.add(expansion)
                        else:
                            # For non-unit cases like pH7, CO2
                            expansion = ' '.join([unit_part.lower()] + number_words)
                            potential_expansions.add(expansion)
                    
                    expansions.update(potential_expansions)
        
        # Handle tokens that are unit abbreviations (without digits)
        elif token.lower() in self.unit_words:
            # Add all possible unit expansions
            for unit_word in self.unit_words[token.lower()]:
                expansions.add(unit_word)
        
        # Handle tokens with symbols only (no digits)
        elif re.search(r'[./+%*=&@#$€£¥°™©®-]', token):
            symbol_positions = []
            for i, char in enumerate(token):
                if char in self.symbol_words:
                    symbol_positions.append((i, char, self.symbol_words[char]))
            
            if symbol_positions:
                symbol_combinations = self._generate_symbol_combinations(symbol_positions)
                for symbol_combo in symbol_combinations:
                    expansion = self._build_expansion_with_symbol_combo_generic(token, symbol_combo)
                    if expansion and expansion != token:
                        expansions.add(expansion)
        
        # If no expansions found, try the regular method
        if not expansions:
            regular_expansion = self.expand_ref_token_semantically(token)
            if regular_expansion and regular_expansion != token:
                expansions.add(regular_expansion)
        
        return list(expansions)

    def _generate_symbol_combinations(self, symbols_info: List[tuple]) -> List[List[tuple]]:
        """Generate all combinations of symbol word choices."""
        if not symbols_info:
            return []
        
        # Extract just the word lists for each symbol
        word_lists = [words for _, _, words in symbols_info]
        
        # Generate all combinations
        combinations = []
        for combo in product(*word_lists):
            # Rebuild the combo with position and symbol info
            combo_with_info = []
            for i, (pos, symbol, _) in enumerate(symbols_info):
                combo_with_info.append((pos, symbol, combo[i]))
            combinations.append(combo_with_info)
        
        return combinations

    def _build_expansion_with_symbol_combo_generic(self, token: str, symbol_combo: List[tuple]) -> str:
        """Build token expansion using specific symbol word combination - generic version."""
        result_parts = []
        current_number = ""
        current_alpha = ""
        char_index = 0
        
        # Create a lookup for symbol replacements by position
        symbol_replacements = {pos: word for pos, symbol, word in symbol_combo}
        
        for char in token:
            if char.isdigit():
                if current_alpha:
                    # Check if it's a unit and get the best expansion
                    if current_alpha.lower() in self.unit_words:
                        result_parts.append(self.unit_words[current_alpha.lower()][0])
                    else:
                        result_parts.append(current_alpha.lower())
                    current_alpha = ""
                current_number += char
            elif char.isalpha():
                if current_number:
                    number_words = self.convert_number_to_words(current_number)
                    result_parts.extend(number_words)
                    current_number = ""
                current_alpha += char
            else:
                # Handle non-alphanumeric characters (symbols)
                if current_number:
                    number_words = self.convert_number_to_words(current_number)
                    result_parts.extend(number_words)
                    current_number = ""
                if current_alpha:
                    if current_alpha.lower() in self.unit_words:
                        result_parts.append(self.unit_words[current_alpha.lower()][0])
                    else:
                        result_parts.append(current_alpha.lower())
                    current_alpha = ""
                
                # Use the specific symbol word from the combination
                if char_index in symbol_replacements:
                    result_parts.append(symbol_replacements[char_index])
                elif char in self.symbol_words:
                    # Fallback to first option if not in combination
                    result_parts.append(self.symbol_words[char][0])
            
            char_index += 1
        
        # Process remaining
        if current_number:
            number_words = self.convert_number_to_words(current_number)
            result_parts.extend(number_words)
        if current_alpha:
            if current_alpha.lower() in self.unit_words:
                result_parts.append(self.unit_words[current_alpha.lower()][0])
            else:
                result_parts.append(current_alpha.lower())
        
        return ' '.join(result_parts) if result_parts else token

    def get_all_possible_expansions_with_units(self, token: str) -> List[str]:
        """Enhanced version that also generates unit variations."""
        base_expansions = self.get_all_possible_expansions(token)
        all_expansions = set(base_expansions)
        
        # For each base expansion, try different unit forms if it contains units
        for expansion in base_expansions:
            words = expansion.split()
            if words:
                last_word = words[-1].lower()
                
                # Check if the last word is a unit that has multiple forms
                unit_found = None
                for unit_abbrev, unit_words in self.unit_words.items():
                    if last_word in [w.lower() for w in unit_words]:
                        unit_found = unit_words
                        break
                
                if unit_found:
                    # Generate versions with all unit forms
                    for unit_form in unit_found:
                        new_expansion = ' '.join(words[:-1] + [unit_form])
                        all_expansions.add(new_expansion)
        
        return list(all_expansions)

    def find_best_expansion_match(self, token: str, target_text: str) -> str:
        """Find the expansion of token that best matches the target text."""
        all_expansions = self.get_all_possible_expansions_with_units(token)
        
        if not all_expansions:
            return token
        
        # Normalize target text for comparison
        normalized_target = self.normalize_text_semantically(target_text.lower())
        
        best_expansion = all_expansions[0]
        best_score = float('inf')
        
        for expansion in all_expansions:
            normalized_expansion = self.normalize_text_semantically(expansion.lower())
            
            # Simple character-level similarity score
            score = self._calculate_edit_distance(normalized_expansion, normalized_target)
            
            if score < best_score:
                best_score = score
                best_expansion = expansion
        
        return best_expansion

    def _calculate_edit_distance(self, s1: str, s2: str) -> int:
        """Calculate edit distance between two strings."""
        if len(s1) < len(s2):
            return self._calculate_edit_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]


# Example usage for testing
if __name__ == "__main__":
    # Test integration with complete aligner
    aligner = EnglishCERAligner(cer_threshold=0.4)
    
    # Test case that was failing before
    reference = "2.8 cm"
    hypothesis = "two point eight centimeters"
    
    print("=== Testing Complete EnglishCERAligner ===")
    print(f"Reference: '{reference}'")
    print(f"Hypothesis: '{hypothesis}'")
    print()
    
    # Test tokenization
    ref_tokens = aligner.tokenize_with_positions(reference)
    hyp_tokens = aligner.tokenize_with_positions(hypothesis)
    print(f"Ref tokens: {[t[0] for t in ref_tokens]}")
    print(f"Hyp tokens: {[t[0] for t in hyp_tokens]}")
    print()
    
    # Test expansions with detailed debug
    print("=== Testing Generic Expansion Methods ===")
    test_tokens = ["2.8", "cm", "750mg", "AMPIGARD", "375MG"]
    
    for token in test_tokens:
        print(f"\nToken: '{token}'")
        print(f"  Token repr: {repr(token)}")
        
        # Test the MAIN condition that's causing the issue
        main_condition = re.search(r'[\d\./\-+%*=&@#$€£¥°™©®]', token)
        print(f"  Main condition (has digits/symbols): {bool(main_condition)}")
        
        # Test individual patterns
        digit_only = re.search(r'\d', token)
        print(f"  Digit only: {bool(digit_only)}")
        
        letter_only = re.search(r'[a-zA-Z]', token)
        print(f"  Letter only: {bool(letter_only)}")
        
        both_check = re.search(r'\d', token) and re.search(r'[a-zA-Z]', token)
        print(f"  Both check: {bool(both_check)}")
        
        # Manual character check
        manual_digits = any(c.isdigit() for c in token)
        print(f"  Manual digit check: {manual_digits}")
        
        # Test if token enters the main if branch
        if re.search(r'[\d\./\-+%*=&@#$€£¥°™©®]', token):
            print(f"  → Enters main if branch")
            
            # Check symbols
            symbols_info = []
            for i, char in enumerate(token):
                if char in aligner.symbol_words:
                    symbols_info.append((i, char, aligner.symbol_words[char]))
            
            print(f"  → Symbols found: {len(symbols_info)}")
            
            if not symbols_info:
                print(f"  → Goes to else branch (no symbols)")
        else:
            print(f"  → Does NOT enter main if branch")
        
        # Old method (single expansion)
        old_expansion = aligner.expand_ref_token_semantically(token)
        print(f"  Old expansion: '{old_expansion}'")
        
        # New method (all expansions)
        all_expansions = aligner.get_all_possible_expansions_with_units(token)
        print(f"  All expansions: {all_expansions}")
    
    print(f"\n=== Testing Full Alignment ===")
    # Test the complete alignment process
    alignments = aligner.align_words_dp(reference, hypothesis)
    
    # Print results
    aligner.print_alignment_visual(alignments)
    
    # Calculate statistics
    stats = aligner.calculate_error_rates(alignments)
    print(f"Error Statistics:")
    print(f"Reference words: {stats['total_ref_words']}")
    print(f"Correct: {stats['word_correct']}")
    print(f"Substitutions: {stats['word_substitutions']}")
    print(f"Deletions: {stats['word_deletions']}")
    print(f"Insertions: {stats['word_insertions']}")
    print(f"Word Error Rate: {stats['wer']:.2%}")
    print(f"Weighted word-level CER: {stats['weighted_word_cer']:.2%}")
    
    print(f"\n=== Expected Results ===")
    print("With the enhanced aligner, you should see:")
    print("- Perfect alignment: 2.8 = 'two point eight', cm = 'centimeters'")
    print("- 750mg = 'seven hundred and fifty MG'") 
    print("- AMPIGARD should remain unchanged (no incorrect processing)")
    print("- Much improved alignment quality overall")