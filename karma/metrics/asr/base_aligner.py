#!/usr/bin/env python3
"""
Base classes for CER-based word alignment with advanced features.
"""

import sys
import re
import editdistance
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
from itertools import product

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

class BaseCERAligner(ABC):
    """Enhanced base class for CER-based word aligners with advanced features."""
    
    def __init__(self, cer_threshold: float = 0.4):
        self.cer_threshold = cer_threshold
        self._initialize_language_specific_mappings()
    
    @abstractmethod
    def _initialize_language_specific_mappings(self):
        """Initialize language-specific mappings."""
        pass
    
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
        
        # Check for acronym-like relationships
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
    
    @abstractmethod
    def normalize_text_semantically(self, text: str) -> str:
        """Normalize text for semantic comparison - language specific."""
        pass
    
    @abstractmethod
    def expand_ref_token_semantically(self, token: str) -> str:
        """Expand a reference token to its semantic word form - language specific."""
        pass
    
    def calculate_character_error_rate(self, ref_text: str, hyp_text: str) -> float:
        """Calculate CER using direct, semantic, and substring-aware comparison."""
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
    
    def get_all_possible_expansions(self, token: str) -> List[str]:
        """Get all possible semantic expansions for a token - can be overridden by subclasses."""
        # Default implementation just returns the single expansion
        expansion = self.expand_ref_token_semantically(token)
        if expansion != token:
            return [expansion]
        return []
    
    def align_words_dp(self, reference: str, hypothesis: str) -> List[WordAlignment]:
        """Use dynamic programming for optimal alignment with enhanced scoring."""
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
                    
                    # First try single expansion
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
                    
                    # Try all possible expansions
                    all_expansions = self.get_all_possible_expansions(ref_token)
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
                
                # Regular alignment combinations with enhanced scoring
                for ref_span in range(1, max_ref_span + 1):
                    for hyp_span in range(1, max_hyp_span + 1):
                        ref_seq = [ref_tokens[ref_start + k][0] for k in range(ref_span)]
                        hyp_seq = [hyp_tokens[hyp_start + k][0] for k in range(hyp_span)]
                        
                        # Calculate similarity with enhanced penalties
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
                                
                                # Enhanced scoring for single ref to multiple hyp
                                if similarity >= 0.99:
                                    score_bonus = similarity + 5.0  # Big bonus for perfect matches
                                elif similarity >= 0.95:
                                    score_bonus = similarity + 3.0  # Good bonus for near-perfect
                                elif hyp_span == 1:
                                    score_bonus = similarity + 1.0
                                else:
                                    # Penalize single ref token to many hyp tokens with poor similarity
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

    def extract_keywords_from_text(self, text: str, annotations: List[Dict]) -> List[str]:
        """
        Extract keywords from text using the provided offset annotations.
        
        Args:
            text: Original text
            annotations: List of annotations with format [keyword, category, [[start, end]]]
            
        Returns:
            List of extracted keywords
        """
        keywords = []
        
        for annotation in annotations:
            keyword_text = annotation[0]
            category = annotation[1]
            offsets = annotation[2]
            
            # Extract text using offsets to verify
            for offset_pair in offsets:
                start, end = offset_pair
                extracted_text = text[start:end]
                # Split multi-word entities into individual words
                words = extracted_text.split()
                keywords.extend(words)
                #keywords.append(extracted_text) #Entire term as a single word
        
        return keywords
    
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
