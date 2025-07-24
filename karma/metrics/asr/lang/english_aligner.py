#!/usr/bin/env python3
"""
Enhanced English-specific CER-based word aligner with all advanced features.
"""

import re
from typing import List, Set
from itertools import product
from karma.metrics.asr.base_aligner import BaseCERAligner

class EnglishCERAligner(BaseCERAligner):
    """English-specific CER-based word aligner with enhanced expansion support."""
    
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
            '.': ['dot', 'point', 'period', 'full stop', ''],
            '/': ['slash', 'over', 'by', 'or', 'per', 'divided by'],
            '-': ['dash', 'hyphen', 'minus', 'negative'],
            '+': ['plus', 'add', 'positive', 'and'],
            '%': ['percent', 'percentage', 'per cent'],
            '*': ['star', 'asterisk', 'times', 'multiply'],
            '=': ['equals', 'equal', 'is', 'equal to'],
            '&': ['and', 'ampersand'],
            '@': ['at', 'at sign'],
            '#': ['hash', 'pound', 'number', 'hashtag'],
            '$': ['dollar', 'dollars'],
            '€': ['euro', 'euros'],
            '£': ['pound', 'pounds'],
            '¥': ['yen'],
            '°': ['degree', 'degrees'],
            '™': ['trademark'],
            '©': ['copyright'],
            '®': ['registered'],
        }
        
        # English unit abbreviations
        self.unit_words = {
            'mm': ['millimeter', 'millimeters', 'millimetre', 'millimetres'],
            'cm': ['centimeter', 'centimeters', 'centimetre', 'centimetres'],
            'm': ['meter', 'meters', 'metre', 'metres'],
            'km': ['kilometer', 'kilometers', 'kilometre', 'kilometres'],
            'g': ['gram', 'grams', 'gramme', 'grammes'],
            'kg': ['kilogram', 'kilograms', 'kilogramme', 'kilogrammes'],
            'mg': ['milligram', 'milligrams'],
            'ml': ['milliliter', 'milliliters', 'millilitre', 'millilitres'],
            'l': ['liter', 'liters', 'litre', 'litres'],
            'sec': ['second', 'seconds'],
            'min': ['minute', 'minutes'],
            'hr': ['hour', 'hours'],
            'db': ['decibel', 'decibels'],
            'hz': ['hertz'],
            'mv': ['millivolt', 'millivolts'],
            'cc': ['cubic centimeter', 'cubic centimeters'],
            'ppm': ['parts per million'],
            'rpm': ['revolutions per minute'],
        }
        
        # Build reverse mappings
        self.word_to_number = {}
        self.word_to_symbol = {}
        self.word_to_unit = {}
        
        for word, digit in self.extended_number_words.items():
            self.word_to_number[word.lower()] = digit
            
        for symbol, words in self.symbol_words.items():
            for word in words:
                if word:  # Skip empty strings
                    self.word_to_symbol[word.lower()] = symbol
                
        for unit, words in self.unit_words.items():
            for word in words:
                self.word_to_unit[word.lower()] = unit

    def normalize_hyphenated_words(self, text: str) -> str:
        """Remove hyphens from hyphenated words."""
        return re.sub(r'\b([a-zA-Z]+)-([a-zA-Z]+)\b', r'\1\2', text)

    def normalize_text_semantically(self, text: str) -> str:
        """Normalize text for semantic comparison - English version."""
        text = text.lower().strip()
        text = self.normalize_hyphenated_words(text)
        
        # Remove common punctuation that shouldn't affect semantic meaning
        text = re.sub(r'[.\-_]', ' ', text)  # Replace punctuation with spaces
        text = re.sub(r'\s+', ' ', text)      # Normalize multiple spaces
        
        # Handle digit/symbol sequences
        if re.search(r'[\d./\-+%*=&@#$€£¥°™©®]', text):
            normalized_parts = []
            current_number = ""
            current_alpha = ""
            
            for char in text:
                if char.isdigit():
                    # Process any accumulated alphabetic characters first
                    if current_alpha:
                        normalized_parts.append(current_alpha)
                        current_alpha = ""
                    current_number += char
                elif char.isalpha():
                    # Process accumulated number first
                    if current_number:
                        number_words = self.convert_number_to_words(current_number)
                        normalized_parts.extend(number_words)
                        current_number = ""
                    current_alpha += char
                elif char in self.symbol_words:
                    # Process accumulated number and alpha first
                    if current_number:
                        number_words = self.convert_number_to_words(current_number)
                        normalized_parts.extend(number_words)
                        current_number = ""
                    if current_alpha:
                        normalized_parts.append(current_alpha)
                        current_alpha = ""
                    # Add symbol word (prefer first non-empty option)
                    symbol_words = self.symbol_words[char]
                    for word in symbol_words:
                        if word:  # Use first non-empty word
                            normalized_parts.append(word)
                            break
                elif char == ' ':
                    # Process accumulated number and alpha, but don't add space
                    if current_number:
                        number_words = self.convert_number_to_words(current_number)
                        normalized_parts.extend(number_words)
                        current_number = ""
                    if current_alpha:
                        normalized_parts.append(current_alpha)
                        current_alpha = ""
            
            # Process any remaining number or alpha at the end
            if current_number:
                number_words = self.convert_number_to_words(current_number)
                normalized_parts.extend(number_words)
            if current_alpha:
                normalized_parts.append(current_alpha)
            
            return ''.join(normalized_parts)
        
        # Handle word sequences - just keep alphabetic chars
        words = text.split()
        normalized_parts = []
        
        for word in words:
            # Remove common punctuation
            word = re.sub(r'[.\-_,;:!?\'"()]', '', word)
            clean_word = ''.join(c for c in word if c.isalpha())
            if clean_word:
                normalized_parts.append(clean_word)
        
        return ''.join(normalized_parts)
    
    def expand_ref_token_semantically(self, token: str) -> str:
        """Expand a reference token to its semantic word form - English version."""
        token = self.normalize_hyphenated_words(token)
        
        # Handle tokens with digits
        if re.search(r'\d', token):
            # Special handling for decimal numbers with units (e.g., 7.5mg)
            decimal_match = re.match(r'^(\d+\.\d+)([a-zA-Z]+)$', token)
            if decimal_match:
                number_part, unit_part = decimal_match.groups()
                number_words = self.convert_decimal_to_words(number_part)
                
                if unit_part.lower() in self.word_to_unit:
                    unit_word = self.unit_words[self.word_to_unit[unit_part.lower()]][0]
                else:
                    unit_word = unit_part.lower()
                
                expanded = ' '.join(number_words + [unit_word])
                if expanded != token:
                    return expanded
            
            # General handling for other digit patterns
            result_parts = []
            current_number = ""
            current_alpha = ""
            in_decimal = False
            
            for char in token:
                if char.isdigit():
                    # Process any accumulated alphabetic characters first
                    if current_alpha:
                        # Check if it's a unit abbreviation
                        if current_alpha.lower() in self.word_to_unit:
                            result_parts.append(self.unit_words[self.word_to_unit[current_alpha.lower()]][0])
                        else:
                            result_parts.append(current_alpha.lower())
                        current_alpha = ""
                    current_number += char
                elif char == '.' and current_number and not in_decimal:
                    # Handle decimal point - continue building the number
                    current_number += char
                    in_decimal = True
                elif char.isalpha():
                    # Process accumulated number first
                    if current_number:
                        if '.' in current_number:
                            number_words = self.convert_decimal_to_words(current_number)
                        else:
                            number_words = self.convert_number_to_words(current_number)
                        result_parts.extend(number_words)
                        current_number = ""
                        in_decimal = False
                    current_alpha += char
                else:
                    # Process accumulated number and alpha
                    if current_number:
                        if '.' in current_number:
                            number_words = self.convert_decimal_to_words(current_number)
                        else:
                            number_words = self.convert_number_to_words(current_number)
                        result_parts.extend(number_words)
                        current_number = ""
                        in_decimal = False
                    if current_alpha:
                        # Check if it's a unit abbreviation
                        if current_alpha.lower() in self.word_to_unit:
                            result_parts.append(self.unit_words[self.word_to_unit[current_alpha.lower()]][0])
                        else:
                            result_parts.append(current_alpha.lower())
                        current_alpha = ""
                    
                    # Process symbol character (but not decimal point)
                    if char in self.symbol_words and char != '.' and self.symbol_words[char]:
                        # Use first non-empty word
                        for word in self.symbol_words[char]:
                            if word:
                                result_parts.append(word)
                                break
            
            # Process any remaining number or alpha at the end
            if current_number:
                if '.' in current_number:
                    number_words = self.convert_decimal_to_words(current_number)
                else:
                    number_words = self.convert_number_to_words(current_number)
                result_parts.extend(number_words)
            if current_alpha:
                # Check if it's a unit abbreviation
                if current_alpha.lower() in self.word_to_unit:
                    result_parts.append(self.unit_words[self.word_to_unit[current_alpha.lower()]][0])
                else:
                    result_parts.append(current_alpha.lower())
            
            if result_parts:
                expanded = ' '.join(result_parts)
                if expanded != token:
                    return expanded
        
        # Check if token is a standalone unit abbreviation
        elif token.lower() in self.word_to_unit:
            return self.unit_words[self.word_to_unit[token.lower()]][0]
        
        # Check if token contains symbols that should be expanded
        elif re.search(r'[./+%*=&@#$€£¥°™©®-]', token):
            expanded_parts = []
            for char in token:
                if char in self.symbol_words and self.symbol_words[char]:
                    # Use first non-empty word
                    for word in self.symbol_words[char]:
                        if word:
                            expanded_parts.append(word)
                            break
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
            return words if words else ["zero"]
        
        if num == 0:
            return ["zero"]
        
        if num < 0:
            return ["negative"] + self.convert_number_to_words(str(-num))
        
        # Handle large numbers with fallback to digit-by-digit
        if num > 999999999999:  # Above 999 billion
            digit_words = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
            words = []
            for digit in str(num):
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
            return words if words else ["zero"]
        
        if num == 0:
            return ["zero"]
        
        if num < 0:
            return ["negative"] + self.convert_number_to_words_with_and(str(-num))
        
        # Handle large numbers with fallback to digit-by-digit
        if num > 999999999999:  # Above 999 billion
            digit_words = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
            words = []
            for digit in str(num):
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

    def convert_decimal_to_words(self, decimal_str: str) -> List[str]:
        """Convert a decimal number string to English words."""
        if not decimal_str:
            return ["zero"]
        
        # Handle negative numbers
        if decimal_str.startswith('-'):
            return ["negative"] + self.convert_decimal_to_words(decimal_str[1:])
        
        # Split into integer and decimal parts
        if '.' in decimal_str:
            integer_part, decimal_part = decimal_str.split('.', 1)
        else:
            integer_part, decimal_part = decimal_str, ""
        
        words = []
        
        # Convert integer part
        if integer_part:
            if integer_part == "0":
                words.append("zero")
            else:
                words.extend(self.convert_number_to_words(integer_part))
        else:
            words.append("zero")
        
        # Convert decimal part
        if decimal_part:
            words.append("point")
            # Convert each digit individually
            digit_words = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
            for digit in decimal_part:
                if digit.isdigit():
                    words.append(digit_words[int(digit)])
        
        return words
    
    def convert_decimal_to_words_with_and(self, decimal_str: str) -> List[str]:
        """Convert a decimal number string to English words with British 'and' style."""
        if not decimal_str:
            return ["zero"]
        
        # Handle negative numbers
        if decimal_str.startswith('-'):
            return ["negative"] + self.convert_decimal_to_words_with_and(decimal_str[1:])
        
        # Split into integer and decimal parts
        if '.' in decimal_str:
            integer_part, decimal_part = decimal_str.split('.', 1)
        else:
            integer_part, decimal_part = decimal_str, ""
        
        words = []
        
        # Convert integer part with British style
        if integer_part:
            if integer_part == "0":
                words.append("zero")
            else:
                words.extend(self.convert_number_to_words_with_and(integer_part))
        else:
            words.append("zero")
        
        # Convert decimal part
        if decimal_part:
            words.append("point")
            # Convert each digit individually
            digit_words = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
            for digit in decimal_part:
                if digit.isdigit():
                    words.append(digit_words[int(digit)])
        
        return words

    def _convert_below_1000(self, num: int) -> List[str]:
        """Convert numbers below 1000 to English words - American style."""
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
    
    def _convert_below_1000_with_and(self, num: int) -> List[str]:
        """Convert numbers below 1000 to English words with 'and' - British style."""
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
    
    # ENHANCED EXPANSION METHODS
    
    def get_all_possible_expansions(self, token: str) -> List[str]:
        """Get all possible semantic expansions for a token."""
        expansions = set()  # Use set to avoid duplicates
        
        # First, try the basic expansion
        basic_expansion = self.expand_ref_token_semantically(token)
        if basic_expansion != token:
            expansions.add(basic_expansion)
        
        # Handle tokens with slashes (like 75mg/40mg)
        if '/' in token:
            parts = token.split('/')
            if len(parts) == 2:
                # Expand each part separately
                left_expansions = self.get_all_possible_expansions(parts[0]) if parts[0] != token else [parts[0]]
                right_expansions = self.get_all_possible_expansions(parts[1]) if parts[1] != token else [parts[1]]
                
                # Combine with different slash words
                slash_words = self.symbol_words.get('/', ['slash', 'over', 'by', 'or', 'per', 'divided by'])
                for left in left_expansions:
                    for right in right_expansions:
                        for slash_word in slash_words:
                            if slash_word:  # Skip empty strings
                                expansion = f"{left} {slash_word} {right}"
                                expansions.add(expansion)
        
        # Handle tokens with digits
        if re.search(r'\d', token) and '/' not in token:  # Don't re-process if already handled above
            # Pattern: Pure decimal number (158.12)
            match = re.match(r'^(\d+\.\d+)$', token)
            if match:
                number_part = match.group(1)
                number_words = self.convert_decimal_to_words(number_part)
                number_words_with_and = self.convert_decimal_to_words_with_and(number_part)
                
                # Add both American and British styles
                expansion = ' '.join(number_words)
                expansions.add(expansion)
                expansion_with_and = ' '.join(number_words_with_and)
                expansions.add(expansion_with_and)
            
            # Pattern: Decimal number followed by unit (7.5mg)
            match = re.match(r'^(\d+\.\d+)([a-zA-Z]+)$', token)
            if match:
                number_part, unit_part = match.groups()
                number_words = self.convert_decimal_to_words(number_part)
                number_words_with_and = self.convert_decimal_to_words_with_and(number_part)
                
                if unit_part.lower() in self.unit_words:
                    # Add all unit variations
                    for unit_variant in self.unit_words[unit_part.lower()]:
                        expansion = ' '.join(number_words + [unit_variant])
                        expansions.add(expansion)
                        # Also add British style with 'and'
                        expansion_with_and = ' '.join(number_words_with_and + [unit_variant])
                        expansions.add(expansion_with_and)
                        
                        # Also add uppercase variant if original was uppercase
                        if unit_part.isupper():
                            expansion_upper = ' '.join(number_words + [unit_part])
                            expansions.add(expansion_upper)
                            expansion_with_and_upper = ' '.join(number_words_with_and + [unit_part])
                            expansions.add(expansion_with_and_upper)
                else:
                    # Not a known unit, just lowercase it
                    expansion = ' '.join(number_words + [unit_part.lower()])
                    expansions.add(expansion)
                    expansion_with_and = ' '.join(number_words_with_and + [unit_part.lower()])
                    expansions.add(expansion_with_and)
            
            # Pattern: Integer number followed by unit (750mg)
            match = re.match(r'^(\d+)([a-zA-Z]+)$', token)
            if match:
                number_part, unit_part = match.groups()
                number_words = self.convert_number_to_words(number_part)
                number_words_with_and = self.convert_number_to_words_with_and(number_part)
                
                if unit_part.lower() in self.unit_words:
                    # Add all unit variations
                    for unit_variant in self.unit_words[unit_part.lower()]:
                        # Lowercase variant
                        expansion = ' '.join(number_words + [unit_variant])
                        expansions.add(expansion)
                        expansion_with_and = ' '.join(number_words_with_and + [unit_variant])
                        expansions.add(expansion_with_and)
                        # Uppercase variant (always generate both cases)
                        expansion_upper = ' '.join(number_words + [unit_part.upper()])
                        expansions.add(expansion_upper)
                        expansion_with_and_upper = ' '.join(number_words_with_and + [unit_part.upper()])
                        expansions.add(expansion_with_and_upper)
                else:
                    # Not a known unit, just lowercase it
                    expansion = ' '.join(number_words + [unit_part.lower()])
                    expansions.add(expansion)
                    expansion_with_and = ' '.join(number_words_with_and + [unit_part.lower()])
                    expansions.add(expansion_with_and)
                    # Uppercase variant (always generate both cases)
                    expansion_upper = ' '.join(number_words + [unit_part.upper()])
                    expansions.add(expansion_upper)
                    expansion_with_and_upper = ' '.join(number_words_with_and + [unit_part.upper()])
                    expansions.add(expansion_with_and_upper)
            
            # Pattern: Unit followed by number (pH7, CO2)
            match = re.match(r'^([a-zA-Z]+)(\d+)$', token)
            if match:
                unit_part, number_part = match.groups()
                number_words = self.convert_number_to_words(number_part)
                
                if unit_part.lower() in self.unit_words:
                    for unit_variant in self.unit_words[unit_part.lower()]:
                        expansion = ' '.join([unit_variant] + number_words)
                        expansions.add(expansion)
                else:
                    expansion = ' '.join([unit_part.lower()] + number_words)
                    expansions.add(expansion)
        
        # Handle unit abbreviations without numbers
        elif token.lower() in self.unit_words:
            for unit_word in self.unit_words[token.lower()]:
                expansions.add(unit_word)
        
        # Handle tokens with symbols
        if re.search(r'[./+%*=&@#$€£¥°™©®-]', token):
            # Find all symbols and their positions
            symbols_info = []
            for i, char in enumerate(token):
                if char in self.symbol_words:
                    symbols_info.append((i, char, self.symbol_words[char]))
            
            if symbols_info:
                # Generate combinations of symbol expansions
                symbol_combinations = self._generate_symbol_combinations(symbols_info)
                for combo in symbol_combinations:
                    expansion = self._build_expansion_with_symbol_combo(token, combo)
                    if expansion != token:
                        expansions.add(expansion)
        
        return list(expansions)
    
    def _generate_symbol_combinations(self, symbols_info: List[tuple]) -> List[List[tuple]]:
        """Generate all combinations of symbol word choices."""
        if not symbols_info:
            return []
        
        # Extract word lists for each symbol
        word_lists = []
        for _, _, words in symbols_info:
            # Filter out empty strings
            non_empty_words = [w for w in words if w]
            if non_empty_words:
                word_lists.append(non_empty_words)
            else:
                word_lists.append([''])  # Use empty string if no non-empty words
        
        # Generate all combinations
        combinations = []
        for combo in product(*word_lists):
            combo_with_info = []
            for i, (pos, symbol, _) in enumerate(symbols_info):
                combo_with_info.append((pos, symbol, combo[i]))
            combinations.append(combo_with_info)
        
        return combinations
    
    def _build_expansion_with_symbol_combo(self, token: str, symbol_combo: List[tuple]) -> str:
        """Build token expansion using specific symbol word combination."""
        result_parts = []
        current_number = ""
        current_alpha = ""
        in_decimal = False
        
        # Create lookup for symbol replacements
        symbol_replacements = {pos: word for pos, symbol, word in symbol_combo}
        
        for i, char in enumerate(token):
            if char.isdigit():
                if current_alpha:
                    if current_alpha.lower() in self.unit_words:
                        result_parts.append(self.unit_words[current_alpha.lower()][0])
                    else:
                        result_parts.append(current_alpha.lower())
                    current_alpha = ""
                current_number += char
            elif char == '.' and current_number and not in_decimal:
                # Handle decimal point - continue building the number
                current_number += char
                in_decimal = True
            elif char.isalpha():
                if current_number:
                    if '.' in current_number:
                        number_words = self.convert_decimal_to_words(current_number)
                    else:
                        number_words = self.convert_number_to_words(current_number)
                    result_parts.extend(number_words)
                    current_number = ""
                    in_decimal = False
                current_alpha += char
            else:
                # Process any accumulated number/alpha
                if current_number:
                    if '.' in current_number:
                        number_words = self.convert_decimal_to_words(current_number)
                    else:
                        number_words = self.convert_number_to_words(current_number)
                    result_parts.extend(number_words)
                    current_number = ""
                    in_decimal = False
                if current_alpha:
                    if current_alpha.lower() in self.unit_words:
                        result_parts.append(self.unit_words[current_alpha.lower()][0])
                    else:
                        result_parts.append(current_alpha.lower())
                    current_alpha = ""
                
                # Add symbol word if it exists and is non-empty (but not decimal point)
                if i in symbol_replacements and symbol_replacements[i] and char != '.':
                    result_parts.append(symbol_replacements[i])
        
        # Process remaining
        if current_number:
            if '.' in current_number:
                number_words = self.convert_decimal_to_words(current_number)
            else:
                number_words = self.convert_number_to_words(current_number)
            result_parts.extend(number_words)
        if current_alpha:
            if current_alpha.lower() in self.unit_words:
                result_parts.append(self.unit_words[current_alpha.lower()][0])
            else:
                result_parts.append(current_alpha.lower())
        
        return ' '.join(result_parts) if result_parts else token