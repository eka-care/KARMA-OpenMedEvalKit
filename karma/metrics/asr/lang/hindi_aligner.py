#!/usr/bin/env python3
"""
Hindi-specific CER-based word aligner.
"""

import re
from typing import List, Dict
from karma.metrics.asr.base_aligner import BaseCERAligner

class HindiCERAligner(BaseCERAligner):
    """Hindi-specific CER-based word aligner with English transliteration support."""
    
    # Class constants - single source of truth
    DIGIT_TO_HINDI = {
        '0': 'शून्य', '1': 'एक', '2': 'दो', '3': 'तीन', '4': 'चार',
        '5': 'पांच', '6': 'छह', '7': 'सात', '8': 'आठ', '9': 'नौ'
    }
    
    DEVANAGARI_TO_ARABIC = {
        '०': '0', '१': '1', '२': '2', '३': '3', '४': '4',
        '५': '5', '६': '6', '७': '7', '८': '8', '९': '9'
    }
    
    def _initialize_language_specific_mappings(self):
        """Initialize Hindi-specific mappings - consolidated."""
        
        # Single comprehensive number mapping (Hindi + English + transliterated)
        self.extended_number_words = {
            # Basic digits
            'शून्य': '0', 'zero': '0', 'जीरो': '0',
            'एक': '1', 'one': '1', 'वन': '1',
            'दो': '2', 'two': '2', 'टू': '2', 'टु': '2',
            'तीन': '3', 'three': '3', 'थ्री': '3',
            'चार': '4', 'four': '4', 'फोर': '4', 'फ़ोर': '4',
            'पांच': '5', 'पाँच': '5', 'five': '5', 'फाइव': '5', 'फ़ाइव': '5',
            'छह': '6', 'छः': '6', 'six': '6', 'सिक्स': '6',
            'सात': '7', 'seven': '7', 'सेवन': '7', 'सेवेन': '7',
            'आठ': '8', 'eight': '8', 'एट': '8', 'एइट': '8',
            'नौ': '9', 'nine': '9', 'नाइन': '9', 'नाईन': '9',
            
            # Teens
            'दस': '10', 'ten': '10', 'टेन': '10',
            'ग्यारह': '11', 'eleven': '11', 'इलेवन': '11', 'एलेवन': '11',
            'बारह': '12', 'twelve': '12', 'ट्वेल्व': '12', 'बारा': '12',
            'तेरह': '13', 'thirteen': '13', 'थर्टीन': '13',
            'चौदह': '14', 'fourteen': '14', 'फोर्टीन': '14',
            'पंद्रह': '15', 'fifteen': '15', 'फिफ्टीन': '15',
            'सोलह': '16', 'sixteen': '16', 'सिक्सटीन': '16',
            'सत्रह': '17', 'seventeen': '17', 'सेवनटीन': '17',
            'अठारह': '18', 'eighteen': '18', 'एटीन': '18', 'एइटीन': '18',
            'उन्नीस': '19', 'nineteen': '19', 'नाइनटीन': '19',
            
            # Tens
            'बीस': '20', 'twenty': '20', 'ट्वेंटी': '20', 'ट्वेन्टी': '20',
            'तीस': '30', 'thirty': '30', 'थर्टी': '30',
            'चालीस': '40', 'forty': '40', 'फोर्टी': '40',
            'पचास': '50', 'fifty': '50', 'फिफ्टी': '50',
            'साठ': '60', 'sixty': '60', 'सिक्सटी': '60',
            'सत्तर': '70', 'seventy': '70', 'सेवंटी': '70', 'सेवेंटी': '70',
            'अस्सी': '80', 'eighty': '80', 'एटी': '80', 'एइटी': '80',
            'नब्बे': '90', 'ninety': '90', 'नाइंटी': '90', 'नाइनटी': '90',
            
            # Place values
            'सौ': '100', 'hundred': '100', 'हंड्रेड': '100', 'हण्ड्रेड': '100',
            'हजार': '1000', 'thousand': '1000', 'थाउजेंड': '1000', 'थाउज़ेंड': '1000',
            'लाख': '100000', 'lakh': '100000', 'लाक': '100000',
            'करोड़': '10000000', 'crore': '10000000', 'क्रोर': '10000000',
            'million': '1000000', 'मिलियन': '1000000',
            'billion': '1000000000', 'बिलियन': '1000000000',
        }
        
        # Hindi symbol words
        self.symbol_words = {
            '.': ['बिंदु', 'पॉइंट', 'dot', 'point', 'डॉट', ''],
            '।': ['दंड', 'पूर्ण विराम', 'दण्ड', 'विराम', 'दंडा', ''],
            '/': ['या', 'स्लैश', 'बटा', 'पर', 'or', 'slash', 'over', 'by', 'ओर'],
            '-': ['डैश', 'हाइफन', 'माइनस', 'dash', 'hyphen', 'minus', 'हाइफ़न', 'माइनस'],
            '+': ['प्लस', 'जोड़', 'plus', 'प्लस'],
            '%': ['प्रतिशत', 'फीसदी', 'percent', 'परसेंट', 'फ़ीसदी'],
            '*': ['तारा', 'स्टार', 'star', 'स्टार'],
            '=': ['बराबर', 'equals', 'इक्वल', 'इक्वल्स'],
            '&': ['और', 'एंड', 'and', 'एण्ड']
        }

        # Unit abbreviations
        self.unit_words = {
            'mm': ['मिलीमीटर', 'millimeter'],
            'cm': ['सेंटीमीटर', 'centimeter'],
            'mg': ['मिलीग्राम', 'milligram'],
            'kg': ['किलोग्राम', 'kilogram'],
            'ml': ['मिलीलीटर', 'milliliter'],
            'gm': ['ग्राम', 'gram'],
            'km': ['किलोमीटर', 'kilometer'],
            'hr': ['घंटा', 'hour'],
            'min': ['मिनट', 'minute'],
            'sec': ['सेकंड', 'second'],
            'db': ['डेसिबल', 'decibel'],
            'hz': ['हर्ट्ज़', 'hertz'],
            'mv': ['मिलीवोल्ट', 'millivolt'],
            'cc': ['सीसी', 'cubic centimeter'],
            'ppm': ['पीपीएम', 'parts per million'],
            'rpm': ['आरपीएम', 'revolutions per minute']
        }
        
        # Build reverse mappings from single source
        self.word_to_number = {}
        self.word_to_unit = {}
 
        # Derive mappings from extended_number_words only
        for word, digit in self.extended_number_words.items():
            self.word_to_number[word.lower()] = digit
                
        for unit, words in self.unit_words.items():
            for word in words:
                self.word_to_unit[word.lower()] = unit
    
    def _is_devanagari(self, char: str) -> bool:
        """Check if character is in Devanagari script range."""
        return '\u0900' <= char <= '\u097F'
    
    def normalize_text_semantically(self, text: str) -> str:
        """Enhanced normalization handling English number structures."""
        text = text.lower().strip()
        
        # Split into tokens
        tokens = text.split()
        
        # Handle multi-token English number structures FIRST
        if len(tokens) >= 2:
            normalized_structure = self._normalize_english_number_structure(tokens)
            if normalized_structure:  # If we found a valid English structure
                return normalized_structure
        
        # Fall back to individual token normalization
        normalized_tokens = []
        for token in tokens:
            normalized_token = self._normalize_single_token(token)
            normalized_tokens.append(normalized_token)
        
        return ''.join(normalized_tokens)
            
    def _normalize_single_token(self, token: str) -> str:
        """Normalize a single token to standard Hindi form - fixed regex."""
        # Remove Hindi punctuation
        clean_token = token.replace('।', '').replace('॥', '')
        
        # Handle pure number tokens (digits only) - FIXED REGEX
        if re.match(r'^[\d०-९\s]+$', clean_token):
            return self._normalize_number_token(clean_token)
        
        # Handle mixed tokens (numbers + letters/symbols) - FIXED REGEX
        if re.search(r'[\d०-९]', clean_token):
            return self._normalize_mixed_token(clean_token)
        
        # Handle pure word tokens
        if re.match(r'^[a-zA-Z\u0900-\u097F]+$', clean_token):
            return self._normalize_word_token(clean_token)
        
        # Handle symbol-containing tokens
        if re.search(r'[./+%*=&-]', clean_token):
            return self._normalize_symbol_token(clean_token)
        
        # Default: return cleaned lowercase
        return ''.join(c.lower() for c in clean_token if c.isalpha() or self._is_devanagari(c))
    
    def _normalize_number_token(self, token: str) -> str:
        """Normalize tokens containing only numbers - fixed for Devanagari digits."""
        result_parts = []
        
        # Handle both Arabic and Devanagari digits
        arabic_digits = re.findall(r'\d+', token)
        for number in arabic_digits:
            hindi_words = self.convert_number_to_words(number)
            result_parts.extend(hindi_words)
        
        # Handle Devanagari digits separately
        devanagari_digits = re.findall(r'[०-९]+', token)
        for devanagari_number in devanagari_digits:
            # Convert Devanagari to Arabic first
            arabic_number = ""
            for char in devanagari_number:
                if char in self.DEVANAGARI_TO_ARABIC:
                    arabic_number += self.DEVANAGARI_TO_ARABIC[char]
            
            if arabic_number:
                hindi_words = self.convert_number_to_words(arabic_number)
                result_parts.extend(hindi_words)
        
        return ''.join(result_parts) if result_parts else token
            
    def _normalize_mixed_token(self, token: str) -> str:
        """Normalize tokens with mixed numbers, letters, and symbols."""
        result_parts = []
        current_number = ""
        current_alpha = ""
        
        for char in token:
            if char.isdigit() or char in self.DEVANAGARI_TO_ARABIC:
                # Flush alpha, accumulate number
                if current_alpha:
                    result_parts.append(self._process_alpha_part(current_alpha))
                    current_alpha = ""
                
                if char in self.DEVANAGARI_TO_ARABIC:
                    current_number += self.DEVANAGARI_TO_ARABIC[char]
                else:
                    current_number += char
                    
            elif char.isalpha() or self._is_devanagari(char):
                # Flush number, accumulate alpha
                if current_number:
                    hindi_words = self.convert_number_to_words(current_number)
                    result_parts.extend(hindi_words)
                    current_number = ""
                current_alpha += char
                
            elif char in self.symbol_words:
                # Flush both, add symbol
                if current_number:
                    hindi_words = self.convert_number_to_words(current_number)
                    result_parts.extend(hindi_words)
                    current_number = ""
                if current_alpha:
                    result_parts.append(self._process_alpha_part(current_alpha))
                    current_alpha = ""
                result_parts.append(self.symbol_words[char][0])  # Use Hindi form
        
        # Flush remaining
        if current_number:
            hindi_words = self.convert_number_to_words(current_number)
            result_parts.extend(hindi_words)
        if current_alpha:
            result_parts.append(self._process_alpha_part(current_alpha))
        
        return ''.join(result_parts)
    
    def _normalize_word_token(self, token: str) -> str:
        """Normalize pure word tokens - enhanced for English structures."""
        clean_word = ''.join(c for c in token if c.isalpha() or self._is_devanagari(c))
        
        # Check if it's a number word - convert to standard Hindi
        if clean_word.lower() in self.word_to_number:
            digit = self.word_to_number[clean_word.lower()]
            hindi_words = self.convert_number_to_words(digit)
            return ''.join(hindi_words)
        
        return clean_word.lower()
    
    def _normalize_symbol_token(self, token: str) -> str:
        """Normalize tokens containing symbols."""
        result_parts = []
        for char in token:
            if char in self.symbol_words:
                result_parts.append(self.symbol_words[char][0])  # Use Hindi form
            elif char.isalpha() or self._is_devanagari(char):
                result_parts.append(char.lower())
        return ''.join(result_parts)

    def _normalize_english_number_structure(self, tokens: List[str]) -> str:
        """Handle English number structures like 'थ्री हंड्रेड' -> 'तीनसौ'."""
        if len(tokens) < 2:
            return ""
        
        # Pattern 1: "X हंड्रेड" (X hundred)
        if len(tokens) == 2 and tokens[1].lower() in ['हंड्रेड', 'हण्ड्रेड', 'hundred']:
            first_word = tokens[0].lower()
            if first_word in self.word_to_number:
                digit = int(self.word_to_number[first_word])
                if 1 <= digit <= 9:  # Valid hundreds digit
                    combined_number = digit * 100
                    hindi_words = self.convert_number_to_words(str(combined_number))
                    return ''.join(hindi_words)
        
        # Pattern 2: "X हंड्रेड Y" (X hundred Y)
        elif len(tokens) == 3 and tokens[1].lower() in ['हंड्रेड', 'हण्ड्रेड', 'hundred']:
            first_word = tokens[0].lower()
            third_word = tokens[2].lower()
            
            if (first_word in self.word_to_number and third_word in self.word_to_number):
                hundreds_digit = int(self.word_to_number[first_word])
                remainder = int(self.word_to_number[third_word])
                
                if 1 <= hundreds_digit <= 9 and 1 <= remainder <= 99:
                    combined_number = hundreds_digit * 100 + remainder
                    hindi_words = self.convert_number_to_words(str(combined_number))
                    return ''.join(hindi_words)
        
        # Pattern 3: "X Y" (tens + ones like "ट्वेंटी फाइव")
        elif len(tokens) == 2:
            first_word = tokens[0].lower()
            second_word = tokens[1].lower()
            
            if (first_word in self.word_to_number and second_word in self.word_to_number):
                first_digit = int(self.word_to_number[first_word])
                second_digit = int(self.word_to_number[second_word])
                
                # English structure: tens + ones (20-99)
                if 20 <= first_digit <= 90 and first_digit % 10 == 0 and 1 <= second_digit <= 9:
                    combined_number = first_digit + second_digit
                    hindi_words = self.convert_number_to_words(str(combined_number))
                    return ''.join(hindi_words)
        
        # Pattern 4: "X थाउजेंड" (X thousand)
        elif len(tokens) == 2 and tokens[1].lower() in ['थाउजेंड', 'थाउज़ेंड', 'thousand']:
            first_word = tokens[0].lower()
            if first_word in self.word_to_number:
                digit = int(self.word_to_number[first_word])
                if 1 <= digit <= 999:
                    combined_number = digit * 1000
                    hindi_words = self.convert_number_to_words(str(combined_number))
                    return ''.join(hindi_words)
        
        # No valid English structure found
        return ""

    def _process_alpha_part(self, alpha_part: str) -> str:
        """Process alphabetic part - check for units or return as-is."""
        if alpha_part.lower() in self.word_to_unit:
            return self.unit_words[self.word_to_unit[alpha_part.lower()]][0]  # Hindi form
        return alpha_part.lower()
    
    def expand_ref_token_semantically(self, token: str) -> str:
        """Expand reference token to match hypothesis patterns."""
        # For reference numbers, we need to generate the MOST LIKELY hypothesis form
        # Since hypothesis tends to be longer, prefer spaced multi-word forms
        
        # Handle pure numbers - expand to spaced Hindi words
        if re.match(r'^\d+$', token):
            try:
                num = int(token)
                
                # For compound numbers that might appear as English structures in hypothesis
                if 21 <= num <= 99:
                    # Numbers like 25 might be "ट्वेंटी फाइव" in hypothesis
                    # So expand to spaced form: "पच्चीस" → "twenty five" structure
                    hindi_words = self.convert_number_to_words(token)
                    return ' '.join(hindi_words)  # Return spaced version
                    
                elif 100 <= num <= 999:
                    # Numbers like 300 might be "थ्री हंड्रेड" in hypothesis  
                    # So expand to English structure form
                    english_structure = self._get_english_structure_for_number(num)
                    if english_structure:
                        return english_structure  # "थ्री हंड्रेड"
                    else:
                        # Fallback to spaced Hindi
                        hindi_words = self.convert_number_to_words(token)
                        return ' '.join(hindi_words)
                        
                elif 1000 <= num <= 9999:
                    # Numbers like 2500 might be "टू थाउजेंड फाइव हंड्रेड"
                    english_structure = self._get_english_structure_for_number(num)
                    if english_structure:
                        return english_structure
                    else:
                        hindi_words = self.convert_number_to_words(token)
                        return ' '.join(hindi_words)
                
                else:
                    # Default: spaced Hindi words
                    hindi_words = self.convert_number_to_words(token)
                    return ' '.join(hindi_words)
                    
            except ValueError:
                pass
        
        # Handle mixed number+unit tokens (like "25mg")
        elif re.search(r'\d', token):
            # Use existing mixed token expansion
            result_parts = []
            current_number = ""
            current_alpha = ""
            
            for char in token:
                if char.isdigit() or char in self.DEVANAGARI_TO_ARABIC:
                    if current_alpha:
                        result_parts.append(self._process_alpha_part(current_alpha))
                        current_alpha = ""
                    if char in self.DEVANAGARI_TO_ARABIC:
                        current_number += self.DEVANAGARI_TO_ARABIC[char]
                    else:
                        current_number += char
                elif char.isalpha() or self._is_devanagari(char):
                    if current_number:
                        # For numbers in mixed tokens, prefer English structure too
                        try:
                            num = int(current_number)
                            english_structure = self._get_english_structure_for_number(num)
                            if english_structure and 10 <= num <= 999:
                                result_parts.extend(english_structure.split())
                            else:
                                hindi_words = self.convert_number_to_words(current_number)
                                result_parts.extend(hindi_words)
                        except ValueError:
                            hindi_words = self.convert_number_to_words(current_number)
                            result_parts.extend(hindi_words)
                        current_number = ""
                    current_alpha += char
                elif char in self.symbol_words:
                    if current_number:
                        hindi_words = self.convert_number_to_words(current_number)
                        result_parts.extend(hindi_words)
                        current_number = ""
                    if current_alpha:
                        result_parts.append(self._process_alpha_part(current_alpha))
                        current_alpha = ""
                    result_parts.append(self.symbol_words[char][0])
            
            # Process remaining
            if current_number:
                try:
                    num = int(current_number)
                    english_structure = self._get_english_structure_for_number(num)
                    if english_structure and 10 <= num <= 999:
                        result_parts.extend(english_structure.split())
                    else:
                        hindi_words = self.convert_number_to_words(current_number)
                        result_parts.extend(hindi_words)
                except ValueError:
                    hindi_words = self.convert_number_to_words(current_number)
                    result_parts.extend(hindi_words)
            if current_alpha:
                result_parts.append(self._process_alpha_part(current_alpha))
            
            if result_parts:
                return ' '.join(result_parts)
        
        # For other tokens, use existing logic
        elif token.lower() in self.word_to_unit:
            return self.unit_words[self.word_to_unit[token.lower()]][0]
        elif re.search(r'[./+%*=&-]', token):
            expanded_parts = []
            for char in token:
                if char in self.symbol_words:
                    expanded_parts.append(self.symbol_words[char][0])
                elif char.isalpha() or self._is_devanagari(char):
                    expanded_parts.append(char.lower())
            if expanded_parts:
                return ' '.join(expanded_parts)
        
        return token
                
    def convert_number_to_words(self, number_str: str) -> List[str]:
        """Convert number string to Hindi words - enhanced for Devanagari."""
        if not number_str:
            return ["शून्य"]
        
        # Convert Devanagari to Arabic first
        arabic_number = ""
        for char in number_str:
            if char in self.DEVANAGARI_TO_ARABIC:
                arabic_number += self.DEVANAGARI_TO_ARABIC[char]
            elif char.isdigit():
                arabic_number += char
            # Skip non-digit characters
        
        if not arabic_number:
            return ["शून्य"]
        
        try:
            num = int(arabic_number)
        except ValueError:
            # Fall back to digit-by-digit if conversion fails
            return [self.DIGIT_TO_HINDI.get(digit, digit) for digit in arabic_number if digit.isdigit()]
            
        if num == 0:
            return ["शून्य"]
        
        # For very large numbers, use digit-by-digit
        if num > 9999999:
            return [self.DIGIT_TO_HINDI.get(digit, digit) for digit in str(num)]
        
        return self._convert_number_to_hindi_words(num)

    def _convert_below_100(self, num: int) -> List[str]:
        """Convert numbers below 100 to Hindi words."""
        if num == 0:
            return []
        
        words = []
        
        # Hundreds
        if num >= 100:
            hundreds_digit = num // 100
            ones_words = ['', 'एक', 'दो', 'तीन', 'चार', 'पांच', 'छह', 'सात', 'आठ', 'नौ']
            words.append(ones_words[hundreds_digit])
            words.append("सौ")
            num %= 100
        
        # Special compound forms for 21-99
        if 21 <= num <= 99:
            special_numbers = {
                21: "इक्कीस", 22: "बाईस", 23: "तेईस", 24: "चौबीस", 25: "पच्चीस",
                26: "छब्बीस", 27: "सत्ताईस", 28: "अट्ठाईस", 29: "उनतीस",
                31: "इकतीस", 32: "बत्तीस", 33: "तैंतीस", 34: "चौंतीस", 35: "पैंतीस",
                36: "छत्तीस", 37: "सैंतीस", 38: "अड़तीस", 39: "उनतालीस",
                41: "इकतालीस", 42: "बयालीस", 43: "तैंतालीस", 44: "चवालीस", 45: "पैंतालीस",
                46: "छियालीस", 47: "सैंतालीस", 48: "अड़तालीस", 49: "उनचास",
                51: "इक्यावन", 52: "बावन", 53: "तिरपन", 54: "चौवन", 55: "पचपन",
                56: "छप्पन", 57: "सत्तावन", 58: "अट्ठावन", 59: "उनसठ",
                61: "इकसठ", 62: "बासठ", 63: "तिरसठ", 64: "चौंसठ", 65: "पैंसठ",
                66: "छियासठ", 67: "सड़सठ", 68: "अड़सठ", 69: "उनहत्तर",
                71: "इकहत्तर", 72: "बहत्तर", 73: "तिहत्तर", 74: "चौहत्तर", 75: "पचहत्तर",
                76: "छिहत्तर", 77: "सतहत्तर", 78: "अठहत्तर", 79: "उन्यासी",
                81: "इक्यासी", 82: "बयासी", 83: "तिरासी", 84: "चौरासी", 85: "पचासी",
                86: "छियासी", 87: "सत्तासी", 88: "अठासी", 89: "नवासी",
                91: "इक्यानवे", 92: "बानवे", 93: "तिरानवे", 94: "चौरानवे", 95: "पचानवे",
                96: "छियानवे", 97: "सत्तानवे", 98: "अठानवे", 99: "निन्यानवे"
            }
            
            if num in special_numbers:
                words.append(special_numbers[num])
                return words
        
        # Regular tens
        if num >= 20:
            tens_digit = num // 10
            tens_words = {2: "बीस", 3: "तीस", 4: "चालीस", 5: "पचास",
                         6: "साठ", 7: "सत्तर", 8: "अस्सी", 9: "नब्बे"}
            if tens_digit in tens_words:
                words.append(tens_words[tens_digit])
            num %= 10
            
        # Teens
        elif num >= 10:
            teen_words = {10: "दस", 11: "ग्यारह", 12: "बारह", 13: "तेरह", 14: "चौदह",
                         15: "पंद्रह", 16: "सोलह", 17: "सत्रह", 18: "अठारह", 19: "उन्नीस"}
            if num in teen_words:
                words.append(teen_words[num])
            num = 0
        
        # Ones
        if num > 0:
            ones_words = ['', 'एक', 'दो', 'तीन', 'चार', 'पांच', 'छह', 'सात', 'आठ', 'नौ']
            if num < len(ones_words):
                words.append(ones_words[num])
        
        return words
    
    def _convert_number_to_hindi_words(self, num: int) -> List[str]:
        """Convert integer to Hindi words using proper Hindi number system."""
        words = []
        remaining = num
        
        # Lakhs and Crores (Hindi numbering system)
        if remaining >= 100000:
            lakh_part = remaining // 100000
            if lakh_part >= 100:
                crore_part = lakh_part // 100
                words.extend(self._convert_below_100(crore_part))
                words.append("करोड़")
                lakh_part %= 100
            
            if lakh_part > 0:
                words.extend(self._convert_below_100(lakh_part))
                words.append("लाख")
            
            remaining %= 100000
        
        # Thousands
        if remaining >= 1000:
            thousand_part = remaining // 1000
            words.extend(self._convert_below_100(thousand_part))
            words.append("हजार")
            remaining %= 1000
        
        # Hundreds, tens, and ones
        if remaining > 0:
            words.extend(self._convert_below_100(remaining))
        
        return words
        
    def get_all_possible_expansions(self, token: str) -> List[str]:
        """Generate ALL possible expansions for reference numbers."""
        expansions = [token]  # Always include original
        
        # For pure numbers, generate multiple expansion strategies
        if re.match(r'^\d+$', token):
            try:
                num = int(token)
                
                # Strategy 1: Standard Hindi compound
                hindi_words = self.convert_number_to_words(token)
                if hindi_words:
                    expansions.append(' '.join(hindi_words))  # Spaced
                    expansions.append(''.join(hindi_words))   # Joined
                
                # Strategy 2: English structure (for 10-999)
                if 10 <= num <= 999:
                    english_structure = self._get_english_structure_for_number(num)
                    if english_structure:
                        expansions.append(english_structure)
                
                # Strategy 3: Individual digits
                digit_words = [self.DIGIT_TO_HINDI[d] for d in token if d.isdigit()]
                if digit_words:
                    expansions.append(' '.join(digit_words))
                    
                # Strategy 4: All variants from mapping
                for word, mapped_digit in self.extended_number_words.items():
                    if mapped_digit == token and word not in expansions:
                        expansions.append(word)
                        
            except ValueError:
                pass
        
        # For number words, add reverse expansions
        elif token.lower() in self.word_to_number:
            digit = self.word_to_number[token.lower()]
            
            # Add numeric form
            expansions.append(digit)
            
            # Add other word variants
            for word, mapped_digit in self.extended_number_words.items():
                if mapped_digit == digit and word != token.lower():
                    expansions.append(word)
        
        return list(dict.fromkeys(expansions))  # Remove duplicates
        
    def _get_english_structure_for_number(self, num: int) -> str:
        """Convert number to English structure using transliterated words."""
        if not 1 <= num <= 9999:  # Extended range
            return ""
        
        parts = []
        
        # Thousands
        if num >= 1000:
            thousands_digit = num // 1000
            if 1 <= thousands_digit <= 9:
                thousands_transliterated = {1: 'वन', 2: 'टू', 3: 'थ्री', 4: 'फोर', 5: 'फाइव',
                                        6: 'सिक्स', 7: 'सेवन', 8: 'एट', 9: 'नाइन'}
                parts.append(thousands_transliterated[thousands_digit])
                parts.append('थाउजेंड')
            num %= 1000
        
        # Hundreds
        if num >= 100:
            hundreds_digit = num // 100
            if 1 <= hundreds_digit <= 9:
                hundreds_transliterated = {1: 'वन', 2: 'टू', 3: 'थ्री', 4: 'फोर', 5: 'फाइव',
                                        6: 'सिक्स', 7: 'सेवन', 8: 'एट', 9: 'नाइन'}
                parts.append(hundreds_transliterated[hundreds_digit])
                parts.append('हंड्रेड')  # Use the transliterated form
            num %= 100
        
        # Tens and ones (20-99)
        if num >= 20:
            tens_digit = num // 10
            tens_transliterated = {2: 'ट्वेंटी', 3: 'थर्टी', 4: 'फोर्टी', 5: 'फिफ्टी',
                                6: 'सिक्सटी', 7: 'सेवंटी', 8: 'एटी', 9: 'नाइंटी'}
            if tens_digit in tens_transliterated:
                parts.append(tens_transliterated[tens_digit])
            
            ones_digit = num % 10
            if ones_digit > 0:
                ones_transliterated = {1: 'वन', 2: 'टू', 3: 'थ्री', 4: 'फोर', 5: 'फाइव',
                                    6: 'सिक्स', 7: 'सेवन', 8: 'एट', 9: 'नाइन'}
                parts.append(ones_transliterated[ones_digit])
        
        # Teens (10-19)
        elif num >= 10:
            teens_transliterated = {10: 'टेन', 11: 'इलेवन', 12: 'ट्वेल्व', 13: 'थर्टीन', 
                                14: 'फोर्टीन', 15: 'फिफ्टीन', 16: 'सिक्सटीन', 17: 'सेवनटीन',
                                18: 'एटीन', 19: 'नाइनटीन'}
            if num in teens_transliterated:
                parts.append(teens_transliterated[num])
        
        # Ones (1-9)
        elif num >= 1:
            ones_transliterated = {1: 'वन', 2: 'टू', 3: 'थ्री', 4: 'फोर', 5: 'फाइव',
                                6: 'सिक्स', 7: 'सेवन', 8: 'एट', 9: 'नाइन'}
            parts.append(ones_transliterated[num])
        
        return ' '.join(parts) if parts else ""        

    def _expand_as_individual_digits(self, token: str) -> str:
        """Expand token treating each digit individually."""
        result_parts = []
        
        for char in token:
            if char.isdigit():
                result_parts.append(self.DIGIT_TO_HINDI[char])
            elif char in self.DEVANAGARI_TO_ARABIC:
                arabic_digit = self.DEVANAGARI_TO_ARABIC[char]
                result_parts.append(self.DIGIT_TO_HINDI[arabic_digit])
            elif char.isalpha() or self._is_devanagari(char):
                result_parts.append(char)
            elif char in self.symbol_words:
                result_parts.append(self.symbol_words[char][0])
        
        return ' '.join(result_parts) if result_parts else token