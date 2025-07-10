#!/usr/bin/env python3
"""
Complete Hindi-specific CER-based word aligner
"""

import re
from typing import List
from karma.metrics.asr.base_aligner import BaseCERAligner

class HindiCERAligner(BaseCERAligner):
    """Hindi-specific CER-based word aligner with English transliteration support."""
    
    def _initialize_language_specific_mappings(self):
        """Initialize Hindi-specific mappings."""
        
        # Hindi number mappings (Devanagari numerals and words)
        self.devanagari_numbers = {
            '०': ['शून्य', 'zero', 'जीरो'],
            '१': ['एक', 'one', 'वन'],
            '२': ['दो', 'two', 'टू'],
            '३': ['तीन', 'three', 'थ्री'],
            '४': ['चार', 'four', 'फोर'],
            '५': ['पांच', 'five', 'फाइव'],
            '६': ['छह', 'six', 'सिक्स'],
            '७': ['सात', 'seven', 'सेवन'],
            '८': ['आठ', 'eight', 'एट'],
            '९': ['नौ', 'nine', 'नाइन'],
        }
        
        # Arabic numerals to Hindi + English transliterated words
        self.arabic_to_hindi = {
            '0': ['शून्य', 'zero', 'जीरो'],
            '1': ['एक', 'one', 'वन'],
            '2': ['दो', 'two', 'टू'],
            '3': ['तीन', 'three', 'थ्री'],
            '4': ['चार', 'four', 'फोर'],
            '5': ['पांच', 'five', 'फाइव'],
            '6': ['छह', 'six', 'सिक्स'],
            '7': ['सात', 'seven', 'सेवन'],
            '8': ['आठ', 'eight', 'एट'],
            '9': ['नौ', 'nine', 'नाइन'],
        }
        
        # Extended number words (Hindi + English + transliterated)
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
            
            # Teens (Hindi + English)
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
            
            # Tens (Hindi + English)
            'बीस': '20', 'twenty': '20', 'ट्वेंटी': '20', 'ट्वेन्टी': '20',
            'तीस': '30', 'thirty': '30', 'थर्टी': '30',
            'चालीस': '40', 'forty': '40', 'फोर्टी': '40',
            'पचास': '50', 'fifty': '50', 'फिफ्टी': '50',
            'साठ': '60', 'sixty': '60', 'सिक्सटी': '60',
            'सत्तर': '70', 'seventy': '70', 'सेवंटी': '70', 'सेवेंटी': '70',
            'अस्सी': '80', 'eighty': '80', 'एटी': '80', 'एइटी': '80',
            'नब्बे': '90', 'ninety': '90', 'नाइंटी': '90', 'नाइनटी': '90',
            
            # Place values (Hindi + English)
            'सौ': '100', 'hundred': '100', 'हंड्रेड': '100', 'हण्ड्रेड': '100',
            'हजार': '1000', 'thousand': '1000', 'थाउजेंड': '1000', 'थाउज़ेंड': '1000',
            'लाख': '100000', 'lakh': '100000', 'लाक': '100000',
            'करोड़': '10000000', 'crore': '10000000', 'क्रोर': '10000000',
            
            # English-only place values that Indians use
            'million': '1000000', 'मिलियन': '1000000',
            'billion': '1000000000', 'बिलियन': '1000000000',
        }
        
        # Hindi symbol words (including English transliterations)
        self.symbol_words = {
            '.': ['बिंदु', 'पॉइंट', 'dot', 'point', 'डॉट', ''],
            '।': ['दंड', 'पूर्ण विराम', 'दण्ड', 'विराम', 'दंडा', ''], # Hindi danda (।)
            '/': ['या', 'स्लैश', 'बटा', 'पर', 'or', 'slash', 'over', 'by', 'ओर'],
            '-': ['डैश', 'हाइफन', 'माइनस', 'dash', 'hyphen', 'minus', 'हाइफ़न', 'माइनस'],
            '+': ['प्लस', 'जोड़', 'plus', 'प्लस'],
            '%': ['प्रतिशत', 'फीसदी', 'percent', 'परसेंट', 'फ़ीसदी'],
            '*': ['तारा', 'स्टार', 'star', 'स्टार'],
            '=': ['बराबर', 'equals', 'इक्वल', 'इक्वल्स'],
            '&': ['और', 'एंड', 'and', 'एण्ड']
        }

        # Unit abbreviations (separate mapping for multi-character units)
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
        
        # Build reverse mappings
        self.word_to_number = {}
        self.word_to_symbol = {}
        self.word_to_unit = {}
 
        # Add all number word mappings (Hindi + English + transliterated)
        for word, digit in self.extended_number_words.items():
            self.word_to_number[word.lower()] = digit
                
        for symbol, words in self.symbol_words.items():
            for word in words:
                self.word_to_symbol[word.lower()] = symbol

        for unit, words in self.unit_words.items():
            for word in words:
                self.word_to_unit[word.lower()] = unit
    
    def _is_devanagari(self, char: str) -> bool:
        """Check if character is in Devanagari script range."""
        return '\u0900' <= char <= '\u097F'
    
    def normalize_text_semantically(self, text: str) -> str:
        """Normalize text for semantic comparison - Hindi version."""
        text = text.lower().strip()
        
        # Handle digit/symbol sequences (both Devanagari and Arabic numerals)
        if re.search(r'[\d०-९\./\-+%*=&]', text):
            normalized_parts = []
            current_number = ""
            current_alpha = ""
            
            for char in text:
                if char.isdigit() or char in '०१२३४५६७८९':
                    # Process any accumulated alphabetic characters first
                    if current_alpha:
                        normalized_parts.append(current_alpha)
                        current_alpha = ""
                    current_number += char
                elif char.isalpha() or self._is_devanagari(char):
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
                    # Add symbol word (prefer Hindi)
                    normalized_parts.append(self.symbol_words[char][0])
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
        
        # Handle word sequences - just keep alphabetic/Devanagari chars
        words = text.split()
        normalized_parts = []
        
        for word in words:
            # Keep alphabetic and Devanagari characters only
            word = word.replace('।', '').replace('॥', '') # Remove Hindi punctuation first
            clean_word = ''.join(c for c in word if c.isalpha() or self._is_devanagari(c))
            if clean_word:
                normalized_parts.append(clean_word)
        
        return ''.join(normalized_parts)
    
    def expand_ref_token_semantically(self, token: str) -> str:
        """Expand a reference token to its semantic word form - Hindi version."""
        # Handle tokens with digits (Arabic or Devanagari)
        if re.search(r'[\d०-९]', token):
            result_parts = []
            current_number = ""
            current_alpha = ""
            
            for char in token:
                if char.isdigit() or char in '०१२३४५६७८९':
                    # Process any accumulated alphabetic characters first
                    if current_alpha:
                        # Check if it's a unit abbreviation
                        if current_alpha.lower() in self.word_to_unit:
                            result_parts.append(self.unit_words[self.word_to_unit[current_alpha.lower()]][0])  # Use Hindi form
                        else:
                            result_parts.append(current_alpha.lower())
                        current_alpha = ""
                    current_number += char
                elif char.isalpha() or self._is_devanagari(char):
                    # Process accumulated number first
                    if current_number:
                        number_words = self.convert_number_to_words(current_number)
                        result_parts.extend(number_words)
                        current_number = ""
                    current_alpha += char
                else:
                    # Process accumulated number and alpha
                    if current_number:
                        number_words = self.convert_number_to_words(current_number)
                        result_parts.extend(number_words)
                        current_number = ""
                    if current_alpha:
                        # Check if it's a unit abbreviation
                        if current_alpha.lower() in self.word_to_unit:
                            result_parts.append(self.unit_words[self.word_to_unit[current_alpha.lower()]][0])  # Use Hindi form
                        else:
                            result_parts.append(current_alpha.lower())
                        current_alpha = ""
                    
                    # Process symbol character
                    if char in self.symbol_words:
                        result_parts.append(self.symbol_words[char][0])
            
            # Process any remaining number or alpha at the end
            if current_number:
                number_words = self.convert_number_to_words(current_number)
                result_parts.extend(number_words)
            if current_alpha:
                # Check if it's a unit abbreviation
                if current_alpha.lower() in self.word_to_unit:
                    result_parts.append(self.unit_words[self.word_to_unit[current_alpha.lower()]][0])  # Use Hindi form
                else:
                    result_parts.append(current_alpha.lower())            
            if result_parts:
                expanded = ' '.join(result_parts)
                if expanded != token:
                    return expanded

        # Check if token is a standalone unit abbreviation
        elif token.lower() in self.word_to_unit:
            return self.unit_words[self.word_to_unit[token.lower()]][0]  # Return Hindi form
        
        # Check if token contains symbols that should be expanded
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
        """Convert a number string to Hindi word form - with English transliterations."""
        if not number_str:
            return ["शून्य"]
        
        # Convert Devanagari digits to Arabic for processing
        arabic_number = ""
        for char in number_str:
            if char in '०१२३४५६७८९':
                devanagari_to_arabic = {'०': '0', '१': '1', '२': '2', '३': '3', '४': '4', 
                                      '५': '5', '६': '6', '७': '7', '८': '8', '९': '9'}
                arabic_number += devanagari_to_arabic[char]
            else:
                arabic_number += char
        
        try:
            num = int(arabic_number)
        except ValueError:
            # If not a valid number, fall back to digit-by-digit
            words = []
            for digit in arabic_number:
                if digit.isdigit() and digit in self.arabic_to_hindi:
                    words.append(self.arabic_to_hindi[digit][0])  # Use Hindi word first
                elif digit in '०१२३४५६७८९':
                    words.append(self.devanagari_numbers[digit][0])  # Use Hindi word first
            return words
            
        if num == 0:
            return ["शून्य"]
        
        # Extended range: Handle up to 99,99,999 (99 lakh 99 thousand 999)
        if num > 9999999:  # Above 99 lakhs, fall back to digit-by-digit
            words = []
            for digit in str(num):
                if digit in self.arabic_to_hindi:
                    words.append(self.arabic_to_hindi[digit][0])  # Use Hindi word
            return words
        
        words = []
        remaining = num
        
        # Lakhs (100,000s) - Hindi numbering system
        if remaining >= 100000:
            lakh_part = remaining // 100000
            if lakh_part >= 100:
                # Handle "करोड़" (crore) for very large numbers
                crore_part = lakh_part // 100
                words.extend(self._convert_below_100(crore_part))
                words.append("करोड़")
                lakh_part %= 100
            
            if lakh_part > 0:
                words.extend(self._convert_below_100(lakh_part))
                words.append("लाख")
            
            remaining %= 100000
        
        # Thousands (1,000s)
        if remaining >= 1000:
            thousand_part = remaining // 1000
            words.extend(self._convert_below_100(thousand_part))
            words.append("हजार")
            remaining %= 1000
        
        # Hundreds, tens, and ones
        if remaining > 0:
            words.extend(self._convert_below_100(remaining))
        
        return words
    
    def _convert_below_100(self, num: int) -> List[str]:
        """Convert numbers below 100 to Hindi words - returns primary Hindi forms."""
        if num == 0:
            return []
        
        words = []
        
        # Hundreds
        if num >= 100:
            hundreds_digit = num // 100
            digit_words = ['', 'एक', 'दो', 'तीन', 'चार', 'पांच', 'छह', 'सात', 'आठ', 'नौ']
            words.append(digit_words[hundreds_digit])
            words.append("सौ")
            num %= 100
        
        # Special cases for Hindi numbers (compound forms)
        if num >= 21 and num <= 99:
            # Hindi has special compound forms for many two-digit numbers
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
        
        # Tens (for 20, 30, 40, etc.)
        if num >= 20:
            tens_digit = num // 10
            tens_words = {2: "बीस", 3: "तीस", 4: "चालीस", 5: "पचास",
                         6: "साठ", 7: "सत्तर", 8: "अस्सी", 9: "नब्बे"}
            if tens_digit in tens_words:
                words.append(tens_words[tens_digit])
            num %= 10
            
        elif num >= 10:
            # Teens
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