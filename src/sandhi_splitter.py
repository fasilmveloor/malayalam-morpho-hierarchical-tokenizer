"""
Sandhi Splitter for Malayalam

This module handles the splitting of compound words and sandhi (junction) rules
in Malayalam. Sandhi is a critical aspect of Malayalam morphology where words
combine at boundaries with various transformations.

Types of Sandhi in Malayalam:
1. സ്വരസന്ധി (Vowel Sandhi) - vowel combinations
2. വ്യഞ്ജനസന്ധി (Consonant Sandhi) - consonant combinations
3. വിസർഗ്ഗസന്ധി (Visarga Sandhi) - involving visarga

Example:
    പാലക്കാട് → പാല + കാട് (compound word)
    ഴ + ഞ → ഴ്ഞ (consonant transformation)
"""

import re
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SandhiRule:
    """Represents a sandhi transformation rule."""
    name: str
    pattern: str  # Regex pattern to detect
    split_pattern: str  # How to split the string
    left_transform: str  # Transformation for left part
    right_transform: str  # Transformation for right part
    examples: List[Tuple[str, str, str]]  # (input, left, right)


class SandhiSplitter:
    """Handles sandhi splitting for Malayalam text."""
    
    def __init__(self):
        self.rules = self._initialize_rules()
        self.common_compounds = self._initialize_common_compounds()
    
    def _initialize_rules(self) -> List[SandhiRule]:
        """Initialize sandhi transformation rules."""
        rules = [
            # Vowel Sandhi Rules
            SandhiRule(
                name="അ-അ സന്ധി (a-a sandhi)",
                pattern=r'(.*)അഅ(.*)',
                split_pattern=r'(.*)ആ(.*)',
                left_transform='അ',
                right_transform='',
                examples=[('വിദ്യാലയം', 'വിദ്യ', 'ആലയം')]
            ),
            
            # Consonant Sandhi Rules
            SandhiRule(
                name="അനുനാസിക സന്ധി (Anusvara sandhi)",
                pattern=r'(.*)ം(.*)',
                split_pattern=r'(.*)ം\s*(.*)',
                left_transform='ം',
                right_transform='',
                examples=[('അങ്ങനെ', 'അങ്ങ്', 'അനെ')]
            ),
            
            # Common consonant doubling
            SandhiRule(
                name="വ്യഞ്ജന ഇരട്ട (Consonant doubling)",
                pattern=r'(.*)്(.)(.*)',
                split_pattern=r'(.*)്\2(.*)',
                left_transform='്',
                right_transform='',
                examples=[('പഠിക്കുന്നു', 'പഠി', 'ക്കുന്നു')]
            ),
        ]
        return rules
    
    def _initialize_common_compounds(self) -> dict:
        """Initialize dictionary of common compound words."""
        return {
            # Place names
            'പാലക്കാട്': ['പാല', 'കാട്'],
            'തിരുവനന്തപുരം': ['തിരു', 'അനന്ത', 'പുരം'],
            'കോഴിക്കോട്': ['കോഴി', 'ക്കോട്'],
            'എറണാകുളം': ['എറണ', 'കുളം'],
            'ഇടുക്കി': ['ഇടു', 'ക്കി'],
            'പത്തനംതിട്ട': ['പത്തനം', 'തിട്ട'],
            
            # Common compounds
            'പാലം': ['പാൽ', 'അം'],  # bridge (milk + place)
            'വിദ്യാലയം': ['വിദ്യ', 'ആലയം'],  # school
            'പ്രധാനമന്ത്രി': ['പ്രധാന', 'മന്ത്രി'],  # prime minister
            'രക്തസമ്മർദ്ദം': ['രക്ത', 'സമ്മർദ്ദം'],  # blood pressure
            
            # Verb compounds
            'പഠിച്ചുകൊണ്ടിരിക്കുന്നു': ['പഠിച്ചു', 'കൊണ്ടിരിക്കുന്നു'],
            'വന്നുകൊണ്ടിരിക്കുന്നു': ['വന്നു', 'കൊണ്ടിരിക്കുന്നു'],
        }
    
    def split_compound(self, word: str) -> List[str]:
        """
        Split a compound word into its components.
        
        Args:
            word: The compound word to split
            
        Returns:
            List of component words
        """
        # Check common compounds dictionary first
        if word in self.common_compounds:
            return self.common_compounds[word]
        
        # Try rule-based splitting
        components = self._rule_based_split(word)
        if len(components) > 1:
            return components
        
        # Return original word if no split found
        return [word]
    
    def _rule_based_split(self, word: str) -> List[str]:
        """Apply sandhi rules to split a word."""
        components = []
        
        # Pattern 1: Words ending in കാട്, പുരം, കുളം etc. (place name suffixes)
        place_suffixes = ['കാട്', 'പുരം', 'കുളം', 'ത്ത്', 'പള്ളി', 'ശ്ശേരി']
        for suffix in place_suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                # Try to split
                prefix = word[:-len(suffix)]
                if len(prefix) >= 2:
                    # Check for connecting vowel
                    if prefix.endswith('്'):
                        # Remove virama and add proper vowel
                        prefix = prefix[:-1] + 'അ'
                    elif prefix.endswith('ാ'):
                        prefix = prefix[:-1] + 'ാ'
                    components = [prefix, suffix]
                    return components
        
        # Pattern 2: Compound verbs with കൊണ്ട്
        compound_verb_markers = ['കൊണ്ട്', 'കൊണ്ടിരിക്കുന്നു', 'വെച്ച്', 'കൊടുത്തു']
        for marker in compound_verb_markers:
            if marker in word:
                parts = word.split(marker, 1)
                if len(parts) == 2 and len(parts[0]) >= 2:
                    components = [parts[0], marker + parts[1]]
                    return components
        
        return [word]
    
    def is_compound(self, word: str) -> bool:
        """Check if a word is likely a compound word."""
        if word in self.common_compounds:
            return True
        
        # Check for compound indicators
        compound_indicators = [
            'കൊണ്ട്',  # instrumental compound
            'കൊണ്ടിരിക്കുന്നു',  # continuous action
            'പ്രധാന',  # main/chief (often in compounds)
        ]
        
        for indicator in compound_indicators:
            if indicator in word and len(word) > len(indicator) + 3:
                return True
        
        return False
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text into words and handle compounds.
        
        Args:
            text: Input text
            
        Returns:
            List of words with compounds split
        """
        # Basic word tokenization (split by spaces and punctuation)
        words = re.findall(r'[\u0D00-\u0D7F]+', text)
        
        result = []
        for word in words:
            if self.is_compound(word):
                components = self.split_compound(word)
                result.extend(components)
            else:
                result.append(word)
        
        return result


def detect_sandhi_boundary(word: str) -> Optional[int]:
    """
    Detect the position of a sandhi boundary in a word.
    
    Returns:
        Index of the boundary, or None if not found
    """
    # Look for patterns that indicate sandhi boundaries
    patterns = [
        r'്ക',  # virama + ka
        r'്ട',  # virama + Ta
        r'്ത',  # virama + ta
        r'്പ',  # virama + pa
        r'്മ',  # virama + ma
        r'ംപ',  # anusvara + pa
    ]
    
    for pattern in patterns:
        match = re.search(pattern, word)
        if match:
            return match.start() + 1
    
    return None
