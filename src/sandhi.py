"""
Malayalam Sandhi Rules and Transformations

This module handles the phonological transformations (Sandhi) that occur
when morphemes combine in Malayalam.

Key Sandhi Types:
1. സ്വരസന്ധി (Vowel Sandhi) - When vowels combine
2. വ്യഞ്ജനസന്ധി (Consonant Sandhi) - When consonants combine
3. ലോപം (Lopa/Elision) - When sounds are dropped

Example Problem:
    പഠിക്ക (padikka) + ുന്നു (unnu) 
    → The 'a' at end of പഠിക്ക should be dropped (lopa)
    → Correct stem: പഠിക്ക് (padikk) + ുന്നു (unnu)
    → Result: പഠിക്കുന്നു ✅

Reference: https://ml.wikipedia.org/wiki/സന്ധി
"""

import re
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class SandhiRule:
    """Represents a sandhi transformation rule."""
    name: str
    description: str
    left_pattern: str  # Pattern at end of left morpheme
    right_pattern: str  # Pattern at start of right morpheme
    left_transform: str  # How to transform left
    right_transform: str  # How to transform right
    examples: List[Tuple[str, str, str]]  # (left, right, result)


class MalayalamSandhi:
    """
    Handles Malayalam sandhi transformations.
    
    Malayalam vowels:
    - അ (a), ആ (ā), ഇ (i), ഈ (ī), ഉ (u), ഊ (ū)
    - ഋ (ṛ), ൠ (ṝ), എ (e), ഏ (ē), ഐ (ai), ഒ (o), ഓ (ō), ഔ (au)
    
    Vowel signs (attached to consonants):
    - ാ (ā), ി (i), ീ (ī), ു (u), ൂ (ū)
    - ൃ (ṛ), െ (e), േ (ē), ൈ (ai), ൊ (o), ോ (ō), ൌ (au)
    
    Virama: ് (suppresses inherent vowel)
    """
    
    # Vowel characters and their signs
    VOWELS = {
        'അ': 'a', 'ആ': 'ā', 'ഇ': 'i', 'ഈ': 'ī', 
        'ഉ': 'u', 'ഊ': 'ū', 'ഋ': 'ṛ', 'ൠ': 'ṝ',
        'എ': 'e', 'ഏ': 'ē', 'ഐ': 'ai', 'ഒ': 'o', 
        'ഓ': 'ō', 'ഔ': 'au'
    }
    
    VOWEL_SIGNS = {
        'ാ': 'ā', 'ി': 'i', 'ീ': 'ī', 'ു': 'u', 'ൂ': 'ū',
        'ൃ': 'ṛ', 'െ': 'e', 'േ': 'ē', 'ൈ': 'ai', 
        'ൊ': 'o', 'ോ': 'ō', 'ൌ': 'au'
    }
    
    # Independent vowels (can appear at word start)
    INDEPENDENT_VOWELS = list(VOWELS.keys())
    
    # Dependent vowels (vowel signs, attach to consonants)
    DEPENDENT_VOWELS = list(VOWEL_SIGNS.keys())
    
    # Virama (cancels inherent vowel)
    VIRAMA = '്'
    
    # Anusvara (nasal)
    ANUSVARA = 'ം'
    
    # Visarga
    VISARGA = 'ഃ'
    
    # Consonants
    CONSONANTS = [
        'ക', 'ഖ', 'ഗ', 'ഘ', 'ങ',  # Velars
        'ച', 'ഛ', 'ജ', 'ഝ', 'ഞ',  # Palatals
        'ട', 'ഠ', 'ഡ', 'ഢ', 'ണ',  # Retroflex
        'ത', 'ഥ', 'ദ', 'ധ', 'ന',  # Dentals
        'പ', 'ഫ', 'ബ', 'ഭ', 'മ',  # Labials
        'യ', 'ര', 'ല', 'വ',       # Semivowels
        'ശ', 'ഷ', 'സ', 'ഹ',       # Sibilants
        'ള', 'ഴ', 'റ',             # Laterals
        'ക്ഷ', 'ത്ര', 'ജ്ഞ',       # Conjuncts used as consonants
    ]
    
    def __init__(self):
        self.rules = self._initialize_rules()
    
    def _initialize_rules(self) -> List[SandhiRule]:
        """Initialize sandhi transformation rules."""
        rules = [
            # 1. LOPA (Elision) - Vowel is dropped
            # അ-സന്ധി (a-sandhi): When 'a' meets another vowel, 'a' is dropped
            SandhiRule(
                name="അ-ലോപം (a-lopa)",
                description="When 'a' at end of first morpheme meets vowel at start of second, 'a' is dropped",
                left_pattern='അ$',  # Ends in 'a'
                right_pattern='^[അആഇഈഉഊഋൠഎഏഐഒഓഔാിീുൂൃെേൈൊോൌ]',  # Starts with vowel
                left_transform='്',  # Replace final 'a' with virama
                right_transform='',  # Keep right as is
                examples=[
                    ('പഠിക്ക', 'ുന്നു', 'പഠിക്കുന്നു'),
                    ('വര', 'ുന്നു', 'വരുന്നു'),
                    ('പോക', 'ുന്നു', 'പോകുന്നു'),
                ]
            ),
            
            # 2. ഇ-സന്ധി (i-sandhi): 'i' + vowel transformation
            SandhiRule(
                name="ഇ-സന്ധി (i-sandhi)",
                description="When 'i' meets another vowel",
                left_pattern='[ിഇ]$',  # Ends in 'i' or 'ി'
                right_pattern='^[അആഇഈഉഊഎഏഐഒഓഔാിീുൂെേൈൊോൌ]',
                left_transform='്',  # Simplified: replace with virama
                right_transform='',
                examples=[
                    ('വന്നി', 'അല്ല', 'വന്നില്ല'),  # Special case
                ]
            ),
            
            # 3. ഉ-സന്ധി (u-sandhi): 'u' + vowel transformation
            SandhiRule(
                name="ഉ-സന്ധി (u-sandhi)",
                description="When 'u' meets another vowel",
                left_pattern='[ുഉ]$',  # Ends in 'u' or 'ു'
                right_pattern='^[അആഇഈഉഊഎഏഐഒഓഔാിീുൂെേൈൊോൌ]',
                left_transform='്',
                right_transform='',
                examples=[
                    ('പഠിച്ചു', 'എന്നു', 'പഠിച്ചെന്നു'),  # Past + quotative
                ]
            ),
            
            # 4. ആ-സന്ധി (ā-sandhi): 'ā' + vowel
            SandhiRule(
                name="ആ-സന്ധി (ā-sandhi)",
                description="When 'ā' meets another vowel",
                left_pattern='ാ$',  # Ends in 'ā' sign
                right_pattern='^[അആഇഈഉഊഎഏഐഒഓഔാിീുൂെേൈൊോൌ]',
                left_transform='്',
                right_transform='',
                examples=[
                    ('വിദ്യാ', 'ആലയം', 'വിദ്യാലയം'),  # Compound
                ]
            ),
        ]
        return rules
    
    def get_final_vowel(self, text: str) -> Optional[Tuple[str, str]]:
        """
        Get the final vowel of a text if it ends with one.
        
        Returns:
            Tuple of (vowel_char, vowel_type) or None
            vowel_type: 'independent', 'dependent', 'inherent'
        """
        if not text:
            return None
        
        last_char = text[-1]
        
        # Check for independent vowel
        if last_char in self.INDEPENDENT_VOWELS:
            return (last_char, 'independent')
        
        # Check for vowel sign (dependent vowel)
        if last_char in self.DEPENDENT_VOWELS:
            return (last_char, 'dependent')
        
        # Check for virama (no vowel)
        if last_char == self.VIRAMA:
            return None
        
        # Check for anusvara
        if last_char == self.ANUSVARA:
            return None
        
        # If ends with consonant, it has inherent 'a'
        if any(text.endswith(c) for c in self.CONSONANTS):
            return ('അ', 'inherent')
        
        return None
    
    def get_initial_vowel(self, text: str) -> Optional[Tuple[str, str]]:
        """
        Get the initial vowel of a text if it starts with one.
        
        Returns:
            Tuple of (vowel_char, vowel_type) or None
        """
        if not text:
            return None
        
        first_char = text[0]
        
        # Check for independent vowel at start
        if first_char in self.INDEPENDENT_VOWELS:
            return (first_char, 'independent')
        
        # Check for vowel sign at start (rare but possible)
        if first_char in self.DEPENDENT_VOWELS:
            return (first_char, 'dependent')
        
        # If starts with consonant, check if there's a vowel sign following
        if first_char in self.CONSONANTS:
            if len(text) > 1 and text[1] in self.DEPENDENT_VOWELS:
                return (text[1], 'dependent')
            elif len(text) > 1 and text[1] == self.VIRAMA:
                return None  # Consonant cluster, no vowel
            else:
                return ('അ', 'inherent')  # Inherent 'a'
        
        return None
    
    def to_stem_form(self, word: str) -> str:
        """
        Convert a word to its stem form (ending with virama if applicable).
        
        This removes the final vowel to prepare for suffix attachment.
        
        Examples:
            പഠിക്ക (padikka) → പഠിക്ക് (padikk) - stem form
            വര (vara) → വര് (var) - stem form
            പോക (pōka) → പോക് (pōk) - stem form
        """
        if not word:
            return word
        
        # Check final vowel
        final_vowel = self.get_final_vowel(word)
        
        if final_vowel is None:
            # Already ends with virama or consonant cluster
            return word
        
        vowel_char, vowel_type = final_vowel
        
        if vowel_type == 'inherent':
            # Ends with consonant having inherent 'a'
            # Add virama to suppress the inherent vowel
            return word + self.VIRAMA
        
        elif vowel_type == 'dependent':
            # Ends with vowel sign, remove it and add virama
            return word[:-1] + self.VIRAMA
        
        elif vowel_type == 'independent':
            # Ends with independent vowel, replace with virama
            return word[:-1] + self.VIRAMA
        
        return word
    
    def apply_sandhi(self, left: str, right: str) -> str:
        """
        Apply sandhi rules to combine two morphemes.
        
        Args:
            left: Left morpheme (e.g., root)
            right: Right morpheme (e.g., suffix)
            
        Returns:
            Combined form after sandhi transformation
        """
        # Get vowels at boundary
        left_vowel = self.get_final_vowel(left)
        right_vowel = self.get_initial_vowel(right)
        
        # Case 1: Left ends with virama - no transformation needed
        if left.endswith(self.VIRAMA):
            # Remove virama if right starts with vowel sign
            if right_vowel and right_vowel[1] == 'dependent':
                # Left ends with virama, right starts with vowel sign
                # Just concatenate (virama is replaced by vowel sign)
                return left[:-1] + right
            elif right and right[0] in self.INDEPENDENT_VOWELS:
                # Left ends with virama, right starts with independent vowel
                # Rare case, just concatenate
                return left + right
            else:
                # Normal case: stem + suffix starting with consonant
                return left + right
        
        # Case 2: Both have vowels - apply lopa (elision)
        if left_vowel and right_vowel:
            # Default: drop left's final vowel (lopa)
            # Convert left to stem form
            left_stem = self.to_stem_form(left)
            
            # If right starts with vowel sign, remove virama and join
            if right_vowel[1] == 'dependent':
                return left_stem[:-1] + right  # Remove virama, add suffix
            else:
                return left_stem + right  # Keep virama
        
        # Case 3: Only left has vowel, right starts with consonant
        if left_vowel and not right_vowel:
            # No sandhi needed, just join
            return left + right
        
        # Case 4: Left ends with consonant (inherent a), right starts with vowel
        if left_vowel and left_vowel[1] == 'inherent' and right_vowel:
            # Convert to stem form
            left_stem = self.to_stem_form(left)
            if right_vowel[1] == 'dependent':
                return left_stem[:-1] + right
            else:
                return left_stem + right
        
        # Default: just concatenate
        return left + right
    
    def split_morphemes_properly(self, word: str, root: str, suffix: str) -> Tuple[str, str]:
        """
        Given a word and its proposed root+suffix split, return proper morphemes
        with correct sandhi handling.
        
        This is used when mlmorph gives us a root that doesn't account for sandhi.
        
        Args:
            word: Original word (e.g., പഠിക്കുന്നു)
            root: Proposed root from analyzer (e.g., പഠിക്കുക)
            suffix: Proposed suffix (e.g., ന്നു)
            
        Returns:
            Tuple of (proper_root, proper_suffix)
        """
        # If the root is already in stem form (ends with virama), return as is
        if root.endswith(self.VIRAMA):
            return (root, suffix)
        
        # Check if word starts with root (without the final vowel)
        root_stem = self.to_stem_form(root)
        
        # Try to find where the split should be
        # The suffix in the word should start where the stem ends
        if word.startswith(root_stem[:-1] if root_stem.endswith(self.VIRAMA) else root_stem):
            # Good split point found
            suffix_in_word = word[len(root_stem)-1:] if root_stem.endswith(self.VIRAMA) else word[len(root_stem):]
            return (root_stem, suffix_in_word)
        
        # Fallback: try to find the split by looking at the word
        # Find where suffix starts
        if suffix and word.endswith(suffix):
            split_point = len(word) - len(suffix)
            proper_root = word[:split_point]
            # Ensure proper root ends with virama if it's a stem
            if proper_root and not proper_root.endswith(self.VIRAMA):
                proper_root = self.to_stem_form(proper_root)
            return (proper_root, suffix)
        
        # Return original if we can't determine proper split
        return (root, suffix)


def get_proper_root_suffix(word: str, analyzer_root: str) -> Tuple[str, str]:
    """
    Convenience function to get proper root-suffix split.
    
    Args:
        word: Original word (പഠിക്കുന്നു)
        analyzer_root: Root from morphological analyzer (പഠിക്കുക)
        
    Returns:
        (proper_stem, suffix) where stem ends with virama
    """
    sandhi = MalayalamSandhi()
    
    # The analyzer root is usually the infinitive form
    # Convert to stem form
    stem = sandhi.to_stem_form(analyzer_root)
    
    # Find where the stem appears in the word
    stem_base = stem[:-1] if stem.endswith('്') else stem
    
    if word.startswith(stem_base):
        # Found the stem at the start
        suffix = word[len(stem_base):]
        return (stem, suffix)
    
    # Try alternative approach: look for common suffix patterns
    common_suffixes = [
        'ുന്നു', 'ിക്കുന്നു', 'ച്ചു', 'ിട്ടു', 'ും', 'ാൻ', 
        'ിയ', 'ുന്ന', 'ിൽ', 'ിന്', 'ിന്റെ', 'ിലെ',
    ]
    
    for suffix in common_suffixes:
        if word.endswith(suffix):
            root_part = word[:-len(suffix)]
            if root_part:
                # Ensure root ends with virama if it's a consonant
                final_vowel = sandhi.get_final_vowel(root_part)
                if final_vowel and final_vowel[1] == 'inherent':
                    root_part = root_part + '്'
                return (root_part, suffix)
    
    # Fallback
    return (stem, word[len(stem_base):] if word.startswith(stem_base) else '')
