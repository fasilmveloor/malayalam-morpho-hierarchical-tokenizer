"""
Hierarchical Vocabulary Manager for Malayalam Tokenizer

This module manages a hierarchical vocabulary structure where tokens are organized
by morphological type:
- Level 1 (1000-1999): Root words / Stems (Semantic Core)
- Level 2 (2000-2999): Tense/Aspect markers
- Level 3 (3000-3999): Case markers and postpositions
- Level 4 (4000-4999): Common function words (pronouns, particles)
- Level 5 (5000-5999): Infixes / Connective particles / Augments
- Level 6 (6000-6999): Subword fallback tokens
- Level 7 (7000+): Character-level tokens

Slot System:
- Slot 0: Roots (Base)
- Slot 1: Augments/Infixes (Middle)
- Slot 2: Tense/Suffix (End)

This prevents the model from hallucinating incorrect morpheme positions.
"""

import json
from typing import Dict, List, Set, Tuple
from collections import Counter
import re


class HierarchicalVocabulary:
    """Manages hierarchical vocabulary for Malayalam morphological tokenizer."""
    
    # Token type prefixes with Slot information
    TOKEN_RANGES = {
        'root': (1000, 1999, 0),      # Slot 0: Base
        'tense': (2000, 2999, 2),     # Slot 2: End
        'case': (3000, 3999, 2),      # Slot 2: End
        'function': (4000, 4999, 0),  # Slot 0: Can appear at start
        'infix': (5000, 5999, 1),     # Slot 1: Middle (NEW!)
        'conjunct': (5000, 5999, 1),  # Slot 1: Middle
        'subword': (6000, 6999, -1),  # No specific slot
        'special': (0, 99, -1),       # Special tokens
    }
    
    # Slot definitions for position validation
    SLOTS = {
        0: 'base',      # Roots, function words
        1: 'middle',    # Infixes, augments
        2: 'suffix',    # Tense, case markers
    }
    
    SPECIAL_TOKENS = {
        '<PAD>': 0,
        '<UNK>': 1,
        '<BOS>': 2,  # Beginning of sequence
        '<EOS>': 3,  # End of sequence
        '<ROOT>': 4,  # Marks start of root
        '<SUFFIX>': 5,  # Marks start of suffix sequence
        '<INFIX>': 6,  # Marks infix/augment
        '<SPACE>': 7,
    }
    
    def __init__(self):
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.token_type: Dict[int, str] = {}  # Maps token ID to its type
        self.token_slot: Dict[int, int] = {}  # Maps token ID to its slot position
        
        # Track next available ID for each type
        self.next_id = {
            'root': 1000,
            'tense': 2000,
            'case': 3000,
            'function': 4000,
            'infix': 5000,
            'conjunct': 5000,
            'subword': 6000,
            'special': 100,
        }
        
        # Initialize special tokens
        for token, token_id in self.SPECIAL_TOKENS.items():
            self.token_to_id[token] = token_id
            self.id_to_token[token_id] = token
            self.token_type[token_id] = 'special'
            self.token_slot[token_id] = -1
    
    def classify_morpheme(self, morpheme: str) -> str:
        """Classify a morpheme into its type based on linguistic rules."""
        
        # Infixes/Augments - These are grammatical bridges between roots and suffixes
        # Examples: ക്ക (kka) in പഠിക്കുന്നു, അ (a) vowel insertion
        infix_markers = [
            'ക്ക', 'ച്ച', 'ത്ത', 'പ്പ', 'ട്ട',  # Consonant doublings (transitivizers)
            'അ', 'ാ', 'ു', 'ൂ',  # Vowel insertions for sandhi
            'യ്', 'വ്',  # Glide insertions
            'ത്ത്', 'ട്ട്',  # Noun inflection markers
        ]
        
        # Common Malayalam tense/aspect suffixes
        tense_markers = [
            'ുന്നു', 'ുകയാണ്', 'ുക', 'ില്ല', 'ില്ലെ', 'ില്ലാ',  # Present/Negative
            'ിച്ചു', 'ിട്ടു', 'ുണ്ട്', 'ി', 'ന്നു',  # Past
            'ും', 'ാം', 'ിക്കാം', 'ണം',  # Future
            'ാൻ', 'ിക്കാൻ', 'ാനായി',  # Infinitive
            'ിയ', 'ിച്ച', 'ുന്ന',  # Relative participle
            'ുകയും', 'ിട്ടുണ്ട്',  # Conjunctive
        ]
        
        # Case markers and postpositions
        case_markers = [
            'ിൽ', 'ിലെ', 'ിലും', 'ിൽനിന്ന്',  # Locative
            'ിന്', 'ിന്റെ', 'ിനെ', 'ിനാൽ',  # Genitive/Accusative
            'ിനോട്', 'ിനോടൊപ്പം',  # Sociative
            'ിൽക്കൂടി', 'ിലൂടെ',  # Through
            'ിന്നും', 'ിനും',  # Ablative
            'ക്ക്', 'ക്കും', 'ക്കായി',  # Dative
            'ിൽവച്ച്', 'ിൽത്തന്നെ',
        ]
        
        # Common function words
        function_words = [
            'ഞാൻ', 'നീ', 'അവൻ', 'അവൾ', 'അവർ', 'ഞങ്ങൾ', 'നിങ്ങൾ', 'അവർ',  # Pronouns
            'ഇത്', 'അത്', 'അവ', 'ഇവ', 'എന്ത്', 'ആര്', 'എവിടെ',  # Demonstratives
            'എന്ന', 'എന്ന്', 'എന്നും', 'ആയ', 'ആയി',  # Connectors
            'ഉം', 'അല്ലെങ്കിൽ', 'പക്ഷേ', 'എന്നാൽ', 'അതിനാൽ',  # Conjunctions
            'വളരെ', 'കൂടുതൽ', 'കുറവ്', 'ഏറ്റവും', 'അധികം',  # Adverbs
            'ഇല്ല', 'വേണ്ട', 'പോലെ', 'പോലുള്ള',  # Negatives/similes
        ]
        
        # Check infixes first (they are typically short)
        if morpheme in infix_markers:
            return 'infix'
        
        # Check each category
        for marker in tense_markers:
            if morpheme.endswith(marker) or morpheme == marker:
                return 'tense'
        
        for marker in case_markers:
            if morpheme.endswith(marker) or morpheme == marker:
                return 'case'
        
        if morpheme in function_words:
            return 'function'
        
        # Check if it's a conjunct consonant pattern (single conjunct)
        if re.match(r'^[ക-ഹ][്][ക-ഹ]$', morpheme):
            return 'infix'  # Changed from conjunct to infix
        
        # Default to root
        return 'root'
    
    def add_token(self, token: str, token_type: str = None) -> int:
        """Add a token to the vocabulary and return its ID."""
        if token in self.token_to_id:
            return self.token_to_id[token]
        
        if token_type is None:
            token_type = self.classify_morpheme(token)
        
        # Check if we have room in this token range
        range_info = self.TOKEN_RANGES.get(token_type, (6000, 6999, -1))
        start, end, slot = range_info[0], range_info[1], range_info[2] if len(range_info) > 2 else -1
        
        if self.next_id[token_type] > end:
            # Overflow to subword range
            token_type = 'subword'
            slot = -1
        
        token_id = self.next_id[token_type]
        self.next_id[token_type] += 1
        
        self.token_to_id[token] = token_id
        self.id_to_token[token_id] = token
        self.token_type[token_id] = token_type
        self.token_slot[token_id] = slot
        
        return token_id
    
    def get_token_id(self, token: str) -> int:
        """Get the ID for a token, or <UNK> if not found."""
        return self.token_to_id.get(token, self.SPECIAL_TOKENS['<UNK>'])
    
    def get_token(self, token_id: int) -> str:
        """Get the token for an ID, or <UNK> if not found."""
        return self.id_to_token.get(token_id, '<UNK>')
    
    def get_token_type(self, token_id: int) -> str:
        """Get the type of a token by its ID."""
        return self.token_type.get(token_id, 'unknown')
    
    def build_from_corpus(self, morphemes: List[str], min_freq: int = 2) -> None:
        """Build vocabulary from a list of morphemes based on frequency."""
        counter = Counter(morphemes)
        
        for morpheme, freq in counter.most_common():
            if freq >= min_freq:
                self.add_token(morpheme)
    
    def get_vocabulary_size(self) -> int:
        """Return the total vocabulary size."""
        return len(self.token_to_id)
    
    def get_tokens_by_type(self, token_type: str) -> List[Tuple[str, int]]:
        """Get all tokens of a specific type."""
        return [
            (token, token_id) 
            for token, token_id in self.token_to_id.items()
            if self.token_type.get(token_id) == token_type
        ]
    
    def save(self, filepath: str) -> None:
        """Save vocabulary to a JSON file."""
        data = {
            'token_to_id': self.token_to_id,
            'id_to_token': {str(k): v for k, v in self.id_to_token.items()},
            'token_type': {str(k): v for k, v in self.token_type.items()},
            'next_id': self.next_id,
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load(self, filepath: str) -> None:
        """Load vocabulary from a JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.token_to_id = data['token_to_id']
        self.id_to_token = {int(k): v for k, v in data['id_to_token'].items()}
        self.token_type = {int(k): v for k, v in data['token_type'].items()}
        self.next_id = data['next_id']
    
    def __len__(self) -> int:
        return self.get_vocabulary_size()
    
    def __contains__(self, token: str) -> bool:
        return token in self.token_to_id
    
    def __repr__(self) -> str:
        type_counts = {}
        for token_type in self.TOKEN_RANGES.keys():
            type_counts[token_type] = len(self.get_tokens_by_type(token_type))
        return f"HierarchicalVocabulary(total={len(self)}, types={type_counts})"
