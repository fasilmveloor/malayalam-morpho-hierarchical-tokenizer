"""
Hierarchical Vocabulary Manager for Malayalam Tokenizer
========================================================

This module manages a hierarchical vocabulary structure where tokens are organized
by morphological type using a slot-based ID assignment system.

Slot System:
- Slot 0: Roots/Base morphemes (can appear at start)
- Slot 1: Infixes/Augments (middle position)
- Slot 2: Suffixes (end position)

This prevents the model from generating invalid morpheme sequences.

Version: 0.9.0-rc (with review fixes)
"""

import json
import logging
import os
from typing import Dict, List, Set, Tuple, Optional
from collections import Counter
from pathlib import Path
import re

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Console handler for development
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


class HierarchicalVocabulary:
    """
    Manages hierarchical vocabulary for Malayalam morphological tokenizer.
    
    Vocabulary Organization:
    - Special tokens: 0-999
    - Roots: 1000-29999 (29,000 slots)
    - Tense markers: 30000-35999 (6,000 slots)
    - Case markers: 36000-41999 (6,000 slots)
    - Function words: 42000-44999 (3,000 slots)
    - Infixes: 45000-47999 (3,000 slots)
    - Subword: 48000-59999 (12,000 slots)
    - Reserved: 60000+
    
    Root Slot Allocation (29,000 slots):
    The "root" category includes all base morphemes that can appear at the start
    of a morphological sequence, not just native Dravidian roots:
    
        - Native Dravidian roots: ~15,000-20,000
          (e.g., പാടുക, എഴുതുക, നടക്കുക)
        
        - Sanskrit tatsama (borrowed roots): ~5,000-8,000
          (e.g., പ്രകാരം, സമയം, വിദ്യ, ജ്ഞാനം)
        
        - Modern loanwords: ~3,000-5,000
          English: സ്കൂൾ (school), ബൾബ് (bulb), കമ്പ്യൂട്ടർ (computer)
          Arabic/Persian: മസ്ജിദ്, ദുനിയ, കിതാബ്
          Portuguese: മേശ (mesa), പീടിക, കസേര
        
        - Proper nouns & technical terms: ~2,000-3,000
          (names, place names, scientific terminology)
        
    This design reflects the reality of modern Malayalam vocabulary, which is
    a rich blend of Dravidian, Sanskrit, and global loanwords. For a practical
    morphological tokenizer, we need slots for ALL base words that appear in
    real text, regardless of etymological origin.
    """
    
    # Expanded token ranges for production scale
    # Format: (start_id, end_id, slot_position)
    TOKEN_RANGES = {
        'special': (0, 999, -1),        # Special tokens
        'root': (1000, 29999, 0),       # Slot 0: Base - 29,000 slots
        'tense': (30000, 35999, 2),     # Slot 2: End - 6,000 slots
        'case': (36000, 41999, 2),      # Slot 2: End - 6,000 slots
        'function': (42000, 44999, 0),  # Slot 0: Can appear at start - 3,000 slots
        'infix': (45000, 47999, 1),     # Slot 1: Middle - 3,000 slots
        'conjunct': (48000, 49999, 1),  # Slot 1: Middle - 2,000 slots
        'subword': (50000, 59999, -1),  # No specific slot - 10,000 slots
        'reserved': (60000, 65535, -1), # Reserved for future use
    }
    
    # Slot definitions for position validation
    SLOTS = {
        0: 'base',      # Roots, function words
        1: 'middle',    # Infixes, augments
        2: 'suffix',    # Tense, case markers
    }
    
    # Special tokens with reserved IDs
    SPECIAL_TOKENS = {
        '<PAD>': 0,
        '<UNK>': 1,
        '<BOS>': 2,      # Beginning of sequence
        '<EOS>': 3,      # End of sequence
        '<ROOT>': 4,     # Marks start of root
        '<SUFFIX>': 5,   # Marks start of suffix sequence
        '<INFIX>': 6,    # Marks infix/augment
        '<SPACE>': 7,
        '<SEP>': 8,      # Separator for compounds
        '<MASK>': 9,     # For masked language modeling
    }
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize the vocabulary manager.
        
        Args:
            data_dir: Optional path to data directory for loading suffix lists
        """
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.token_type: Dict[int, str] = {}
        self.token_slot: Dict[int, int] = {}
        self.data_dir = Path(data_dir) if data_dir else None
        
        # Track next available ID for each type
        self.next_id = {
            'special': 100,
            'root': 1000,
            'tense': 30000,
            'case': 36000,
            'function': 42000,
            'infix': 45000,
            'conjunct': 48000,
            'subword': 50000,
            'reserved': 60000,
        }
        
        # Load suffix lists from file or use defaults
        self._load_suffix_lists()
        
        # Initialize special tokens
        self._init_special_tokens()
        
        logger.info(f"Initialized HierarchicalVocabulary with ranges: {self._get_range_summary()}")
    
    def _init_special_tokens(self) -> None:
        """Initialize special tokens in vocabulary."""
        for token, token_id in self.SPECIAL_TOKENS.items():
            self.token_to_id[token] = token_id
            self.id_to_token[token_id] = token
            self.token_type[token_id] = 'special'
            self.token_slot[token_id] = -1
    
    def _get_range_summary(self) -> str:
        """Get a summary of token ranges."""
        parts = []
        for token_type, (start, end, slot) in self.TOKEN_RANGES.items():
            capacity = end - start + 1
            parts.append(f"{token_type}: {capacity:,}")
        return ", ".join(parts)
    
    def _load_suffix_lists(self) -> None:
        """Load suffix lists from file or use comprehensive defaults."""
        
        # Try to load from data directory
        if self.data_dir:
            suffix_file = self.data_dir / 'suffix_lists.json'
            if suffix_file.exists():
                try:
                    with open(suffix_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    self.infix_markers = data.get('infix_markers', [])
                    self.tense_markers = data.get('tense_markers', [])
                    self.case_markers = data.get('case_markers', [])
                    self.function_words = data.get('function_words', [])
                    logger.info(f"Loaded suffix lists from {suffix_file}")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load suffix lists: {e}. Using defaults.")
        
        # Comprehensive default lists
        self._init_default_suffix_lists()
    
    def _init_default_suffix_lists(self) -> None:
        """Initialize comprehensive default suffix lists."""
        
        # Infixes/Augments - Grammatical bridges between roots and suffixes
        self.infix_markers = [
            # Consonant doublings (transitivizers)
            'ക്ക', 'ച്ച', 'ത്ത', 'പ്പ', 'ട്ട', 'ക്ക്', 'ച്ച്', 'ത്ത്', 'പ്പ്', 'ട്ട്',
            'ഗ്ഗ', 'ദ്ദ', 'ബ്ബ', 'മ്മ', 'ന്ന', 'ല്ല', 'വ്വ',
            # Vowel insertions for sandhi
            'അ', 'ാ', 'ി', 'ീ', 'ു', 'ൂ', 'ൃ', 'െ', 'േ', 'ൈ', 'ൊ', 'ോ', 'ൌ',
            # Glide insertions
            'യ്', 'വ്', 'യ', 'വ',
            # Noun inflection markers (ം → ത്ത് transformation)
            'ത്ത്', 'ട്ട്', 'ക്ക്',
            # Sandhi connectors
            '്', '്യ', '്വ',
        ]
        
        # Comprehensive tense/aspect suffixes
        self.tense_markers = [
            # Present tense
            'ുന്നു', 'ുന്ന', 'കുന്നു', 'കുന്ന', 'യുന്നു', 'യുന്ന',
            'ുകയാണ്', 'ുകയായിരുന്നു', 'ുകെ', 'ുന്നത്',
            # Past tense
            'ി', 'ു', 'ച്ചു', 'ത്തു', 'ന്നു', 'ണ്ടു', 'ട്ടു', 'പ്പു',
            'ിച്ചു', 'ിത്തു', 'ിന്നു', 'ിട്ടു', 'ിപ്പു',
            'ുണ്ടായി', 'ുണ്ടായിരുന്നു',
            # Future tense
            'ും', 'ാം', 'ിക്കും', 'ിക്കാം', 'കും', 'കാം', 'യും', 'യാം',
            # Imperative/Necessity
            'ണം', 'ിക്കണം', 'ാണം', 'കണം', 'വേണം',
            # Infinitive
            'ാൻ', 'ിക്കാൻ', 'ിക്കുവാൻ', 'കാൻ', 'യാൻ',
            'ാനായി', 'ിക്കാനായി',
            # Relative participle
            'ിയ', 'ിച്ച', 'ുന്ന', 'ുന്നത്', 'ിട്ട', 'ിട്ടുള്ള',
            'ാത്ത', 'ില്ലാത്ത', 'ാത',
            # Negative
            'ില്ല', 'ില്ലെ', 'ില്ലാ', 'ാത്ത', 'ാതെ', 'ാതും', 'ിട്ടില്ല',
            'ുന്നില്ല', 'ിക്കുന്നില്ല',
            # Conjunctive participles
            'ുകയും', 'ിട്ടും', 'ുമ്പോൾ', 'ുമ്പോൾത്തന്നെ',
            # Verbal nouns
            'ൽ', 'കൽ', 'ിക്കൽ', 'ിപ്പ്', 'ിക്കുക', 'ിക്കൽ',
            # Conditional
            'ിയാൽ', 'ുകയാണെങ്കിൽ', 'െങ്കിൽ', 'ാൽ',
        ]
        
        # Comprehensive case markers and postpositions
        self.case_markers = [
            # Locative (ിൽ variants)
            'ിൽ', 'ിലെ', 'ിലും', 'ിൽനിന്ന്', 'ിൽക്കൂടി', 'ിലൂടെ', 'ിൽവച്ച്', 'ിൽത്തന്നെ',
            'ത്തിൽ', 'ത്തിലെ', 'ത്തിലും', 'ട്ടിൽ', 'ട്ടിലെ',
            # Genitive (ിന്റെ variants)
            'ിന്റെ', 'ിന്റെയും', 'ന്റെ', 'ന്റെയും', 'ുടെ', 'ുടെയും',
            'യുടെ', 'യുടെയും', 'ത്തിന്റെ', 'ട്ടിന്റെ',
            # Accusative (ിനെ variants)
            'ിനെ', 'ിനേ', 'നെ', 'നേ', 'യെ', 'യേ',
            # Dative (ക്ക് variants)
            'ിന്', 'ിനും', 'ിനാൽ', 'ിനായി', 'ിനോട്', 'ിനോടൊപ്പം',
            'ക്ക്', 'ക്കും', 'ക്കായി', 'ക്കാനായി',
            # Ablative
            'ിൽനിന്ന്', 'ിൽനിന്നും', 'ിൽക്കൂടി', 'നിന്ന്', 'നിന്നും',
            # Sociative
            'ിനോടൊപ്പം', 'ോടൊപ്പം', 'ോടൊന്നിച്ച്', 'ോടൊപ്പംത്തന്നെ',
            # Instrumental
            'ിനാൽ', 'കൊണ്ട്', 'കൊണ്ടും', 'മൂലം', 'മൂലമായി',
            # Vocative
            'േ', 'ാരേ', 'ൻേ', 'ിലേ',
        ]
        
        # Common function words
        self.function_words = [
            # Personal pronouns
            'ഞാൻ', 'നീ', 'അവൻ', 'അവൾ', 'അവർ', 'അത്', 'അവ', 'ഇവ',
            'ഞങ്ങൾ', 'നിങ്ങൾ', 'അവർ', 'താൻ', 'താൻതന്നെ',
            # Demonstratives
            'ഇത്', 'അത്', 'അവ', 'ഇവ', 'എന്ത്', 'ആര്', 'എവിടെ', 'എപ്പോൾ',
            'ഇവിടെ', 'അവിടെ', 'ഇങ്ങനെ', 'അങ്ങനെ',
            # Interrogatives
            'എന്ത്', 'എവിടെ', 'എപ്പോൾ', 'എങ്ങനെ', 'എന്തുകൊണ്ട്', 'ആര്', 'ആരെ',
            # Connectors and conjunctions
            'എന്ന', 'എന്ന്', 'എന്നും', 'ആയ', 'ആയി',
            'ഉം', 'അല്ലെങ്കിൽ', 'പക്ഷേ', 'എന്നാൽ', 'അതിനാൽ', 'അതുകൊണ്ട്',
            'പിന്നെ', 'എന്നാൽ', 'ഇനി', 'അതിനുശേഷം', 'മാത്രം',
            # Adverbs
            'വളരെ', 'കൂടുതൽ', 'കുറവ്', 'ഏറ്റവും', 'അധികം', 'ഇന്ന്', 'ഇന്നത്തെ',
            'നാളെ', 'ഇപ്പോൾ', 'അപ്പോൾ', 'എപ്പോഴും', 'എപ്പോഴും',
            # Negatives and conditionals
            'ഇല്ല', 'വേണ്ട', 'പോലെ', 'പോലുള്ള', 'ഇല്ലാതെ', 'ഇല്ലാത്ത',
            # Emphatics
            'തന്നെ', 'മാത്രം', 'പോലും', 'പോലും', 'ഉണ്ട്', 'ആണ്',
        ]
        
        logger.debug(f"Loaded {len(self.infix_markers)} infix markers, "
                    f"{len(self.tense_markers)} tense markers, "
                    f"{len(self.case_markers)} case markers, "
                    f"{len(self.function_words)} function words")
    
    def save_suffix_lists(self, filepath: str) -> None:
        """Save current suffix lists to a JSON file."""
        data = {
            'infix_markers': self.infix_markers,
            'tense_markers': self.tense_markers,
            'case_markers': self.case_markers,
            'function_words': self.function_words,
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved suffix lists to {filepath}")
    
    def classify_morpheme(self, morpheme: str) -> str:
        """
        Classify a morpheme into its type based on linguistic rules.
        
        Args:
            morpheme: The morpheme string to classify
            
        Returns:
            Classification string: 'infix', 'tense', 'case', 'function', or 'root'
        """
        if not morpheme:
            logger.warning("Empty morpheme passed to classify_morpheme")
            return 'root'
        
        # Check infixes first (they are typically short and specific)
        if morpheme in self.infix_markers:
            return 'infix'
        
        # Check each category
        for marker in self.tense_markers:
            if morpheme.endswith(marker) or morpheme == marker:
                return 'tense'
        
        for marker in self.case_markers:
            if morpheme.endswith(marker) or morpheme == marker:
                return 'case'
        
        if morpheme in self.function_words:
            return 'function'
        
        # Check if it's a conjunct consonant pattern (single conjunct)
        if re.match(r'^[ക-ഹ][്][ക-ഹ]$', morpheme):
            return 'infix'
        
        # Default to root
        return 'root'
    
    def add_token(self, token: str, token_type: str = None) -> int:
        """
        Add a token to the vocabulary and return its ID.
        
        Args:
            token: The token string to add
            token_type: Optional explicit type; auto-detected if None
            
        Returns:
            The assigned token ID
        """
        if not token:
            logger.warning("Attempted to add empty token, returning UNK")
            return self.SPECIAL_TOKENS['<UNK>']
        
        if token in self.token_to_id:
            return self.token_to_id[token]
        
        if token_type is None:
            token_type = self.classify_morpheme(token)
        
        # Validate token_type
        if token_type not in self.TOKEN_RANGES:
            logger.warning(f"Unknown token type '{token_type}', defaulting to 'subword'")
            token_type = 'subword'
        
        # Get range info
        start, end, slot = self.TOKEN_RANGES[token_type]
        
        # Check if we have room in this token range
        if self.next_id[token_type] > end:
            logger.warning(f"Token range overflow for type '{token_type}'. "
                          f"Falling back to subword range.")
            token_type = 'subword'
            slot = -1
            start, end, _ = self.TOKEN_RANGES['subword']
        
        token_id = self.next_id[token_type]
        self.next_id[token_type] += 1
        
        self.token_to_id[token] = token_id
        self.id_to_token[token_id] = token
        self.token_type[token_id] = token_type
        self.token_slot[token_id] = slot
        
        logger.debug(f"Added token '{token}' with ID {token_id}, type '{token_type}'")
        
        return token_id
    
    def get_token_id(self, token: str) -> int:
        """
        Get the ID for a token, or <UNK> if not found.
        
        Args:
            token: The token string
            
        Returns:
            The token ID or <UNK> ID if not found
        """
        if not token:
            return self.SPECIAL_TOKENS['<UNK>']
        return self.token_to_id.get(token, self.SPECIAL_TOKENS['<UNK>'])
    
    def get_token(self, token_id: int) -> str:
        """
        Get the token for an ID, or <UNK> if not found.
        
        Args:
            token_id: The token ID
            
        Returns:
            The token string or '<UNK>' if not found
        """
        if token_id is None or token_id < 0:
            return '<UNK>'
        return self.id_to_token.get(token_id, '<UNK>')
    
    def get_token_type(self, token_id: int) -> str:
        """
        Get the type of a token by its ID.
        
        Args:
            token_id: The token ID
            
        Returns:
            The token type string
        """
        return self.token_type.get(token_id, 'unknown')
    
    def get_token_slot(self, token_id: int) -> int:
        """
        Get the slot position for a token.
        
        Args:
            token_id: The token ID
            
        Returns:
            The slot position (-1 for no restriction)
        """
        return self.token_slot.get(token_id, -1)
    
    def build_from_corpus(self, morphemes: List[str], min_freq: int = 2) -> None:
        """
        Build vocabulary from a list of morphemes based on frequency.
        
        Args:
            morphemes: List of morpheme strings
            min_freq: Minimum frequency threshold for inclusion
        """
        if not morphemes:
            logger.warning("Empty morpheme list passed to build_from_corpus")
            return
        
        counter = Counter(morphemes)
        added_count = 0
        
        for morpheme, freq in counter.most_common():
            if freq >= min_freq:
                self.add_token(morpheme)
                added_count += 1
        
        logger.info(f"Added {added_count} tokens from corpus with min_freq={min_freq}")
    
    def get_vocabulary_size(self) -> int:
        """Return the total vocabulary size."""
        return len(self.token_to_id)
    
    def get_tokens_by_type(self, token_type: str) -> List[Tuple[str, int]]:
        """
        Get all tokens of a specific type.
        
        Args:
            token_type: The type to filter by
            
        Returns:
            List of (token, token_id) tuples
        """
        return [
            (token, token_id) 
            for token, token_id in self.token_to_id.items()
            if self.token_type.get(token_id) == token_type
        ]
    
    def get_capacity_info(self) -> Dict[str, Dict[str, int]]:
        """
        Get capacity information for each token type.
        
        Returns:
            Dictionary with capacity details for each type
        """
        info = {}
        for token_type, (start, end, slot) in self.TOKEN_RANGES.items():
            capacity = end - start + 1
            used = self.next_id[token_type] - start
            remaining = capacity - used
            info[token_type] = {
                'start': start,
                'end': end,
                'capacity': capacity,
                'used': used,
                'remaining': remaining,
                'utilization': used / capacity if capacity > 0 else 0
            }
        return info
    
    def save(self, filepath: str) -> None:
        """
        Save vocabulary to a JSON file.
        
        Args:
            filepath: Path to save the vocabulary
        """
        data = {
            'token_to_id': self.token_to_id,
            'id_to_token': {str(k): v for k, v in self.id_to_token.items()},
            'token_type': {str(k): v for k, v in self.token_type.items()},
            'token_slot': {str(k): v for k, v in self.token_slot.items()},
            'next_id': self.next_id,
            'version': '0.9.0-rc'
        }
        
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved vocabulary ({len(self.token_to_id)} tokens) to {filepath}")
    
    def load(self, filepath: str) -> None:
        """
        Load vocabulary from a JSON file.
        
        Args:
            filepath: Path to load the vocabulary from
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Vocabulary file not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.token_to_id = data['token_to_id']
        self.id_to_token = {int(k): v for k, v in data['id_to_token'].items()}
        self.token_type = {int(k): v for k, v in data.get('token_type', {}).items()}
        self.token_slot = {int(k): v for k, v in data.get('token_slot', {}).items()}
        
        # Handle both old and new format for next_id
        if 'next_id' in data:
            self.next_id = data['next_id']
        
        logger.info(f"Loaded vocabulary ({len(self.token_to_id)} tokens) from {filepath}")
    
    def __len__(self) -> int:
        return self.get_vocabulary_size()
    
    def __contains__(self, token: str) -> bool:
        return token in self.token_to_id
    
    def __repr__(self) -> str:
        type_counts = {}
        for token_type in self.TOKEN_RANGES.keys():
            type_counts[token_type] = len(self.get_tokens_by_type(token_type))
        return f"HierarchicalVocabulary(total={len(self)}, types={type_counts})"
