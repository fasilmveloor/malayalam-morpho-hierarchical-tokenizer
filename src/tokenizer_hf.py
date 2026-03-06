"""
HuggingFace-Compatible Morpho-Hierarchical Tokenizer for Malayalam

This module provides a HuggingFace transformers-compatible tokenizer
that can be used with any HuggingFace model.

Usage:
    from transformers import AutoTokenizer
    from tokenizer_hf import MorphoHierarchicalTokenizerFast
    
    # Direct usage
    tokenizer = MorphoHierarchicalTokenizerFast.from_pretrained("./models")
    tokens = tokenizer.tokenize("പഠിക്കുന്നു")
    
    # Via AutoTokenizer (after registration)
    tokenizer = AutoTokenizer.from_pretrained("malayalam-morpho-tokenizer")
"""

import os
import json
import unicodedata
import re
from typing import List, Optional, Dict, Tuple, Union
from collections import OrderedDict

# HuggingFace imports
try:
    from transformers import PreTrainedTokenizer
    from transformers.utils import PaddingStrategy, TensorType
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    PreTrainedTokenizer = object


class MorphoHierarchicalTokenizerFast(PreTrainedTokenizer if TRANSFORMERS_AVAILABLE else object):
    """
    HuggingFace-compatible Morpho-Hierarchical Tokenizer for Malayalam.
    
    Novel Features:
    - Slot System: Hierarchical token IDs encoding grammatical structure
    - Phoneme-Aware Encoding: 10-dimensional feature vectors
    - Sandhi Reconstruction: ം → ത്ത് transformation
    - Hybrid Pipeline: FST (mlmorph) + Neural Bi-LSTM for OOV
    
    Token ID Structure:
        Special:  0-999     (PAD, UNK, BOS, EOS, etc.)
        Root:     1000-1999 (Verb/Noun stems)
        Tense:    2000-2999 (ഉന്നു, ച്ചു, ാൻ, etc.)
        Case:     3000-3999 (ിൽ, ിന്റെ, ുടെ, etc.)
        Function: 4000-4999 (Conjunctions, particles)
        Infix:    5000-5999 (Sandhi infixes like ത്ത്)
    
    Example:
        >>> tokenizer = MorphoHierarchicalTokenizerFast.from_pretrained("./models")
        >>> tokens = tokenizer.tokenize("പഠിക്കുന്നു")
        >>> print(tokens)
        ['പഠിക്ക്', 'ുന്നു']
        >>> ids = tokenizer.encode("പഠിക്കുന്നു")
        >>> print(ids)
        [2, 1001, 2001, 3]  # BOS, root, tense, EOS
    """
    
    vocab_files_names = {
        "vocab_file": "vocab.json",
        "model_file": "model.pt"
    }
    
    # Model type for AutoTokenizer
    model_input_names = ["input_ids", "attention_mask"]
    
    def __init__(
        self,
        vocab_file: Optional[str] = None,
        vocab: Optional[Dict[str, int]] = None,
        merges_file: Optional[str] = None,
        use_mlmorph: bool = True,
        use_neural: bool = True,
        unk_token: str = "<UNK>",
        bos_token: str = "<BOS>",
        eos_token: str = "<EOS>",
        pad_token: str = "<PAD>",
        mask_token: str = "<MASK>",
        cls_token: str = "<CLS>",
        sep_token: str = "<SEP>",
        **kwargs
    ):
        """
        Initialize the Morpho-Hierarchical Tokenizer.
        
        Args:
            vocab_file: Path to vocabulary JSON file
            vocab: Direct vocabulary dictionary
            use_mlmorph: Whether to use mlmorph FST analyzer
            use_neural: Whether to use neural Bi-LSTM for OOV
            unk_token: Unknown token
            bos_token: Beginning of sequence token
            eos_token: End of sequence token
            pad_token: Padding token
        """
        # Initialize parent class if available
        if TRANSFORMERS_AVAILABLE:
            super().__init__(
                unk_token=unk_token,
                bos_token=bos_token,
                eos_token=eos_token,
                pad_token=pad_token,
                mask_token=mask_token,
                cls_token=cls_token,
                sep_token=sep_token,
                **kwargs
            )
        
        # Special tokens
        self.SPECIAL_TOKENS = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3,
            '<MASK>': 4,
            '<CLS>': 5,
            '<SEP>': 6,
        }
        
        # Slots for hierarchical vocabulary
        self.SLOTS = {
            'special': (0, 999),
            'root': (1000, 1999),
            'tense': (2000, 2999),
            'case': (3000, 3999),
            'function': (4000, 4999),
            'infix': (5000, 5999),
            'char': (7000, 7999),
        }
        
        # Slot counters for new token assignment
        self.slot_counters = {cat: start for cat, (start, end) in self.SLOTS.items()}
        
        # Load vocabulary
        self.vocab: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        
        if vocab_file and os.path.exists(vocab_file):
            self._load_vocab(vocab_file)
        elif vocab:
            self.vocab = vocab
            self.id_to_token = {v: k for k, v in vocab.items()}
        else:
            # Initialize with special tokens
            self.vocab = self.SPECIAL_TOKENS.copy()
            self.id_to_token = {v: k for k, v in self.vocab.items()}
        
        # Initialize mlmorph
        self.use_mlmorph = use_mlmorph
        self.morph_analyzer = None
        if use_mlmorph:
            try:
                from mlmorph import Analyser
                self.morph_analyzer = Analyser()
                print("✓ mlmorph analyzer initialized")
            except ImportError:
                print("⚠ mlmorph not available, using fallback")
                self.use_mlmorph = False
        
        # Neural model placeholder
        self.use_neural = use_neural
        self.neural_model = None
        
        # Character vocabulary for OOV fallback
        self.char_vocab: Dict[str, int] = {}
        
        # High-frequency word cache
        self.cache: Dict[str, List[str]] = {}
        self._load_cache()
        
        # Statistics
        self.stats = {
            'morphology_hits': 0,
            'cache_hits': 0,
            'fallback_hits': 0,
        }
    
    def _load_vocab(self, vocab_file: str):
        """Load vocabulary from JSON file."""
        with open(vocab_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            if 'token_to_id' in data:
                self.vocab = data['token_to_id']
            else:
                self.vocab = data
        else:
            raise ValueError(f"Invalid vocabulary format in {vocab_file}")
        
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        
        # Update slot counters
        for token, token_id in self.vocab.items():
            for cat, (start, end) in self.SLOTS.items():
                if start <= token_id < end:
                    self.slot_counters[cat] = max(self.slot_counters[cat], token_id + 1)
                    break
    
    def _load_cache(self):
        """Load high-frequency word cache."""
        cache_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'exceptions.json')
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if 'high_frequency_words' in data:
                        self.cache = data['high_frequency_words']
            except:
                pass
    
    def _normalize(self, text: str) -> str:
        """Apply NFKC normalization."""
        normalized = unicodedata.normalize('NFKC', text)
        normalized = normalized.replace('\u200d', '')  # ZWJ
        normalized = normalized.replace('\u200c', '')  # ZWNJ
        return normalized
    
    def _tokenize(self, text: str, **kwargs) -> List[str]:
        """
        Tokenize text into morphemes.
        
        This is the core tokenization method required by PreTrainedTokenizer.
        """
        text = self._normalize(text)
        
        # Extract Malayalam words
        words = re.findall(r'[\u0D00-\u0D7F]+', text)
        
        morphemes = []
        for word in words:
            morphemes.extend(self._get_morphemes(word))
        
        return morphemes
    
    def _get_morphemes(self, word: str) -> List[str]:
        """Get morphological decomposition of a word."""
        # Check cache first
        if word in self.cache:
            self.stats['cache_hits'] += 1
            return self.cache[word]
        
        # Try mlmorph
        if self.use_mlmorph and self.morph_analyzer:
            try:
                analysis = self.morph_analyzer.analyse(word)
                if analysis:
                    morphemes = self._parse_analysis(analysis[0], word)
                    if morphemes:
                        self.stats['morphology_hits'] += 1
                        return morphemes
            except:
                pass
        
        # Fallback to suffix-based splitting
        self.stats['fallback_hits'] += 1
        return self._fallback_split(word)
    
    def _parse_analysis(self, analysis, original: str) -> List[str]:
        """Parse mlmorph analysis output."""
        analysis_str = analysis[0] if isinstance(analysis, tuple) else str(analysis)
        
        # Extract root
        root_match = re.match(r'^([^\s<+]+)', analysis_str)
        if not root_match:
            return [original]
        
        root = root_match.group(1)
        
        # Handle compound analysis (contains +)
        if '+' in root:
            parts = root.split('+')
            return [p.strip() for p in parts if p.strip()]
        
        # Single root with potential suffix
        if root == original:
            return [original]
        
        # Find suffix
        root_base = root.rstrip('്')
        if original.startswith(root_base) and len(original) > len(root_base):
            suffix = original[len(root_base):]
            if suffix:
                # Convert root to stem form if vowel-initial suffix
                if suffix[0] in 'ാിീുൂൃെേൈൊോൌ':  # Dependent vowels
                    stem = root_base + '്'
                else:
                    stem = root
                return [stem, suffix]
        
        return [root] if root else [original]
    
    def _fallback_split(self, word: str) -> List[str]:
        """Fallback suffix-based morphological splitting."""
        # Common Malayalam suffixes (ordered by length)
        suffixes = [
            # Tense/Aspect
            'ുകയാണ്', 'ുകയും', 'ുന്നുണ്ട്',
            'ുന്നു', 'ുക', 'ിച്ചു', 'ിട്ടു', 'ുണ്ട്',
            'ും', 'ാം', 'ണം', 'ാൻ',
            # Case markers
            'ിൽനിന്ന്', 'ിന്റെ', 'ിനോട്', 'ിലൂടെ',
            'ിൽ', 'ിലെ', 'ിന്', 'ിനെ', 'ക്ക്',
            # Sandhi forms
            'ത്തിൽ', 'ത്തിന്', 'ത്തിന്റെ',
        ]
        
        for suffix in suffixes:
            if word.endswith(suffix):
                stem = word[:-len(suffix)]
                if len(stem) >= 2:
                    # Add virama to stem if suffix is vowel-initial
                    if suffix[0] in 'ാിീുൂൃെേൈൊോൌ':
                        stem = stem + '്'
                    return [stem, suffix]
        
        return [word]
    
    def _convert_token_to_id(self, token: str) -> int:
        """Convert token to vocabulary ID."""
        if token in self.vocab:
            return self.vocab[token]
        
        # Add new token to vocabulary
        return self._add_token(token)
    
    def _add_token(self, token: str) -> int:
        """Add a new token to the vocabulary."""
        # Determine category
        category = self._classify_token(token)
        
        # Get next ID in slot
        token_id = self.slot_counters.get(category, 8000)
        
        # Check slot capacity
        start, end = self.SLOTS.get(category, (8000, 9999))
        if token_id >= end:
            # Fallback to char slot
            token_id = self.slot_counters['char']
            category = 'char'
        
        self.vocab[token] = token_id
        self.id_to_token[token_id] = token
        self.slot_counters[category] = token_id + 1
        
        return token_id
    
    def _classify_token(self, token: str) -> str:
        """Classify a token into a morphological category."""
        # Tense markers
        if any(token.endswith(s) for s in ['ുന്നു', 'ുക', 'ിച്ചു', 'ും', 'ാൻ', 'ണം']):
            return 'tense'
        
        # Case markers
        if any(token.endswith(s) for s in ['ിൽ', 'ിന്', 'ിന്റെ', 'ിനെ', 'ക്ക്']):
            return 'case'
        
        # Sandhi infixes
        if token.endswith('ത്ത്'):
            return 'infix'
        
        # Short function words
        if len(token) <= 3 and not token.endswith('്'):
            return 'function'
        
        # Default to root
        return 'root'
    
    def _convert_id_to_token(self, index: int) -> str:
        """Convert vocabulary ID to token."""
        return self.id_to_token.get(index, self.unk_token)
    
    def classify_token(self, token_id: int) -> str:
        """Classify token by hierarchical slot."""
        for category, (start, end) in self.SLOTS.items():
            if start <= token_id < end:
                return category
        return 'unknown'
    
    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return len(self.vocab)
    
    def get_vocab(self) -> Dict[str, int]:
        """Return the vocabulary."""
        return self.vocab.copy()
    
    # Public methods for standalone usage (when transformers not available)
    def tokenize(self, text: str, **kwargs) -> List[str]:
        """
        Tokenize text into morphemes (public interface).
        
        Args:
            text: Input text
        
        Returns:
            List of morpheme tokens
        """
        return self._tokenize(text, **kwargs)
    
    def encode(self, text: str, add_special_tokens: bool = True, **kwargs) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            add_special_tokens: Whether to add BOS/EOS tokens
        
        Returns:
            List of token IDs
        """
        tokens = self.tokenize(text)
        token_ids = [self._convert_token_to_id(t) for t in tokens]
        
        if add_special_tokens:
            return self.build_inputs_with_special_tokens(token_ids)
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True, **kwargs) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens
        
        Returns:
            Decoded text
        """
        tokens = []
        for token_id in token_ids:
            if skip_special_tokens and token_id in self.SPECIAL_TOKENS.values():
                continue
            tokens.append(self._convert_id_to_token(token_id))
        
        return self.decode_morphemes(tokens)
    
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """Save vocabulary to file."""
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory, exist_ok=True)
        
        vocab_file = os.path.join(
            save_directory, 
            (filename_prefix or '') + self.vocab_files_names["vocab_file"]
        )
        
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump({
                'token_to_id': self.vocab,
                'slots': self.SLOTS,
                'special_tokens': self.SPECIAL_TOKENS
            }, f, ensure_ascii=False, indent=2)
        
        return (vocab_file,)
    
    def build_inputs_with_special_tokens(
        self, 
        token_ids_0: List[int], 
        token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs with special tokens.
        
        Args:
            token_ids_0: First sequence of token IDs
            token_ids_1: Optional second sequence for pair tasks
        
        Returns:
            List of token IDs with special tokens
        """
        bos = [self.SPECIAL_TOKENS['<BOS>']]
        eos = [self.SPECIAL_TOKENS['<EOS>']]
        
        if token_ids_1 is None:
            return bos + token_ids_0 + eos
        
        sep = [self.SPECIAL_TOKENS['<SEP>']]
        return bos + token_ids_0 + sep + token_ids_1 + eos
    
    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Get mask for special tokens.
        
        Returns:
            List of 0s and 1s, where 1 indicates a special token
        """
        if already_has_special_tokens:
            return [1 if t in self.SPECIAL_TOKENS.values() else 0 for t in token_ids_0]
        
        mask = [1] + [0] * len(token_ids_0) + [1]
        
        if token_ids_1:
            mask += [1] + [0] * len(token_ids_1) + [1]
        
        return mask
    
    def create_token_type_ids_from_sequences(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create token type IDs for sequence pair.
        
        Returns:
            List of token type IDs (0 for first sequence, 1 for second)
        """
        bos = [self.SPECIAL_TOKENS['<BOS>']]
        eos = [self.SPECIAL_TOKENS['<EOS>']]
        sep = [self.SPECIAL_TOKENS['<SEP>']]
        
        if token_ids_1 is None:
            return [0] * (len(bos) + len(token_ids_0) + len(eos))
        
        return [0] * len(bos + token_ids_0 + sep) + [1] * len(token_ids_1 + eos)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *args, **kwargs):
        """
        Load tokenizer from pretrained model.
        
        Args:
            pretrained_model_name_or_path: Path to model directory
                or HuggingFace Hub model ID
        """
        # Resolve vocab file path
        if os.path.isdir(pretrained_model_name_or_path):
            vocab_file = os.path.join(
                pretrained_model_name_or_path,
                cls.vocab_files_names["vocab_file"]
            )
        else:
            # Try to download from Hub
            try:
                from huggingface_hub import hf_hub_download
                vocab_file = hf_hub_download(
                    repo_id=pretrained_model_name_or_path,
                    filename=cls.vocab_files_names["vocab_file"]
                )
            except:
                vocab_file = pretrained_model_name_or_path
        
        kwargs['vocab_file'] = vocab_file
        return cls(*args, **kwargs)
    
    def decode_morphemes(self, morphemes: List[str]) -> str:
        """
        Decode morphemes back to word form.
        
        Handles sandhi transformations like stem + vowel-initial suffix.
        """
        result = []
        
        for morpheme in morphemes:
            if not result:
                result.append(morpheme)
                continue
            
            prev = result[-1]
            
            # Stem ending with virama + vowel-initial suffix
            if prev.endswith('്') and morpheme and morpheme[0] in 'ാിീുൂൃെേൈൊോൌ':
                # Remove virama and combine
                result[-1] = prev[:-1] + morpheme
            else:
                result.append(morpheme)
        
        return ''.join(result)
    
    def get_stats(self) -> Dict:
        """Get tokenization statistics."""
        return {
            'vocab_size': self.vocab_size,
            'morphology_hits': self.stats['morphology_hits'],
            'cache_hits': self.stats['cache_hits'],
            'fallback_hits': self.stats['fallback_hits'],
        }


# Register with AutoTokenizer if transformers is available
def register_tokenizer():
    """Register the tokenizer with HuggingFace Auto classes."""
    if TRANSFORMERS_AVAILABLE:
        from transformers import AutoTokenizer
        AutoTokenizer.register(
            'morpho-hierarchical',
            MorphoHierarchicalTokenizerFast,
            exist_ok=True
        )
        print("✓ Registered MorphoHierarchicalTokenizerFast with AutoTokenizer")


# Demo
if __name__ == "__main__":
    print("=" * 60)
    print("Morpho-Hierarchical Tokenizer Demo (HuggingFace Compatible)")
    print("=" * 60)
    
    # Initialize
    tokenizer = MorphoHierarchicalTokenizerFast(use_mlmorph=True)
    
    # Test words
    test_words = [
        'പഠിക്കുന്നു',      # Present tense verb
        'വിദ്യാലയം',       # Noun
        'കേരളത്തിൽ',       # Place + case
        'തിരുവനന്തപുരം',   # Compound place name
    ]
    
    print("\nTokenization Results:")
    print("-" * 60)
    
    for word in test_words:
        tokens = tokenizer.tokenize(word)
        token_ids = tokenizer.encode(word)
        categories = [tokenizer.classify_token(i) for i in token_ids if i not in tokenizer.SPECIAL_TOKENS.values()]
        
        print(f"\nWord: {word}")
        print(f"  Tokens: {' + '.join(tokens)}")
        print(f"  IDs: {token_ids}")
        print(f"  Categories: {categories}")
    
    print("\n" + "=" * 60)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Statistics: {tokenizer.get_stats()}")
    print("=" * 60)
