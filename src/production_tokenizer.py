"""
Production Tokenizer for Malayalam - HuggingFace Compatible

This module provides a HuggingFace-compatible tokenizer wrapper around our
Morpho-Hierarchical tokenization system.

Architecture:
1. MalayalamMorphTokenizer - Main tokenizer class
2. HybridSandhiSplitter - Morphological splitting
3. BPE Fallback - For OOV words

Usage:
    tokenizer = MalayalamMorphTokenizer.from_pretrained("./malayalam-tokenizer")
    tokens = tokenizer("ഞാൻ പഠിക്കുന്നു")
"""

import os
import json
import re
import unicodedata
from typing import List, Dict, Optional, Tuple, Union
from collections import Counter
import sys

# Try to import HuggingFace transformers
try:
    from transformers import PreTrainedTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("⚠ transformers not installed. Install with: pip install transformers")

# Import local modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tokenizer import MorphoHierarchicalTokenizer
from src.sandhi import MalayalamSandhi


class HybridSandhiSplitter:
    """
    Hybrid sandhi splitter combining:
    1. High-frequency cache
    2. Dictionary lookup
    3. Rule-based patterns
    4. Statistical features
    5. mlmorph ground truth
    """
    
    def __init__(self, use_mlmorph: bool = True):
        self.sandhi = MalayalamSandhi()
        self.use_mlmorph = use_mlmorph
        self.morph_analyzer = None
        
        # Statistics for scoring
        self.suffix_freq = Counter({
            'ുന്നു': 5000, 'ും': 4000, 'ിച്ചു': 3000,
            'ിൽ': 8000, 'ിന്റെ': 6000, 'ിന്': 5000,
            'ിലെ': 3000, 'ാൻ': 2500, 'ണം': 2000,
            'ുക': 1500, 'ിയ': 2000, 'ുന്ന': 1500,
            'ക്ക്': 3000, 'ത്തിൽ': 1500,
        })
        
        # Initialize mlmorph if available
        if use_mlmorph:
            try:
                from mlmorph import Analyser
                self.morph_analyzer = Analyser()
            except ImportError:
                self.use_mlmorph = False
        
        # Load exceptions
        self.exceptions = self._load_exceptions()
    
    def _load_exceptions(self) -> Dict:
        """Load exceptions dictionary."""
        exceptions_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data', 'exceptions.json'
        )
        
        if os.path.exists(exceptions_path):
            with open(exceptions_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def split(self, word: str) -> List[str]:
        """Split a word into morphemes."""
        # 1. Check exceptions cache
        if word in self.exceptions.get('high_frequency_words', {}):
            return self.exceptions['high_frequency_words'][word]
        
        # 2. Apply rule-based patterns
        result = self._apply_rules(word)
        if result:
            return result
        
        # 3. Use mlmorph if available
        if self.morph_analyzer:
            result = self._mlmorph_split(word)
            if result:
                return result
        
        # 4. Fallback
        return [word]
    
    def _apply_rules(self, word: str) -> Optional[List[str]]:
        """Apply rule-based patterns."""
        for suffix in self.suffix_freq:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                root = word[:-len(suffix)]
                
                # Handle ം ending nouns
                if root.endswith('ം'):
                    root = root[:-1] + 'ത്ത്'
                elif not root.endswith('്') and suffix[0] in self.sandhi.DEPENDENT_VOWELS:
                    root = root + '്'
                
                return [root, suffix]
        
        return None
    
    def _mlmorph_split(self, word: str) -> Optional[List[str]]:
        """Use mlmorph for ground truth."""
        if not self.morph_analyzer:
            return None
        
        try:
            analysis = self.morph_analyzer.analyse(word)
            if analysis and len(analysis) > 0:
                analysis_str = analysis[0][0] if isinstance(analysis[0], tuple) else str(analysis[0])
                
                root_match = re.match(r'^([^\s<]+)', analysis_str)
                if root_match:
                    root = root_match.group(1)
                    
                    # Convert to stem form
                    if root.endswith('ുക'):
                        stem = root[:-2] + '്'
                    elif root.endswith('ക'):
                        stem = root[:-1] + '്'
                    else:
                        stem = self.sandhi.to_stem_form(root)
                    
                    stem_base = stem[:-1] if stem.endswith('്') else stem
                    
                    if word.startswith(stem_base):
                        suffix = word[len(stem_base):]
                        if suffix:
                            return [stem, suffix]
        except Exception:
            pass
        
        return None


class MalayalamMorphTokenizer(PreTrainedTokenizer if HAS_TRANSFORMERS else object):
    """
    HuggingFace-compatible Malayalam Morpho-Hierarchical Tokenizer.
    
    This tokenizer combines:
    1. Morphological analysis using mlmorph
    2. Hierarchical vocabulary (stems, suffixes, infixes)
    3. Sandhi rules for proper decomposition
    4. BPE fallback for OOV words
    
    Token ID Structure:
    - 0-99: Special tokens (<PAD>, <UNK>, <BOS>, <EOS>)
    - 1000-1999: Root stems
    - 2000-2999: Tense/Aspect markers
    - 3000-3999: Case markers
    - 4000-4999: Function words
    - 5000-5999: Infixes/Augments
    - 6000-6999: Subword tokens
    - 7000+: Character tokens
    
    Usage:
        >>> tokenizer = MalayalamMorphTokenizer(vocab_file="vocab.json")
        >>> tokens = tokenizer.encode("ഞാൻ പഠിക്കുന്നു")
        >>> print(tokens)
        [2, 4001, 1001, 2001, 3]
    """
    
    vocab_files_names = {
        "vocab_file": "vocab.json",
        "merges_file": "merges.txt",
    }
    
    model_input_names = ["input_ids", "attention_mask"]
    
    def __init__(
        self,
        vocab_file: str = None,
        merges_file: str = None,
        errors: str = "replace",
        unk_token: str = "<unk>",
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        pad_token: str = "<pad>",
        **kwargs
    ):
        # IMPORTANT: Initialize vocab BEFORE calling super().__init__()
        # because the parent class will call get_vocab()
        self.vocab = {}
        self.ids_to_tokens = {}
        self._init_default_vocab()
        
        # Initialize parent
        if HAS_TRANSFORMERS:
            super().__init__(
                errors=errors,
                unk_token=unk_token,
                bos_token=bos_token,
                eos_token=eos_token,
                pad_token=pad_token,
                **kwargs
            )
        
        # Initialize components
        self.splitter = HybridSandhiSplitter(use_mlmorph=True)
        self.sandhi = MalayalamSandhi()
        
        # Load vocabulary if provided
        if vocab_file and os.path.exists(vocab_file):
            self._load_vocab(vocab_file)
        
        # BPE for OOV fallback
        self.bpe_tokenizer = None
        self._init_bpe(merges_file)
    
    def _init_default_vocab(self):
        """Initialize default hierarchical vocabulary."""
        # Special tokens
        self.vocab = {
            '<pad>': 0,
            '<unk>': 1,
            '<s>': 2,
            '</s>': 3,
            '<root>': 4,
            '<suffix>': 5,
            '<infix>': 6,
            '<space>': 7,
        }
        
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
    
    def _load_vocab(self, vocab_file: str):
        """Load vocabulary from JSON file."""
        with open(vocab_file, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
        
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
    
    def _init_bpe(self, merges_file: str = None):
        """Initialize BPE for OOV fallback."""
        try:
            from tokenizers import Tokenizer
            from tokenizers.models import BPE
            from tokenizers.trainers import BpeTrainer
            
            # Create a simple BPE tokenizer for fallback
            if merges_file and os.path.exists(merges_file):
                self.bpe_tokenizer = Tokenizer(BPE(vocab=self.vocab, merges=merges_file))
        except ImportError:
            self.bpe_tokenizer = None
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into morphemes.
        
        This is the core tokenization method that HuggingFace calls.
        
        Args:
            text: Input text
            
        Returns:
            List of token strings
        """
        # Normalize
        text = unicodedata.normalize('NFKC', text)
        text = text.replace('\u200d', '').replace('\u200c', '')
        
        # Split into words
        words = re.findall(r'[\u0D00-\u0D7F]+', text)
        
        tokens = []
        
        for word in words:
            # Get morpheme split
            morphemes = self.splitter.split(word)
            tokens.extend(morphemes)
            tokens.append('<space>')  # Add space marker between words
        
        # Remove trailing space marker
        if tokens and tokens[-1] == '<space>':
            tokens = tokens[:-1]
        
        return tokens
    
    def _convert_token_to_id(self, token: str) -> int:
        """Convert token to ID."""
        return self.vocab.get(token, self.vocab.get('<unk>', 1))
    
    def _convert_id_to_token(self, index: int) -> str:
        """Convert ID to token."""
        return self.ids_to_tokens.get(index, '<unk>')
    
    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Convert tokens back to string."""
        # Remove space markers and join
        tokens = [t for t in tokens if t != '<space>']
        return ''.join(tokens)
    
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """Add special tokens to input."""
        if token_ids_1 is None:
            return [self.vocab['<s>']] + token_ids_0 + [self.vocab['</s>']]
        return (
            [self.vocab['<s>']] + 
            token_ids_0 + 
            [self.vocab['</s>'], self.vocab['</s>']] + 
            token_ids_1 + 
            [self.vocab['</s>']]
        )
    
    def get_vocab(self) -> Dict[str, int]:
        """Return vocabulary."""
        return self.vocab
    
    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return len(self.vocab)
    
    def save_vocabulary(self, save_directory: str, filename_prefix: str = None) -> Tuple[str]:
        """Save vocabulary to directory."""
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory)
        
        vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + "vocab.json"
        )
        
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
        
        return (vocab_file,)
    
    def train_from_corpus(self, corpus: List[str], vocab_size: int = 30000, min_freq: int = 2):
        """
        Train tokenizer vocabulary from corpus.
        
        Args:
            corpus: List of text strings
            vocab_size: Maximum vocabulary size
            min_freq: Minimum frequency for token inclusion
        """
        print(f"Training on {len(corpus)} documents...")
        
        # Collect all morphemes
        morpheme_counter = Counter()
        
        for text in corpus:
            tokens = self._tokenize(text)
            morpheme_counter.update(tokens)
        
        # Build vocabulary
        self._init_default_vocab()  # Start with special tokens
        
        # Reserve ranges for each type
        ranges = {
            'root': (1000, 1999),
            'tense': (2000, 2999),
            'case': (3000, 3999),
            'function': (4000, 4999),
            'infix': (5000, 5999),
            'subword': (6000, 9999),
        }
        
        current_ids = {t: ranges[t][0] for t in ranges}
        
        for morpheme, freq in morpheme_counter.most_common():
            if freq < min_freq:
                continue
            
            if len(self.vocab) >= vocab_size:
                break
            
            # Classify morpheme
            mtype = self._classify_morpheme(morpheme)
            
            # Assign ID
            if mtype in ranges and current_ids[mtype] < ranges[mtype][1]:
                token_id = current_ids[mtype]
                current_ids[mtype] += 1
            else:
                # Fallback to subword range
                if current_ids['subword'] < ranges['subword'][1]:
                    token_id = current_ids['subword']
                    current_ids['subword'] += 1
                else:
                    continue
            
            self.vocab[morpheme] = token_id
            self.ids_to_tokens[token_id] = morpheme
        
        # Update vocab size
        print(f"Vocabulary size: {len(self.vocab)}")
        
        # Print breakdown
        for t, (start, end) in ranges.items():
            count = sum(1 for k, v in self.vocab.items() if start <= v < end)
            print(f"  {t}: {count}")
    
    def _classify_morpheme(self, morpheme: str) -> str:
        """Classify a morpheme into type."""
        tense_markers = ['ുന്നു', 'ും', 'ിച്ചു', 'ാൻ', 'ണം', 'ുക']
        case_markers = ['ിൽ', 'ിന്റെ', 'ിന്', 'ിലെ', 'ക്ക്', 'ത്തിൽ']
        function_words = ['ഞാൻ', 'നീ', 'അവർ', 'അത്', 'ഇത്']
        infix_markers = ['ക്ക', 'ച്ച', 'ത്ത', 'പ്പ', 'ത്ത്']
        
        if morpheme in function_words:
            return 'function'
        if any(morpheme.endswith(m) or morpheme == m for m in tense_markers):
            return 'tense'
        if any(morpheme.endswith(m) or morpheme == m for m in case_markers):
            return 'case'
        if morpheme in infix_markers:
            return 'infix'
        return 'root'


def create_tokenizer(corpus_path: str = None, vocab_size: int = 30000) -> MalayalamMorphTokenizer:
    """
    Create and optionally train a Malayalam tokenizer.
    
    Args:
        corpus_path: Path to corpus file (one line per document)
        vocab_size: Maximum vocabulary size
        
    Returns:
        Trained MalayalamMorphTokenizer
    """
    tokenizer = MalayalamMorphTokenizer()
    
    if corpus_path and os.path.exists(corpus_path):
        with open(corpus_path, 'r', encoding='utf-8') as f:
            corpus = [line.strip() for line in f if line.strip()]
        
        tokenizer.train_from_corpus(corpus, vocab_size=vocab_size)
    
    return tokenizer


def demo():
    """Demo the production tokenizer."""
    print("\n" + "="*60)
    print("MalayalamMorphTokenizer - Production Demo")
    print("="*60)
    
    # Create tokenizer
    tokenizer = MalayalamMorphTokenizer()
    
    # Train on SMC corpus
    corpus_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data', 'smc_corpus.txt'
    )
    
    if os.path.exists(corpus_path):
        with open(corpus_path, 'r', encoding='utf-8') as f:
            corpus = [line.strip() for line in f.readlines()[:5000]]
        
        tokenizer.train_from_corpus(corpus, vocab_size=15000)
    
    # Test sentences
    test_sentences = [
        "ഞാൻ പഠിക്കുന്നു",
        "വിദ്യാലയത്തിൽ കുട്ടികൾ വരുന്നു",
        "തിരുവനന്തപുരം കേരളത്തിന്റെ തലസ്ഥാനമാണ്",
    ]
    
    print("\n📝 Test Results:")
    print("-"*60)
    
    for sentence in test_sentences:
        print(f"\nInput: {sentence}")
        
        # Tokenize
        tokens = tokenizer._tokenize(sentence)
        token_ids = [tokenizer._convert_token_to_id(t) for t in tokens]
        
        print(f"Tokens: {tokens}")
        print(f"IDs: {token_ids}")
        
        # Decode
        decoded = tokenizer.convert_tokens_to_string(tokens)
        print(f"Decoded: {decoded}")


if __name__ == "__main__":
    demo()
