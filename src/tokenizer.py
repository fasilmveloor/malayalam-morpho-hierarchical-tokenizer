"""
Morpho-Hierarchical Tokenizer for Malayalam

A novel tokenization approach combining:
1. Morphological analysis using mlmorph
2. Hierarchical vocabulary structure (roots, suffixes, function words)
3. Sandhi splitting for compound words
4. Proper stem extraction (virama-ending roots)
5. Unigram fallback for OOV handling

Architecture:
    Input Text
        ↓
    Unicode Normalization (NFKC)
        ↓
    Sandhi Splitting (compound words)
        ↓
    Morphological Analysis (mlmorph)
        ↓
    Stem Extraction (virama-ending roots) ← NEW!
        ↓
    Hierarchical Token Assignment
        ↓
    Unigram Fallback (for OOV)
        ↓
    Token IDs

Key Innovation: Roots are stored in STEM form (ending with ്/virama)
to properly handle sandhi transformations.

Example:
    പഠിക്കുന്നു → പഠിക്ക് (stem) + ുന്നു (suffix)
    Not: പഠിക്ക + ുന്നു (incorrect - extra 'a' sound)

Author: Malayalam NLP Research
"""

import unicodedata
import re
import logging
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# Import local modules
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vocabulary import HierarchicalVocabulary
from src.sandhi_splitter import SandhiSplitter
from src.sandhi import MalayalamSandhi, get_proper_root_suffix


@dataclass
class TokenInfo:
    """Information about a single token."""
    text: str
    token_id: int
    token_type: str
    morpheme: Optional[str] = None
    is_oov: bool = False
    subword_ids: Optional[List[int]] = None


class MorphoHierarchicalTokenizer:
    """
    A novel morphological tokenizer for Malayalam that combines:
    - mlmorph for morphological analysis
    - Proper stem extraction (virama-ending roots)
    - Hierarchical vocabulary (roots, suffixes, case markers, etc.)
    - Sandhi splitting
    - Character-level fallback for OOV
    
    Key Innovation:
    Roots are stored in STEM form (ending with ്) to properly handle
    vowel sandhi transformations when morphemes combine.
    
    Example:
        പഠിക്കുന്നു → പഠിക്ക് (stem) + ുന്നു (suffix)
        This allows proper reconstruction:
        പഠിക്ക് + ുന്നു = പഠിക്കുന്നു ✅
        പഠിക്ക് + ണം = പഠിക്കണം ✅
    """
    
    def __init__(self, vocab_size: int = 8000, use_mlmorph: bool = True):
        """
        Initialize the tokenizer.
        
        Args:
            vocab_size: Maximum vocabulary size
            use_mlmorph: Whether to use mlmorph for morphological analysis
        """
        self.vocab_size = vocab_size
        self.use_mlmorph = use_mlmorph
        self.vocab = HierarchicalVocabulary()
        self.sandhi_splitter = SandhiSplitter()
        self.sandhi = MalayalamSandhi()  # NEW: Sandhi handler
        
        # Initialize mlmorph if available
        self.morph_analyzer = None
        if use_mlmorph:
            try:
                from mlmorph import Analyser
                self.morph_analyzer = Analyser()
                print("✓ mlmorph (Analyser) initialized successfully")
            except ImportError:
                print("⚠ mlmorph not available, using fallback mode")
                self.use_mlmorph = False
        
        # Character-level vocabulary for fallback
        self.char_vocab: Dict[str, int] = {}
        self.next_char_id = 7000  # Start character IDs from 7000
        
        # High-frequency word cache for speed optimization
        # This bypasses the slow FST calls for common words
        self.high_freq_cache: Dict[str, List[str]] = {}
        self.cache_hits = 0
        self._load_high_freq_cache()
        
        # Statistics
        self.stats = {
            'total_tokens': 0,
            'morphology_hits': 0,
            'oov_hits': 0,
            'char_fallback_hits': 0,
            'cache_hits': 0,
        }
    
    def _load_high_freq_cache(self) -> None:
        """Load high-frequency words from exceptions.json for fast lookup."""
        import json
        import os
        
        exceptions_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data', 'exceptions.json'
        )
        
        if os.path.exists(exceptions_path):
            try:
                with open(exceptions_path, 'r', encoding='utf-8') as f:
                    exceptions = json.load(f)
                    
                # Load high-frequency words
                if 'high_frequency_words' in exceptions:
                    self.high_freq_cache = exceptions['high_frequency_words']
                    print(f"✓ Loaded {len(self.high_freq_cache)} high-frequency words into cache")
            except Exception as e:
                print(f"⚠ Could not load exceptions.json: {e}")
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize text using NFKC normalization.
        
        This handles:
        - Different Unicode representations of same character
        - Normalization of chillu characters
        - Consistent representation of vowel signs
        """
        # NFKC normalization
        normalized = unicodedata.normalize('NFKC', text)
        
        # Remove zero-width joiners and non-joiners that may cause issues
        normalized = normalized.replace('\u200d', '')  # ZWJ
        normalized = normalized.replace('\u200c', '')  # ZWNJ
        
        return normalized
    
    def get_morphemes(self, word: str) -> List[str]:
        """
        Get morpheme decomposition of a word.
        
        Args:
            word: A single Malayalam word
            
        Returns:
            List of morphemes (root + suffixes)
        """
        # Handle empty input
        if not word:
            logger.warning("Empty word passed to get_morphemes")
            return ['<UNK>']
        
        # First, check high-frequency cache for speed
        if word in self.high_freq_cache:
            self.stats['cache_hits'] += 1
            logger.debug(f"Cache hit for '{word}'")
            return self.high_freq_cache[word]
        
        if self.morph_analyzer is None:
            return self._fallback_morpheme_split(word)
        
        try:
            analysis = self.morph_analyzer.analyse(word)
            
            if analysis and len(analysis) > 0:
                # mlmorph returns list of tuples: [('root<pos><tense>', weight), ...]
                # Take the best analysis (first one)
                analysis_item = analysis[0]
                if isinstance(analysis_item, tuple):
                    analysis_str = analysis_item[0]  # Get the analysis string
                else:
                    analysis_str = str(analysis_item)
                
                # Parse the analysis
                morphemes = self._parse_morph_analysis(analysis_str, word)
                
                if morphemes:
                    self.stats['morphology_hits'] += 1
                    logger.debug(f"mlmorph analysis: '{word}' → {morphemes}")
                    return morphemes
        except Exception as e:
            logger.debug(f"mlmorph failed for '{word}': {e}")
        
        return self._fallback_morpheme_split(word)
    
    def _parse_morph_analysis(self, analysis: str, original_word: str) -> List[str]:
        """
        Parse mlmorph analysis output into morphemes with proper stem extraction.
        
        Key improvement: Roots are converted to STEM form (ending with virama ്)
        to properly handle sandhi transformations.
        
        Example:
            mlmorph output: 'പഠിക്കുക<v><present>'
            Original word: പഠിക്കുന്നു
            Our output: ['പഠിക്ക്', 'ുന്നു']  # Stem form!
        """
        morphemes = []
        
        # mlmorph format: root<pos><tense> or root+suffix<pos><tense>
        # Example: 'പഠിക്കുക<v><present>' means root is പഠിക്കുക (infinitive)
        
        # Extract the root word (part before first <)
        root_match = re.match(r'^([^\s<]+)', analysis)
        if root_match:
            root = root_match.group(1)
            
            # The root might contain + for compound analysis
            if '+' in root:
                parts = root.split('+')
                raw_morphemes = [p.strip() for p in parts if p.strip()]
            else:
                raw_morphemes = [root]
            
            # Convert each morpheme to proper form
            if len(raw_morphemes) == 1 and raw_morphemes[0] != original_word:
                # Single root with suffix - need to find proper split
                root = raw_morphemes[0]
                
                # Use proper stem extraction
                stem, suffix = get_proper_root_suffix(original_word, root)
                
                if stem and suffix:
                    morphemes = [stem, suffix]
                elif stem:
                    morphemes = [stem]
                else:
                    # Fallback: try to find where root ends in original word
                    # Convert root to stem form
                    stem_form = self.sandhi.to_stem_form(root)
                    stem_base = stem_form[:-1] if stem_form.endswith('്') else stem_form
                    
                    if original_word.startswith(stem_base):
                        suffix = original_word[len(stem_base):]
                        if suffix:
                            morphemes = [stem_form, suffix]
                        else:
                            morphemes = [stem_form]
                    else:
                        # Root doesn't match - use fallback
                        return self._fallback_morpheme_split(original_word)
            else:
                morphemes = raw_morphemes
        
        # If we didn't get good morphemes, use fallback
        if not morphemes:
            return self._fallback_morpheme_split(original_word)
        
        return morphemes
    
    def _fallback_morpheme_split(self, word: str) -> List[str]:
        """
        Fallback morpheme splitting when mlmorph fails.
        
        Uses common suffix patterns and produces STEM forms (virama-ending).
        
        Example:
            പഠിക്കുന്നു → ['പഠിക്ക്', 'ുന്നു']  # Stem form!
        """
        morphemes = []
        
        # Common Malayalam suffixes in order of length (longest first)
        # These are stored WITH their vowel signs (not in stem form)
        common_suffixes = [
            # Tense/Aspect suffixes (vowel-initial, as they appear in words)
            'ുകയാണ്', 'ുകയും', 'ുന്നുണ്ട്', 'ിട്ടുണ്ട്',
            'ുന്നു', 'ുകയും', 'ിച്ചു', 'ിട്ടു', 'ുണ്ട്',
            'ുക', 'ില്ല', 'ില്ലെ', 'ില്ലാ', 'ാനായി',
            'ും', 'ാം', 'ണം', 'ാൻ', 'ിയ', 'ുന്ന',
            
            # Case suffixes
            'ിൽനിന്ന്', 'ിനോടൊപ്പം', 'ിൽക്കൂടി',
            'ിന്റെ', 'ിനോട്', 'ിലൂടെ', 'ിൽവച്ച്',
            'ിൽ', 'ിലെ', 'ിലും', 'ിന്', 'ിനെ', 'ിനാൽ',
            'ിനും', 'ക്ക്', 'ക്കും', 'ക്കായി',
            
            # Other common endings
            'ക്കാട്', 'പുരം', 'കുളം', 'ശ്ശേരി',
        ]
        
        remaining = word
        suffixes_found = []
        
        # Find suffixes from the end
        for suffix in common_suffixes:
            if remaining.endswith(suffix):
                stem = remaining[:-len(suffix)]
                if len(stem) >= 2:
                    suffixes_found.insert(0, suffix)
                    remaining = stem
                    break  # Take the longest matching suffix
        
        if suffixes_found:
            # Convert stem to proper form
            stem = remaining
            
            # Check if stem ends with a consonant (has inherent 'a')
            final_vowel = self.sandhi.get_final_vowel(stem)
            
            if final_vowel and final_vowel[1] == 'inherent':
                # Stem ends with consonant having inherent 'a'
                # For vowel-initial suffix, convert to stem form (add virama)
                if suffixes_found and suffixes_found[0][0] in self.sandhi.DEPENDENT_VOWELS:
                    # Suffix starts with vowel sign - add virama to stem
                    stem = stem + '്'
                # else: keep as-is (consonant-initial suffix)
            elif final_vowel and final_vowel[1] == 'dependent':
                # Stem ends with vowel sign
                # For vowel-initial suffix, replace vowel with virama
                if suffixes_found and suffixes_found[0][0] in self.sandhi.DEPENDENT_VOWELS:
                    stem = stem[:-1] + '്'
            
            morphemes = [stem] + suffixes_found
        else:
            # No suffix found - return word as single morpheme
            morphemes = [word]
        
        self.stats['oov_hits'] += 1
        return morphemes if morphemes else [word]
    
    def tokenize_word(self, word: str, depth: int = 0) -> List[TokenInfo]:
        """
        Tokenize a single word into hierarchical tokens.
        
        Args:
            word: A single Malayalam word
            depth: Recursion depth (used to prevent infinite recursion)
            
        Returns:
            List of TokenInfo objects
        """
        # Prevent infinite recursion
        MAX_DEPTH = 3
        if depth > MAX_DEPTH:
            # Fallback to simple morpheme splitting
            morphemes = self.get_morphemes(word)
            return self._create_tokens_for_morphemes(morphemes)
        
        tokens = []
        
        # Normalize the word
        word = self.normalize_text(word)
        
        # Check if it's a compound word
        if depth == 0 and self.sandhi_splitter.is_compound(word):
            components = self.sandhi_splitter.split_compound(word)
            for comp in components:
                tokens.extend(self.tokenize_word(comp, depth=depth+1))
            return tokens
        
        # Get morpheme decomposition
        morphemes = self.get_morphemes(word)
        
        return self._create_tokens_for_morphemes(morphemes)
    
    def _create_tokens_for_morphemes(self, morphemes: List[str]) -> List[TokenInfo]:
        """Create TokenInfo objects for a list of morphemes."""
        tokens = []
        
        for morpheme in morphemes:
            # Try to get from vocabulary
            if morpheme in self.vocab:
                token_id = self.vocab.get_token_id(morpheme)
                token_type = self.vocab.get_token_type(token_id)
                tokens.append(TokenInfo(
                    text=morpheme,
                    token_id=token_id,
                    token_type=token_type,
                    morpheme=morpheme,
                    is_oov=False
                ))
            else:
                # OOV - add to vocabulary or use character fallback
                if len(self.vocab) < self.vocab_size:
                    token_id = self.vocab.add_token(morpheme)
                    token_type = self.vocab.get_token_type(token_id)
                    tokens.append(TokenInfo(
                        text=morpheme,
                        token_id=token_id,
                        token_type=token_type,
                        morpheme=morpheme,
                        is_oov=False
                    ))
                else:
                    # Character-level fallback
                    char_tokens = self._tokenize_chars(morpheme)
                    tokens.extend(char_tokens)
        
        self.stats['total_tokens'] += len(tokens)
        return tokens
    
    def _tokenize_chars(self, text: str) -> List[TokenInfo]:
        """Tokenize at character level for OOV handling."""
        tokens = []
        
        # Split into Malayalam character units
        # A character unit can be: consonant + virama + consonant + vowel sign
        char_pattern = r'[\u0D00-\u0D7F][\u0D00-\u0D7F]*'
        chars = re.findall(char_pattern, text)
        
        if not chars:
            # Fallback to individual code points
            chars = list(text)
        
        for char in chars:
            if char not in self.char_vocab:
                self.char_vocab[char] = self.next_char_id
                self.next_char_id += 1
            
            token_id = self.char_vocab[char]
            tokens.append(TokenInfo(
                text=char,
                token_id=token_id,
                token_type='char',
                morpheme=None,
                is_oov=True
            ))
        
        self.stats['char_fallback_hits'] += len(tokens)
        return tokens
    
    def tokenize(self, text: str) -> List[int]:
        """
        Tokenize text into token IDs.
        
        Args:
            text: Input text (can be multiple words)
            
        Returns:
            List of token IDs
        """
        # Handle None or empty input
        if not text:
            logger.debug("Empty or None input, returning BOS+EOS")
            return [self.vocab.SPECIAL_TOKENS['<BOS>'], self.vocab.SPECIAL_TOKENS['<EOS>']]
        
        try:
            # Normalize text
            text = self.normalize_text(text)
            
            # Split into words
            words = re.findall(r'[\u0D00-\u0D7F]+', text)
            
            if not words:
                logger.debug("No Malayalam words found in input")
                return [self.vocab.SPECIAL_TOKENS['<BOS>'], self.vocab.SPECIAL_TOKENS['<EOS>']]
            
            token_ids = []
            
            # Add BOS token
            token_ids.append(self.vocab.SPECIAL_TOKENS['<BOS>'])
            
            for word in words:
                tokens = self.tokenize_word(word)
                for token in tokens:
                    token_ids.append(token.token_id)
            
            # Add EOS token
            token_ids.append(self.vocab.SPECIAL_TOKENS['<EOS>'])
            
            logger.debug(f"Tokenized {len(words)} words into {len(token_ids)} tokens")
            return token_ids
            
        except Exception as e:
            logger.error(f"Error tokenizing text: {e}")
            return [self.vocab.SPECIAL_TOKENS['<BOS>'], self.vocab.SPECIAL_TOKENS['<EOS>']]
    
    def tokenize_detailed(self, text: str) -> List[TokenInfo]:
        """
        Tokenize text with detailed information.
        
        Args:
            text: Input text
            
        Returns:
            List of TokenInfo objects with full details
        """
        text = self.normalize_text(text)
        words = re.findall(r'[\u0D00-\u0D7F]+', text)
        
        all_tokens = []
        for word in words:
            tokens = self.tokenize_word(word)
            all_tokens.extend(tokens)
        
        return all_tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to text with proper sandhi handling.
        
        When a stem (ending with virama) is followed by a vowel-initial suffix,
        the virama is removed and the suffix's vowel sign is applied.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Decoded text
        """
        tokens = []
        for token_id in token_ids:
            if token_id in self.vocab.id_to_token:
                tokens.append(self.vocab.id_to_token[token_id])
            elif token_id in self.char_vocab:
                # Reverse lookup in char vocab
                for char, cid in self.char_vocab.items():
                    if cid == token_id:
                        tokens.append(char)
                        break
        
        # Apply sandhi rules when combining tokens
        result = []
        for i, token in enumerate(tokens):
            if not result:
                result.append(token)
                continue
            
            prev = result[-1]
            
            # Check if previous token is a stem (ends with virama)
            # and current token starts with a vowel sign
            if prev.endswith('്'):
                # Get next token's initial character
                if token and token[0] in self.sandhi.DEPENDENT_VOWELS:
                    # Remove virama and append (sandhi: virama replaced by vowel)
                    result[-1] = prev[:-1] + token
                elif token and token[0] in self.sandhi.INDEPENDENT_VOWELS:
                    # Independent vowel after stem - rare, keep virama
                    result.append(token)
                else:
                    # Consonant-initial suffix - keep virama and append
                    result.append(token)
            else:
                # No virama at end - apply general sandhi
                combined = self.sandhi.apply_sandhi(prev, token)
                if combined != prev + token:
                    # Sandhi applied - replace last token with combined
                    result[-1] = combined
                else:
                    # No sandhi needed
                    result.append(token)
        
        return ''.join(result)
    
    def train(self, corpus: List[str], min_freq: int = 2) -> None:
        """
        Train the tokenizer on a corpus.
        
        Args:
            corpus: List of text strings
            min_freq: Minimum frequency for a token to be included
        """
        print(f"Training on {len(corpus)} documents...")
        
        all_morphemes = []
        
        for text in corpus:
            text = self.normalize_text(text)
            words = re.findall(r'[\u0D00-\u0D7F]+', text)
            
            for word in words:
                # Handle compounds
                if self.sandhi_splitter.is_compound(word):
                    components = self.sandhi_splitter.split_compound(word)
                    for comp in components:
                        morphemes = self.get_morphemes(comp)
                        all_morphemes.extend(morphemes)
                else:
                    morphemes = self.get_morphemes(word)
                    all_morphemes.extend(morphemes)
        
        # Build vocabulary from morphemes
        self.vocab.build_from_corpus(all_morphemes, min_freq)
        
        print(f"Vocabulary size: {len(self.vocab)}")
        print(f"  - Roots: {len(self.vocab.get_tokens_by_type('root'))}")
        print(f"  - Tense markers: {len(self.vocab.get_tokens_by_type('tense'))}")
        print(f"  - Case markers: {len(self.vocab.get_tokens_by_type('case'))}")
        print(f"  - Function words: {len(self.vocab.get_tokens_by_type('function'))}")
    
    def get_stats(self) -> Dict:
        """Get tokenizer statistics."""
        return {
            **self.stats,
            'vocab_size': len(self.vocab),
            'char_vocab_size': len(self.char_vocab),
            'morphology_coverage': (
                self.stats['morphology_hits'] / max(1, self.stats['total_tokens']) * 100
            ),
        }
    
    def save(self, path: str) -> None:
        """Save tokenizer to directory."""
        import json
        import os
        
        os.makedirs(path, exist_ok=True)
        
        # Save vocabulary
        self.vocab.save(os.path.join(path, 'vocab.json'))
        
        # Save char vocab
        with open(os.path.join(path, 'char_vocab.json'), 'w', encoding='utf-8') as f:
            json.dump(self.char_vocab, f, ensure_ascii=False, indent=2)
        
        # Save config
        config = {
            'vocab_size': self.vocab_size,
            'use_mlmorph': self.use_mlmorph,
            'stats': self.stats,
        }
        with open(os.path.join(path, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        
        print(f"Tokenizer saved to {path}")
    
    def load(self, path: str) -> None:
        """Load tokenizer from directory."""
        import json
        import os
        
        # Load vocabulary
        self.vocab.load(os.path.join(path, 'vocab.json'))
        
        # Load char vocab
        with open(os.path.join(path, 'char_vocab.json'), 'r', encoding='utf-8') as f:
            self.char_vocab = json.load(f)
        
        # Update next_char_id
        if self.char_vocab:
            self.next_char_id = max(self.char_vocab.values()) + 1
        
        print(f"Tokenizer loaded from {path}")


# Convenience function
def create_tokenizer(vocab_size: int = 8000) -> MorphoHierarchicalTokenizer:
    """Create a new MorphoHierarchicalTokenizer instance."""
    return MorphoHierarchicalTokenizer(vocab_size=vocab_size)
