"""
Neural Sandhi Splitter for Malayalam

A hybrid approach combining:
1. Rule-based patterns for common sandhi
2. Statistical features for ambiguous cases  
3. Ground truth from mlmorph when available

This is a prototype that demonstrates how a neural approach could work
for sandhi splitting in Malayalam.

Architecture:
    Input Word
        ↓
    Check High-Frequency Cache
        ↓
    Apply Rule-Based Patterns
        ↓
    Statistical Feature Analysis
        ↓
    mlmorph Ground Truth (if needed)
        ↓
    Output Split
"""

import re
from typing import List, Tuple, Optional, Dict
from collections import Counter
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sandhi_dictionary import get_split, lookup_compound
from src.sandhi import MalayalamSandhi


class NeuralSandhiSplitter:
    """
    A hybrid neural-style sandhi splitter for Malayalam.
    
    Uses:
    1. Dictionary lookup (fast path)
    2. Rule-based patterns (medium path)
    3. Statistical features (slow path)
    4. mlmorph ground truth (fallback)
    """
    
    def __init__(self, use_mlmorph: bool = True):
        self.sandhi = MalayalamSandhi()
        self.use_mlmorph = use_mlmorph
        self.morph_analyzer = None
        
        # Statistical model parameters
        self.boundary_features = {}
        self.suffix_freq = Counter()
        self.root_freq = Counter()
        
        # Initialize mlmorph if available
        if use_mlmorph:
            try:
                from mlmorph import Analyser
                self.morph_analyzer = Analyser()
                print("✓ NeuralSandhiSplitter: mlmorph initialized")
            except ImportError:
                print("⚠ NeuralSandhiSplitter: mlmorph not available")
                self.use_mlmorph = False
        
        # Load training statistics
        self._load_statistics()
    
    def _load_statistics(self):
        """Load pre-computed statistics for the statistical model."""
        # Common suffix frequencies (from Malayalam corpus analysis)
        self.suffix_freq = Counter({
            'ുന്നു': 5000,
            'ും': 4000,
            'ിച്ചു': 3000,
            'ിൽ': 8000,
            'ിന്റെ': 6000,
            'ിന്': 5000,
            'ിലെ': 3000,
            'ാൻ': 2500,
            'ണം': 2000,
            'ുക': 1500,
            'ിയ': 2000,
            'ുന്ന': 1500,
            'ക്ക്': 3000,
            'ത്തിൽ': 1500,
            'ത്തിന്റെ': 1000,
            'ത്തിലെ': 800,
        })
        
        # Boundary features - characters that often indicate split points
        self.boundary_features = {
            # Before these, often a split point
            'before': {
                'ു': 0.8,  # vowel sign u
                'ി': 0.7,   # vowel sign i
                'ാ': 0.6,   # vowel sign aa
                'ത്ത്': 0.9,  # noun inflection
                'ച്ച്': 0.9,  # past tense marker
            },
            # After these, often a split point
            'after': {
                '്': 0.9,   # virama - strong indicator
                'ം': 0.7,   # anusvara - before case markers
                'ർ': 0.5,   # chillu r
                'ൽ': 0.5,   # chillu l
                'ൺ': 0.5,   # chillu n
            }
        }
    
    def split(self, word: str) -> List[str]:
        """
        Split a compound word using the hybrid approach.
        
        Args:
            word: Input word (possibly compound)
            
        Returns:
            List of component morphemes
        """
        # Step 1: Check dictionary (fastest)
        dict_split = get_split(word)
        if dict_split and len(dict_split) > 1:
            return dict_split
        
        # Step 2: Check if it's a single morpheme
        if self._is_single_morpheme(word):
            return [word]
        
        # Step 3: Apply rule-based patterns
        rule_split = self._apply_rules(word)
        if rule_split and len(rule_split) > 1:
            return rule_split
        
        # Step 4: Use statistical features
        stat_split = self._statistical_split(word)
        if stat_split and len(stat_split) > 1:
            return stat_split
        
        # Step 5: Use mlmorph ground truth
        if self.morph_analyzer:
            mlmorph_split = self._mlmorph_split(word)
            if mlmorph_split:
                return mlmorph_split
        
        # Fallback: return as single word
        return [word]
    
    def _is_single_morpheme(self, word: str) -> bool:
        """Check if a word is likely a single morpheme."""
        # Very short words are usually single morphemes
        if len(word) < 4:
            return True
        
        # Check if word ends with common single-word patterns
        single_patterns = [
            'ം$',      # Words ending in anusvara (often nouns)
            '്$',      # Words ending in virama (stems)
        ]
        
        for pattern in single_patterns:
            if re.search(pattern, word):
                return True
        
        return False
    
    def _apply_rules(self, word: str) -> Optional[List[str]]:
        """Apply rule-based sandhi patterns."""
        
        # Pattern 1: Words ending in common suffixes
        for suffix, freq in self.suffix_freq.most_common():
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                root = word[:-len(suffix)]
                
                # Validate: root should not be empty
                if len(root) >= 2:
                    # Apply sandhi transformation
                    if root.endswith('ം'):
                        # Noun inflection: ം → ത്ത്
                        root = root[:-1] + 'ത്ത്'
                    elif not root.endswith('്') and suffix[0] in self.sandhi.DEPENDENT_VOWELS:
                        # Add virama to root for vowel-initial suffix
                        root = root + '്'
                    
                    return [root, suffix]
        
        # Pattern 2: Place names ending in common patterns
        place_suffixes = ['പുരം', 'കാട്', 'കുളം', 'നാട്', 'ക്കര', 'മല', 'ത്തറ', 'പള്ളി']
        for suffix in place_suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                root = word[:-len(suffix)]
                if len(root) >= 2:
                    return [root, suffix]
        
        # Pattern 3: Compound words with ത്ത് infix (from ം nouns)
        if 'ത്തി' in word:
            idx = word.find('ത്തി')
            if idx > 2:
                # Could be noun + case marker
                root = word[:idx]
                suffix = word[idx:]
                # The original noun would have ended in ം
                original_root = root + 'ം' if not root.endswith('ം') else root
                return [root + '്', suffix]
        
        return None
    
    def _statistical_split(self, word: str) -> Optional[List[str]]:
        """Use statistical features to find split points."""
        
        # Calculate split scores at each position
        scores = []
        
        for i in range(2, len(word) - 2):
            left = word[:i]
            right = word[i:]
            
            score = 0
            
            # Feature 1: Does left end with virama?
            if left.endswith('്'):
                score += 0.9
            
            # Feature 2: Does right start with vowel sign?
            if right and right[0] in self.sandhi.DEPENDENT_VOWELS:
                score += 0.7
            
            # Feature 3: Is right a common suffix?
            for suffix, freq in self.suffix_freq.items():
                if right.startswith(suffix):
                    score += min(1.0, freq / 5000)
                    break
            
            # Feature 4: Length balance (prefer roughly equal splits)
            length_ratio = min(len(left), len(right)) / max(len(left), len(right))
            score += length_ratio * 0.3
            
            scores.append((i, score))
        
        # Find best split point
        if scores:
            best_idx, best_score = max(scores, key=lambda x: x[1])
            
            if best_score > 0.7:  # Threshold
                left = word[:best_idx]
                right = word[best_idx:]
                
                # Validate and adjust
                if left and right:
                    # If left doesn't end with virama and right starts with vowel,
                    # add virama to left
                    if not left.endswith('്') and right[0] in self.sandhi.DEPENDENT_VOWELS:
                        if not left.endswith('ം'):  # Not an anusvara case
                            left = left + '്'
                    
                    return [left, right]
        
        return None
    
    def _mlmorph_split(self, word: str) -> Optional[List[str]]:
        """Use mlmorph for ground truth splitting."""
        if not self.morph_analyzer:
            return None
        
        try:
            analysis = self.morph_analyzer.analyse(word)
            
            if analysis and len(analysis) > 0:
                analysis_item = analysis[0]
                if isinstance(analysis_item, tuple):
                    analysis_str = analysis_item[0]
                else:
                    analysis_str = str(analysis_item)
                
                # Parse the analysis
                root_match = re.match(r'^([^\s<]+)', analysis_str)
                if root_match:
                    root = root_match.group(1)
                    
                    # Convert to stem form
                    if root != word:
                        stem = self.sandhi.to_stem_form(root)
                        stem_base = stem[:-1] if stem.endswith('്') else stem
                        
                        if word.startswith(stem_base):
                            suffix = word[len(stem_base):]
                            if suffix:
                                return [stem, suffix]
        
        except Exception:
            pass
        
        return None
    
    def train_on_corpus(self, corpus: List[str]):
        """
        Train the statistical model on a corpus.
        
        This updates suffix and root frequencies.
        """
        print(f"Training on {len(corpus)} lines...")
        
        for line in corpus:
            words = re.findall(r'[\u0D00-\u0D7F]+', line)
            
            for word in words:
                # Get split
                split = self.split(word)
                
                if len(split) > 1:
                    # Update frequencies
                    root = split[0]
                    suffix = ''.join(split[1:])
                    
                    self.root_freq[root] += 1
                    self.suffix_freq[suffix] += 1
        
        print(f"Trained: {len(self.root_freq)} roots, {len(self.suffix_freq)} suffixes")


def demo():
    """Demo the neural sandhi splitter."""
    print("\n" + "="*60)
    print("Neural Sandhi Splitter Demo")
    print("="*60)
    
    splitter = NeuralSandhiSplitter()
    
    test_words = [
        # Place names
        "തിരുവനന്തപുരം",
        "പാലക്കാട്",
        "വിദ്യാലയം",
        "വിദ്യാലയത്തിൽ",
        
        # Verb forms
        "പഠിക്കുന്നു",
        "പഠിക്കണം",
        "പഠിച്ചു",
        "വരുന്നു",
        "വന്നു",
        
        # Compound nouns
        "പ്രധാനമന്ത്രി",
        "രക്തസമ്മർദ്ദം",
        
        # Case marked nouns
        "വീട്ടിൽ",
        "കേരളത്തിൽ",
        "വിദ്യാർത്ഥിയുടെ",
    ]
    
    print()
    for word in test_words:
        split = splitter.split(word)
        dict_entry = lookup_compound(word)
        
        source = ""
        if dict_entry:
            source = "📖 Dictionary"
        elif len(split) > 1:
            source = "🧠 Hybrid"
        else:
            source = "📝 Single"
        
        print(f"{word}")
        print(f"  → {' + '.join(split)}")
        print(f"  Source: {source}")
        print()


if __name__ == "__main__":
    demo()
