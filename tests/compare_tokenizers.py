"""
Comparison Test: Morpho-Hierarchical vs Baseline Tokenizers

Compares our tokenizer against:
1. Standard BPE (sentencepiece)
2. Standard Unigram (sentencepiece)
3. Character-level tokenization

Metrics compared:
- Tokenization efficiency
- Morphological alignment
- OOV rate
- Compression ratio
"""

import os
import sys
import time
import json
from typing import List, Dict, Tuple
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tokenizer import MorphoHierarchicalTokenizer

try:
    import sentencepiece as spm
    HAS_SP = True
except ImportError:
    HAS_SP = False
    print("⚠ sentencepiece not available for comparison")


class BaselineComparison:
    """Compare Morpho-Hierarchical tokenizer against baselines."""
    
    def __init__(self, vocab_size: int = 5000):
        self.vocab_size = vocab_size
        self.results = {}
    
    def extract_words(self, text: str) -> List[str]:
        """Extract Malayalam words from text."""
        import re
        return re.findall(r'[\u0D00-\u0D7F]+', text)
    
    def train_bpe(self, corpus: List[str]) -> bool:
        """Train a BPE tokenizer using sentencepiece."""
        if not HAS_SP:
            return False
        
        # Write corpus to temp file
        temp_file = "./data/temp_corpus.txt"
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(corpus))
        
        # Train BPE model
        model_prefix = "./data/bpe_model"
        spm.SentencePieceTrainer.train(
            input=temp_file,
            model_prefix=model_prefix,
            vocab_size=self.vocab_size,
            model_type='bpe',
            character_coverage=1.0,
        )
        
        self.bpe_model = spm.SentencePieceProcessor()
        self.bpe_model.load(f"{model_prefix}.model")
        return True
    
    def train_unigram(self, corpus: List[str]) -> bool:
        """Train a Unigram tokenizer using sentencepiece."""
        if not HAS_SP:
            return False
        
        temp_file = "./data/temp_corpus.txt"
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(corpus))
        
        model_prefix = "./data/unigram_model"
        spm.SentencePieceTrainer.train(
            input=temp_file,
            model_prefix=model_prefix,
            vocab_size=self.vocab_size,
            model_type='unigram',
            character_coverage=1.0,
        )
        
        self.unigram_model = spm.SentencePieceProcessor()
        self.unigram_model.load(f"{model_prefix}.model")
        return True
    
    def count_morphological_tokens(self, tokens: List[str]) -> int:
        """
        Count tokens that align with morphological boundaries.
        
        This is a heuristic that checks if tokens are likely morphemes.
        """
        morphological_count = 0
        
        # Common morpheme patterns
        suffix_patterns = [
            'ുന്നു', 'ിച്ചു', 'ും', 'ാൻ', 'ിയ', 'ുന്ന',
            'ിൽ', 'ിന്', 'ിന്റെ', 'ക്ക്', 'ിലെ',
            'ാണ്', 'ായ', 'ില്ല', 'ുണ്ട്',
        ]
        
        for token in tokens:
            # Check if token is a known suffix or root
            for pattern in suffix_patterns:
                if token.endswith(pattern) or token == pattern:
                    morphological_count += 1
                    break
        
        return morphological_count
    
    def evaluate_tokenizer(self, name: str, tokenize_func, corpus: List[str]) -> Dict:
        """Evaluate a tokenizer on corpus."""
        total_tokens = 0
        total_words = 0
        unique_tokens = set()
        morphological_tokens = 0
        
        start_time = time.time()
        
        for line in corpus:
            words = self.extract_words(line)
            total_words += len(words)
            
            for word in words:
                tokens = tokenize_func(word)
                total_tokens += len(tokens)
                unique_tokens.update(tokens)
                
                # Check morphological alignment
                if isinstance(tokens[0], str):
                    morphological_tokens += self.count_morphological_tokens(tokens)
        
        processing_time = time.time() - start_time
        
        return {
            'total_words': total_words,
            'total_tokens': total_tokens,
            'unique_tokens': len(unique_tokens),
            'avg_tokens_per_word': total_tokens / total_words if total_words else 0,
            'compression_ratio': total_words / total_tokens if total_tokens else 0,
            'morphological_alignment': morphological_tokens / total_tokens if total_tokens else 0,
            'processing_time': processing_time,
            'words_per_second': total_words / processing_time if processing_time else 0,
        }
    
    def run_comparison(self, corpus: List[str]) -> Dict:
        """Run full comparison across all tokenizers."""
        print("\n" + "="*70)
        print("  TOKENIZER COMPARISON")
        print("="*70)
        
        results = {}
        
        # 1. Morpho-Hierarchical Tokenizer
        print("\n[1/4] Testing Morpho-Hierarchical Tokenizer...")
        morpho_tokenizer = MorphoHierarchicalTokenizer(vocab_size=self.vocab_size)
        morpho_tokenizer.train(corpus, min_freq=2)
        
        def morpho_tokenize(word):
            tokens = morpho_tokenizer.tokenize_word(word)
            return [t.text for t in tokens]
        
        results['morpho-hierarchical'] = self.evaluate_tokenizer(
            'Morpho-Hierarchical', morpho_tokenize, corpus
        )
        results['morpho-hierarchical']['vocab_breakdown'] = {
            t: len(morpho_tokenizer.vocab.get_tokens_by_type(t))
            for t in ['root', 'tense', 'case', 'function', 'subword']
        }
        
        # 2. BPE Tokenizer
        if HAS_SP:
            print("\n[2/4] Testing BPE Tokenizer...")
            if self.train_bpe(corpus):
                def bpe_tokenize(word):
                    return self.bpe_model.encode(word, out_type=str)
                
                results['bpe'] = self.evaluate_tokenizer(
                    'BPE', bpe_tokenize, corpus
                )
        
        # 3. Unigram Tokenizer
        if HAS_SP:
            print("\n[3/4] Testing Unigram Tokenizer...")
            if self.train_unigram(corpus):
                def unigram_tokenize(word):
                    return self.unigram_model.encode(word, out_type=str)
                
                results['unigram'] = self.evaluate_tokenizer(
                    'Unigram', unigram_tokenize, corpus
                )
        
        # 4. Character-level Tokenizer
        print("\n[4/4] Testing Character-level Tokenizer...")
        def char_tokenize(word):
            import re
            # return re.findall(r'[\u0D00-\u0D7F][\u0D00-\u0D7F]*|.', word)
            return list(word)
        
        results['character'] = self.evaluate_tokenizer(
            'Character', char_tokenize, corpus
        )
        
        self.results = results
        return results
    
    def print_comparison(self):
        """Print comparison table."""
        if not self.results:
            print("No results to display.")
            return
        
        print("\n" + "="*90)
        print("  COMPARISON RESULTS")
        print("="*90)
        
        # Header
        print(f"\n{'Metric':<30}", end='')
        for name in self.results.keys():
            print(f"{name:>18}", end='')
        print()
        print("-"*90)
        
        # Metrics
        metrics = [
            ('Total Words', 'total_words', 'd'),
            ('Total Tokens', 'total_tokens', 'd'),
            ('Unique Tokens', 'unique_tokens', 'd'),
            ('Avg Tokens/Word', 'avg_tokens_per_word', '.2f'),
            ('Compression Ratio', 'compression_ratio', '.4f'),
            ('Morphological Alignment', 'morphological_alignment', '.2%'),
            ('Processing Time (s)', 'processing_time', '.2f'),
            ('Words/Second', 'words_per_second', ',.0f'),
        ]
        
        for label, key, fmt in metrics:
            print(f"{label:<30}", end='')
            for name, data in self.results.items():
                value = data.get(key, 0)
                if fmt == 'd':
                    print(f"{value:>18,}", end='')
                elif fmt == ',.0f':
                    print(f"{value:>18,.0f}", end='')
                elif fmt == '.2%':
                    print(f"{value:>18.2%}", end='')
                else:
                    print(f"{value:>18{fmt}}", end='')
            print()
        
        print("\n" + "="*90)
        
        # Vocabulary breakdown for Morpho-Hierarchical
        if 'morpho-hierarchical' in self.results:
            print("\n📚 Morpho-Hierarchical Vocabulary Breakdown:")
            for token_type, count in self.results['morpho-hierarchical'].get('vocab_breakdown', {}).items():
                print(f"   {token_type}: {count:,}")
    
    def save_results(self, filepath: str):
        """Save comparison results to JSON."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {filepath}")


def demonstrate_differences():
    """Show tokenization differences on sample words."""
    print("\n" + "="*70)
    print("  TOKENIZATION DIFFERENCES DEMONSTRATION")
    print("="*70)
    
    sample_words = [
        "പഠിക്കുന്നു",  # learning (present tense)
        "വിദ്യാലയത്തിൽ",  # in school
        "തിരുവനന്തപുരം",  # Thiruvananthapuram (compound)
        "എഴുതിയിരിക്കുന്നു",  # has written (perfect)
    ]
    
    # Initialize tokenizers
    morpho = MorphoHierarchicalTokenizer(vocab_size=3000)
    morpho.train([
        "ഞാൻ പഠിക്കുന്നു",
        "അവർ വിദ്യാലയത്തിൽ പോകുന്നു",
        "തിരുവനന്തപുരം കേരളത്തിന്റെ തലസ്ഥാനമാണ്",
        "അദ്ദേഹം കത്ത് എഴുതിയിരിക്കുന്നു",
    ] * 100, min_freq=1)
    
    for word in sample_words:
        print(f"\n📝 Word: {word}")
        
        # Morpho-Hierarchical
        tokens = morpho.tokenize_word(word)
        token_strs = [t.text for t in tokens]
        types = [t.token_type for t in tokens]
        print(f"   Morpho-Hierarchical: {' | '.join(token_strs)}")
        print(f"                        Types: {types}")
        
        # Character-level
        import re
        chars = re.findall(r'[\u0D00-\u0D7F][\u0D00-\u0D7F]*', word)
        print(f"   Character-level:      {' | '.join(chars)}")


def main():
    """Run comparison tests."""
    from test_tokenizer import SMCCorpusLoader
    
    print("\n" + "="*70)
    print("  MALAYALAM TOKENIZER COMPARISON TEST")
    print("="*70)
    
    # Show tokenization differences
    demonstrate_differences()
    
    # Load corpus
    print("\n" + "="*60)
    print("LOADING CORPUS FOR COMPARISON")
    print("="*60)
    
    loader = SMCCorpusLoader(cache_dir="./data")
    corpus = loader.load(max_lines=5000)
    
    if not corpus:
        print("Using sample corpus...")
        corpus = [
            "മലയാളം ഒരു ദ്രാവിഡ ഭാഷയാണ്",
            "കേരളം ഇന്ത്യയിലെ ഒരു സംസ്ഥാനമാണ്",
        ] * 500
    
    print(f"Loaded {len(corpus)} lines")
    
    # Run comparison
    comparison = BaselineComparison(vocab_size=5000)
    comparison.run_comparison(corpus)
    comparison.print_comparison()
    comparison.save_results("./data/comparison_results.json")


if __name__ == "__main__":
    main()
