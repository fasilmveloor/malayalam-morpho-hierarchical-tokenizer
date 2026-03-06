"""
Test Harness for Malayalam Morpho-Hierarchical Tokenizer

Tests the tokenizer against the SMC (Swathanthra Malayalam Computing) corpus
and generates performance metrics.

Metrics evaluated:
1. Tokenization efficiency (tokens per word)
2. Morphology coverage (% words with morphological analysis)
3. Vocabulary utilization
4. OOV rate
5. Compression ratio
"""

import os
import sys
import json
import time
import requests
from typing import List, Dict, Tuple
from collections import Counter
from dataclasses import dataclass, asdict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tokenizer import MorphoHierarchicalTokenizer
from src.vocabulary import HierarchicalVocabulary


@dataclass
class TestResults:
    """Container for test results."""
    total_words: int
    total_tokens: int
    unique_tokens: int
    oov_words: int
    morphology_coverage: float
    compression_ratio: float
    avg_tokens_per_word: float
    vocab_breakdown: Dict[str, int]
    processing_time: float
    words_per_second: float


class SMCCorpusLoader:
    """Downloads and loads the SMC Malayalam corpus."""
    
    # GitLab API for the SMC corpus repository
    REPO_API = "https://gitlab.com/api/v4/projects/smc%2Fcorpus"
    
    def __init__(self, cache_dir: str = "./data"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_file = os.path.join(cache_dir, "smc_corpus.txt")
    
    def download(self, max_files: int = 50) -> bool:
        """Download the corpus from SMC GitLab repository."""
        if os.path.exists(self.cache_file):
            # Check if file has content
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if len(content) > 1000:
                    print(f"✓ Corpus already cached at {self.cache_file}")
                    return True
        
        print("Downloading SMC corpus from GitLab...")
        
        all_text = []
        
        try:
            # Get list of files in text folder
            tree_url = f"{self.REPO_API}/repository/tree?path=text"
            response = requests.get(tree_url, timeout=30)
            
            if response.status_code != 200:
                print(f"✗ Failed to get file list: {response.status_code}")
                return False
            
            files = response.json()
            
            # Download text files
            downloaded = 0
            for file_info in files:
                if file_info['type'] == 'blob' and file_info['name'].endswith('.txt'):
                    if downloaded >= max_files:
                        break
                    
                    file_url = f"{self.REPO_API}/repository/files/text%2F{file_info['name']}/raw?ref=master"
                    try:
                        file_resp = requests.get(file_url, timeout=60)
                        if file_resp.status_code == 200:
                            all_text.append(file_resp.text)
                            downloaded += 1
                            if downloaded % 10 == 0:
                                print(f"  Downloaded {downloaded} files...")
                    except Exception as e:
                        print(f"  Warning: Failed to download {file_info['name']}: {e}")
                        continue
            
            # Also try subdirectories for more content
            subdirs = ['ml-wiki', 'oscar']  # High-content folders
            for subdir in subdirs:
                subdir_url = f"{self.REPO_API}/repository/tree?path=text%2F{subdir}"
                try:
                    subdir_resp = requests.get(subdir_url, timeout=30)
                    if subdir_resp.status_code == 200:
                        subdir_files = subdir_resp.json()
                        for file_info in subdir_files[:10]:  # Limit per subdir
                            if file_info['type'] == 'blob':
                                file_url = f"{self.REPO_API}/repository/files/text%2F{subdir}%2F{file_info['name']}/raw?ref=master"
                                try:
                                    file_resp = requests.get(file_url, timeout=60)
                                    if file_resp.status_code == 200:
                                        all_text.append(file_resp.text)
                                        downloaded += 1
                                except:
                                    continue
                except:
                    continue
            
            if all_text:
                combined = '\n'.join(all_text)
                with open(self.cache_file, 'w', encoding='utf-8') as f:
                    f.write(combined)
                print(f"✓ Downloaded {downloaded} files to {self.cache_file}")
                print(f"  Total size: {len(combined):,} characters")
                return True
            else:
                print("✗ No content downloaded")
                return False
                
        except Exception as e:
            print(f"✗ Download failed: {e}")
            return False
    
    def load(self, max_lines: int = None) -> List[str]:
        """Load corpus as list of lines."""
        if not os.path.exists(self.cache_file):
            if not self.download():
                return []
        
        with open(self.cache_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if max_lines:
            lines = lines[:max_lines]
        
        # Filter empty lines and non-Malayalam content
        lines = [
            line.strip() for line in lines 
            if line.strip() and any('\u0D00' <= c <= '\u0D7F' for c in line)
        ]
        
        return lines


class TokenizerTester:
    """Test harness for the Malayalam tokenizer."""
    
    def __init__(self, tokenizer: MorphoHierarchicalTokenizer):
        self.tokenizer = tokenizer
        self.results = None
    
    def extract_words(self, text: str) -> List[str]:
        """Extract Malayalam words from text."""
        import re
        return re.findall(r'[\u0D00-\u0D7F]+', text)
    
    def run_test(self, corpus: List[str], train: bool = True) -> TestResults:
        """
        Run comprehensive test on corpus.
        
        Args:
            corpus: List of text lines
            train: Whether to train the tokenizer on corpus first
        """
        print(f"\n{'='*60}")
        print("Running Tokenizer Test")
        print(f"{'='*60}")
        
        # Train tokenizer if requested
        if train:
            print("\n[1/4] Training tokenizer...")
            start_time = time.time()
            self.tokenizer.train(corpus, min_freq=2)
            train_time = time.time() - start_time
            print(f"    Training time: {train_time:.2f}s")
        
        # Extract all words
        print("\n[2/4] Extracting words from corpus...")
        all_words = []
        for line in corpus:
            words = self.extract_words(line)
            all_words.extend(words)
        
        word_counter = Counter(all_words)
        unique_words = len(word_counter)
        total_words = len(all_words)
        
        print(f"    Total words: {total_words:,}")
        print(f"    Unique words: {unique_words:,}")
        
        # Tokenize and collect statistics
        print("\n[3/4] Tokenizing corpus...")
        start_time = time.time()
        
        all_tokens = []
        oov_words = 0
        words_with_morphology = 0
        
        for i, word in enumerate(all_words):
            tokens = self.tokenizer.tokenize_word(word)
            all_tokens.extend([t.token_id for t in tokens])
            
            # Check if any token was OOV
            if any(t.is_oov for t in tokens):
                oov_words += 1
            
            # Check if morphology was used (non-OOV, non-char tokens)
            if any(t.token_type not in ['char', 'special'] for t in tokens):
                words_with_morphology += 1
            
            if (i + 1) % 10000 == 0:
                print(f"    Processed {i+1:,} words...")
        
        processing_time = time.time() - start_time
        
        # Calculate metrics
        print("\n[4/4] Calculating metrics...")
        
        token_counter = Counter(all_tokens)
        unique_tokens = len(token_counter)
        
        compression_ratio = total_words / len(all_tokens) if all_tokens else 0
        avg_tokens_per_word = len(all_tokens) / total_words if total_words else 0
        morphology_coverage = (words_with_morphology / total_words * 100) if total_words else 0
        
        # Vocabulary breakdown
        vocab_breakdown = {}
        for token_type in ['root', 'tense', 'case', 'function', 'conjunct', 'subword', 'special']:
            vocab_breakdown[token_type] = len(
                self.tokenizer.vocab.get_tokens_by_type(token_type)
            )
        
        # Create results object
        self.results = TestResults(
            total_words=total_words,
            total_tokens=len(all_tokens),
            unique_tokens=unique_tokens,
            oov_words=oov_words,
            morphology_coverage=morphology_coverage,
            compression_ratio=compression_ratio,
            avg_tokens_per_word=avg_tokens_per_word,
            vocab_breakdown=vocab_breakdown,
            processing_time=processing_time,
            words_per_second=total_words / processing_time if processing_time > 0 else 0
        )
        
        return self.results
    
    def print_results(self) -> None:
        """Print test results in a formatted way."""
        if self.results is None:
            print("No results available. Run test first.")
            return
        
        r = self.results
        
        print(f"\n{'='*60}")
        print("TEST RESULTS")
        print(f"{'='*60}")
        
        print(f"\n📊 Corpus Statistics:")
        print(f"   Total words processed: {r.total_words:,}")
        print(f"   Total tokens generated: {r.total_tokens:,}")
        print(f"   Unique tokens used: {r.unique_tokens:,}")
        
        print(f"\n📈 Performance Metrics:")
        print(f"   Morphology coverage: {r.morphology_coverage:.2f}%")
        print(f"   OOV word rate: {(r.oov_words/r.total_words*100):.2f}%")
        print(f"   Compression ratio: {r.compression_ratio:.4f}")
        print(f"   Avg tokens/word: {r.avg_tokens_per_word:.2f}")
        
        print(f"\n📚 Vocabulary Breakdown:")
        for token_type, count in r.vocab_breakdown.items():
            print(f"   {token_type}: {count:,}")
        
        print(f"\n⏱️ Performance:")
        print(f"   Processing time: {r.processing_time:.2f}s")
        print(f"   Words/second: {r.words_per_second:,.0f}")
        
        print(f"\n{'='*60}")
    
    def save_results(self, filepath: str) -> None:
        """Save results to JSON file."""
        if self.results is None:
            print("No results to save.")
            return
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(asdict(self.results), f, indent=2)
        print(f"Results saved to {filepath}")


def demo_tokenization():
    """Demonstrate tokenization with sample Malayalam text."""
    print("\n" + "="*60)
    print("TOKENIZATION DEMO")
    print("="*60)
    
    tokenizer = MorphoHierarchicalTokenizer()
    
    # Sample Malayalam sentences
    samples = [
        "ഞാൻ പഠിക്കുന്നു",  # I am learning
        "അവർ വിദ്യാലയത്തിൽ പോകുന്നു",  # They go to school
        "മലയാളം ഒരു സുന്ദരമായ ഭാഷയാണ്",  # Malayalam is a beautiful language
        "പാലക്കാട് കേരളത്തിലെ ഒരു ജില്ലയാണ്",  # Palakkad is a district in Kerala
    ]
    
    for text in samples:
        print(f"\n📝 Input: {text}")
        
        # Get detailed tokens
        tokens = tokenizer.tokenize_detailed(text)
        
        print("   Tokens:")
        for token in tokens:
            oov_marker = " [OOV]" if token.is_oov else ""
            print(f"     {token.text} → {token.token_id} ({token.token_type}){oov_marker}")
        
        # Get token IDs
        token_ids = tokenizer.tokenize(text)
        print(f"   Token IDs: {token_ids}")


def main():
    """Main test function."""
    print("\n" + "="*70)
    print("  MALAYALAM MORPHO-HIERARCHICAL TOKENIZER - TEST SUITE")
    print("="*70)
    
    # Demo first
    demo_tokenization()
    
    # Load corpus
    print("\n" + "="*60)
    print("LOADING SMC CORPUS")
    print("="*60)
    
    loader = SMCCorpusLoader(cache_dir="./data")
    corpus = loader.load(max_lines=10000)  # Limit for faster testing
    
    if not corpus:
        print("✗ Could not load corpus. Using sample data for testing.")
        corpus = [
            "മലയാളം ഒരു ദ്രാവിഡ ഭാഷയാണ്",
            "കേരളം ഇന്ത്യയിലെ ഒരു സംസ്ഥാനമാണ്",
            "തിരുവനന്തപുരം കേരളത്തിന്റെ തലസ്ഥാനമാണ്",
            "ഞാൻ പഠിക്കുന്നു",
            "അവർ വരുന്നു",
            "നമ്മൾ പോകുന്നു",
        ] * 100  # Repeat to simulate larger corpus
    
    print(f"✓ Loaded {len(corpus)} lines from corpus")
    
    # Create and test tokenizer
    tokenizer = MorphoHierarchicalTokenizer(vocab_size=5000)
    tester = TokenizerTester(tokenizer)
    
    # Run test
    results = tester.run_test(corpus, train=True)
    
    # Print and save results
    tester.print_results()
    tester.save_results("./data/test_results.json")
    
    # Save tokenizer
    tokenizer.save("./data/tokenizer")
    
    print("\n" + "="*60)
    print("✓ Testing complete!")
    print("="*60)


if __name__ == "__main__":
    main()
