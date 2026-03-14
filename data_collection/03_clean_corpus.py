#!/usr/bin/env python3
"""
Step 3: Clean corpus and extract unique Malayalam words
Usage: python 03_clean_corpus.py

Input:  raw_data/wiki_text.txt + raw_data/smc_corpus.txt + raw_data/smc_wordlist.txt
Output: clean_data/word_frequency.json, clean_data/word_list.txt
"""

import re
import json
import unicodedata
from collections import Counter
from pathlib import Path

MALAYALAM_RANGE = (0x0D00, 0x0D7F)

# Malayalam consonants
CONSONANTS = set('കഖഗഘങചഛജഝഞടഠഡഢണതഥദധനപഫബഭമയരലവശഷസഹളഴറ')
CHILLU = set('ർൽൾൻൺ')

def is_malayalam_word(word):
    """Check if word is valid Malayalam."""
    if len(word) < 2:
        return False
    
    # Must have consonant or chillu
    if not any(c in CONSONANTS or c in CHILLU for c in word):
        return False
    
    # Must be 90% Malayalam
    ml_count = sum(1 for c in word if MALAYALAM_RANGE[0] <= ord(c) <= MALAYALAM_RANGE[1])
    if len(word) == 0:
        return False
    return ml_count / len(word) > 0.9

def extract_words_from_text(text, source_name=""):
    """Extract Malayalam words from text."""
    word_counter = Counter()
    
    # Split into sentences first
    sentences = re.split(r'[।\n\.!?]+', text)
    
    sentence_count = 0
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        
        # Check if sentence is Malayalam-heavy
        ml_chars = sum(1 for c in sent if MALAYALAM_RANGE[0] <= ord(c) <= MALAYALAM_RANGE[1])
        total_alpha = sum(1 for c in sent if c.isalpha() or '\u0D00' <= c <= '\u0D7F')
        
        if total_alpha > 0 and ml_chars / total_alpha > 0.6:
            sentence_count += 1
            words = re.split(r'[\s.,;:!?()\[\]{}"\'«»—–-]+', sent)
            
            for word in words:
                word = unicodedata.normalize('NFC', word.strip())
                if is_malayalam_word(word):
                    word_counter[word] += 1
    
    print(f"  {source_name}: {sentence_count:,} sentences, {sum(word_counter.values()):,} words, {len(word_counter):,} unique")
    return word_counter

def process_wordlist(file_path, source_name=""):
    """Process a wordlist file (one word per line)."""
    word_counter = Counter()
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            word = unicodedata.normalize('NFC', line.strip())
            if is_malayalam_word(word):
                word_counter[word] += 1
    
    print(f"  {source_name}: {len(word_counter):,} words")
    return word_counter

def main():
    """Process all corpora and combine."""
    output_dir = Path("clean_data")
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("CLEANING CORPORA")
    print("=" * 60)
    
    # Combined word counter
    all_words = Counter()
    
    # 1. Process Wikipedia extracted text
    print("\n[1] Processing Wikipedia text...")
    wiki_path = Path("raw_data/wiki_text.txt")
    if wiki_path.exists():
        with open(wiki_path, 'r', encoding='utf-8') as f:
            wiki_text = f.read()
        wiki_words = extract_words_from_text(wiki_text, "Wikipedia")
        all_words.update(wiki_words)
    else:
        print(f"  ✗ Not found: {wiki_path}")
    
    # 2. Process SMC Corpus
    print("\n[2] Processing SMC Corpus...")
    smc_corpus_path = Path("raw_data/smc_corpus.txt")
    if smc_corpus_path.exists():
        with open(smc_corpus_path, 'r', encoding='utf-8') as f:
            smc_text = f.read()
        smc_words = extract_words_from_text(smc_text, "SMC Corpus")
        all_words.update(smc_words)
    else:
        print(f"  ✗ Not found: {smc_corpus_path}")
    
    # 3. Process SMC Wordlist
    print("\n[3] Processing SMC Wordlist...")
    smc_wordlist_path = Path("raw_data/smc_wordlist.txt")
    if smc_wordlist_path.exists():
        smc_list_words = process_wordlist(smc_wordlist_path, "SMC Wordlist")
        all_words.update(smc_list_words)
    else:
        print(f"  ✗ Not found: {smc_wordlist_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("COMBINED RESULTS")
    print("=" * 60)
    print(f"  Total unique words: {len(all_words):,}")
    print(f"  Total word tokens:  {sum(all_words.values()):,}")
    
    # Top words
    print(f"\n  Top 10 words:")
    for word, count in all_words.most_common(10):
        print(f"    {word}: {count:,}")
    
    # Save outputs
    print("\n" + "-" * 60)
    print("SAVING OUTPUTS")
    print("-" * 60)
    
    # Word frequency JSON
    freq_path = output_dir / "word_frequency.json"
    with open(freq_path, 'w', encoding='utf-8') as f:
        json.dump(dict(all_words.most_common()), f, ensure_ascii=False, indent=2)
    print(f"  ✓ {freq_path}")
    
    # Word list TXT
    list_path = output_dir / "word_list.txt"
    with open(list_path, 'w', encoding='utf-8') as f:
        for word, _ in all_words.most_common():
            f.write(word + '\n')
    print(f"  ✓ {list_path}")
    
    # Stats
    stats = {
        'unique_words': len(all_words),
        'total_tokens': sum(all_words.values()),
        'top_100': all_words.most_common(100)
    }
    stats_path = output_dir / "stats.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"  ✓ {stats_path}")
    
    print("\n✓ Corpus cleaning complete!")
    print("  Next: python 04_combine_corpora.py")

if __name__ == "__main__":
    main()
