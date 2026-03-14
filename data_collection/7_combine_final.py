#!/usr/bin/env python3
"""
Step 7: Combine all valid words for final output
Usage: python 7_combine_final.py

Input:  validated_data/words_valid.txt + pattern_validated/words_pattern_valid.txt
Output: final_corpus/words_final.txt
"""

import json
from pathlib import Path

def combine_final():
    """Combine mlmorph-valid + pattern-valid words."""
    output_dir = Path("final_corpus")
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 50)
    print("COMBINING ALL VALID WORDS")
    print("=" * 50)
    
    all_words = set()
    
    # Load mlmorph valid words
    print("\n[1] Loading mlmorph-validated words...")
    mlmorph_path = Path("validated_data/words_valid.txt")
    if mlmorph_path.exists():
        with open(mlmorph_path, 'r', encoding='utf-8') as f:
            mlmorph_words = {line.strip() for line in f if line.strip()}
        all_words.update(mlmorph_words)
        print(f"    mlmorph valid: {len(mlmorph_words):,}")
    else:
        print("    ✗ File not found")
        mlmorph_words = set()
    
    # Load pattern valid words
    print("\n[2] Loading pattern-validated words...")
    pattern_path = Path("pattern_validated/words_pattern_valid.txt")
    if pattern_path.exists():
        with open(pattern_path, 'r', encoding='utf-8') as f:
            pattern_words = {line.strip() for line in f if line.strip()}
        all_words.update(pattern_words)
        print(f"    pattern valid: {len(pattern_words):,}")
    else:
        print("    ✗ File not found")
        pattern_words = set()
    
    # Stats
    overlap = len(mlmorph_words) + len(pattern_words) - len(all_words)
    
    print(f"\n[3] Combined Statistics:")
    print(f"    mlmorph valid:  {len(mlmorph_words):>10,}")
    print(f"    pattern valid:  {len(pattern_words):>10,}")
    print(f"    overlap:        {overlap:>10,}")
    print(f"    total unique:   {len(all_words):>10,}")
    
    # Save
    output_file = output_dir / "words_final.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        for word in sorted(all_words):
            f.write(word + '\n')
    
    # Stats
    stats = {
        'mlmorph_valid': len(mlmorph_words),
        'pattern_valid': len(pattern_words),
        'overlap': overlap,
        'total_unique': len(all_words),
    }
    with open(output_dir / "stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n{'='*50}")
    print("FINAL CORPUS READY")
    print(f"{'='*50}")
    print(f"  ✓ {len(all_words):,} words")
    print(f"  ✓ Saved: {output_file}")
    print(f"\n  Next step: Run mlmorph for training labels")
    
    return all_words

if __name__ == "__main__":
    combine_final()
    