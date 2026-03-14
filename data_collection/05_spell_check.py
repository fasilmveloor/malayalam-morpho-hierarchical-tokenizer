#!/usr/bin/env python3
"""
Step 5: Spell check using mlmorph
Usage: python 5_spell_check.py

Input:  combined_data/words_for_validation.txt
Output: validated_data/words_valid.txt, validated_data/words_needs_review.txt
"""

import json
from pathlib import Path
from mlmorph import Analyser

def spell_check(input_path, output_dir):
    """Validate words using mlmorph."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load words
    with open(input_path, 'r', encoding='utf-8') as f:
        words = [line.strip() for line in f if line.strip()]
    
    print(f"Spell checking {len(words):,} words with mlmorph...")
    
    # Initialize mlmorph
    print("Initializing mlmorph...")
    analyser = Analyser()
    
    valid_words = []
    review_words = []
    stats = {'total': len(words), 'valid': 0, 'review': 0, 'error': 0}
    
    for i, word in enumerate(words):
        if (i + 1) % 50000 == 0:
            print(f"  Progress: {i+1:,} / {len(words):,} | Valid: {stats['valid']:,}")
        
        try:
            analyses = analyser.analyse(word)
            
            if analyses and len(analyses) > 0:
                valid_words.append(word)
                stats['valid'] += 1
            else:
                review_words.append(word)
                stats['review'] += 1
                
        except Exception as e:
            review_words.append(word)
            stats['error'] += 1
    
    # Save valid words
    with open(output_dir / "words_valid.txt", 'w', encoding='utf-8') as f:
        for word in valid_words:
            f.write(word + '\n')
    
    # Save review words
    with open(output_dir / "words_needs_review.txt", 'w', encoding='utf-8') as f:
        for word in review_words:
            f.write(word + '\n')
    
    # Stats
    with open(output_dir / "stats.json", 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n{'='*50}")
    print("SPELL CHECK RESULTS")
    print(f"{'='*50}")
    print(f"  Total:      {stats['total']:>12,}")
    print(f"  Valid:      {stats['valid']:>12,} ({stats['valid']/stats['total']*100:.1f}%)")
    print(f"  Review:     {stats['review']:>12,} ({stats['review']/stats['total']*100:.1f}%)")
    print(f"  Errors:     {stats['error']:>12,}")
    print(f"{'='*50}")
    print(f"\n✓ Valid: {output_dir / 'words_valid.txt'}")
    print(f"✓ Review: {output_dir / 'words_needs_review.txt'}")
    
    return valid_words, review_words

if __name__ == "__main__":
    spell_check("combined_data/words_for_validation.txt", "validated_data")
