#!/usr/bin/env python3
"""
Step 4: Prepare word list for spell checking
Usage: python 04_combine_corpora.py

Input:  clean_data/word_list.txt
Output: combined_data/words_for_validation.txt
"""

import json
from pathlib import Path

# Transliterations to remove
TRANSLITERATIONS = {
    'എസ്', 'എം', 'എൽ', 'എക്സ്', 'ബി', 'സി', 'ഡി', 'ടി', 'പി',
    'എൻ', 'ആർ', 'കെ', 'യു', 'വി', 'ഡബ്ല്യു', 'എച്ച്',
    'എ', 'ബി', 'സി', 'ഡി', 'ഇ', 'എഫ്', 'ജി', 'എച്ച്',
    'ഐ', 'ജെ', 'കെ', 'എൽ', 'എം', 'എൻ', 'ഒ', 'പി',
    'ക്യു', 'ആർ', 'എസ്', 'ടി', 'യു', 'വി', 'ഡബ്ല്യൂ', 'എക്സ്',
    'വൈ', 'സെഡ്',
}

def main():
    """Prepare word list for spell checking."""
    output_dir = Path("combined_data")
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("PREPARING WORD LIST FOR VALIDATION")
    print("=" * 60)
    
    # Load cleaned words
    input_path = Path("clean_data/word_list.txt")
    
    if not input_path.exists():
        print(f"✗ Not found: {input_path}")
        print("  Run: python 03_clean_corpus.py first")
        return
    
    with open(input_path, 'r', encoding='utf-8') as f:
        words = [line.strip() for line in f if line.strip()]
    
    print(f"\nInput: {len(words):,} words")
    
    # Remove transliterations
    cleaned = [w for w in words if w not in TRANSLITERATIONS]
    removed = len(words) - len(cleaned)
    
    print(f"Removed transliterations: {removed:,}")
    print(f"Final: {len(cleaned):,} words")
    
    # Save
    output_path = output_dir / "words_for_validation.txt"
    with open(output_path, 'w', encoding='utf-8') as f:
        for word in cleaned:
            f.write(word + '\n')
    
    # Stats
    stats = {
        'input': len(words),
        'transliterations_removed': removed,
        'output': len(cleaned),
    }
    with open(output_dir / "stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n✓ Saved: {output_path}")
    print("  Next: python 05_spell_check.py")

if __name__ == "__main__":
    main()
