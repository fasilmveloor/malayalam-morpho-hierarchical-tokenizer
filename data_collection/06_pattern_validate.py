#!/usr/bin/env python3
"""
Step 6: Pattern-based validation for review words
Usage: python 6_pattern_validate.py

Input:  validated_data/words_needs_review.txt
Output: pattern_validated/words_pattern_valid.txt
"""

import re
import json
from pathlib import Path

# Malayalam Unicode range
ML_RANGE = (0x0D00, 0x0D7F)

# Valid starting characters
VALID_START = set(
    'അആഇഈഉഊഋഎഏഐഒഓഔകഖഗഘങചഛജഝഞടഠഡഢണതഥദധനപഫബഭമയരലവശഷസഹളഴറ'
    'ർൽൾൻൺ'  # Chillu
)

# Wiki markup to reject
WIKI_MARKERS = set('*>|/%_#&@!=-')

def is_valid_malayalam(word):
    """Pattern-based validation."""
    if not word or len(word) < 2:
        return False, "too_short"
    
    # Reject wiki artifacts
    if word[0] in WIKI_MARKERS:
        return False, "wiki_marker"
    
    if word.startswith('ം'):
        return False, "broken_anusvara"
    
    if word[0] not in VALID_START:
        return False, "invalid_start"
    
    # Must be >90% Malayalam
    ml_count = sum(1 for c in word if ML_RANGE[0] <= ord(c) <= ML_RANGE[1])
    if ml_count / len(word) < 0.9:
        return False, "not_malayalam"
    
    # Must have consonant or chillu
    consonants = set('കഖഗഘങചഛജഝഞടഠഡഢണതഥദധനപഫബഭമയരലവശഷസഹളഴറർൽൾൻൺ')
    if not any(c in consonants for c in word):
        return False, "no_consonant"
    
    return True, "valid"

def pattern_validate(input_path, output_dir):
    """Validate words using character patterns."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    with open(input_path, 'r', encoding='utf-8') as f:
        words = [line.strip() for line in f if line.strip()]
    
    print(f"Pattern validating {len(words):,} words...")
    
    valid = []
    invalid = []
    reasons = {}
    
    for word in words:
        is_valid, reason = is_valid_malayalam(word)
        reasons[reason] = reasons.get(reason, 0) + 1
        
        if is_valid:
            valid.append(word)
        else:
            invalid.append((word, reason))
    
    # Save valid
    with open(output_dir / "words_pattern_valid.txt", 'w', encoding='utf-8') as f:
        for word in valid:
            f.write(word + '\n')
    
    # Save invalid with reasons
    with open(output_dir / "words_pattern_invalid.txt", 'w', encoding='utf-8') as f:
        for word, reason in invalid:
            f.write(f"{word}\t# {reason}\n")
    
    # Stats
    stats = {
        'total': len(words),
        'valid': len(valid),
        'invalid': len(invalid),
        'rejection_reasons': {k: v for k, v in reasons.items() if k != 'valid'}
    }
    with open(output_dir / "stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n{'='*50}")
    print("PATTERN VALIDATION RESULTS")
    print(f"{'='*50}")
    print(f"  Total:   {len(words):>12,}")
    print(f"  Valid:   {len(valid):>12,} ({len(valid)/len(words)*100:.1f}%)")
    print(f"  Invalid: {len(invalid):>12,}")
    print(f"\nRejection reasons:")
    for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
        if reason != 'valid':
            print(f"  {reason:20} {count:>8,}")
    print(f"\n✓ Valid: {output_dir / 'words_pattern_valid.txt'}")
    
    return valid

if __name__ == "__main__":
    pattern_validate("validated_data/words_needs_review.txt", "pattern_validated")
