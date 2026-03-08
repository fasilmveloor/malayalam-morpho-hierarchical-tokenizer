# Architecture Documentation

## Malayalam Morpho-Hierarchical Tokenizer

This document provides a comprehensive overview of the system architecture, design decisions, and implementation details.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Core Components](#2-core-components)
3. [Data Flow](#3-data-flow)
4. [Neural Architecture](#4-neural-architecture)
5. [Vocabulary System](#5-vocabulary-system)
6. [Sandhi Processing](#6-sandhi-processing)
7. [API Reference](#7-api-reference)
8. [Extension Points](#8-extension-points)

---

## 1. System Overview

### 1.1 Design Philosophy

The Morpho-Hierarchical Tokenizer is built on three core principles:

1. **Linguistic Grounding**: Tokenization should reflect morphological structure, not just statistical patterns
2. **Graceful Degradation**: The system should work even when components are unavailable
3. **Hierarchical Organization**: Token IDs should encode grammatical information

### 1.2 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT TEXT                                   │
│                  "പഠിക്കുന്നു"                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   NORMALIZATION LAYER                           │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  • NFKC Unicode Normalization                           │    │
│  │  • Zero-width character removal (ZWJ, ZWNJ)             │    │
│  │  • Whitespace normalization                             │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   SANDHI SPLITTER                               │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐       │
│  │  Dictionary   │→ │   mlmorph     │→ │   Neural      │       │
│  │  (instant)    │  │   FST         │  │   Bi-LSTM     │       │
│  │  100% acc     │  │   95% acc     │  │   85% acc     │       │
│  └───────────────┘  └───────────────┘  └───────────────┘       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                MORPHOLOGICAL ANALYZER                           │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  • Root extraction from FST analysis                    │    │
│  │  • Suffix identification                                │    │
│  │  • Stem form conversion (add ് for verb roots)          │    │
│  │  • Fallback: Rule-based suffix stripping                │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              HIERARCHICAL TOKEN ASSIGNMENT                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Token          │  Category  │  Slot    │  Token ID     │    │
│  │  ─────────────────────────────────────────────────────  │    │
│  │  പഠിക്ക്        │  root      │  1xxx    │  1001         │    │
│  │  ുന്നു          │  tense     │  30xxx   │  30001        │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      OUTPUT                                     │
│              [2, 1001, 30001, 3]                                │
│           (BOS, root, tense, EOS)                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Core Components

### 2.1 MorphoHierarchicalTokenizer

**Location:** `src/tokenizer.py`

The main tokenizer class that orchestrates all components.

```python
class MorphoHierarchicalTokenizer:
    """
    Main tokenizer combining:
    - Unicode normalization
    - Sandhi splitting
    - Morphological analysis
    - Hierarchical token assignment
    """
    
    def __init__(self, vocab_size=8000, use_mlmorph=True):
        self.vocab = HierarchicalVocabulary()
        self.sandhi_splitter = SandhiSplitter()
        self.morph_analyzer = None  # mlmorph.Analyser if available
        
    def tokenize(self, text: str) -> List[int]:
        """Main tokenization method."""
        
    def tokenize_detailed(self, text: str) -> List[TokenInfo]:
        """Detailed tokenization with metadata."""
        
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text."""
```

### 2.2 HierarchicalVocabulary

**Location:** `src/vocabulary.py`

Manages the slot-based vocabulary system.

```python
class HierarchicalVocabulary:
    """
    Slot-based vocabulary with hierarchical ID assignment.
    
    Slots:
        special:  0-999        (1,000 slots)
        root:     1000-29999   (29,000 slots)
        tense:    30000-35999  (6,000 slots)
        case:     36000-41999  (6,000 slots)
        function: 42000-44999  (3,000 slots)
        infix:    45000-47999  (3,000 slots)
        conjunct: 48000-49999  (2,000 slots)
        subword:  50000-59999  (10,000 slots)
        reserved: 60000+       (future expansion)
    """
    
    TOKEN_RANGES = {
        'special': (0, 999, -1),        # Special tokens
        'root': (1000, 29999, 0),       # Slot 0: Base - 29,000 slots
        'tense': (30000, 35999, 2),     # Slot 2: End - 6,000 slots
        'case': (36000, 41999, 2),      # Slot 2: End - 6,000 slots
        'function': (42000, 44999, 0),  # Slot 0: Can start - 3,000 slots
        'infix': (45000, 47999, 1),     # Slot 1: Middle - 3,000 slots
        'conjunct': (48000, 49999, 1),  # Slot 1: Middle - 2,000 slots
        'subword': (50000, 59999, -1),  # No specific slot - 10,000 slots
        'reserved': (60000, 65535, -1), # Reserved for future use
    }
```

### 2.3 HybridSandhiSplitter

**Location:** `src/hybrid_sandhi.py`

Three-tier fallback system for sandhi splitting.

```python
class HybridSandhiSplitter:
    """
    Pipeline:
        1. Dictionary lookup (instant, 100% accurate)
        2. mlmorph FST (medium, high accuracy)
        3. Neural Bi-LSTM (generalizes to OOV)
    """
    
    def split(self, word: str) -> List[str]:
        """
        Split compound word into components.
        
        Example:
            തിരുവനന്തപുരം → ['തിരു', 'അനന്തപുരം']
        """
```

### 2.4 MalayalamSandhi

**Location:** `src/sandhi.py`

Implements Malayalam sandhi rules.

```python
class MalayalamSandhi:
    """
    Sandhi transformation rules for Malayalam.
    
    Key transformations:
        - Anusvara (ം) → ത്ത് before case markers
        - Stem form: add ് for verb roots
        - Vowel sandhi: combine stems with vowel-initial suffixes
    """
    
    DEPENDENT_VOWELS = 'ാിീുൂൃെേൈൊോൌ'
    INDEPENDENT_VOWELS = 'അആഇഈഉഊഋഎഏഐഒഓഔ'
    
    def to_stem_form(self, word: str) -> str:
        """Convert word to stem form (ending with ്)."""
        
    def apply_sandhi(self, part1: str, part2: str) -> str:
        """Apply sandhi rules to combine two parts."""
```

---

## 3. Data Flow

### 3.1 Tokenization Flow

```
Input: "പഠിക്കുന്നു"
    │
    ├── Normalize → "പഠിക്കുന്നു" (NFKC)
    │
    ├── Check compound? → No (single word)
    │
    ├── Get morphemes:
    │   ├── Check cache → Miss
    │   ├── mlmorph.analyse():
    │   │   └── Returns: "പഠിക്കുക<v><present>"
    │   └── Parse analysis:
    │       ├── Root: പഠിക്കുക
    │       ├── Original: പഠിക്കുന്നു
    │       └── Split: പഠിക്ക് (stem) + ുന്നു (suffix)
    │
    ├── Token assignment:
    │   ├── പഠിക്ക് → root → ID: 1001
    │   └── ുന്നു → tense → ID: 30001
    │
    └── Output: [2, 1001, 30001, 3]
              (BOS, root, tense, EOS)
```

### 3.2 Decoding Flow

```
Input: [2, 1001, 30001, 3]
    │
    ├── ID → Token:
    │   ├── 2 → <BOS> (skip)
    │   ├── 1001 → പഠിക്ക്
    │   ├── 30001 → ുന്നു
    │   └── 3 → <EOS> (skip)
    │
    ├── Sandhi reconstruction:
    │   ├── പഠിക്ക് ends with ്
    │   ├── ുന്നു starts with ു (vowel sign)
    │   └── Combine: പഠിക്ക് + ുന്നു → പഠിക്കുന്നു
    │
    └── Output: "പഠിക്കുന്നു"
```

---

## 4. Neural Architecture

### 4.1 Phoneme-BiLSTM

**Location:** `src/phoneme_sandhi.py`

A Bi-LSTM model enhanced with phoneme features for sandhi split prediction.

```
Input Layer
    │
    ├── Character IDs ──→ Embedding (32-dim)
    │                            │
    └── Phoneme Features (10-dim)│
                                 │
                    ┌────────────┴────────────┐
                    │   Concatenate (42-dim)  │
                    └────────────┬────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │   Bi-LSTM Layer 1 (96)  │
                    │   Bidirectional         │
                    └────────────┬────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │   Bi-LSTM Layer 2 (96)  │
                    │   Bidirectional         │
                    └────────────┬────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │   Dense (192 → 96)      │
                    │   ReLU + Dropout(0.2)   │
                    └────────────┬────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │   Output (96 → 1)       │
                    │   Sigmoid               │
                    └────────────┬────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │   Split Probabilities   │
                    │   (per character)       │
                    └─────────────────────────┘
```

### 4.2 Phoneme Feature Encoding

```python
class MalayalamPhonemeEncoder:
    """
    10-dimensional phoneme feature vector:
    
    [0] Is_Vowel        - Independent vowels (അ, ആ, ഇ, ...)
    [1] Is_VowelSign    - Dependent vowels (ാ, ി, ീ, ...)
    [2] Is_Consonant    - Base consonants (ക, ഖ, ഗ, ...)
    [3] Is_Virama       - Chandrakkala (്) - CRITICAL for sandhi
    [4] Is_AnuSvara     - Anusvara (ം) - transforms in sandhi
    [5] Is_Chillu       - Chillu letters (ൽ, ർ, ൻ, ...)
    [6] Is_Conjunct     - Part of conjunct cluster
    [7] Is_Digit        - Malayalam/Arabic digits
    [8] Is_Punctuation  - Punctuation marks
    [9] Is_Other        - Other characters
    """
```

### 4.3 BIO Tagger

**Location:** `src/bio_sandhi.py`

BIO (Begin-Inside) tagging for morpheme boundaries.

```
Word: പഠിക്കുന്നു
      ││││││└─ I (inside morpheme)
      │││││└── I (inside morpheme)
      ││││└─── I (inside morpheme)
      │││└──── B (boundary - split here!)
      ││└───── I (inside morpheme)
      │└────── I (inside morpheme)
      └─────── B (beginning)

BIO Tags: [B, I, I, I, B, I, I]
           │           │
           └─ morpheme ─┘── morpheme
```

### 4.4 BiLSTM-CRF

**Location:** `src/bilstm_crf.py`

Bi-LSTM with CRF layer for constrained decoding.

```
Input Sequence
      │
      ▼
┌─────────────┐
│  Bi-LSTM    │  → Emission scores
└─────────────┘
      │
      ▼
┌─────────────┐
│  CRF Layer  │  → Transition constraints
└─────────────┘
      │
      ▼
Optimal Tag Sequence
(Viterbi decoding)
```

---

## 5. Vocabulary System

### 5.1 Slot-Based Organization

```
┌─────────────────────────────────────────────────────────────┐
│                    VOCABULARY LAYOUT                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  0-999     SPECIAL TOKENS                                   │
│            ┌──────────────────────────────────────┐         │
│            │ 0: <PAD>                             │         │
│            │ 1: <UNK>                             │         │
│            │ 2: <BOS>                             │         │
│            │ 3: <EOS>                             │         │
│            │ 4: <ROOT>                            │         │
│            │ 5: <SUFFIX>                          │         │
│            │ 6: <INFIX>                           │         │
│            │ 7: <SPACE>                           │         │
│            │ 8: <SEP>                             │         │
│            │ 9: <MASK>                            │         │
│            └──────────────────────────────────────┘         │
│                                                             │
│  1000-29999   ROOT TOKENS (29,000 slots)                    │
│            ┌──────────────────────────────────────┐         │
│            │ 1000: പഠിക്ക് (study)               │         │
│            │ 1001: വിദ്യാലയം (school)            │         │
│            │ 1002: കേരളം (Kerala)                │         │
│            │ ...                                  │         │
│            │                                      │         │
│            │ Includes:                            │         │
│            │ • Native Dravidian roots (~15-20K)   │         │
│            │ • Sanskrit tatsama (~5-8K)           │         │
│            │ • Modern loanwords (~3-5K)           │         │
│            │ • Proper nouns (~2-3K)               │         │
│            └──────────────────────────────────────┘         │
│                                                             │
│  30000-35999  TENSE MARKERS (6,000 slots)                   │
│            ┌──────────────────────────────────────┐         │
│            │ 30000: ുന്നു (present)              │         │
│            │ 30001: ച്ചു (past)                  │         │
│            │ 30002: ും (future)                  │         │
│            │ ...                                  │         │
│            └──────────────────────────────────────┘         │
│                                                             │
│  36000-41999  CASE MARKERS (6,000 slots)                    │
│            ┌──────────────────────────────────────┐         │
│            │ 36000: ിൽ (locative)                │         │
│            │ 36001: ിന്റെ (genitive)             │         │
│            │ 36002: ക്ക് (dative)                │         │
│            │ ...                                  │         │
│            └──────────────────────────────────────┘         │
│                                                             │
│  42000-44999  FUNCTION WORDS (3,000 slots)                  │
│  45000-47999  INFIX/SANDHI (3,000 slots)                    │
│  48000-49999  CONJUNCT (2,000 slots)                        │
│  50000-59999  SUBWORD (10,000 slots)                        │
│  60000+       RESERVED (future expansion)                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 Token Classification

```python
def classify_token(token_id: int) -> str:
    """Classify token by its ID range."""
    if 0 <= token_id < 1000:
        return 'special'
    elif 1000 <= token_id < 30000:
        return 'root'
    elif 30000 <= token_id < 36000:
        return 'tense'
    elif 36000 <= token_id < 42000:
        return 'case'
    elif 42000 <= token_id < 45000:
        return 'function'
    elif 45000 <= token_id < 48000:
        return 'infix'
    elif 48000 <= token_id < 50000:
        return 'conjunct'
    elif 50000 <= token_id < 60000:
        return 'subword'
    else:
        return 'unknown'
```

---

## 6. Sandhi Processing

### 6.1 Sandhi Rules

#### Anusvara Transformation

```
Rule: ം + case_marker → ത്ത് + case_marker

Examples:
  വിദ്യാലയം + ിൽ → വിദ്യാലയത്തിൽ
  കേരളം + ിൽ → കേരളത്തിൽ
  പുസ്തകം + ിൽ → പുസ്തകത്തിൽ
```

#### Stem Form Conversion

```
Rule: Verb roots end with ് (virama)

Examples:
  പഠിക്കുക → പഠിക്ക് (study)
  വരുക → വര് (come)
  ചെയ്യുക → ചെയ്യ് (do)
```

#### Vowel Sandhi

```
Rule: Stem (ends with ്) + Vowel-initial suffix
      → Remove ്, combine

Examples:
  പഠിക്ക് + ുന്നു → പഠിക്കുന്നു
  വര് + ുന്നു → വരുന്നു
```

### 6.2 Reconstruction Logic

```python
def decode_morphemes(morphemes: List[str]) -> str:
    """
    Reconstruct word from morpheme sequence.
    
    Algorithm:
        1. Start with first morpheme
        2. For each subsequent morpheme:
           a. If previous ends with ് AND current starts with vowel sign
              → Remove ്, combine
           b. Else → Just append
    """
    result = []
    for morpheme in morphemes:
        if result and result[-1].endswith('്'):
            if morpheme[0] in DEPENDENT_VOWELS:
                # Sandhi: remove virama, combine
                result[-1] = result[-1][:-1] + morpheme
            else:
                result.append(morpheme)
        else:
            result.append(morpheme)
    return ''.join(result)
```

---

## 7. API Reference

### 7.1 MorphoHierarchicalTokenizer

```python
class MorphoHierarchicalTokenizer:
    """
    Main tokenizer class.
    
    Methods:
        tokenize(text: str) -> List[int]
            Tokenize text to token IDs.
        
        tokenize_detailed(text: str) -> List[TokenInfo]
            Tokenize with full metadata.
        
        decode(token_ids: List[int]) -> str
            Decode token IDs to text.
        
        get_morphemes(word: str) -> List[str]
            Get morpheme decomposition.
        
        save(path: str) -> None
            Save tokenizer state.
        
        load(path: str) -> None
            Load tokenizer state.
        
        get_stats() -> Dict
            Get tokenization statistics.
    """
```

### 7.2 TokenInfo Dataclass

```python
@dataclass
class TokenInfo:
    """Information about a single token."""
    text: str              # Token text
    token_id: int          # Token ID
    token_type: str        # Category (root, tense, case, etc.)
    morpheme: Optional[str] # Original morpheme
    is_oov: bool           # Is out-of-vocabulary?
    subword_ids: Optional[List[int]]  # For character fallback
```

### 7.3 MorphoHierarchicalTokenizerFast (HuggingFace)

```python
class MorphoHierarchicalTokenizerFast(PreTrainedTokenizer):
    """
    HuggingFace-compatible tokenizer.
    
    Methods:
        tokenize(text: str) -> List[str]
            Tokenize to string tokens.
        
        encode(text: str) -> List[int]
            Encode to token IDs.
        
        decode(token_ids: List[int]) -> str
            Decode to text.
        
        classify_token(token_id: int) -> str
            Get token category.
        
        save_vocabulary(save_directory: str) -> Tuple[str]
            Save vocabulary files.
    """
```

---

## 8. Extension Points

### 8.1 Adding New Sandhi Rules

```python
# In src/sandhi.py

class MalayalamSandhi:
    def __init__(self):
        self.custom_rules = []
    
    def add_rule(self, pattern: str, replacement: str):
        """Add a custom sandhi transformation rule."""
        self.custom_rules.append((pattern, replacement))
```

### 8.2 Adding New Token Categories

```python
# In src/vocabulary.py

class HierarchicalVocabulary:
    TOKEN_RANGES = {
        # Existing slots...
        'root': (1000, 29999, 0),
        # ...
        
        # Add new slot in reserved range
        'technical': (60000, 64999, 0),
    }
```

### 8.3 Custom Neural Models

```python
# Create custom model
class CustomSandhiModel(nn.Module):
    def __init__(self, ...):
        pass
    
    def forward(self, x):
        pass

# Register with tokenizer
tokenizer.neural_model = CustomSandhiModel()
```

### 8.4 Integration with Other Languages

The architecture can be adapted for other morphologically rich languages:

1. **Tamil**: Similar agglutinative structure
2. **Kannada**: Related Dravidian language
3. **Telugu**: Dravidian with different script
4. **Hindi**: Indo-Aryan with sandhi rules

---

## Appendix A: File Formats

### vocab.json

```json
{
  "token_to_id": {
    "<PAD>": 0,
    "<UNK>": 1,
    "പഠിക്ക്": 1000,
    "ുന്നു": 30000
  },
  "slots": {
    "special": [0, 999],
    "root": [1000, 29999],
    "tense": [30000, 35999]
  }
}
```

### Model Files

```
best_sandhi_model.pt
├── model_state_dict   # Trained weights
├── char2idx          # Character vocabulary
├── config            # Model configuration
└── training_stats    # Training metrics
```

---

## Appendix B: Performance Optimization

### Caching Strategy

```python
# High-frequency word cache
cache = {
    "പഠിക്കുന്നു": ["പഠിക്ക്", "ുന്നു"],
    "വരുന്നു": ["വര്", "ുന്നു"],
    # ... most frequent 1000 words
}

# Check cache before expensive FST call
if word in cache:
    return cache[word]
```

### Batch Processing

```python
def tokenize_batch(texts: List[str]) -> List[List[int]]:
    """Efficient batch tokenization."""
    # Pre-normalize all texts
    normalized = [normalize(t) for t in texts]
    
    # Batch morphological analysis
    words = [extract_words(t) for t in normalized]
    flat_words = [w for ws in words for w in ws]
    
    # Single pass through analyzer
    analyses = batch_analyze(flat_words)
    
    # Reconstruct results
    return [encode(w, analyses) for w in words]
```

---

*Last updated: March 2026*
