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
│  │  ുന്നു          │  tense     │  2xxx    │  2001         │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      OUTPUT                                     │
│              [2, 1001, 2001, 3]                                 │
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
        special:  0-999
        root:     1000-1999
        tense:    2000-2999
        case:     3000-3999
        function: 4000-4999
        infix:    5000-5999
        char:     7000-7999
    """
    
    SLOTS = {
        'special': (0, 999),
        'root': (1000, 1999),
        'tense': (2000, 2999),
        'case': (3000, 3999),
        'function': (4000, 4999),
        'infix': (5000, 5999),
        'char': (7000, 7999),
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
    │   └── ുന്നു → tense → ID: 2001
    │
    └── Output: [2, 1001, 2001, 3]
              (BOS, root, tense, EOS)
```

### 3.2 Decoding Flow

```
Input: [2, 1001, 2001, 3]
    │
    ├── ID → Token:
    │   ├── 2 → <BOS> (skip)
    │   ├── 1001 → പഠിക്ക്
    │   ├── 2001 → ുന്നു
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
│            │ 4: <MASK>                            │         │
│            │ 5: <CLS>                             │         │
│            │ 6: <SEP>                             │         │
│            └──────────────────────────────────────┘         │
│                                                             │
│  1000-1999 ROOT TOKENS                                      │
│            ┌──────────────────────────────────────┐         │
│            │ 1000: പഠിക്ക് (study)               │         │
│            │ 1001: വിദ്യാലയം (school)            │         │
│            │ 1002: കേരളം (Kerala)                │         │
│            │ ...                                  │         │
│            └──────────────────────────────────────┘         │
│                                                             │
│  2000-2999 TENSE MARKERS                                    │
│            ┌──────────────────────────────────────┐         │
│            │ 2000: ുന്നു (present)               │         │
│            │ 2001: ച്ചു (past)                   │         │
│            │ 2002: ും (future)                   │         │
│            │ ...                                  │         │
│            └──────────────────────────────────────┘         │
│                                                             │
│  3000-3999 CASE MARKERS                                     │
│            ┌──────────────────────────────────────┐         │
│            │ 3000: ിൽ (locative)                 │         │
│            │ 3001: ിന്റെ (genitive)              │         │
│            │ 3002: ക്ക് (dative)                 │         │
│            │ ...                                  │         │
│            └──────────────────────────────────────┘         │
│                                                             │
│  4000-4999 FUNCTION WORDS                                   │
│  5000-5999 INFIX/SANDHI                                     │
│  7000-7999 CHARACTER FALLBACK                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 Token Classification

```python
def classify_token(token_id: int) -> str:
    """Classify token by its ID range."""
    if 0 <= token_id < 1000:
        return 'special'
    elif 1000 <= token_id < 2000:
        return 'root'
    elif 2000 <= token_id < 3000:
        return 'tense'
    elif 3000 <= token_id < 4000:
        return 'case'
    elif 4000 <= token_id < 5000:
        return 'function'
    elif 5000 <= token_id < 6000:
        return 'infix'
    elif 7000 <= token_id < 8000:
        return 'char'
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
    SLOTS = {
        # Existing slots...
        'root': (1000, 1999),
        # ...
        
        # Add new slot
        'proper_noun': (6000, 6999),
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
    "ുന്നു": 2000
  },
  "slots": {
    "special": [0, 999],
    "root": [1000, 1999]
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

*Last updated: 2024*
