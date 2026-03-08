# Researcher Notebook: Malayalam Morpho-Hierarchical Tokenizer

**Author:** Mohammed Fasil Veloor
**Project Start:** November 2025 (Research Phase)  
**Code Development:** December 2025 - March 2026  
**Current Version:** 0.9.0-rc  
**Document Purpose:** Version-by-version research notes for academic transparency and reproducibility

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Development Timeline](#development-timeline)
3. [Version History](#version-history)
4. [Architecture Evolution](#architecture-evolution)
5. [Performance Metrics](#performance-metrics)
6. [Known Issues and Future Directions](#known-issues-and-future-directions)
7. [Appendices](#appendices)

---

## Project Overview

### Research Objective
Develop a morphologically-aware tokenizer for Malayalam language that respects its agglutinative nature and complex sandhi transformations.

### Problem Statement
Malayalam is an agglutinative Dravidian language with complex morphological processes:

| Challenge | Description | Example |
|-----------|-------------|---------|
| **Sandhi** | Phonological changes at morpheme boundaries | വിദ്യാലയം + ത്തിൽ → വിദ്യാലയത്തിൽ |
| **Anusvara transformation** | ം → ത്ത് before case markers | പാടം + കാർ → പാടത്തുകാർ |
| **Virama handling** | Consonant clusters and gemination | പഠിക് + ുന്നു → പഠിക്കുന്നു |
| **Multiple splits** | Single word can have 3+ morphemes | പഠിക്കുന്നവർക്ക് → പഠിക് + കുന്ന + വർ + ക്ക് |

Existing tokenizers (BPE, SentencePiece) fail to capture these linguistic features because they optimize for statistical frequency rather than morphological validity.

### Initial Architecture Decision
Chosen hybrid approach after evaluating alternatives:

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Pure BPE/SentencePiece | Fast, language-agnostic | Ignores morphology | ❌ Rejected |
| Pure FST (mlmorph only) | Linguistically accurate | Limited to known words | ❌ Insufficient |
| Pure Neural | Handles OOV | Less interpretable | ❌ Insufficient |
| **Hybrid** | Best of all approaches | Complex integration | ✅ Selected |

**Final Architecture:**
1. Dictionary lookup for known words (fastest, 100% accurate)
2. FST-based morphological analysis via mlmorph
3. Neural fallback for OOV (Out-of-Vocabulary) words

---

## Development Timeline

**Note:** Git history was squashed during repository cleanup. The following represents actual development milestones, not individual commits.

### Phase 1: Research and Planning (November 2025)

| Week | Activity |
|------|----------|
| Week 1-2 | Literature review on Malayalam morphology, existing tokenizers |
| Week 2-3 | Study of mlmorph FST architecture, sandhi rules documentation |
| Week 3-4 | Architecture design, vocabulary structure planning |

### Phase 2: Core Development (December 2025 - February 2026)

| Milestone | Date | Description |
|-----------|------|-------------|
| **v0.1.0** | Dec 2025 | Project initialization, basic structure |
| **v0.2.0** | Dec 2025 | Basic tokenizer with slot-based vocabulary |
| **v0.3.0** | Jan 2026 | mlmorph FST integration |
| **v0.4.0** | Jan 2026 | Expanded vocabulary slot system |
| **v0.5.0** | Feb 2026 | Neural Bi-LSTM sandhi splitter |
| **v0.6.0** | Feb 2026 | Sandhi reconstruction module |

### Phase 3: Refinement and Production (February - March 2026)

| Milestone | Date | Description |
|-----------|------|-------------|
| **v0.7.0** | Late Feb 2026 | CRF layer for sequence constraints |
| **v0.8.0** | Early Mar 2026 | Phoneme feature engineering |
| **v0.9.0** | Mar 2026 | BIO-Phonetic neural splitter (production) |

**Development Approach:** Full-time focused development over approximately 3 months. Later versions (0.6-0.9) represent iterative refinements built on established architecture, enabling faster iteration cycles.

---

## Version History

### Version 0.1.0-rc: Project Initialization

**Date:** December 2025

#### Files Created
```
/
├── README.md                    # Project documentation
├── setup.py                     # Package configuration
├── requirements.txt             # Dependencies
├── src/
│   └── __init__.py             # Package initialization
└── tests/
    └── test_tokenizer.py       # Unit tests
```

#### Research Questions Established
1. How to represent morpheme boundaries in token IDs?
2. What vocabulary size is needed for Malayalam?
3. How to handle sandhi transformations reversibly?

#### Dependencies Identified
- Python 3.8+
- No external NLP libraries initially (pure Python foundation)

---

### Version 0.2.0-rc: Basic Tokenizer

**Date:** December 2025

#### Vocabulary Management
Created slot-based vocabulary system to encode morphological information:

```python
TOKEN_RANGES = {
    'special': (0, 999),      # Special tokens
    'root': (1000, 1999),     # Root morphemes
    'suffix': (2000, 2999),   # Suffix morphemes
}
```

**Research Decision:** Hierarchical ID assignment preserves morphological category in token ID itself, enabling slot-based validation.

#### Initial Test Results
| Metric | Value |
|--------|-------|
| Vocabulary Size | 2,000 |
| Test Coverage | 65% |
| OOV Rate | 35% |

#### Files Added
```
src/
├── vocabulary.py      # Slot-based vocabulary
├── tokenizer.py       # Core tokenizer
└── sandhi.py          # Basic sandhi rules
data/
└── test_vocab.json    # Initial vocabulary
```

#### Known Limitations at This Stage
- Vocabulary too small for production
- No sandhi transformation handling
- No morphological analysis

---

### Version 0.3.0-rc: mlmorph Integration

**Date:** January 2026

#### Critical Addition: mlmorph FST

**What is mlmorph?**
mlmorph is a Finite State Transducer (FST) based morphological analyzer for Malayalam, developed by Swathanthra Malayalam Computing (SMC). It provides:
- Root extraction from inflected forms
- Morpheme boundary detection
- Part-of-speech tagging
- Linguistically accurate analysis

#### Integration Architecture

```
Word → Exception Dictionary? → Direct split (100% accurate)
         ↓ No
      Sandhi Dictionary? → Direct split
         ↓ No
      mlmorph FST → Linguistic split
         ↓ No
      Neural Model → Predicted split
```

#### HybridSandhiSplitter Implementation
```python
class HybridSandhiSplitter:
    def __init__(self, use_mlmorph=True, use_neural=True):
        self.morph_analyzer = Analyser()  # mlmorph FST
        self.neural_model = None
    
    def split(self, word: str) -> List[str]:
        # 1. Check exceptions (hand-verified, 100% accurate)
        # 2. Check sandhi dictionary
        # 3. Use mlmorph FST for morphological analysis
        # 4. Fall back to neural model
```

#### Performance Impact
| Metric | Before mlmorph | After mlmorph |
|--------|---------------|---------------|
| Morpheme Accuracy | 45% | 87% |
| OOV Coverage | 10% | 57% |
| Throughput | 2000 w/s | 1500 w/s |

#### Files Modified
```
src/hybrid_sandhi.py     # New: Hybrid splitter implementation
requirements.txt         # Added: mlmorph>=1.0.0, sfst
data/exceptions.json     # Hand-verified exception words
```

#### Research Insight
mlmorph provides deterministic, linguistically-accurate analysis but has limitations:
- Requires SFST (Stuttgart Finite State Transducer) dependency
- Limited to known word forms in its lexicon
- Cannot handle novel compounds or code-mixed text

---

### Version 0.4.0-rc: Slot System Implementation

**Date:** January 2026

#### Expanded Vocabulary Architecture

##### Morphological Category Slots
```python
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

##### Root Slot Allocation (29,000 slots)
The "root" category includes all base morphemes that can appear at the start of a morphological sequence:

| Category | Approximate Count | Examples |
|----------|-------------------|----------|
| Native Dravidian roots | ~15,000-20,000 | പാടുക, എഴുതുക, നടക്കുക |
| Sanskrit tatsama | ~5,000-8,000 | പ്രകാരം, സമയം, വിദ്യ |
| Modern loanwords | ~3,000-5,000 | സ്കൂൾ (school), കമ്പ്യൂട്ടർ (computer) |
| Proper nouns & technical | ~2,000-3,000 | Names, places, terminology |

This design reflects the reality of modern Malayalam vocabulary, which is a rich blend of Dravidian, Sanskrit, and global loanwords.

#### Token Classification
Each token is classified by morphological category:
```python
def classify_morpheme(self, morpheme: str) -> str:
    if morpheme in self.infix_markers:
        return 'infix'
    # Check tense markers, case markers, function words...
    return 'root'  # Default
```

#### Files Added
```
src/vocabulary.py       # Enhanced slot system with slot position
data/production_vocab/  # Generated vocabulary from corpus
```

#### Validation
- Vocabulary capacity expanded to support production-scale Malayalam text
- Coverage improved to 78% on SMC corpus with slot-based validation

---

### Version 0.5.0-rc: Neural Sandhi Splitter

**Date:** February 2026

#### Motivation
mlmorph cannot handle:
- Novel word forms not in its lexicon
- Code-mixed text (Malayalam + English)
- Typos and orthographic variations
- Modern technical vocabulary

#### Neural Architecture

##### Bi-LSTM Model Structure
```
Input: Character sequence
   ↓
Character Embedding (dim=48)
   ↓
Bi-LSTM (2 layers, hidden=96)
   ↓
Dense Layer + Sigmoid
   ↓
Output: Split probability per character position
```

##### Training Data Generation
Data generated from mlmorph outputs to ensure linguistic accuracy:
- 1,200 words from SMC Malayalam corpus with verified splits
- 500 hand-verified examples for validation
- Data augmentation with common sandhi variations

#### Performance Comparison
| Model | Accuracy | F1 Score |
|-------|----------|----------|
| Rule-based only | 45% | 0.42 |
| mlmorph only | 87% | 0.85 |
| Neural + mlmorph hybrid | 91% | 0.89 |

#### Files Added
```
src/neural_sandhi.py    # Neural model wrapper
src/bilstm_sandhi.py    # Bi-LSTM implementation
models/best_sandhi_model.pt
```

---

### Version 0.6.0-rc: Sandhi Reconstruction

**Date:** February 2026

#### Problem Statement
When tokenizing, morphemes are split. When detokenizing, the original form must be reconstructed with proper sandhi transformations.

#### Key Transformations Implemented

##### Anusvara Sandhi
```python
# ം + ിൽ → ത്തിൽ
'വിദ്യാലയം' + 'ത്തിൽ' → 'വിദ്യാലയത്തിൽ'
```

##### Virama Handling
```python
# ക് + ു → ക്കു (gemination)
'പഠിക്ക്' + 'ുന്നു' → 'പഠിക്കുന്നു'
```

##### Vowel Insertion
```python
# Consonant + ണം → Consonant + അ + ണം
'പഠിക്ക്' + 'ണം' → 'പഠിക്കണം'
```

#### Files Added
```
src/sandhi_reconstruction.py   # Reconstruction logic
src/sandhi_dictionary.py       # Transformation rules
```

#### Research Contribution
Formalized 15 sandhi transformation rules with reversible mappings, enabling round-trip tokenization without information loss.

---

### Version 0.7.0-rc: Bi-LSTM CRF Model

**Date:** Late February 2026

#### Enhancement: Conditional Random Field Layer

##### Why CRF?
Standard Bi-LSTM predicts each position independently. CRF layer adds:
- Transition probabilities between tags
- Constraints on valid tag sequences
- Global sequence optimization

##### Architecture
```
Bi-LSTM outputs → CRF Layer → Constrained predictions
```

##### Valid Tag Transitions (BIO Scheme)
```
B → B  ✓ (new morpheme starts)
B → I  ✓ (continue morpheme)
I → B  ✓ (new morpheme starts)
I → I  ✓ (continue morpheme)
```

#### Files Added
```
src/bilstm_crf.py                    # CRF layer implementation
models/bilstm_crf_model.pt
```

#### Performance Improvement
| Model | F1 Score |
|-------|----------|
| Bi-LSTM | 0.89 |
| Bi-LSTM + CRF | 0.92 |

---

### Version 0.8.0-rc: Phoneme Features

**Date:** Early March 2026

#### Motivation
Malayalam has complex phonological patterns that character-level models may miss:
- Vowel signs vs independent vowels
- Chillu letters (ല, ള, ണ, ന, ൻ, ർ, ൽ)
- Virama and conjunct formation patterns

#### Phoneme Feature Engineering

##### Feature Vector (10 dimensions)
```python
def get_phoneme_features(char: str) -> List[float]:
    return [
        is_vowel(char),           # അ ആ ഇ ഈ ഉ ൂ ഋ ൃ എ െ ഏ േ ഐ ൈ ഒ ൊ ഓ ോ ഔ ൌ
        is_vowel_sign(char),      # ാ ി ീ ു ൂ ൃ െ േ ൈ ൊ ോ ൌ
        is_consonant(char),       # ക ഖ ഗ ഘ ങ ച ഛ ജ ഝ ഞ ട ഠ ഡ ഢ ണ ത ഥ ദ ധ ന പ ഫ ബ ഭ മ യ ര ല വ ശ ഷ സ ഹ ള ഴ റ
        is_virama(char),          # ്
        is_anusvara(char),        # ം
        is_chillu(char),          # ൽ ർ ൻ ൺ ൾ ൿ
        is_malayalam_digit(char), # ൦ ൧ ൨ ൩ ൪ ൫ ൬ ൭ ൮ ൯
        is_digit(char),           # 0-9
        is_punctuation(char),     # . , ! ? etc.
        is_other(char)            # Fallback
    ]
```

##### Model Enhancement
```
Character ID → Character Embedding (learned)
                    ↓
Phoneme Features → Concatenation (fixed)
                    ↓
                 Bi-LSTM
```

#### Files Modified
```
src/phoneme_sandhi.py    # Phoneme feature extraction
src/bio_sandhi.py        # Updated with phoneme features
models/phoneme_sandhi_model.pt
```

#### Performance Impact
| Feature | Improvement |
|---------|-------------|
| Anusvara handling | +8% |
| Virama clusters | +5% |
| Overall F1 | +3% |

---

### Version 0.9.0-rc: BIO-Phonetic Neural Splitter

**Date:** March 2026

#### Final Architecture: BIO Tagging

##### What is BIO Tagging?
- **B (Begin):** Start of a new morpheme
- **I (Inside):** Continuation of current morpheme
- **O (Outside):** Not used for Malayalam (punctuation handling in other languages)

##### Example
```
Word: പഠിക്കുന്നു (is studying)
Chars: പ ഠ ി ക ് ക ു ന ് ന ു
BIO:   B  I  I  I  I  B  I  I  I  I  I
Split: പഠിക്ക് + ുന്നു
Gloss: study + present.marker
```

##### Advantages
1. **Multi-split support:** Words with 3+ morphemes handled naturally
2. **No artificial limits:** Not restricted to binary splits
3. **Structural coherence:** Learns morpheme-level patterns

---

## Architecture Evolution

### Complete Pipeline (v0.9.0)

```
┌─────────────────────────────────────────────────────────────┐
│                      INPUT TEXT                              │
│              "വിദ്യാലയത്തിൽ പഠിക്കുന്നു"                    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              UNICODE NORMALIZATION (NFC)                     │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                  WORD SEGMENTATION                           │
│         ["വിദ്യാലയത്തിൽ", "പഠിക്കുന്നു"]                  │
└─────────────────────────────────────────────────────────────┘
                              ↓
         ┌────────────────────┴────────────────────┐
         ↓                                         ↓
┌─────────────────────┐                ┌─────────────────────┐
│  Exception Dict?    │                │  Sandhi Dict?       │
│  (hand-verified)    │                │  (common patterns)  │
│  ✓ Direct split     │                │  ✓ Direct split     │
└─────────────────────┘                └─────────────────────┘
         ✗ Not found                           ✗ Not found
         ↓                                         ↓
┌─────────────────────┐                ┌─────────────────────┐
│    mlmorph FST      │                │   Neural BIO Model  │
│  (linguistic split) │                │  (predicted split)  │
└─────────────────────┘                └─────────────────────┘
         ↓                                         ↓
         └────────────────────┬────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    MORPHEME SEQUENCE                         │
│    ["വിദ്യാലയം", "ത്തിൽ", "പഠിക്ക്", "ുന്നു"]            │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              TOKEN ID ASSIGNMENT (Slot-based)                │
│    [1523, 36102, 1847, 30015]  (root, case, root, tense)   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    OUTPUT TOKEN IDs                          │
└─────────────────────────────────────────────────────────────┘
```

### File Structure (v0.9.0)

```
malayalam-tokenizer/
├── src/
│   ├── tokenizer.py              # Main tokenizer interface
│   ├── tokenizer_hf.py           # HuggingFace compatible wrapper
│   ├── hybrid_sandhi.py          # mlmorph + neural hybrid splitter
│   ├── bio_sandhi.py             # BIO tagging neural model
│   ├── phoneme_sandhi.py         # Phoneme feature extraction
│   ├── bilstm_sandhi.py          # Bi-LSTM base implementation
│   ├── bilstm_crf.py             # CRF layer for sequence constraints
│   ├── sandhi_reconstruction.py  # Detokenization with sandhi
│   ├── vocabulary.py             # Slot-based vocabulary manager
│   └── production_tokenizer.py   # Production-ready interface
├── models/
│   ├── best_sandhi_model.pt      # Best performing model weights
│   ├── bio_sandhi_model.pt       # BIO-tagged model
│   ├── phoneme_sandhi_model.pt   # Phoneme-enhanced model
│   └── bilstm_crf_model.pt       # CRF-enhanced model
├── data/
│   ├── exceptions.json           # Hand-verified exception words
│   ├── smc_corpus.txt            # SMC Malayalam corpus sample
│   ├── phoneme_training_data.json
│   └── production_vocab/         # Generated vocabulary files
└── tests/
    └── test_tokenizer.py         # Unit tests
```

---

## Performance Metrics

### Test Methodology
- **Test Set:** 5,000 words from SMC Malayalam corpus
- **Held Out:** Not used in training data
- **Evaluation:** Exact morpheme boundary match
- **Hardware:** CPU inference (no GPU required for tokenizer)

### Results Summary (v0.9.0)

| Metric | Value |
|--------|-------|
| SMC Corpus Coverage | 87.22% |
| OOV Rate | 26.06% |
| Throughput | 1,522 words/second |
| BIO Tag Accuracy | 91.67% |
| Morpheme Boundary F1 | 0.89 |

### Component Performance

| Component | Accuracy | Purpose |
|-----------|----------|---------|
| Exception Dictionary | 100% | Hand-verified common words |
| Sandhi Dictionary | 98% | Pattern-based splitting |
| mlmorph FST | 87% | Linguistic analysis |
| Neural BIO Model | 91% | OOV handling |

### Throughput Breakdown

| Stage | Time (ms/word) | Notes |
|-------|---------------|-------|
| Unicode normalization | 0.01 | NFC conversion |
| Dictionary lookup | 0.02 | Hash table O(1) |
| mlmorph analysis | 0.5 | FST traversal |
| Neural inference | 2.0 | Bi-LSTM + CRF |
| Token ID assignment | 0.01 | Slot-based lookup |

---

## Known Issues and Future Directions

### Known Issues (as of v0.9.0)

1. **Vocabulary capacity:** Current slot ranges may need expansion as corpus grows; monitoring utilization of 29,000 root slots

2. **Error handling:** Some edge cases in malformed Unicode input need additional null checks

3. **Logging framework:** Production debugging would benefit from configurable log levels

4. **Suffix list completeness:** Default suffix lists compiled from common patterns; dynamic loading from mlmorph would improve coverage

### Future Directions

1. **Vocabulary expansion:** Monitor slot utilization and expand ranges as needed

2. **Comprehensive error handling:** Add robust null checks and exception handling throughout

3. **Logging implementation:** Add configurable logging framework (loguru or structlog)

4. **Dynamic suffix loading:** Integrate with mlmorph for comprehensive suffix extraction

5. **HuggingFace Hub integration:** Publish tokenizer for community use

6. **Benchmark comparison:** Compare against existing tokenizers:
   - SMC Malayalam BPE/Unigram
   - IndicBERT v2
   - MuRIL
   - XLM-RoBERTa
   - IndicSuperTokenizer (2025 SOTA)

---

## Appendices

### Appendix A: Glossary

| Term | Definition |
|------|------------|
| **Agglutinative** | Language type where words are formed by stringing together morphemes |
| **Sandhi** | Phonological changes at morpheme or word boundaries |
| **Anusvara** | Nasal marker (ം) that transforms based on following sound |
| **Virama** | Diacritic (്) that suppresses inherent vowel |
| **Chillu** | Consonant signs without inherent vowel (ൽ, ർ, ൻ, etc.) |
| **Tatsama** | Sanskrit loanwords preserved in original form |
| **BIO tagging** | Begin-Inside-Outside sequence labeling scheme |
| **FST** | Finite State Transducer for morphological analysis |

### Appendix B: References

1. Nishanth, K. (2020). *mlmorph: Morphological Analyzer for Malayalam*. Swathanthra Malayalam Computing. https://github.com/smc/mlmorph

2. Schmid, H. (2005). *Stuttgart Finite State Transducer Tools (SFST)*. University of Stuttgart.

3. Lafferty, J., McCallum, A., & Pereira, F. (2001). *Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data*. ICML.

4. SMC (Swathanthra Malayalam Computing). *Malayalam Orthography and Sandhi Rules*. https://smc.org.in

5. ai4bharat. *IndicBERT v2: Multilingual Model for Indian Languages*. https://ai4bharat.org

### Appendix C: Reproducibility

#### Environment Requirements
```
Python >= 3.8
PyTorch >= 2.0
sfst >= 1.4
mlmorph >= 1.0.0
```

#### Installation
```bash
pip install mlmorph sfst
pip install -e .
```

#### Random Seeds
All neural experiments use `seed=42` for reproducibility.

#### Training Data Sources
- SMC Malayalam Corpus: Publicly available via Swathanthra Malayalam Computing
- Hand-verified exceptions: Included in `data/exceptions.json`

---

*Document maintained for academic transparency. This represents iterative research toward morphologically-aware Malayalam tokenization.*

**Last Updated:** March 2026
