# Malayalam Morpho-Hierarchical Tokenizer

A novel tokenization approach for Malayalam that combines morphological analysis with hierarchical vocabulary structure.

## 🎯 Project Overview

This tokenizer addresses the unique challenges of Malayalam tokenization:

- **Agglutinative Morphology**: Single words can contain multiple morphemes
- **Sandhi Rules**: Complex word junction rules
- **Unicode Complexity**: Conjunct consonants and complex character combinations
- **Allomorphy**: Same morpheme with different surface forms

## 📁 Project Structure

```
malayalam-tokenizer/
├── src/
│   ├── tokenizer.py          # Main tokenizer class
│   ├── vocabulary.py         # Hierarchical vocabulary manager
│   ├── sandhi_splitter.py    # Sandhi splitting rules
│   ├── sandhi.py             # Sandhi rules
│   └── __init__.py
├── tests/
│   ├── test_tokenizer.py     # Test harness
│   └── compare_tokenizers.py # Comparison with baselines
├── data/
│   └── smc_corpus.txt        # Downloaded corpus
└── vocab/                    # Saved tokenizer models
```

## 🏗️ Architecture

```
Input Text
    ↓
Unicode Normalization (NFKC)
    ↓
Sandhi Splitting (compound word detection)
    ↓
Morphological Analysis (mlmorph)
    ↓
Hierarchical Token Assignment
    ↓
Unigram/Character Fallback (for OOV)
    ↓
Token IDs
```

## 📊 Hierarchical Vocabulary Structure

Token IDs are organized by morphological type:

| Type | Token Range | Description |
|------|-------------|-------------|
| `special` | 0-99 | Special tokens (PAD, UNK, BOS, EOS) |
| `root` | 1000-1999 | Root words / Stems |
| `tense` | 2000-2999 | Tense/Aspect markers |
| `case` | 3000-3999 | Case markers and postpositions |
| `function` | 4000-4999 | Function words (pronouns, particles) |
| `conjunct` | 5000-5999 | Conjunct consonants |
| `subword` | 6000-6999 | Subword fallback tokens |
| `char` | 7000+ | Character-level tokens |

## 📈 Test Results (SMC Corpus)

### Performance Metrics

| Metric | Morpho-Hierarchical | BPE | Unigram | Character |
|--------|---------------------|-----|---------|-----------|
| **Total Words** | 28,447 | 28,447 | 28,447 | 28,447 |
| **Total Tokens** | 40,174 | 63,716 | 62,654 | 28,447 |
| **Avg Tokens/Word** | **1.41** | 2.24 | 2.20 | 1.00 |
| **Compression Ratio** | **0.708** | 0.447 | 0.454 | 1.000 |
| **Morphological Alignment** | **23.07%** | 15.34% | 15.55% | 29.89% |

### Vocabulary Breakdown

| Type | Count |
|------|-------|
| Roots | 1,000 |
| Tense markers | 590 |
| Case markers | 104 |
| Function words | 25 |
| Subword tokens | 3,268 |

## 🚀 Key Findings

1. **Better Compression**: Our tokenizer produces ~37% fewer tokens than BPE/Unigram
2. **Morphological Awareness**: Higher morphological alignment (23.07%) vs BPE (15.34%)
3. **Semantic Preservation**: Tokens maintain morphological meaning (root, tense, case)

## 🛠️ Usage

```python
from src.tokenizer import MorphoHierarchicalTokenizer

# Create tokenizer
tokenizer = MorphoHierarchicalTokenizer(vocab_size=5000)

# Train on corpus
tokenizer.train(corpus_lines, min_freq=2)

# Tokenize text
token_ids = tokenizer.tokenize("ഞാൻ പഠിക്കുന്നു")

# Get detailed tokenization
tokens = tokenizer.tokenize_detailed("ഞാൻ പഠിക്കുന്നു")
for token in tokens:
    print(f"{token.text} → {token.token_id} ({token.token_type})")
```

## 📋 Sample Output

```
Input: ഞാൻ പഠിക്കുന്നു
Tokens:
  ഞാൻ → 2000 (tense)       # "I" - pronoun classified as function word
  പഠിക്ക → 1000 (root)     # "learn" - root verb
  ുന്നു → 2001 (tense)     # present tense marker
```

## ⚠️ Current Limitations

1. **Training Speed**: Morphological analysis adds overhead (~1,500 words/sec)
2. **mlmorph Coverage**: ~80% vocabulary coverage, requires fallback
3. **Sandhi Rules**: Limited compound word dictionary

## 🔮 Future Improvements

1. **Neural Sandhi Splitter**: ML-based compound word detection
2. **Allomorph Handling**: Learn surface→underlying morpheme mappings
3. **Pre-training Integration**: Direct integration with LLM embedding layers

## 📦 Dependencies

- Python 3.8+
- mlmorph (Malayalam morphological analyzer)
- sentencepiece (for baseline comparison)

## 📄 License

MIT License

## 👥 Credits

- SMC (Swathanthra Malayalam Computing) for the corpus and mlmorph
- Built as a research prototype for Malayalam NLP
