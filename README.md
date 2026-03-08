# Malayalam Morpho-Hierarchical Tokenizer

<div align="center">

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-HuggingFace-blue)](https://huggingface.co/)

**A Novel Morphologically-Aware Tokenizer for Malayalam**

*Combining Finite State Transducers with Phoneme-Aware Bi-LSTM for Agglutinative Languages*

[Quick Start](#-quick-start) • [Benchmarks](#-performance) • [Documentation](#-documentation) • [Citation](#-citation)

</div>

---

## 📖 Overview

Malayalam is a morphologically rich Dravidian language where words are formed through complex agglutination processes. Standard tokenizers like BPE and Unigram fail to capture the linguistic structure, often splitting words at arbitrary subword boundaries that don't align with morphemes.

This project introduces a **Morpho-Hierarchical Tokenizer** that:

- **Leverages linguistic structure** through Finite State Transducers (mlmorph)
- **Handles OOV words** with a Phoneme-Aware Bi-LSTM neural network
- **Organizes tokens hierarchically** using a Slot System for grammatical categories
- **Achieves 87.22% morphology coverage** with only 3,265 vocabulary tokens

### Key Innovations

| Feature | Description |
|---------|-------------|
| **Slot System** | Hierarchical token IDs encoding grammatical role (Root=1000-29999, Tense=30000-35999, Case=36000-41999) |
| **Phoneme Features** | 10-dimensional vector encoding Virama, Vowel, Consonant categories |
| **BIO Tagging** | 91.67% accuracy on morpheme boundary detection |
| **Sandhi Reconstruction** | ം → ത്ത് transformation for canonical form restoration |
| **Hybrid Pipeline** | Dictionary → FST → Neural fallback chain |

---

## 📊 Performance

### Benchmark Results (SMC Malayalam Corpus)

| Tokenizer | Tokens/Word | Vocab Size | Throughput | Morph. Aligned |
|-----------|-------------|------------|------------|----------------|
| **★ Morpho-Hierarchical** | 1.71 | **3,265** ✓ | **37,449 w/s** ✓ | ✅ Yes |
| MuRIL | **1.33** | 197,258 | 34,477 w/s | ❌ No |
| IndicBERT v2 | 1.41 | 250,000 | 24,033 w/s | ❌ No |
| SMC Malayalam Unigram | 1.57 | 16,000 | 17,173 w/s | ⚠️ Partial |
| SMC Malayalam BPE | 1.66 | 16,000 | 19,336 w/s | ❌ No |
| XLM-RoBERTa | 1.66 | 250,002 | 16,144 w/s | ❌ No |
| mBERT | 3.71 | 119,547 | 29,803 w/s | ❌ No |

### Vocabulary Efficiency

**Vocab Efficiency Score** = Tokens/Word × Vocab Size (lower is better)

| Tokenizer | Calculation | Score | vs Ours |
|-----------|-------------|-------|---------|
| **★ Morpho-Hierarchical** | 1.71 × 3,265 | **5,583** | — |
| SMC Malayalam Unigram | 1.57 × 16,000 | 25,120 | 4.5x larger |
| MuRIL | 1.33 × 197,258 | 262,353 | 47x larger |
| IndicBERT v2 | 1.41 × 250,000 | 352,500 | 63x larger |

**Key Insight**: MuRIL achieves better raw compression (1.33 tokens/word) but requires 60x larger vocabulary. Our tokenizer achieves comparable compression with dramatically smaller vocabulary, enabling faster training, lower memory footprint, and better deployment efficiency.

### Why This Matters

| Factor | Large Vocab (MuRIL) | Small Vocab (Ours) |
|--------|---------------------|---------------------|
| **Embedding layer size** | ~790 MB | ~13 MB |
| **Training memory** | High GPU RAM required | Fits on CPU |
| **Inference latency** | O(vocab) softmax | Faster lookup |
| **Model portability** | Large model files | Lightweight deployment |

### Morphological Alignment

Unlike statistical tokenizers that split at arbitrary character boundaries, our tokenizer respects morphological structure:

```
Word: പഠിക്കുന്നു (is studying)

❌ MuRIL: ['പ', 'ഠ', 'ി', 'ക', '്', 'ക', 'ു', 'ന', '്', 'ന', 'ു']
          → 11 tokens, no morphological meaning

✅ Ours:  ['പഠിക്ക്', 'ുന്നു']
          → 2 tokens: [root:study] + [tense:present]
          → Linguistically interpretable
```

**Benefits of morphological alignment:**
- Downstream tasks (POS tagging, NER, parsing) can leverage token structure
- Interpretable tokens aid debugging and analysis
- Better generalization to unseen word forms through morphological rules

### When to Choose Which

| Use Case | Recommended Tokenizer |
|----------|----------------------|
| **Resource-constrained deployment** (mobile, edge) | ★ Morpho-Hierarchical |
| **Fast training with limited GPU memory** | ★ Morpho-Hierarchical |
| **Downstream morphological tasks** (parsing, NER) | ★ Morpho-Hierarchical |
| **Maximum compression regardless of vocab size** | MuRIL |
| **Fine-tuning existing MuRIL models** | MuRIL (compatibility) |

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/fasilmveloor/malayalam-morpho-hierarchical-tokenizer.git
cd malayalam-morpho-hierarchical-tokenizer

# Install dependencies
pip install -r requirements.txt

# Or install mlmorph separately (optional but recommended)
pip install mlmorph torch transformers
```

### Basic Usage

```python
from src.tokenizer import MorphoHierarchicalTokenizer

# Initialize tokenizer
tokenizer = MorphoHierarchicalTokenizer(use_mlmorph=True)

# Tokenize text
text = "പഠിക്കുന്നു വിദ്യാലയത്തിൽ"
tokens = tokenizer.tokenize(text)
print(tokens)  # Token IDs

# Get detailed tokenization
detailed = tokenizer.tokenize_detailed(text)
for token in detailed:
    print(f"{token.text} → ID:{token.token_id} Type:{token.token_type}")

# Decode back
decoded = tokenizer.decode(tokens)
print(decoded)
```

### HuggingFace Integration

```python
from src.tokenizer_hf import MorphoHierarchicalTokenizerFast

# Initialize
tokenizer = MorphoHierarchicalTokenizerFast(use_mlmorph=True)

# Tokenize
tokens = tokenizer.tokenize("പഠിക്കുന്നു")
# Output: ['പഠിക്ക്', 'ുന്നു']

# Encode
ids = tokenizer.encode("പഠിക്കുന്നു")
# Output: [2, 1001, 30001, 3]  # BOS, root, tense, EOS

# Classify tokens
category = tokenizer.classify_token(30001)
# Output: 'tense'
```

---

## 🏗️ Architecture

### Tokenization Pipeline

```
Input Text
    ↓
┌─────────────────────────────────────┐
│  Unicode Normalization (NFKC)       │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Sandhi Splitting                   │
│  ┌─────────────────────────────┐    │
│  │ Dictionary Lookup (Fast)    │    │
│  │ ↓ (miss)                    │    │
│  │ mlmorph FST (Medium)        │    │
│  │ ↓ (miss)                    │    │
│  │ Neural Bi-LSTM (OOV)        │    │
│  └─────────────────────────────┘    │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Morphological Analysis             │
│  - Root extraction                  │
│  - Suffix identification            │
│  - Stem form conversion             │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Hierarchical Token Assignment      │
│  - Slot classification              │
│  - Token ID allocation              │
│  - Vocabulary update                │
└─────────────────────────────────────┘
    ↓
Token IDs
```

### Slot System

| Slot | Category | ID Range | Slots | Examples |
|------|----------|----------|-------|----------|
| 0 | Special | 0-999 | 1,000 | `<PAD>`, `<UNK>`, `<BOS>`, `<EOS>` |
| 1 | Root | 1000-29999 | 29,000 | പഠിക്ക്, വിദ്യാലയം, സ്കൂൾ |
| 2 | Tense | 30000-35999 | 6,000 | ുന്നു, ച്ചു, ും |
| 3 | Case | 36000-41999 | 6,000 | ിൽ, ിന്റെ, ക്ക് |
| 4 | Function | 42000-44999 | 3,000 | എന്ന, എങ്കിൽ |
| 5 | Infix | 45000-47999 | 3,000 | ത്ത് (sandhi) |
| 6 | Conjunct | 48000-49999 | 2,000 | Conjunct consonants |
| 7 | Subword | 50000-59999 | 10,000 | Character-level fallback |
| 8 | Reserved | 60000+ | — | Future expansion |

### Root Slot Allocation (29,000 slots)

The root category includes all base morphemes that can appear at the start of a morphological sequence:

| Category | Approximate Count | Examples |
|----------|-------------------|----------|
| Native Dravidian roots | ~15,000-20,000 | പാടുക, എഴുതുക, നടക്കുക |
| Sanskrit tatsama | ~5,000-8,000 | പ്രകാരം, സമയം, വിദ്യ |
| Modern loanwords | ~3,000-5,000 | സ്കൂൾ (school), കമ്പ്യൂട്ടർ (computer) |
| Proper nouns & technical | ~2,000-3,000 | Names, places, terminology |

This design reflects the reality of modern Malayalam vocabulary, which is a rich blend of Dravidian, Sanskrit, and global loanwords.

---

## 📁 Project Structure

```
malayalam-tokenizer/
├── 📂 src/
│   ├── tokenizer.py              # Main tokenizer implementation
│   ├── tokenizer_hf.py           # HuggingFace-compatible version
│   ├── vocabulary.py             # Hierarchical vocabulary management
│   ├── sandhi.py                 # Sandhi rules and transformations
│   ├── sandhi_splitter.py        # Compound word splitting
│   ├── hybrid_sandhi.py          # Hybrid splitter (Dict+FST+Neural)
│   ├── phoneme_sandhi.py         # Phoneme-enhanced Bi-LSTM
│   ├── bio_sandhi.py             # BIO tagging model
│   ├── bilstm_crf.py             # Bi-LSTM with CRF layer
│   └── sandhi_reconstruction.py  # Canonical form restoration
│
├── 📂 models/
│   ├── best_sandhi_model.pt      # Trained sandhi model
│   ├── phoneme_sandhi_model.pt   # Phoneme-enhanced model
│   ├── bio_sandhi_model.pt       # BIO tagger model
│   └── bilstm_crf_model.pt       # CRF-enhanced model
│
├── 📂 data/
│   ├── smc_corpus.txt            # SMC corpus sample
│   ├── exceptions.json           # Exception dictionary
│   └── training_data.json        # Training examples
│
├── 📂 tests/
│   ├── test_tokenizer.py         # Unit tests
│   └── compare_tokenizers.py     # Benchmark comparison
│
├── 📂 docs/
│   ├── ARCHITECTURE.md           # System architecture
│   ├── BENCHMARKS.md             # Detailed benchmarks
│   ├── HUGGINGFACE_INTEGRATION.md # HF integration guide
│   ├── MODEL_CARD.md             # HuggingFace model card
│   └── TESTING_CHECKLIST.md      # Testing checklist
│
├── 📄 README.md                  # This file
├── 📄 DISCLAIMER.md              # Usage disclaimer
├── 📄 RESEARCHER_NOTEBOOK.md     # Development notes
├── 📄 requirements.txt           # Dependencies
├── 📄 LICENSE                    # MIT License
│
├── 📓 Malayalam_Morpho_Hierarchical_Tokenizer.ipynb
│   # Complete tutorial notebook
│
└── 📓 Malayalam_Tokenizer_Validation_Colab.ipynb
    # Validation notebook for Colab
```

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| **[BENCHMARKS.md](docs/BENCHMARKS.md)** | Detailed benchmark analysis and methodology |
| **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** | Detailed system architecture |
| **[HUGGINGFACE_INTEGRATION.md](docs/HUGGINGFACE_INTEGRATION.md)** | HuggingFace integration guide |
| **[MODEL_CARD.md](docs/MODEL_CARD.md)** | HuggingFace model card |
| **[TESTING_CHECKLIST.md](docs/TESTING_CHECKLIST.md)** | Comprehensive testing checklist |
| **[RESEARCHER_NOTEBOOK.md](RESEARCHER_NOTEBOOK.md)** | Development timeline and notes |
| **[DISCLAIMER.md](DISCLAIMER.md)** | Usage disclaimer and limitations |

---

## 🔬 Research & Citation

### Novel Contributions

1. **Slot System**: Hierarchical token IDs encoding grammatical structure
2. **Phoneme-Aware Bi-LSTM**: Explicit encoding of Virama, Vowel, Consonant categories
3. **Anusvara Reconstruction**: Specific solution for ം → ത്ത് transformation
4. **Hybrid Pipeline**: FST (mlmorph) + Neural for OOV handling
5. **Vocabulary Efficiency**: Achieving competitive compression with 60x smaller vocabulary

### Citation

```bibtex
@misc{malayalam-morpho-tokenizer,
  title={A Hybrid Morpho-Hierarchical Tokenizer for Agglutinative Languages: 
         Combining Finite State Transducers with Phoneme-Aware Bi-LSTM for Malayalam},
  author={Mohammed Fasil Veloor},
  year={2026},
  publisher={GitHub},
  url={https://github.com/fasilmveloor/malayalam-morpho-hierarchical-tokenizer}
}
```

### Target Venues

- **ACL** (Association for Computational Linguistics)
- **EMNLP** (Conference on Empirical Methods in NLP)
- **LREC** (Language Resources and Evaluation Conference)
- **DravidianLangTech** (Workshop on Dravidian Language Technology)

---

## 🧪 Testing

### Run Tests Locally

```bash
# Run unit tests
python -m pytest tests/

# Run specific test
python tests/test_tokenizer.py

# Compare with baselines
python tests/compare_tokenizers.py
```

### Validate on Colab

1. Upload `Malayalam_Tokenizer_Validation_Colab.ipynb` to Google Colab
2. Run all cells
3. Check test results in the summary

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone and install dev dependencies
git clone https://github.com/fasilmveloor/malayalam-morpho-hierarchical-tokenizer.git
cd malayalam-morpho-hierarchical-tokenizer
pip install -e ".[dev]"

# Run linting
flake8 src/

# Run tests
pytest tests/ -v
```

---

## 📋 Requirements

```
torch>=2.0
transformers>=4.30
mlmorph>=1.0
sentencepiece>=0.1.99
numpy>=1.21
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **mlmorph** - Malayalam Morphological Analyzer by Santhosh Thottingal
- **SMC** (Swathanthra Malayalam Computing) - For corpus and linguistic resources
- **HuggingFace** - For the transformers library and model hosting
- **Google MuRIL** - For multilingual Indian language representations
- **ai4bharat** - For IndicBERT models

---

## 📞 Contact

- **GitHub Issues**: [Report a bug](https://github.com/fasilmveloor/malayalam-morpho-hierarchical-tokenizer/issues)
- **Discussions**: [Join the discussion](https://github.com/fasilmveloor/malayalam-morpho-hierarchical-tokenizer/discussions)

---

<div align="center">

**Made with ❤️ for Malayalam NLP**

[⬆ Back to Top](#malayalam-morpho-hierarchical-tokenizer)

</div>
