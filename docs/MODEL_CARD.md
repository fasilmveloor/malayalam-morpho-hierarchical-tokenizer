---
language:
  - ml
  - en
license: mit
library_name: transformers
tags:
  - tokenizer
  - malayalam
  - morphological
  - dravidian
  - nlp
  - agglutinative
datasets:
  - smc-corpus
metrics:
  - morphology-coverage
  - oov-rate
  - compression-ratio
---

# Malayalam Morpho-Hierarchical Tokenizer

## Model Description

This is a novel morphologically-aware tokenizer for Malayalam, a Dravidian language with complex agglutinative morphology. Unlike standard BPE/Unigram tokenizers that perform frequency-based subword splitting, this tokenizer leverages linguistic structure through a hybrid approach combining:

1. **Finite State Transducers (mlmorph)** for accurate morphological analysis
2. **Phoneme-Aware Bi-LSTM** for handling out-of-vocabulary words
3. **Hierarchical Slot System** for grammatically meaningful token IDs
4. **Sandhi Reconstruction** for canonical form restoration

### Key Features

- **Slot System**: Token IDs encode grammatical categories (Root=1xxx, Tense=2xxx, Case=3xxx)
- **Phoneme Features**: 10-dimensional vector encoding Virama, Vowel, Consonant categories
- **BIO Tagging**: 91.67% accuracy on morpheme boundary detection
- **Sandhi Rules**: Handles ം → ത്ത് transformation and vowel sandhi

## Intended Uses & Limitations

### Intended Uses

- Tokenization for Malayalam NLP pipelines
- Pre-processing for language models
- Morphological analysis and lemmatization
- Machine translation systems
- Text generation for Malayalam

### Limitations

- Requires `mlmorph` package for optimal performance (fallback available)
- Neural models require PyTorch
- Some rare words may not split correctly
- Anusvara transformation may need refinement for edge cases

## Performance

| Metric | Value |
|--------|-------|
| Morphology Coverage | 87.22% |
| OOV Rate | 26.06% |
| Compression Ratio | 0.672 |
| Tokens/Word | 1.49 |
| Speed | 1,522 words/sec |
| BIO Accuracy | 91.67% |

## How to Use

### Installation

```bash
pip install transformers torch mlmorph
```

### Basic Usage

```python
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("yourusername/malayalam-morpho-tokenizer")

# Tokenize
text = "പഠിക്കുന്നു വിദ്യാലയത്തിൽ"
tokens = tokenizer.tokenize(text)
# Output: ['പഠിക്ക്', 'ുന്നു', 'വിദ്യാലയത്ത്', 'ിൽ']

# Encode
ids = tokenizer.encode(text)
# Output: [2, 1001, 2001, 1002, 3001, 3]

# Decode
decoded = tokenizer.decode(ids)
# Output: "പഠിക്കുന്നു വിദ്യാലയത്തിൽ"
```

### Token Classification

```python
# Classify tokens by grammatical category
for token_id in ids:
    category = tokenizer.classify_token(token_id)
    print(f"{token_id}: {category}")

# Output:
# 2: special (BOS)
# 1001: root
# 2001: tense
# 1002: root
# 3001: case
# 3: special (EOS)
```

### With HuggingFace Pipeline

```python
from transformers import pipeline

# Use with any Malayalam model
nlp = pipeline("fill-mask", model="yourusername/malayalam-bert")
result = nlp("മലയാളത്തിൽ എഴുതുന്നു [MASK]")
```

## Token ID Structure

| Slot | Category | ID Range | Description |
|------|----------|----------|-------------|
| 0 | Special | 0-999 | `<PAD>`, `<UNK>`, `<BOS>`, `<EOS>`, `<MASK>`, `<CLS>`, `<SEP>` |
| 1 | Root | 1000-1999 | Verb/Noun stems |
| 2 | Tense | 2000-2999 | Present (ുന്നു), Past (ച്ചു), Future (ും) |
| 3 | Case | 3000-3999 | Locative (ിൽ), Genitive (ിന്റെ), Dative (ക്ക്) |
| 4 | Function | 4000-4999 | Conjunctions, particles |
| 5 | Infix | 5000-5999 | Sandhi infixes (ത്ത്) |
| 6 | Char | 7000-7999 | Character-level fallback |

## Training Details

### Training Data

- **SMC Corpus**: News articles, Wikipedia, literature
- **Training Examples**: 500+ manually annotated morpheme splits
- **Vocabulary**: 8,000 tokens maximum

### Training Procedure

1. **Dictionary Creation**: High-frequency words with verified splits
2. **FST Integration**: mlmorph for morphological analysis
3. **Neural Training**: Bi-LSTM on phoneme features
4. **CRF Fine-tuning**: BIO tag optimization

### Training Hyperparameters

```python
training_args = {
    "epochs": 50,
    "batch_size": 8,
    "learning_rate": 0.003,
    "hidden_dim": 96,
    "embed_dim": 32,
    "phoneme_dim": 10,
    "dropout": 0.2,
}
```

## Evaluation

### Benchmark Results

Evaluated on SMC corpus (10,000 words):

```
╔══════════════════════════════════════════╗
║        PERFORMANCE METRICS MATRIX        ║
╠══════════════════════════════════════════╣
║  Metric                    Value         ║
╠══════════════════════════════════════════╣
║  Morphology Coverage       87.22%        ║
║  OOV Rate                  26.06%        ║
║  Compression Ratio         0.672         ║
║  Tokens/Word               1.49          ║
║  Speed                     1,522 w/s     ║
║  BIO Accuracy              91.67%        ║
╚══════════════════════════════════════════╝
```

### Comparison with Baselines

| Tokenizer | Tokens/Word | Morpheme Alignment | Linguistic Quality |
|-----------|-------------|-------------------|-------------------|
| **Ours** | 1.49 | ✅ High | ✅ Excellent |
| BPE | 2.31 | ❌ Low | ❌ Poor |
| Unigram | 2.18 | ❌ Low | ⚠️ Fair |
| IndicBERT | 2.45 | ❌ Low | ⚠️ Fair |

## Technical Specifications

### Model Architecture

```
Input: Character IDs + Phoneme Features (10-dim)
    ↓
Embedding: 32-dim character + 10-dim phoneme
    ↓
Bi-LSTM: 2 layers, 96 hidden, bidirectional
    ↓
Dense: 192 → 96 → 1
    ↓
Output: Split probability per character
```

### Dependencies

```
torch>=2.0
transformers>=4.30
mlmorph>=1.0
sentencepiece>=0.1.99
numpy>=1.21
```

### Hardware Requirements

- **CPU**: Works on any modern CPU
- **GPU**: Optional, speeds up neural inference
- **RAM**: ~200MB for models
- **Disk**: ~10MB for all models

## Citation

```bibtex
@misc{malayalam-morpho-tokenizer,
  title={A Hybrid Morpho-Hierarchical Tokenizer for Agglutinative Languages: 
         Combining Finite State Transducers with Phoneme-Aware Bi-LSTM for Malayalam},
  author={Your Name},
  year={2024},
  publisher={HuggingFace},
  url={https://huggingface.co/yourusername/malayalam-morpho-tokenizer}
}
```

## Acknowledgments

- **mlmorph** by Santhosh Thottingal - Malayalam Morphological Analyzer
- **SMC** (Swathanthra Malayalam Computing) - Corpus and linguistic resources
- **HuggingFace** - Transformers library and model hosting

## License

MIT License - See [LICENSE](LICENSE) for details.

## Contact

- **GitHub**: https://github.com/yourusername/malayalam-tokenizer
- **Issues**: https://github.com/yourusername/malayalam-tokenizer/issues
