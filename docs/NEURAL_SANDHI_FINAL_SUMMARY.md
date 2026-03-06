# Neural Sandhi Splitter - Final Implementation Summary

## Overview

Successfully implemented a **three-tier Neural Sandhi Splitter** with phoneme feature engineering, BIO tagging, and sandhi reconstruction logic for Malayalam morphological analysis.

---

## 🏗️ Architecture Comparison

### 1. Baseline Bi-LSTM
```
Character IDs → Embedding (48-dim) → Bi-LSTM → Dense → Sigmoid
```
- **Parameters**: 359,713
- **Training**: 100 epochs
- **Accuracy**: 88.24%

### 2. Phoneme-Enhanced Bi-LSTM
```
Char IDs + Phoneme Features (10-dim) → Concat → Bi-LSTM → Dense → Sigmoid
```
- **Parameters**: 353,857
- **Training**: 100 epochs
- **Accuracy**: 89.22%
- **Innovation**: Explicit phonetic category encoding

### 3. BIO-Tagging Bi-LSTM
```
Char IDs + Phoneme Features → Bi-LSTM → Dense → Softmax (B/I)
```
- **Parameters**: Similar
- **Training**: 100 epochs
- **Accuracy**: 91.67%
- **Innovation**: Sequence labeling approach

---

## 🔬 Phoneme Feature Engineering

### 10-Dimensional Feature Vector

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | Is_Vowel | Independent vowels (അ, ആ, ഇ, ...) |
| 1 | Is_VowelSign | Dependent vowel signs (ാ, ി, ീ, ...) |
| 2 | Is_Consonant | Consonants (ക, ഖ, ഗ, ...) |
| 3 | **Is_Virama** | **Chandrakkala (്) - CRITICAL** |
| 4 | Is_AnuSvara | Anusvara (ം) - transforms in sandhi |
| 5 | Is_Chillu | Chillu letters (ൽ, ർ, ൻ, ൺ, ൿ) |
| 6 | Is_Conjunct | Part of conjunct cluster |
| 7 | Is_Digit | Malayalam/Arabic digits |
| 8 | Is_Punctuation | Punctuation marks |
| 9 | Is_Other | Other characters |

### Example Encoding
```
Word: പഠിക്കുന്നു

Char  Phoneme Features (V, VS, C, Vir, Anu, Ch...)
പ    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  ← Consonant
ഠ    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  ← Consonant
ി    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  ← Vowel Sign
ക    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  ← Consonant
്    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  ← VIRAMA (split point indicator!)
ക    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  ← Consonant
...
```

---

## 🏷️ BIO Tagging System

### Tag Definitions
- **B (Begin)**: First character of a morpheme
- **I (Inside)**: Continuation of a morpheme

### Example BIO Sequence
```
Word:    പ ഠ ി ക ് ക ു ൻ ് ൻ ു
BIO:     B  I  I  I  B  I  I  I  I  I  I
         ↑                    ↑
         |                    └── Split point (B tag)
         └── First char always B
```

### Advantages
1. **Structural coherence**: Model learns morpheme-level patterns
2. **Multi-split handling**: Natural support for 3+ component words
3. **Constrained output**: BIO grammar prevents invalid sequences

---

## 🔄 Sandhi Reconstruction

### Canonical Form Restoration

| Surface Form | Canonical Form | Rule |
|--------------|----------------|------|
| വിദ്യാലയത്ത് | വിദ്യാലയം | anusvara_infix |
| കേരളത്ത് | കേരളം | anusvara_infix |
| പഠിക്ക് | പഠിക്ക് | stem_form |
| വീട് | വീട് | stem_form |

### Reconstruction Rules

1. **Anusvara Infix**: ം + case marker → ത്ത് + case marker
   - വിദ്യാലയം + ഇൽ → വിദ്യാലയത്തിൽ

2. **Stem Form**: Verbs retain virama in stem form
   - പഠിക്ക് + ുന്നു → പഠിക്കുന്നു

3. **Vowel Insertion**: Consonant + vowel suffix → vowel insertion
   - പറ + യുന്നു → പറയുന്നു

---

## 📊 Evaluation Results

### SMC Corpus (15,885 unique words)

| Metric | Value |
|--------|-------|
| Multi-component words | 79.2% |
| Single-component words | 20.8% |
| Avg components/word | 1.86 |
| Vocabulary reduction | 13.6% |

### Split Source Distribution

| Source | Count | Percentage |
|--------|-------|------------|
| mlmorph FST | 9,107 | 57.3% |
| Neural Bi-LSTM | 3,464 | **21.8%** |
| Dictionary | 11 | 0.1% |
| Fallback | 3,303 | 20.8% |

### Model Comparison

| Model | Val Accuracy | Key Advantage |
|-------|--------------|---------------|
| Baseline | 88.24% | Simple, fast |
| Phoneme | 89.22% | Phonetic awareness |
| **BIO** | **91.67%** | Structural coherence |

---

## 📁 Files Created

```
malayalam-tokenizer/
├── src/
│   ├── bilstm_sandhi.py          # Baseline Bi-LSTM
│   ├── phoneme_sandhi.py         # Phoneme-enhanced model
│   ├── bio_sandhi.py             # BIO-tagging model
│   ├── hybrid_sandhi.py          # Hybrid pipeline integration
│   └── sandhi_reconstruction.py  # Canonical form restoration
├── models/
│   ├── best_sandhi_model.pt      # Baseline weights
│   ├── phoneme_sandhi_model.pt   # Phoneme model weights
│   └── bio_sandhi_model.pt       # BIO model weights
└── data/
    ├── sandhi_training_data.json      # Training examples
    ├── phoneme_training_data.json     # Phoneme training data
    └── neural_sandhi_evaluation.json  # Evaluation results
```

---

## 🚀 Usage

### Basic Splitting
```python
from hybrid_sandhi import HybridSandhiSplitter

splitter = HybridSandhiSplitter()
components = splitter.split('പഠിക്കുന്നു')
# ['പഠിക്ക്', 'ുന്നു']
```

### With Reconstruction
```python
from sandhi_reconstruction import SandhiReconstructor

recon = SandhiReconstructor()
result = recon.reconstruct_word('വിദ്യാലയത്തിൽ', ['വിദ്യാലയത്ത്', 'ിൽ'])
# {'surface': 'വിദ്യാലയത്ത് + ിൽ', 'canonical': 'വിദ്യാലയം + ിൽ'}
```

---

## 🎯 Key Achievements

1. **Neural Generalization Layer**: Handles 21.8% of words that mlmorph cannot process
2. **Phoneme Awareness**: Model explicitly learns sandhi-critical phonetic categories
3. **BIO Tagging**: 91.67% accuracy with structural coherence
4. **Reconstruction Logic**: Restores canonical forms for vocabulary consistency
5. **Hybrid Pipeline**: Dictionary → FST → Neural ensures both precision and coverage

---

## 🔮 Future Improvements

1. **Expand Training Data**: Current 95 examples → Target 500+ with linguist verification
2. **Transfer Learning**: Pre-train on Sanskrit morphology, fine-tune on Malayalam
3. **CRF Layer**: Add Conditional Random Field for BIO sequence optimization
4. **Attention Mechanism**: Self-attention for long-distance sandhi dependencies
5. **Dialect Handling**: Regional variations in sandhi patterns

---

## 📚 References

- Malayalam Unicode Block: U+0D00–U+0D7F
- mlmorph: FST-based Malayalam morphological analyzer
- BIO Tagging: Standard NLP sequence labeling technique
- Sandhi: Sanskrit/Malayalam phonological transformation rules
