# Training Strategy Decision Matrix

## Option 1: Fine-Tune Existing Model (Recommended First Step)

### Advantages
- **Fast iteration**: 2-4 hours to see results
- **Lower compute**: Single GPU or even CPU
- **Leverages pre-trained knowledge**: Language understanding already exists
- **Quick validation**: Can test tokenizer quality immediately

### Recommended Models
| Model | Size | Why |
|-------|------|-----|
| **indic-bert** | 110M | Already trained on Malayalam corpus |
| **muril-base** | 110M | Multilingual, strong Dravidian support |
| **malayalam-bert** | 110M | Community model, if available |

### Steps
1. Replace tokenizer with our Morpho-Hierarchical tokenizer
2. Fine-tune on Malayalam text (Wikipedia + SMC corpus)
3. Evaluate on downstream tasks (NER, classification)

---

## Option 2: Train From Scratch (Long-term Goal)

### Advantages
- **Full control**: Architecture optimized for Malayalam
- **No tokenization mismatch**: Model learns our morpheme boundaries
- **Smaller model possible**: Can train efficient 50-100M model
- **Research contribution**: Novel tokenizer + model combo

### Requirements
| Resource | Amount |
|----------|--------|
| Training data | 5-10 GB Malayalam text |
| Compute | 4-8 GPUs for 1-2 weeks |
| Time | 1-2 weeks |
| Budget | $500-2000 (cloud) |

### Steps
1. Collect more Malayalam text (Wikipedia dump, news, books)
2. Pre-train with our tokenizer from scratch
3. Evaluate and iterate

---

## Recommendation: Hybrid Approach

### Phase 1: Fine-Tune First (Week 1-2)
```
indic-bert + Morpho-Hierarchical Tokenizer
          ↓
    Fine-tune on Malayalam corpus
          ↓
    Evaluate: NER, classification, perplexity
```

### Phase 2: Train From Scratch (Month 2-3)
```
Collect 5GB+ Malayalam text
          ↓
Pre-train 100M parameter model
          ↓
    Compare with Phase 1 results
```

---

## Quick Start: Fine-Tuning Setup

```python
# 1. Load indic-bert
from transformers import AutoModelForMaskedLM, AutoTokenizer

model = AutoModelForMaskedLM.from_pretrained("ai4bharat/indic-bert")
# Replace tokenizer with ours
# Fine-tune on Malayalam corpus

# 2. Evaluate
# - Perplexity on held-out text
# - Downstream task performance
# - Compare with original indic-bert
```

---

## Success Metrics

| Metric | Fine-Tune Target | Scratch Target |
|--------|------------------|----------------|
| Perplexity ↓ | < 15 | < 10 |
| Vocab Efficiency ↑ | +15% vs baseline | +25% vs baseline |
| NER F1 ↑ | 85%+ | 90%+ |
| Training Time | 2-4 hours | 1-2 weeks |
| Compute Cost | <$50 | $500-2000 |

---

## Recommendation

**Start with Fine-Tuning indic-bert** using our tokenizer.

This gives us:
1. Quick validation of tokenizer quality
2. Immediate usable model for downstream tasks
3. Research baseline for comparison
4. Time to collect more training data for scratch training
