# Malayalam Tokenizer Benchmarks

## Iteration: v0.1.0-beta (2026-03-06)
**Dataset:** SMC Malayalam Corpus (28,447 words)
**Goal:** Minimize fertility rate while maintaining morphological alignment.

### Comparison Table
| Metric | Morpho-Hierarchical (v1) | Unigram (Baseline) | BPE |
| :--- | :--- | :--- | :--- |
| **Avg Tokens/Word** | **1.36** | 2.20 | 2.24 |
| **Compression Ratio** | 0.7356 | 0.4540 | 0.4465 |
| **Unique Tokens** | 11,692 | 4,753 | 4,572 |
| **Throughput** | 2,126 w/s | 181,363 w/s | 146,599 w/s |

### Analysis
The v1-beta successfully solves the "Anusvara split" and "Swarachinnam fragmentation" issues identified in early R&D. 
Next steps: Optimize the `get_malayalam_syllables` regex logic to increase processing speed.

---

## [v0.2.0-beta] - 2026-03-06 (Hybrid Neural Transition)
**Focus:** Hybrid Neural-Statistical Logic & Hardware Validation
**Environment:** Local Workstation (Transitioned from Google Colab)

### Comparison Table
| Metric | Morpho-Hierarchical (v0.2.0) | BPE | Unigram | Character |
| :--- | :--- | :--- | :--- | :--- |
| **Avg Tokens/Word** | **1.49** | 2.24 | 2.20 | 9.54 |
| **Morph. Alignment** | **23.29%** | 15.34% | 15.55% | 0.00% |
| **Unique Tokens** | 12,306 | 4,572 | 4,753 | 67 |
| **Throughput** | 2,166 w/s | 150,149 w/s | 183,055 w/s | 85,684 w/s |

### Architectural Analysis
1. **Efficiency:** Maintained a 0.67 compression ratio, proving the Hybrid model's consistency across hardware environments.
2. **Alignment:** Achieved the highest morphological alignment recorded to date (23.29%), justifying the added complexity of the `sandhi_dictionary` and `neural_splitter`.
3. **Hardware Note:** Local execution provides a more stable throughput baseline of ~2.1k w/s.

## [v0.3.0-beta] - 2026-03-06 (Neural Era)
**Focus:** Neural Generalization via Bi-LSTM Fallback
**Architecture:** Character-level Bi-LSTM (48-dim Embeddings, 2-layer)
**Dependencies:** PyTorch, mlmorph

### Comparison Table
| Metric | v0.3.0-Beta (Neural) | v0.2.0-Beta (Hybrid) | BPE (Standard) |
| :--- | :--- | :--- | :--- |
| **Avg Tokens/Word** | **1.52** | 1.49 | 2.24 |
| **Morph. Alignment** | **28.45%** | 23.29% | 15.34% |
| **OOV Recovery** | **+21.8%** | N/A | 0% |
| **Vocab Reduction** | **13.6%** | N/A | N/A |
| **Throughput** | **~850 w/s** | 2,166 w/s | 150k w/s |

### Analysis
* **The Intelligence Win:** The Bi-LSTM handles the 21.8% of words that were previously OOV for the FST. This pushed Morphological Alignment to an all-time high of **28.45%**.
* **The Complexity Cost:** Avg Tokens/Word increased slightly to 1.52 as the neural model identifies more granular sub-morphemes. 
* **The Bottleneck:** Throughput dropped significantly due to PyTorch inference overhead. Optimization (TorchScript/ONNX) is now a Priority 1 task.


