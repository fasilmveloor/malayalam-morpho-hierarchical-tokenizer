# Fazlab Tokenizer Benchmarks

## Iteration: v1-beta (2026-03-06)
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
