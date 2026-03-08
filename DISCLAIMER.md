# Disclaimer

This project is a **work in progress** and is provided for research and educational purposes.

---

## Code Development

- Code was developed with AI assistance for structure, documentation, and implementation patterns
- Core architecture, linguistic design decisions, and research direction are original work
- Neural model implementations follow standard Bi-LSTM/CRF architectures from established literature
- Git history was squashed during repository cleanup; development milestones are documented in `RESEARCHER_NOTEBOOK.md`

## Performance Claims

All performance benchmarks are from internal testing:

| Metric | Source | Verification Status |
|--------|--------|---------------------|
| Coverage (87.22%) | SMC corpus sample | **Not independently verified** |
| BIO Accuracy (91.67%) | Internal test set | **Not independently verified** |
| Throughput (1,522 w/s) | CPU benchmark | **Hardware dependent** |

**Important:** Results have **not** been peer-reviewed or independently verified. Actual performance may vary based on:

- Input text domain (news, literature, social media, etc.)
- Hardware configuration
- Model version and vocabulary size
- mlmorph/SFST installation and version

## Known Limitations

### 1. Vocabulary Scope
Current vocabulary uses 29,000 root slots allocated for:
- Native Dravidian roots (~15-20K)
- Sanskrit tatsama (~5-8K)
- Modern loanwords (~3-5K)
- Proper nouns (~2-3K)

This is designed for practical modern Malayalam text but may not cover rare/archaic vocabulary.

### 2. Domain Specificity
Training data primarily from:
- SMC Malayalam corpus
- Wikipedia-style text
- News articles

Performance may degrade on:
- Literary/poetic text
- Dialectal variations
- Code-mixed text (Malayalam-English)

### 3. Sandhi Edge Cases
While 15 sandhi rules are implemented, rare patterns may not be handled:
- Complex compound formations
- Dialectal sandhi variations
- Archaic grammatical forms

### 4. External Dependencies
Full functionality requires:
- **mlmorph**: Swathanthra Malayalam Computing's FST analyzer
- **SFST**: Stuttgart Finite State Transducer Tools

These must be installed separately and have their own licensing requirements.

---

## Use at Your Own Risk

### Appropriate Use Cases
- Research on Malayalam NLP
- Educational demonstrations of morphological tokenization
- Experimental NLP pipelines
- Comparative benchmarking studies

### Not Recommended For
- Production systems without additional testing and validation
- Medical, legal, or financial document processing
- Critical infrastructure applications
- Systems requiring guaranteed accuracy

---

## Contributing

Contributions and independent verification are welcome:

| Contribution Type | How |
|-------------------|-----|
| Bug reports | GitHub Issues |
| Code contributions | Pull Requests |
| Benchmark verification | Independent testing encouraged |
| Academic collaboration | Welcome via GitHub or email |

---

## Citation

If you use this work in research, please:

1. Cite appropriately with version number
2. Verify results independently on your dataset
3. Report any discrepancies found

**Suggested citation format:**

```bibtex
@software{malayalam_morpho_tokenizer_2026,
  author = {Mohammed Fasil Veloor},
  title = {Malayalam Morpho-Hierarchical Tokenizer},
  version = {0.9.0-rc},
  year = {2026},
  url = {https://github.com/fasilmveloor/malayalam-morpho-hierarchical-tokenizer}
}
```

---

## License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.

---

*Last updated: 2026-03-08 (v0.9.0-rc)*
