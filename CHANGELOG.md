# Malayalam Morpho-Hierarchical Tokenizer - Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-03-06

### Added
- Initial release of Malayalam Morpho-Hierarchical Tokenizer
- Core tokenizer with morphological analysis (`src/tokenizer.py`)
- HuggingFace-compatible tokenizer class (`src/tokenizer_hf.py`)
- Hierarchical vocabulary system with slot-based ID assignment
- Hybrid sandhi splitter (Dictionary → FST → Neural)
- Phoneme-enhanced Bi-LSTM model for OOV handling
- BIO tagging model for morpheme boundary detection
- Bi-LSTM with CRF layer for constrained decoding
- Sandhi reconstruction logic (ം → ത്ത് transformation)
- Comprehensive test suite
- Jupyter notebook tutorials
- Validation notebook for Google Colab
- Complete documentation

### Performance
- 87.22% morphology coverage on SMC corpus
- 26.06% OOV rate
- 1,522 words/second throughput
- 91.67% BIO tagging accuracy

### Models
- `best_sandhi_model.pt` - Main sandhi splitting model
- `phoneme_sandhi_model.pt` - Phoneme-enhanced Bi-LSTM
- `bio_sandhi_model.pt` - BIO tagger for boundaries
- `bilstm_crf_model.pt` - CRF-enhanced model

### Documentation
- README.md with quick start guide
- ARCHITECTURE.md with system design
- MODEL_CARD.md for HuggingFace
- HUGGINGFACE_INTEGRATION.md
- TESTING_CHECKLIST.md
- CONTRIBUTING.md

## [0.5.0] - 2024-02-15

### Added
- Neural sandhi splitter with Bi-LSTM architecture
- Phoneme feature encoding (10-dimensional)
- High-frequency word cache for optimization
- Fallback morpheme splitting rules

### Changed
- Improved sandhi transformation accuracy
- Optimized mlmorph integration

## [0.4.0] - 2024-02-01

### Added
- Slot-based vocabulary system
- Token classification by grammatical category
- Encode-decode roundtrip support

### Fixed
- Unicode normalization issues
- Virama handling in sandhi combination

## [0.3.0] - 2024-01-15

### Added
- mlmorph FST integration
- Basic morphological analysis
- Case marker detection

### Changed
- Refactored tokenizer architecture

## [0.2.0] - 2024-01-01

### Added
- Basic tokenizer implementation
- Vocabulary management
- Simple suffix stripping

## [0.1.0] - 2023-12-15

### Added
- Project initialization
- Basic project structure
- Initial documentation

---

## Version History

| Version | Date | Description |
|---------|------|-------------|
| 1.0.0 | 2024-03-06 | Initial public release |
| 0.5.0 | 2024-02-15 | Neural models added |
| 0.4.0 | 2024-02-01 | Slot system implemented |
| 0.3.0 | 2024-01-15 | mlmorph integration |
| 0.2.0 | 2024-01-01 | Basic tokenizer |
| 0.1.0 | 2023-12-15 | Project start |

---

## Roadmap

### v1.1.0 (Planned)
- [ ] Expand training data to 500+ examples
- [ ] Add CRF layer optimization
- [ ] Benchmark against indic-bert tokenizer
- [ ] Linguistic verification by scholars

### v1.2.0 (Planned)
- [ ] Transfer learning from Sanskrit
- [ ] Self-attention for long-distance sandhi
- [ ] Multilingual support (Tamil, Kannada)

### v2.0.0 (Future)
- [ ] Full HuggingFace Hub integration
- [ ] Pre-trained language model
- [ ] API endpoint for tokenization service

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.
