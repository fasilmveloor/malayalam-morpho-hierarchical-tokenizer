# Malayalam Morpho-Hierarchical Tokenizer - Testing Checklist

## Download & Setup


---

## 1. Environment Setup Tests

### 1.1 Google Colab Setup
- [ ] Open Google Colab (colab.research.google.com)
- [ ] Upload `Malayalam_Tokenizer_Validation_Colab.ipynb`
- [ ] Run "Install Dependencies" cell
- [ ] Verify PyTorch version printed
- [ ] Verify GPU/CPU device detected

### 1.2 Package Download
- [ ] Run "Download Tokenizer Package" cell
- [ ] Verify download completes without errors
- [ ] Verify extraction to `/content/malayalam-tokenizer`
- [ ] Check directory structure matches expected

### 1.3 Module Import
- [ ] Run "Import Modules" cell
- [ ] Verify `✓ Core modules imported successfully`
- [ ] Run "Initialize Tokenizer" cell
- [ ] Verify mlmorph initializes (or fallback message appears)
- [ ] Check vocabulary size is displayed

---

## 2. Core Functionality Tests

### 2.1 Basic Tokenization
| Test Word | Expected Split | Pass/Fail |
|-----------|---------------|-----------|
| പഠിക്കുന്നു | പഠിക്ക് + ുന്നു | ☐ |
| വിദ്യാലയം | Single token | ☐ |
| കേരളത്തിൽ | കേരളത്ത് + ിൽ | ☐ |
| ഭാരതനാട്യം | Compound split | ☐ |
| അധ്യാപിക | Single/Root | ☐ |

### 2.2 Sandhi Splitting
| Compound Word | Expected Components | Pass/Fail |
|---------------|---------------------|-----------|
| തിരുവനന്തപുരം | തിരു + അനന്തപുരം | ☐ |
| പാലക്കാട് | പാല + ക്കാട് | ☐ |
| ഭാരതനാട്യം | ഭാരത + നാട്യം | ☐ |
| പ്രധാനമന്ത്രി | പ്രധാന + മന്ത്രി | ☐ |

### 2.3 Case Marker Detection
| Word | Expected Marker | Detected | Pass/Fail |
|------|-----------------|----------|-----------|
| വീട്ടിൽ | ിൽ (Locative) | ___ | ☐ |
| വീട്ടുകാരന്റെ | ന്റെ (Genitive) | ___ | ☐ |
| വീടിന് | ിന് (Dative) | ___ | ☐ |
| വീടിനെ | ിനെ (Accusative) | ___ | ☐ |
| കേരളത്തിൽ | ിൽ (Locative) | ___ | ☐ |

### 2.4 Tense Marker Detection
| Word | Expected Tense | Detected | Pass/Fail |
|------|----------------|----------|-----------|
| പഠിക്കുന്നു | Present (ുന്നു) | ___ | ☐ |
| പഠിച്ചു | Past (ച്ചു) | ___ | ☐ |
| പഠിക്കും | Future (ും) | ___ | ☐ |
| വരുന്നു | Present (ുന്നു) | ___ | ☐ |
| വന്നു | Past | ___ | ☐ |

---

## 3. Edge Case Tests

### 3.1 OOV Handling
| Word | Behavior | Pass/Fail |
|------|----------|-----------|
| കുട്ടികളുടെയെല്ലാം | Produces tokens | ☐ |
| പുതിയതരം | Produces tokens | ☐ |
| അസാധാരണമായ | Produces tokens | ☐ |
| വിചിത്രസുന്ദരി | Produces tokens | ☐ |

### 3.2 Anusvara Transformation (ം → ത്ത്)
| Input | Expected Behavior | Pass/Fail |
|-------|-------------------|-----------|
| വിദ്യാലയം + ിൽ | വിദ്യാലയത്തിൽ | ☐ |
| കേരളം + ിൽ | കേരളത്തിൽ | ☐ |
| പുസ്തകം + ിൽ | പുസ്തകത്തിൽ | ☐ |

### 3.3 Long Compounds
| Word | Time (ms) | Pass/Fail |
|------|-----------|-----------|
| തിരുവനന്തപുരം | ___ | ☐ |
| ഭാരതീയജനതാപാർട്ടി | ___ | ☐ |
| സ്വാതന്ത്ര്യസമരസേനാനി | ___ | ☐ |
| വിദ്യാഭ്യാസവകുപ്പ് | ___ | ☐ |

### 3.4 Encode-Decode Consistency
| Original | Decoded | Match | Pass/Fail |
|----------|---------|-------|-----------|
| പഠിക്കുന്നു | ___ | ☐ Yes ☐ No | ☐ |
| വിദ്യാലയം | ___ | ☐ Yes ☐ No | ☐ |
| കേരളത്തിൽ | ___ | ☐ Yes ☐ No | ☐ |
| അധ്യാപിക | ___ | ☐ Yes ☐ No | ☐ |

---

## 4. Performance Benchmarks

### 4.1 Speed Test (500 words)
| Metric | Expected | Actual | Pass/Fail |
|--------|----------|--------|-----------|
| Words/sec | > 500 | ___ | ☐ |
| Total time | < 2s | ___ | ☐ |
| Tokens/word | 1.3-2.0 | ___ | ☐ |

### 4.2 Morphology Coverage
| Metric | Expected | Actual | Pass/Fail |
|--------|----------|--------|-----------|
| Coverage % | > 80% | ___ | ☐ |
| OOV rate | < 30% | ___ | ☐ |
| Cache hit rate | ___ | ___ | ☐ |

### 4.3 Memory Usage
| Metric | Expected | Actual | Pass/Fail |
|--------|----------|--------|-----------|
| Model load time | < 5s | ___ | ☐ |
| Memory footprint | < 500MB | ___ | ☐ |

---

## 5. Comparison Tests (BPE vs Unigram vs Ours)

### 5.1 Token Count Comparison
| Word | Ours | BPE | Unigram | Best |
|------|------|-----|---------|------|
| പഠിക്കുന്നു | ___ | ___ | ___ | ___ |
| വിദ്യാലയത്തിൽ | ___ | ___ | ___ | ___ |
| കേരളത്തിൽ | ___ | ___ | ___ | ___ |
| തിരുവനന്തപുരം | ___ | ___ | ___ | ___ |

### 5.2 Linguistic Quality
| Criterion | Ours | BPE | Unigram |
|-----------|------|-----|---------|
| Morpheme boundaries | ☐ | ☐ | ☐ |
| Root preservation | ☐ | ☐ | ☐ |
| Suffix isolation | ☐ | ☐ | ☐ |

---

## 6. HuggingFace Integration Tests

### 6.1 Tokenizer Class
- [ ] `MorphoHierarchicalTokenizerFast` imports without errors
- [ ] `tokenize()` method works
- [ ] `encode()` method works
- [ ] `decode()` method works
- [ ] `classify_token()` returns correct category

### 6.2 Slot System Verification
| Token Type | ID Range | Example | Pass/Fail |
|------------|----------|---------|-----------|
| Special | 0-999 | PAD=0, UNK=1 | ☐ |
| Root | 1000-1999 | ___ | ☐ |
| Tense | 2000-2999 | ___ | ☐ |
| Case | 3000-3999 | ___ | ☐ |
| Function | 4000-4999 | ___ | ☐ |
| Infix | 5000-5999 | ___ | ☐ |

### 6.3 Export Functionality
- [ ] `save_vocabulary()` creates vocab.json
- [ ] `tokenizer_config.json` is valid
- [ ] `char_vocab.json` is created
- [ ] Exported files can be reloaded

---

## 7. Neural Model Tests

### 7.1 Model Loading
- [ ] `best_sandhi_model.pt` loads without errors
- [ ] `phoneme_sandhi_model.pt` loads without errors
- [ ] `bio_sandhi_model.pt` loads without errors

### 7.2 Model Inference
| Model | Test Word | Prediction | Pass/Fail |
|-------|-----------|------------|-----------|
| BiLSTM Sandhi | പഠിക്കുന്നു | ___ | ☐ |
| Phoneme BiLSTM | വിദ്യാലയം | ___ | ☐ |
| BIO Tagger | കേരളത്തിൽ | ___ | ☐ |

---

## 8. Real-World Test Cases

### 8.1 News Article Sample
```
കേരളത്തിൽ പുതിയ വിദ്യാഭ്യാസ നയം നടപ്പാക്കുന്നു. 
തിരുവനന്തപുരത്ത് സർക്കാർ ആശുപത്രിയിൽ പുതിയ സൗകര്യങ്ങൾ ആരംഭിച്ചു.
```

- [ ] All words tokenized
- [ ] Compound words split correctly
- [ ] Case markers isolated
- [ ] Verb forms decomposed

### 8.2 Literary Text Sample
```
അധ്യാപിക വിദ്യാലയത്തിൽ പഠിക്കുന്ന കുട്ടികളെ സഹായിച്ചു.
```

- [ ] All words tokenized
- [ ] Proper morphological decomposition

### 8.3 Formal Document Sample
```
ഭാരതസർക്കാർ പുതിയ വിദ്യാഭ്യാസ നയം പ്രഖ്യാപിച്ചു.
```

- [ ] All words tokenized
- [ ] Complex compounds handled

---

## 9. Error Handling Tests

### 9.1 Invalid Input
| Input | Behavior | Pass/Fail |
|-------|----------|-----------|
| Empty string | Returns empty/special tokens | ☐ |
| English text "hello" | Handles gracefully | ☐ |
| Mixed "hello മലയാളം" | Tokenizes Malayalam only | ☐ |
| Numbers "123" | Handles gracefully | ☐ |
| Special chars "!@#" | Handles gracefully | ☐ |

### 9.2 Boundary Cases
| Input | Behavior | Pass/Fail |
|-------|----------|-----------|
| Single char "അ" | Returns valid token | ☐ |
| Very long word (50+ chars) | Processes without crash | ☐ |
| Repeated word 100x | No memory leak | ☐ |

---

## 10. Final Validation

### 10.1 Test Summary
| Category | Tests | Passed | Failed | Pass Rate |
|----------|-------|--------|--------|-----------|
| Setup | 8 | ___ | ___ | ___% |
| Core Functionality | 19 | ___ | ___ | ___% |
| Edge Cases | 15 | ___ | ___ | ___% |
| Performance | 8 | ___ | ___ | ___% |
| Comparison | 9 | ___ | ___ | ___% |
| HuggingFace | 10 | ___ | ___ | ___% |
| Neural Models | 6 | ___ | ___ | ___% |
| Real-World | 10 | ___ | ___ | ___% |
| Error Handling | 10 | ___ | ___ | ___% |
| **TOTAL** | **95** | ___ | ___ | ___% |

### 10.2 Known Limitations
- [ ] mlmorph requires installation (fallback available)
- [ ] Neural models need PyTorch
- [ ] Some rare words may not split correctly
- [ ] Anusvara transformation may need refinement

### 10.3 Sign-off
- **Tester:** _______________
- **Date:** _______________
- **Environment:** ☐ Colab ☐ Local
- **Overall Result:** ☐ PASS ☐ FAIL

---

## Quick Start Commands

```bash
# Download
wget https://files.catbox.moe/2jh47c.zip -O malayalam-tokenizer.zip

# Extract
unzip malayalam-tokenizer.zip

# Test
cd malayalam-tokenizer
python -c "from src.tokenizer_hf import MorphoHierarchicalTokenizerFast; t = MorphoHierarchicalTokenizerFast(); print(t.tokenize('പഠിക്കുന്നു'))"
```

---

## Contact & Support

For issues or questions:
1. Check the README.md
2. Review HUGGINGFACE_INTEGRATION.md
3. Run the validation notebook
