# Malayalam Corpus Collection Pipeline

Complete pipeline for collecting and validating Malayalam text corpus.

## 📁 Files

```
data_collection/
├── 01_download_corpus.py      # Download Wikipedia + SMC
├── 02_extract_wikipedia.py    # Extract text from wiki dump
├── 03_clean_corpus.py         # Clean all corpora + extract words
├── 04_combine_corpora.py      # Remove transliterations, prepare for validation
├── 05_spell_check.py          # Validate with mlmorph
├── 06_pattern_validate.py     # Pattern validation
├── 07_combine_final.py        # Final output
├── check_artifacts.sh         # Check wiki markers
└── README.md
```

## 📊 Data Flow

```
Wikipedia dump (187MB)
        ↓
    02_extract
        ↓
wiki_text.txt (35M words)
        ↓
    ┌───┴───┐
    │  03   │ ← Also processes: smc_corpus.txt + smc_wordlist.txt
    │ clean │
    └───┬───┘
        ↓
word_list.txt (~1.9M unique)
        ↓
    ┌───┴───┐
    │  04   │ ← Remove transliterations
    │ prep  │
    └───┬───┘
        ↓
words_for_validation.txt
        ↓
    ┌───┴───┐
    │  05   │ ← mlmorph spell check
    │ spell │
    └───┬───┘
        ↓
┌───────────────┬──────────────────┐
│  words_valid  │ words_needs_review│
│   (~53%)      │     (~47%)        │
└───────┬───────┴────────┬─────────┘
        │                │
        │         ┌──────┴──────┐
        │         │     06      │ ← Pattern validation
        │         │  pattern    │
        │         └──────┬──────┘
        │                │
        │         words_pattern_valid
        │                │
        └────────┬───────┘
                 ↓
           ┌─────┴─────┐
           │     07    │
           │  combine  │
           └─────┬─────┘
                 ↓
        words_final.txt (~1.8M)
```

## 🚀 Quick Start

```bash
# Step 1: Download all corpora (Wikipedia + SMC)
python 01_download_corpus.py

# Step 2: Extract text from Wikipedia dump
python 02_extract_wikipedia.py

# Step 3: Clean and extract words from all sources
python 03_clean_corpus.py

# Step 4: Prepare for validation (remove transliterations)
python 04_combine_corpora.py

# Step 5: Spell check with mlmorph
python 05_spell_check.py

# Step 6: Pattern validation for review words
python 06_pattern_validate.py

# Step 7: Combine all valid words
python 07_combine_final.py
```

## 📊 Expected Output

| Step | Input | Output |
|------|-------|--------|
| 01 | - | mlwiki dump + SMC files |
| 02 | mlwiki dump | wiki_text.txt (~35M words) |
| 03 | wiki + smc_corpus + smc_wordlist | word_list.txt (~1.9M) |
| 04 | word_list.txt | words_for_validation.txt |
| 05 | words_for_validation | words_valid (~53%) + review (~47%) |
| 06 | words_needs_review | words_pattern_valid |
| **07** | **all valid** | **words_final.txt (~1.8M)** |

## 📦 Requirements

```bash
pip install mlmorph
```

## 📝 Data Sources

| Source | Type | URL |
|--------|------|-----|
| Wikipedia | Dump | `dumps.wikimedia.org/mlwiki/latest/` |
| SMC Wordlist | Word list | `gitlab.com/smc/corpus/-/raw/master/wordlist.txt` |
| SMC Corpus | Text corpus | `gitlab.com/smc/corpus/-/raw/master/corpus.txt` |

## 🎯 Next Step

Use `final_corpus/words_final.txt` to generate training labels:

```bash
python generate_mlmorph_labels.py --input final_corpus/words_final.txt
```
