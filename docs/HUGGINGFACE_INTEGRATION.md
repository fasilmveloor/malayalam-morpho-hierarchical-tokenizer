# HuggingFace Integration Plan for Malayalam Morpho-Hierarchical Tokenizer

## Overview

This document outlines the strategy for integrating the Malayalam Morpho-Hierarchical Tokenizer into the HuggingFace ecosystem.

## Integration Options

### Option 1: Custom PreTrainedTokenizer (Recommended)

Create a custom tokenizer class that inherits from `PreTrainedTokenizer`:

```python
from transformers import PreTrainedTokenizer
from typing import List, Optional
import os
import json

class MorphoHierarchicalTokenizerFast(PreTrainedTokenizer):
    """
    HuggingFace-compatible Morpho-Hierarchical Tokenizer for Malayalam.
    
    Novel Features:
    - Slot System: Hierarchical token IDs encoding grammatical structure
    - Phoneme-Aware Encoding: 10-dimensional feature vectors
    - Sandhi Reconstruction: ം → ത്ത് transformation
    - Hybrid Pipeline: FST (mlmorph) + Neural Bi-LSTM for OOV
    """
    
    vocab_files_names = {
        "vocab_file": "vocab.json",
        "config_file": "tokenizer_config.json"
    }
    
    def __init__(
        self,
        vocab_file: str = None,
        use_mlmorph: bool = True,
        use_neural: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Load vocabulary
        if vocab_file and os.path.exists(vocab_file):
            with open(vocab_file, 'r', encoding='utf-8') as f:
                vocab_data = json.load(f)
                self.vocab = vocab_data.get('token_to_id', {})
                self.id_to_token = {v: k for k, v in self.vocab.items()}
        
        # Initialize morphological analyzer
        self.use_mlmorph = use_mlmorph
        if use_mlmorph:
            try:
                from mlmorph import Analyser
                self.morph_analyzer = Analyser()
            except ImportError:
                self.morph_analyzer = None
        
        # Slots for hierarchical vocabulary
        self.SLOTS = {
            'special': (0, 999),
            'root': (1000, 1999),
            'tense': (2000, 2999),
            'case': (3000, 3999),
            'function': (4000, 4999),
            'infix': (5000, 5999),
        }
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into morphemes."""
        import re
        
        # Normalize text
        text = self._normalize(text)
        
        # Extract Malayalam words
        words = re.findall(r'[\u0D00-\u0D7F]+', text)
        
        morphemes = []
        for word in words:
            morphemes.extend(self._get_morphemes(word))
        
        return morphemes
    
    def _normalize(self, text: str) -> str:
        """NFKC normalization."""
        import unicodedata
        normalized = unicodedata.normalize('NFKC', text)
        normalized = normalized.replace('\u200d', '')  # ZWJ
        normalized = normalized.replace('\u200c', '')  # ZWNJ
        return normalized
    
    def _get_morphemes(self, word: str) -> List[str]:
        """Get morphological decomposition."""
        if self.morph_analyzer:
            try:
                analysis = self.morph_analyzer.analyse(word)
                if analysis:
                    # Parse and return morphemes
                    return self._parse_analysis(analysis[0], word)
            except:
                pass
        
        # Fallback to suffix-based splitting
        return self._fallback_split(word)
    
    def _parse_analysis(self, analysis, original: str) -> List[str]:
        """Parse mlmorph analysis output."""
        import re
        analysis_str = analysis[0] if isinstance(analysis, tuple) else str(analysis)
        root_match = re.match(r'^([^\s<]+)', analysis_str)
        
        if root_match:
            root = root_match.group(1)
            if root != original:
                # Find suffix
                base = root.rstrip('്')
                if original.startswith(base):
                    suffix = original[len(base):]
                    if suffix:
                        return [root, suffix]
                return [root]
        
        return [original]
    
    def _fallback_split(self, word: str) -> List[str]:
        """Fallback suffix-based splitting."""
        suffixes = ['ുന്നു', 'ുക', 'ിൽ', 'ിന്റെ', 'ിന്', 'ത്തിൽ']
        
        for suffix in suffixes:
            if word.endswith(suffix):
                stem = word[:-len(suffix)]
                if len(stem) >= 2:
                    return [stem, suffix]
        
        return [word]
    
    def _convert_token_to_id(self, token: str) -> int:
        """Convert token to vocabulary ID."""
        return self.vocab.get(token, self.unk_token_id)
    
    def _convert_id_to_token(self, index: int) -> str:
        """Convert vocabulary ID to token."""
        return self.id_to_token.get(index, self.unk_token)
    
    def classify_token(self, token_id: int) -> str:
        """Classify token by hierarchical slot."""
        for category, (start, end) in self.SLOTS.items():
            if start <= token_id < end:
                return category
        return 'unknown'
    
    @property
    def vocab_size(self) -> int:
        return len(self.vocab)
    
    def save_vocabulary(self, save_directory: str, filename_prefix: str = None) -> tuple:
        """Save vocabulary to file."""
        vocab_path = os.path.join(save_directory, (filename_prefix or '') + 'vocab.json')
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump({
                'token_to_id': self.vocab,
                'slots': self.SLOTS
            }, f, ensure_ascii=False, indent=2)
        return (vocab_path,)
```

### Option 2: Tokenizers Library Integration

Create a custom tokenizer using the HuggingFace `tokenizers` library:

```python
from tokenizers import Tokenizer, models, pre_tokenizers, decoders
from tokenizers.trainers import Trainer

class MorphoTokenizer:
    def __init__(self):
        self.tokenizer = Tokenizer(models.Unigram())
        self.tokenizer.pre_tokenizer = pre_tokenizers.CustomPreTokenizer(
            MalayalamMorphemePreTokenizer()
        )

class MalayalamMorphemePreTokenizer:
    """Custom pre-tokenizer that applies morphological analysis."""
    
    def split(self, normalized):
        # Apply mlmorph analysis
        # Return morpheme-level splits
        pass
```

### Option 3: SentencePiece Integration

Export vocabulary to SentencePiece format for use with existing models:

```python
import sentencepiece as spm

# Train SentencePiece model with morphological constraints
spm.SentencePieceTrainer.train(
    input='corpus.txt',
    model_prefix='malayalam_morpho',
    vocab_size=8000,
    user_defined_symbols=morpheme_list,  # Force morpheme boundaries
    character_coverage=1.0
)
```

## Recommended Approach

**Option 1 (Custom PreTrainedTokenizer)** is recommended because:

1. **Full Control**: Complete control over tokenization logic
2. **mlmorph Integration**: Direct integration with FST-based analyzer
3. **Neural Fallback**: Can load Bi-LSTM models for OOV handling
4. **Slot System**: Preserves hierarchical token ID structure
5. **Transformers Compatible**: Works with all HuggingFace models

## Implementation Steps

### Phase 1: Core Integration
1. Create `MorphoHierarchicalTokenizerFast` class
2. Implement vocabulary loading/saving
3. Add morphological analysis integration
4. Test with HuggingFace pipelines

### Phase 2: Model Support
1. Create tokenizer registration
2. Add to HuggingFace Hub
3. Test with IndicBERT, mBERT, XLM-R
4. Benchmark performance

### Phase 3: Documentation
1. Create model card
2. Write usage examples
3. Add API documentation
4. Publish to HuggingFace Hub

## File Structure

```
malayalam-tokenizer/
├── src/
│   ├── __init__.py
│   ├── tokenizer_hf.py          # HuggingFace-compatible tokenizer
│   ├── tokenizer.py             # Original implementation
│   ├── vocabulary.py
│   └── sandhi.py
├── models/
│   ├── vocab.json
│   ├── tokenizer_config.json
│   └── best_sandhi_model.pt
├── tests/
│   └── test_hf_integration.py
└── setup.py
```

## HuggingFace Hub Publishing

```bash
# Login to HuggingFace
huggingface-cli login

# Create repository
huggingface-cli repo create malayalam-morpho-tokenizer --organization your-org

# Upload files
huggingface-cli upload malayalam-morpho-tokenizer ./models/
```

## Usage Example

```python
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("your-org/malayalam-morpho-tokenizer")

# Tokenize
text = "പഠിക്കുന്നു വിദ്യാലയത്തിൽ"
tokens = tokenizer.tokenize(text)
# Output: ['പഠിക്ക്', 'ുന്നു', 'വിദ്യാലയത്ത്', 'ിൽ']

# Get token IDs
ids = tokenizer.encode(text)
# Output: [2, 1001, 2001, 1002, 3001, 3]

# Classify tokens
for token_id in ids:
    category = tokenizer.classify_token(token_id)
    print(f"{token_id}: {category}")
```

## Performance Considerations

1. **Caching**: Implement LRU cache for frequent morphological analyses
2. **Batch Processing**: Support batch tokenization for efficiency
3. **Model Size**: Keep neural models small (< 5MB) for fast loading
4. **Fallback Speed**: Ensure fallback is fast when mlmorph unavailable

## Testing Strategy

1. **Unit Tests**: Test individual components
2. **Integration Tests**: Test with HuggingFace pipelines
3. **Regression Tests**: Ensure backward compatibility
4. **Performance Tests**: Benchmark against BPE/Unigram

## References

- HuggingFace Tokenizers: https://huggingface.co/docs/tokenizers/
- PreTrainedTokenizer: https://huggingface.co/docs/transformers/main_classes/tokenizer
- mlmorph: https://github.com/libindic/mlmorph
