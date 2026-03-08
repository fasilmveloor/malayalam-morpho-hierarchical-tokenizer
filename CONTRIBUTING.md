# Contributing to Malayalam Morpho-Hierarchical Tokenizer

First off, thank you for considering contributing to this project! 🎉

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)

---

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/0/code_of conduct/). By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

---

## Getting Started

### Prerequisites

- Python 3.8+
- Git
- Basic understanding of Malayalam language and morphology
- Familiarity with NLP concepts

### Fork and Clone

```bash
# Fork the repository on GitHub
# Then clone your fork
git clone https://github.com/fasilmveloor/ml-morpho-hierarchical-tokenizer.git
cd ml-morpho-hierarchical-tokenizer

# Add upstream remote
git remote add upstream https://github.com/fasilmveloor/ml-morpho-hierarchical-tokenizer.git
```

---

## Development Setup

### Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

### Install Dependencies

```bash
# Install production dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt

# Install in editable mode
pip install -e .
```

### Install Optional Dependencies

```bash
# For mlmorph support
pip install mlmorph

# For development tools
pip install pytest flake8 black isort
```

---

## How to Contribute

### Types of Contributions

1. **Bug Reports** 🐛
   - Report issues via GitHub Issues
   - Include reproduction steps
   - Provide sample input/output

2. **Feature Requests** 💡
   - Suggest new features via GitHub Issues
   - Explain use case and expected behavior

3. **Code Contributions** 💻
   - Fix bugs
   - Implement new features
   - Improve performance

4. **Documentation** 📚
   - Improve README
   - Add code comments
   - Write tutorials

5. **Linguistic Contributions** 📖
   - Add sandhi rules
   - Verify morphological splits
   - Expand vocabulary

### Areas Needing Contribution

- [ ] Expand training data (500+ examples goal)
- [ ] Add CRF layer for BIO optimization
- [ ] Linguistic verification by Malayalam scholars
- [ ] Benchmark against more tokenizers
- [ ] Add support for other Dravidian languages
- [ ] Improve documentation
- [ ] Add more test cases

---

## Pull Request Process

### Before Submitting

1. **Update from upstream**
   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow coding standards
   - Add tests
   - Update documentation

4. **Run tests**
   ```bash
   pytest tests/ -v
   flake8 src/
   ```

5. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add new sandhi rule for X"
   ```

   Follow [Conventional Commits](https://www.conventionalcommits.org/):
   - `feat:` - New feature
   - `fix:` - Bug fix
   - `docs:` - Documentation changes
   - `test:` - Adding tests
   - `refactor:` - Code refactoring
   - `chore:` - Maintenance tasks

### Submit PR

1. Push to your fork
   ```bash
   git push origin feature/your-feature-name
   ```

2. Create Pull Request on GitHub

3. Fill in the PR template:
   - Description of changes
   - Related issue number
   - Type of change
   - Testing performed

### PR Review Process

- PRs require at least one approval
- CI tests must pass
- Code coverage should not decrease
- Documentation must be updated

---

## Coding Standards

### Python Style Guide

Follow [PEP 8](https://pep8.org/) with these specifics:

```python
# Use 4 spaces for indentation
def tokenize(text: str) -> List[int]:
    """Tokenize text into morpheme IDs."""
    pass

# Maximum line length: 100 characters
# Use type hints for function signatures
def get_morphemes(word: str) -> List[str]:
    """Get morphological decomposition of a word."""
    pass

# Use descriptive variable names
morpheme_boundaries = find_boundaries(word)  # Good
mb = find_b(w)  # Bad
```

### Code Formatting

We use `black` and `isort` for formatting:

```bash
# Format code
black src/

# Sort imports
isort src/

# Or both
black src/ && isort src/
```

### Linting

```bash
# Run flake8
flake8 src/ --max-line-length=100

# Run with specific checks
flake8 src/ --select=E,W,F
```

### Type Hints

```python
from typing import List, Dict, Optional, Tuple

class MorphoTokenizer:
    def __init__(self, vocab_size: int = 8000) -> None:
        self.vocab: Dict[str, int] = {}
        self.morphemes: List[str] = []
    
    def tokenize(self, text: str) -> List[int]:
        """Return list of token IDs."""
        pass
    
    def get_morpheme(self, word: str) -> Optional[List[str]]:
        """Return morphemes or None if not found."""
        pass
```

---

## Testing Guidelines

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_tokenizer.py

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run verbose
pytest tests/ -v -s
```

### Writing Tests

```python
import pytest
from src.tokenizer import MorphoHierarchicalTokenizer

class TestTokenizer:
    @pytest.fixture
    def tokenizer(self):
        return MorphoHierarchicalTokenizer(use_mlmorph=False)
    
    def test_basic_tokenization(self, tokenizer):
        """Test basic word tokenization."""
        result = tokenizer.tokenize("പഠിക്കുന്നു")
        assert len(result) > 0
    
    def test_sandhi_splitting(self, tokenizer):
        """Test compound word splitting."""
        morphemes = tokenizer.get_morphemes("തിരുവനന്തപുരം")
        assert len(morphemes) >= 2
    
    @pytest.mark.parametrize("word,expected", [
        ("പഠിക്കുന്നു", "tense"),
        ("വീട്ടിൽ", "case"),
    ])
    def test_token_classification(self, tokenizer, word, expected):
        """Test token category classification."""
        tokens = tokenizer.tokenize_detailed(word)
        assert tokens[-1].token_type == expected
```

### Test Coverage

- Maintain >80% code coverage
- Cover edge cases and error conditions
- Include parametrized tests for multiple cases

---

## Documentation

### Code Documentation

```python
class MorphoHierarchicalTokenizer:
    """
    A morphological tokenizer for Malayalam.
    
    This tokenizer combines Finite State Transducers with Neural Networks
    to provide linguistically meaningful tokenization.
    
    Attributes:
        vocab_size: Maximum vocabulary size.
        use_mlmorph: Whether to use mlmorph for analysis.
        vocab: Hierarchical vocabulary mapping.
    
    Example:
        >>> tokenizer = MorphoHierarchicalTokenizer()
        >>> tokens = tokenizer.tokenize("പഠിക്കുന്നു")
        >>> print(tokens)
        [2, 1001, 2001, 3]
    
    Note:
        Requires mlmorph package for optimal performance.
        Falls back to rule-based splitting if unavailable.
    """
```

### Updating Documentation

1. Update relevant `.md` files
2. Update docstrings in code
3. Update Jupyter notebooks if needed
4. Add examples for new features

---

## Linguistic Contributions

### Adding Sandhi Rules

```python
# In src/sandhi.py

# Add new sandhi transformation
SANDHI_RULES = {
    # Existing rules...
    
    # Add new rule
    'ം + വ': ('മ്പ്', 'v→p transformation'),
}
```

### Adding Vocabulary

```python
# In data/exceptions.json

{
    "high_frequency_words": {
        "നിയമസഭ": ["നിയമ", "സഭ"],
        # Add new words...
    }
}
```

### Training Data Format

```json
[
    {
        "word": "പഠിക്കുന്നു",
        "split_positions": [4],
        "morphemes": ["പഠിക്ക്", "ുന്നു"],
        "categories": ["root", "tense"],
        "verified": true
    }
]
```

---

## Release Process

1. Update version in `setup.py` and `__init__.py`
2. Update `CHANGELOG.md`
3. Run full test suite
4. Create release PR
5. Tag release after merge

---

## Getting Help

- **GitHub Discussions**: For questions and ideas
- **GitHub Issues**: For bug reports and feature requests
- **Documentation**: Check README.md and ARCHITECTURE.md

---

## Recognition

Contributors will be listed in:
- README.md acknowledgments
- Release notes
- Project documentation

Thank you for contributing! 🙏
