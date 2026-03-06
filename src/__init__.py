"""
Malayalam Morpho-Hierarchical Tokenizer

A novel tokenization approach for Malayalam that combines:
- Morphological analysis (mlmorph)
- Hierarchical vocabulary structure
- Sandhi splitting
- Unigram/character fallback
"""

from .tokenizer import (
    MorphoHierarchicalTokenizer,
    TokenInfo,
    create_tokenizer
)
from .vocabulary import HierarchicalVocabulary
from .sandhi_splitter import SandhiSplitter

__version__ = '0.1.0'
__all__ = [
    'MorphoHierarchicalTokenizer',
    'TokenInfo',
    'HierarchicalVocabulary',
    'SandhiSplitter',
    'create_tokenizer',
]
