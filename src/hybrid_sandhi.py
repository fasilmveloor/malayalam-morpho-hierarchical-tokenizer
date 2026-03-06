"""
Hybrid Sandhi Splitter for Malayalam Tokenizer

Combines three approaches in order of precision:
1. Dictionary lookup (fastest, most accurate for known words)
2. mlmorph FST (medium speed, high accuracy)
3. Neural Bi-LSTM (slower, generalizes to OOV words)

Usage:
    from hybrid_sandhi import HybridSandhiSplitter
    
    splitter = HybridSandhiSplitter()
    components = splitter.split('പഠിക്കുന്നു')
    # Returns: ['പഠിക്ക്', 'ഉന്നു']
"""

import os
import sys
from typing import List, Tuple, Optional
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn


class SimpleBiLSTM(nn.Module):
    """Simplified Bi-LSTM for sandhi prediction."""
    
    def __init__(self, vocab_size: int, embed_dim: int = 48, hidden_dim: int = 96):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=2, batch_first=True,
            bidirectional=True, dropout=0.2
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, mask=None):
        x = self.embed(x)
        x, _ = self.lstm(x)
        x = self.fc(x).squeeze(-1)
        if mask is not None:
            x = x * mask
        return x


class NeuralSandhiModel:
    """Wrapper for the neural sandhi model."""
    
    def __init__(self, model_path: str = None):
        self.model = None
        self.encoder = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if model_path is None:
            model_path = str(Path(__file__).parent.parent / 'models' / 'best_sandhi_model.pt')
        
        self._load_model(model_path)
    
    def _load_model(self, model_path: str):
        """Load the trained model."""
        if not os.path.exists(model_path):
            print(f"⚠ Neural model not found at {model_path}")
            return
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Build encoder
            self.encoder = type('Encoder', (), {
                'char2idx': checkpoint['char2idx'],
                'vocab_size': len(checkpoint['char2idx'])
            })()
            
            # Build model
            config = checkpoint['config']
            self.model = SimpleBiLSTM(
                vocab_size=config['vocab_size'],
                embed_dim=config['embed_dim'],
                hidden_dim=config['hidden_dim']
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            print(f"✓ Neural sandhi model loaded from {model_path}")
        except Exception as e:
            print(f"⚠ Failed to load neural model: {e}")
    
    def encode(self, text: str) -> List[int]:
        """Encode text to character indices."""
        return [self.encoder.char2idx.get(c, 1) for c in text]
    
    def predict_splits(self, word: str, threshold: float = 0.3) -> List[str]:
        """Predict split points and return components."""
        if self.model is None:
            return [word]
        
        with torch.no_grad():
            chars = self.encode(word)
            max_len = 30
            
            # Pad
            padded = chars[:max_len] + [0] * (max_len - len(chars))
            mask = [1] * min(len(chars), max_len) + [0] * (max_len - min(len(chars), max_len))
            
            # Predict
            char_tensor = torch.tensor([padded], dtype=torch.long).to(self.device)
            mask_tensor = torch.tensor([mask], dtype=torch.float).to(self.device)
            
            probs = self.model(char_tensor, mask_tensor)[0][:len(word)]
            preds = (probs > threshold).int().tolist()
        
        # Split at predicted positions
        components = []
        start = 0
        for i, p in enumerate(preds):
            if p == 1 and i < len(word) - 1:
                components.append(word[start:i+1])
                start = i + 1
        components.append(word[start:])
        
        return components if len(components) > 1 else [word]


class HybridSandhiSplitter:
    """
    Hybrid sandhi splitter combining multiple approaches.
    
    Pipeline:
        1. Check exceptions dictionary (instant, 100% accurate)
        2. Check sandhi dictionary (fast, accurate for known compounds)
        3. Use mlmorph FST (medium, high accuracy)
        4. Use neural model (generalizes to unknown words)
    
    Usage:
        splitter = HybridSandhiSplitter()
        result = splitter.split('പഠിക്കുന്നു')
        # ['പഠിക്ക്', 'ഉന്നു']
    """
    
    def __init__(self, use_neural: bool = True, use_mlmorph: bool = True):
        self.use_neural = use_neural
        self.use_mlmorph = use_mlmorph
        
        # Initialize components
        self.exceptions = self._load_exceptions()
        self.sandhi_dict = self._load_sandhi_dict()
        self.morph_analyzer = None
        self.neural_model = None
        
        # Initialize mlmorph
        if use_mlmorph:
            try:
                from mlmorph import Analyser
                self.morph_analyzer = Analyser()
                print("✓ mlmorph initialized")
            except ImportError:
                print("⚠ mlmorph not available")
                self.use_mlmorph = False
        
        # Initialize neural model
        if use_neural:
            self.neural_model = NeuralSandhiModel()
            if self.neural_model.model is None:
                self.use_neural = False
        
        # Statistics
        self.stats = {
            'exception_hits': 0,
            'dict_hits': 0,
            'mlmorph_hits': 0,
            'neural_hits': 0,
            'fallback': 0
        }
    
    def _load_exceptions(self) -> dict:
        """Load exceptions dictionary."""
        exceptions_path = Path(__file__).parent.parent / 'data' / 'exceptions.json'
        if exceptions_path.exists():
            import json
            with open(exceptions_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(f"✓ Loaded {len(data)} exceptions")
                return data
        return {}
    
    def _load_sandhi_dict(self) -> dict:
        """Load sandhi dictionary."""
        # Common sandhi patterns
        return {
            # Place names
            'തിരുവനന്തപുരം': ['തിരു', 'അനന്തപുരം'],
            'പാലക്കാട്': ['പാല', 'ക്കാട്'],
            'കോഴിക്കോട്': ['കോഴി', 'ക്കോട്'],
            'തൃശ്ശൂർ': ['തൃശ്ശ', 'ഊർ'],
            'കൊല്ലം': ['കൊല്ലം'],
            'ആലപ്പുഴ': ['ആല', 'പ്പുഴ'],
            'കണ്ണൂർ': ['കണ്ണൂർ'],
            'കാസർഗോഡ്': ['കാസർ', 'ഗോഡ്'],
            'ഇടുക്കി': ['ഇടുക്കി'],
            'വയനാട്': ['വയന', 'നാട്'],
            'മലപ്പുറം': ['മല', 'പ്പുറം'],
            'പത്തനംതിട്ട': ['പത്തനം', 'തിട്ട'],
            'കോട്ടയം': ['കോട്ട', 'യം'],
            'എറണാകുളം': ['എറണാകു', 'ളം'],
            
            # Compound words
            'ഭാരതനാട്യം': ['ഭാരത', 'നാട്യം'],
            'രമാവൈദ്യനാഥൻ': ['രമാ', 'വൈദ്യനാഥൻ'],
            'പ്രധാനമന്ത്രി': ['പ്രധാന', 'മന്ത്രി'],
            'രക്തസമ്മർദ്ദം': ['രക്ത', 'സമ്മർദ്ദം'],
            'വിദ്യാഭ്യാസം': ['വിദ്യാ', 'ഭ്യാസം'],
            'സ്വാതന്ത്ര്യം': ['സ്വ', 'തന്ത്ര്യം'],
            'പ്രവൃത്തി': ['പ്ര', 'വൃത്തി'],
        }
    
    def split(self, word: str) -> List[str]:
        """
        Split a word using the hybrid approach.
        
        Args:
            word: Input word
        
        Returns:
            List of morphological components
        """
        # 1. Check exceptions
        if word in self.exceptions:
            self.stats['exception_hits'] += 1
            return self.exceptions[word].get('components', [word])
        
        # 2. Check sandhi dictionary
        if word in self.sandhi_dict:
            self.stats['dict_hits'] += 1
            return self.sandhi_dict[word]
        
        # 3. Try mlmorph
        if self.use_mlmorph and self.morph_analyzer:
            mlmorph_result = self._mlmorph_split(word)
            if mlmorph_result and len(mlmorph_result) > 1:
                self.stats['mlmorph_hits'] += 1
                return mlmorph_result
        
        # 4. Use neural model
        if self.use_neural and self.neural_model:
            neural_result = self.neural_model.predict_splits(word)
            if len(neural_result) > 1:
                self.stats['neural_hits'] += 1
                return neural_result
        
        # Fallback: return as single component
        self.stats['fallback'] += 1
        return [word]
    
    def _mlmorph_split(self, word: str) -> Optional[List[str]]:
        """Use mlmorph for morphological splitting."""
        try:
            analysis = self.morph_analyzer.analyse(word)
            if analysis and len(analysis) > 0:
                # Parse analysis
                analysis_str = analysis[0][0] if isinstance(analysis[0], tuple) else str(analysis[0])
                
                # Extract root
                import re
                root_match = re.match(r'^([^\s<]+)', analysis_str)
                if root_match:
                    root = root_match.group(1)
                    
                    if root != word:
                        # Find suffix
                        import unicodedata
                        # Normalize for comparison
                        root_base = root.rstrip('്')
                        word_start = word[:len(root_base)] if word.startswith(root_base) else word[:len(root)]
                        
                        if len(root_base) < len(word):
                            suffix = word[len(root_base):]
                            if suffix:
                                return [root, suffix]
        except Exception:
            pass
        
        return None
    
    def get_stats(self) -> dict:
        """Get splitting statistics."""
        total = sum(self.stats.values())
        if total == 0:
            return self.stats
        
        return {
            **self.stats,
            'total': total,
            'exception_rate': self.stats['exception_hits'] / total,
            'dict_rate': self.stats['dict_hits'] / total,
            'mlmorph_rate': self.stats['mlmorph_hits'] / total,
            'neural_rate': self.stats['neural_hits'] / total,
        }
    
    def reset_stats(self):
        """Reset statistics."""
        for key in self.stats:
            self.stats[key] = 0


def demo():
    """Demo the hybrid sandhi splitter."""
    print("\n" + "="*60)
    print("Hybrid Sandhi Splitter Demo")
    print("="*60)
    
    splitter = HybridSandhiSplitter()
    
    test_words = [
        # Place names
        'തിരുവനന്തപുരം',
        'പാലക്കാട്',
        'കേരളത്തിൽ',
        
        # Verb forms
        'പഠിക്കുന്നു',
        'പഠിച്ചു',
        'പഠിക്കണം',
        
        # Noun forms
        'വിദ്യാലയത്തിൽ',
        'വീട്ടിൽ',
        'അധ്യാപികയുടെ',
        
        # Compound words
        'ഭാരതനാട്യം',
        'രമാവൈദ്യനാഥൻ',
        'പ്രധാനമന്ത്രി',
        
        # Unknown words
        'പുതിയവാക്ക്',
        'മലയാളപ്പഠനം',
    ]
    
    print()
    for word in test_words:
        components = splitter.split(word)
        print(f"  {word}")
        print(f"    → {' + '.join(components)}")
        print()
    
    # Show stats
    print("\nSplitting Statistics:")
    stats = splitter.get_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2%}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    demo()
