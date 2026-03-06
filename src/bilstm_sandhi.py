"""
Bi-LSTM Neural Sandhi Splitter for Malayalam

Architecture:
    Input: Character sequence (പഠിക്കുന്നു)
        ↓
    Character Embedding Layer
        ↓
    Bidirectional LSTM (captures left & right context)
        ↓
    Dense Layer with Sigmoid
        ↓
    Output: Split point predictions [0,0,1,0,0,1,0,0,0]
        ↓
    Split: ['പഠിക്ക്', 'ന്നു'] → ['പഠിക്ക്', 'ഉന്നു']

Training Data Generation:
    1. Extract words from SMC corpus
    2. Use mlmorph for ground truth morphological splits
    3. Create character-level labels (0=no split, 1=split after this char)

Key Features:
    - Character-level: Handles OOV words naturally
    - Bidirectional: Captures sandhi context from both sides
    - Phoneme-aware: Special handling for virama, chillu, anusvara
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import re
import json
import pickle
from typing import List, Tuple, Dict, Optional
from collections import Counter
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Check for GPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔧 Using device: {DEVICE}")


class MalayalamCharEncoder:
    """
    Character-level encoder for Malayalam text.
    Handles all Unicode characters in the Malayalam block (U+0D00-U+0D7F).
    """
    
    # Malayalam Unicode range
    MALAYALAM_START = 0x0D00
    MALAYALAM_END = 0x0D7F
    
    # Special tokens
    PAD_TOKEN = '<PAD>'
    UNK_TOKEN = '<UNK>'
    SOS_TOKEN = '<SOS>'  # Start of sequence
    EOS_TOKEN = '<EOS>'  # End of sequence
    
    # Phoneme categories for feature engineering
    VOWELS = set('അആഇഈഉഊഋൠഎഏഐഒഓഔഃ')
    VOWEL_SIGNS = set('ാിീുൂൃെേൈൊോൌൗ')
    CONSONANTS = set('കഖഗഘങചഛജഝഞടഠഡഢണതഥദധനപഫബഭമയരലവശഷസഹളഴറ')
    CHILLU = set('ൽർൻൺൿ')  # Chillu letters
    VIRAMA = '്'
    ANUSVARA = 'ം'
    
    def __init__(self):
        self.char_to_idx: Dict[str, int] = {}
        self.idx_to_char: Dict[int, str] = {}
        self.vocab_size = 0
        self._build_vocab()
    
    def _build_vocab(self):
        """Build character vocabulary from Malayalam Unicode block."""
        # Special tokens
        special_tokens = [self.PAD_TOKEN, self.UNK_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN]
        for i, token in enumerate(special_tokens):
            self.char_to_idx[token] = i
            self.idx_to_char[i] = token
        
        # Malayalam characters
        idx = len(special_tokens)
        for codepoint in range(self.MALAYALAM_START, self.MALAYALAM_END + 1):
            char = chr(codepoint)
            self.char_to_idx[char] = idx
            self.idx_to_char[idx] = char
            idx += 1
        
        # Common ASCII characters that might appear
        for char in ' 0123456789.,!?;:\'"-()[]{}':
            if char not in self.char_to_idx:
                self.char_to_idx[char] = idx
                self.idx_to_char[idx] = char
                idx += 1
        
        self.vocab_size = len(self.char_to_idx)
        print(f"✓ Vocabulary built: {self.vocab_size} characters")
    
    def encode(self, text: str) -> List[int]:
        """Convert text to list of indices."""
        return [self.char_to_idx.get(c, self.char_to_idx[self.UNK_TOKEN]) for c in text]
    
    def decode(self, indices: List[int]) -> str:
        """Convert indices back to text."""
        return ''.join(self.idx_to_char.get(i, self.UNK_TOKEN) for i in indices)
    
    def get_phoneme_features(self, char: str) -> List[float]:
        """Get phoneme features for a character."""
        features = [
            float(char in self.VOWELS),
            float(char in self.VOWEL_SIGNS),
            float(char in self.CONSONANTS),
            float(char in self.CHILLU),
            float(char == self.VIRAMA),
            float(char == self.ANUSVARA),
        ]
        return features


class SandhiDataset(Dataset):
    """Dataset for sandhi split point prediction."""
    
    def __init__(
        self,
        words: List[str],
        split_labels: List[List[int]],
        encoder: MalayalamCharEncoder,
        max_len: int = 50
    ):
        self.words = words
        self.split_labels = split_labels
        self.encoder = encoder
        self.max_len = max_len
    
    def __len__(self):
        return len(self.words)
    
    def __getitem__(self, idx):
        word = self.words[idx]
        labels = self.split_labels[idx]
        
        # Encode characters
        char_ids = self.encoder.encode(word)
        
        # Pad/truncate to max_len
        seq_len = min(len(char_ids), self.max_len)
        padded_chars = char_ids[:self.max_len] + [0] * (self.max_len - len(char_ids))
        padded_labels = labels[:self.max_len] + [0] * (self.max_len - len(labels))
        
        # Create mask for actual characters
        mask = [1] * seq_len + [0] * (self.max_len - seq_len)
        
        return {
            'char_ids': torch.tensor(padded_chars, dtype=torch.long),
            'labels': torch.tensor(padded_labels, dtype=torch.float),
            'mask': torch.tensor(mask, dtype=torch.float),
            'seq_len': seq_len,
            'word': word
        }


class BiLSTMSandhiSplitter(nn.Module):
    """
    Bi-LSTM model for sandhi split point prediction.
    
    Architecture:
        Input: Character indices [batch, seq_len]
        ↓
        Embedding Layer [batch, seq_len, embed_dim]
        ↓
        Bidirectional LSTM [batch, seq_len, hidden_dim * 2]
        ↓
        Dropout Layer
        ↓
        Linear Layer [batch, seq_len, 1]
        ↓
        Sigmoid → Split probabilities [batch, seq_len]
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        # Character embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, char_ids, mask=None):
        """
        Forward pass.
        
        Args:
            char_ids: [batch, seq_len] character indices
            mask: [batch, seq_len] mask for valid positions
        
        Returns:
            split_probs: [batch, seq_len] split probabilities
        """
        # Embedding
        embeds = self.embedding(char_ids)  # [batch, seq_len, embed_dim]
        
        # Bi-LSTM
        lstm_out, _ = self.lstm(embeds)  # [batch, seq_len, hidden_dim * 2]
        
        # Output layers
        out = self.dropout(lstm_out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out).squeeze(-1)  # [batch, seq_len]
        out = self.sigmoid(out)
        
        # Apply mask
        if mask is not None:
            out = out * mask
        
        return out
    
    def predict_splits(self, char_ids: torch.Tensor, mask: torch.Tensor, threshold: float = 0.5) -> List[List[int]]:
        """Predict split points for a batch of words."""
        self.eval()
        with torch.no_grad():
            probs = self.forward(char_ids, mask)
            predictions = (probs > threshold).int()
        
        # Convert to list of split positions
        batch_splits = []
        for i in range(predictions.shape[0]):
            seq_len = int(mask[i].sum().item())
            splits = predictions[i, :seq_len].cpu().tolist()
            batch_splits.append(splits)
        
        return batch_splits


class SandhiTrainer:
    """Trainer for the Bi-LSTM Sandhi Splitter."""
    
    def __init__(
        self,
        model: BiLSTMSandhiSplitter,
        encoder: MalayalamCharEncoder,
        learning_rate: float = 0.001
    ):
        self.model = model.to(DEVICE)
        self.encoder = encoder
        self.criterion = nn.BCELoss(reduction='none')
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': []
        }
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_samples = 0
        
        for batch in dataloader:
            char_ids = batch['char_ids'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            mask = batch['mask'].to(DEVICE)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(char_ids, mask)
            
            # Compute loss (only for non-padded positions)
            loss = self.criterion(outputs, labels)
            loss = (loss * mask).sum() / mask.sum()
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item() * char_ids.size(0)
            total_samples += char_ids.size(0)
        
        return total_loss / total_samples
    
    def evaluate(self, dataloader: DataLoader, threshold: float = 0.5) -> Tuple[float, float, float]:
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0
        total_samples = 0
        correct = 0
        total_positions = 0
        
        with torch.no_grad():
            for batch in dataloader:
                char_ids = batch['char_ids'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)
                mask = batch['mask'].to(DEVICE)
                
                # Forward pass
                outputs = self.model(char_ids, mask)
                
                # Loss
                loss = self.criterion(outputs, labels)
                loss = (loss * mask).sum() / mask.sum()
                total_loss += loss.item() * char_ids.size(0)
                total_samples += char_ids.size(0)
                
                # Accuracy
                predictions = (outputs > threshold).float()
                correct += ((predictions == labels) * mask).sum().item()
                total_positions += mask.sum().item()
        
        avg_loss = total_loss / total_samples
        accuracy = correct / total_positions if total_positions > 0 else 0
        
        return avg_loss, accuracy, 0.0  # precision placeholder


class GroundTruthGenerator:
    """
    Generate ground truth split labels using mlmorph.
    This creates training data for the neural network.
    """
    
    def __init__(self):
        self.morph_analyzer = None
        self._init_mlmorph()
        
        # Common suffix patterns for labeling
        self.suffix_patterns = [
            'ുന്നു', 'ുന്ന', 'ുക', 'ും', 'ിച്ചു', 'ിച്ച', 'ിയ', 'ാൻ', 'ണം',
            'ിൽ', 'ിന്', 'ിന്റെ', 'ിലെ', 'ിക്ക്', 'ത്തിൽ', 'ത്തിന്റെ', 'ത്തിലെ',
            'ുകയാണ്', 'ുകയുണ്ടായി', 'പ്പെട്ടു', 'പ്പെട്ട'
        ]
    
    def _init_mlmorph(self):
        """Initialize mlmorph analyzer."""
        try:
            from mlmorph import Analyser
            self.morph_analyzer = Analyser()
            print("✓ mlmorph initialized for ground truth generation")
        except ImportError:
            print("⚠ mlmorph not available, using rule-based generation")
    
    def generate_split_labels(self, word: str) -> Tuple[List[str], List[int]]:
        """
        Generate split labels for a word.
        
        Returns:
            components: List of morpheme components
            labels: Binary labels for each character position (1 = split after this char)
        """
        components = [word]  # Default: no split
        labels = [0] * len(word)
        
        # Try mlmorph analysis
        if self.morph_analyzer:
            analysis = self.morph_analyzer.analyse(word)
            if analysis and len(analysis) > 0:
                analysis_str = analysis[0][0] if isinstance(analysis[0], tuple) else str(analysis[0])
                
                # Parse analysis to find root and suffixes
                # Format: <root><suffix1><suffix2>...
                parts = self._parse_analysis(word, analysis_str)
                if len(parts) > 1:
                    components = parts
                    labels = self._create_labels(word, parts)
                    return components, labels
        
        # Fallback: use suffix patterns
        for suffix in self.suffix_patterns:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                root = word[:-len(suffix)]
                if len(root) >= 2:
                    # Apply sandhi transformation
                    if root.endswith('ം'):
                        root = root[:-1] + 'ത്ത്'
                    elif not root.endswith('്'):
                        root = root + '്'
                    
                    components = [root, suffix]
                    labels = self._create_labels(word, [word[:-len(suffix)], suffix])
                    break
        
        return components, labels
    
    def _parse_analysis(self, word: str, analysis: str) -> List[str]:
        """Parse mlmorph analysis to extract components."""
        # Extract root
        root_match = re.match(r'^([^\s<]+)', analysis)
        if not root_match:
            return [word]
        
        root = root_match.group(1)
        
        # Check if root is different from word
        if root == word:
            return [word]
        
        # Extract suffixes from analysis
        suffixes = re.findall(r'<([^>]+)>', analysis)
        
        # Filter for actual morphological tags
        morph_tags = []
        for tag in suffixes:
            if tag in ['n', 'v', 'adj', 'adv']:  # POS tags
                continue
            if tag.startswith(('v_', 'n_', 'adj_')):  # Inflection tags
                morph_tags.append(tag)
        
        if not morph_tags:
            # Try to match root with word
            if word.startswith(root[:-1] if root.endswith('്') else root):
                suffix = word[len(root)-1:] if root.endswith('്') else word[len(root):]
                if suffix:
                    return [root, suffix]
        
        return [word]
    
    def _create_labels(self, word: str, components: List[str]) -> List[int]:
        """Create binary labels from components."""
        labels = [0] * len(word)
        
        pos = 0
        for i, comp in enumerate(components[:-1]):
            pos += len(comp)
            if pos < len(word):
                labels[pos - 1] = 1  # Split after this position
        
        return labels
    
    def generate_training_data(
        self,
        corpus_path: str,
        output_path: str,
        max_words: int = 10000
    ) -> Tuple[List[str], List[List[int]]]:
        """
        Generate training data from corpus.
        
        Args:
            corpus_path: Path to corpus file
            output_path: Path to save generated data
            max_words: Maximum number of words to process
        
        Returns:
            words: List of words
            labels: List of label sequences
        """
        print(f"\n📊 Generating training data from {corpus_path}")
        
        # Read corpus
        with open(corpus_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Extract Malayalam words
        words = list(set(re.findall(r'[\u0D00-\u0D7F]+', text)))
        words = [w for w in words if len(w) >= 3][:max_words]
        
        print(f"   Found {len(words)} unique words")
        
        # Generate labels
        training_words = []
        training_labels = []
        multi_component_count = 0
        
        for word in words:
            components, labels = self.generate_split_labels(word)
            if len(components) > 1:
                multi_component_count += 1
            training_words.append(word)
            training_labels.append(labels)
        
        print(f"   Multi-component words: {multi_component_count} ({100*multi_component_count/len(words):.1f}%)")
        
        # Save training data
        data = {
            'words': training_words,
            'labels': training_labels,
            'stats': {
                'total_words': len(training_words),
                'multi_component': multi_component_count
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"   ✓ Saved to {output_path}")
        
        return training_words, training_labels


def split_word_with_model(
    model: BiLSTMSandhiSplitter,
    encoder: MalayalamCharEncoder,
    word: str,
    threshold: float = 0.5
) -> List[str]:
    """
    Split a word using the trained model.
    
    Args:
        model: Trained Bi-LSTM model
        encoder: Character encoder
        word: Word to split
        threshold: Split probability threshold
    
    Returns:
        List of split components
    """
    model.eval()
    
    # Encode
    char_ids = encoder.encode(word)
    char_tensor = torch.tensor([char_ids], dtype=torch.long).to(DEVICE)
    mask = torch.tensor([[1] * len(char_ids)], dtype=torch.float).to(DEVICE)
    
    # Predict
    with torch.no_grad():
        probs = model(char_tensor, mask)
        predictions = (probs[0, :len(word)] > threshold).int().tolist()
    
    # Split at predicted positions
    components = []
    start = 0
    for i, pred in enumerate(predictions):
        if pred == 1 and i < len(word) - 1:
            components.append(word[start:i+1])
            start = i + 1
    components.append(word[start:])
    
    return components if len(components) > 1 else [word]


def train_model(
    corpus_path: str = None,
    epochs: int = 20,
    batch_size: int = 32,
    embed_dim: int = 64,
    hidden_dim: int = 128,
    max_words: int = 15000
):
    """Main training function."""
    
    # Paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data'
    models_dir = project_root / 'models'
    models_dir.mkdir(exist_ok=True)
    
    if corpus_path is None:
        corpus_path = str(data_dir / 'smc_corpus.txt')
    
    # Initialize encoder
    encoder = MalayalamCharEncoder()
    
    # Generate/load training data
    training_data_path = str(data_dir / 'sandhi_training_data.json')
    
    generator = GroundTruthGenerator()
    words, labels = generator.generate_training_data(
        corpus_path, training_data_path, max_words
    )
    
    # Split into train/val
    split_idx = int(0.9 * len(words))
    train_words = words[:split_idx]
    train_labels = labels[:split_idx]
    val_words = words[split_idx:]
    val_labels = labels[split_idx:]
    
    print(f"\n📊 Dataset split:")
    print(f"   Training: {len(train_words)} words")
    print(f"   Validation: {len(val_words)} words")
    
    # Create datasets
    train_dataset = SandhiDataset(train_words, train_labels, encoder)
    val_dataset = SandhiDataset(val_words, val_labels, encoder)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    model = BiLSTMSandhiSplitter(
        vocab_size=encoder.vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=2,
        dropout=0.3
    )
    
    print(f"\n🧠 Model architecture:")
    print(f"   Vocabulary size: {encoder.vocab_size}")
    print(f"   Embedding dim: {embed_dim}")
    print(f"   Hidden dim: {hidden_dim}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize trainer
    trainer = SandhiTrainer(model, encoder)
    
    # Training loop
    print(f"\n🚂 Training for {epochs} epochs...")
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        train_loss = trainer.train_epoch(train_loader)
        val_loss, val_acc, _ = trainer.evaluate(val_loader)
        
        trainer.history['train_loss'].append(train_loss)
        trainer.history['val_loss'].append(val_loss)
        trainer.history['val_accuracy'].append(val_acc)
        
        # Learning rate scheduling
        trainer.scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'encoder_vocab': encoder.char_to_idx,
                'config': {
                    'embed_dim': embed_dim,
                    'hidden_dim': hidden_dim,
                    'vocab_size': encoder.vocab_size
                }
            }, str(models_dir / 'best_sandhi_model.pt'))
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"   Epoch {epoch+1:2d}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'encoder_vocab': encoder.char_to_idx,
        'config': {
            'embed_dim': embed_dim,
            'hidden_dim': hidden_dim,
            'vocab_size': encoder.vocab_size
        }
    }, str(models_dir / 'final_sandhi_model.pt'))
    
    print(f"\n✓ Model saved to {models_dir}")
    
    # Test the model
    print("\n" + "="*60)
    print("Testing Neural Sandhi Splitter")
    print("="*60)
    
    test_words = [
        'പഠിക്കുന്നു',
        'വിദ്യാലയത്തിൽ',
        'പഠിച്ചു',
        'പഠിക്കണം',
        'വന്നു',
        'കേരളത്തിൽ',
        'ഭാരതനാട്യം',
        'രമാവൈദ്യനാഥൻ',
    ]
    
    for word in test_words:
        components = split_word_with_model(model, encoder, word)
        print(f"\n{word}")
        print(f"  → {' + '.join(components)}")
    
    return model, encoder, trainer.history


def load_trained_model(model_path: str = None):
    """Load a trained model."""
    
    project_root = Path(__file__).parent.parent
    if model_path is None:
        model_path = str(project_root / 'models' / 'best_sandhi_model.pt')
    
    checkpoint = torch.load(model_path, map_location=DEVICE)
    
    # Initialize encoder
    encoder = MalayalamCharEncoder()
    encoder.char_to_idx = checkpoint['encoder_vocab']
    encoder.idx_to_char = {v: k for k, v in encoder.char_to_idx.items()}
    encoder.vocab_size = len(encoder.char_to_idx)
    
    # Initialize model
    config = checkpoint['config']
    model = BiLSTMSandhiSplitter(
        vocab_size=config['vocab_size'],
        embed_dim=config['embed_dim'],
        hidden_dim=config['hidden_dim']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    
    print(f"✓ Model loaded from {model_path}")
    
    return model, encoder


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Bi-LSTM Neural Sandhi Splitter')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--max-words', type=int, default=15000, help='Max words for training')
    parser.add_argument('--test', action='store_true', help='Test the model')
    parser.add_argument('--word', type=str, help='Single word to split')
    
    args = parser.parse_args()
    
    if args.train:
        train_model(epochs=args.epochs, max_words=args.max_words)
    elif args.test:
        model, encoder = load_trained_model()
        
        test_words = [
            'പഠിക്കുന്നു', 'വിദ്യാലയത്തിൽ', 'പഠിച്ചു', 'പഠിക്കണം',
            'കേരളത്തിൽ', 'ഭാരതനാട്യം'
        ]
        
        print("\n🧪 Testing loaded model:")
        for word in test_words:
            components = split_word_with_model(model, encoder, word)
            print(f"  {word} → {' + '.join(components)}")
    elif args.word:
        model, encoder = load_trained_model()
        components = split_word_with_model(model, encoder, args.word)
        print(f"{' + '.join(components)}")
    else:
        # Default: train
        train_model(epochs=20, max_words=15000)
