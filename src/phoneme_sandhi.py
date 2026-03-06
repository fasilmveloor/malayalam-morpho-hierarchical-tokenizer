"""
Phoneme-Enhanced Neural Sandhi Splitter for Malayalam

Key Innovation:
Instead of just character embeddings, we add explicit phoneme features that
directly encode the linguistic categories that drive sandhi rules:

    - Is_Vowel: അ, ആ, ഇ, ഈ, ഉ, ഊ, ഋ, ൠ, എ, ഏ, ഐ, ഒ, ഓ, ഔ
    - Is_VowelSign: ാ, ി, ീ, ു, ൂ, ൃ, െ, േ, ൈ, ൊ, ോ, ൌ, ൗ
    - Is_Consonant: ക, ഖ, ഗ, ഘ, ങ, ച, ഛ, ജ, ഝ, ഞ, ...
    - Is_Virama: ് (THE critical sandhi marker!)
    - Is_AnuSvara: ം (transforms to ത്ത് in sandhi)
    - Is_Chillu: ൽ, ർ, ൻ, ൺ, ൿ (final forms)
    - Is_Conjunct: Combined consonants like ക്ക, ത്ത, ന്ന

Architecture:
    Input: Character + Phoneme Features
        ↓
    Char Embedding (32-dim) + Phoneme Features (10-dim) → Concat
        ↓
    Bidirectional LSTM (128 hidden)
        ↓
    Dense Layer with Sigmoid → Split predictions

This makes the model "aware" of the phonetic categories that govern sandhi,
accelerating training and improving accuracy on rare patterns.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import os
import re
from typing import List, Tuple, Dict, Optional
from pathlib import Path

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MalayalamPhonemeEncoder:
    """
    Encodes Malayalam characters with both identity and phoneme features.
    
    Phoneme Features (10-dimensional binary vector):
        0: Is_Vowel        - Independent vowels (അ, ആ, ഇ, ...)
        1: Is_VowelSign    - Dependent vowel signs (ാ, ി, ീ, ...)
        2: Is_Consonant    - Consonants (ക, ഖ, ഗ, ...)
        3: Is_Virama       - Chandrakkala (്) - CRITICAL for sandhi!
        4: Is_AnuSvara     - Anusvara (ം) - transforms in sandhi
        5: Is_Chillu       - Chillu letters (ൽ, ർ, ൻ, ൺ, ൿ)
        6: Is_Conjunct     - Common conjuncts (ക്ക, ത്ത, ന്ന, ...)
        7: Is_Digit        - Arabic-Malayalam digits
        8: Is_Punctuation  - Punctuation marks
        9: Is_Other        - Other characters
    """
    
    # Phoneme categories
    VOWELS = set('അആഇഈഉഊഋൠഎഏഐഒഓഔഐഔ')
    VOWEL_SIGNS = set('ാിീുൂൃെേൈൊോൌൗ')
    CONSONANTS = set('കഖഗഘങചഛജഝഞടഠഡഢണതഥദധനപഫബഭമയരലവശഷസഹളഴറഩറ')
    CHILLU = set('ൽർൻൺൿ')  # Chillu letters
    VIRAMA = '്'
    ANUSVARA = 'ം'
    VISARGA = 'ഃ'
    
    # Common conjunct patterns (for feature detection)
    CONJUNCTS = ['ക്ക', 'ത്ത', 'ന്ന', 'മ്മ', 'പ്പ', 'വ്വ', 'ല്ല', 'ണ്ണ', 'ട്ട', 'ങ്ങ',
                 'ച്ച', 'ഞ്ഞ', 'യ്യ', 'ശ്ശ', 'സ്സ', 'ദ്ദ', 'ബ്ബ', 'ഡ്ഡ', 'ഗ്ഗ', 'ഘ്ഘ']
    
    NUM_FEATURES = 10
    
    def __init__(self):
        self.char_to_idx: Dict[str, int] = {}
        self.idx_to_char: Dict[int, str] = {}
        self._build_vocab()
    
    def _build_vocab(self):
        """Build character vocabulary."""
        # Special tokens
        self.char_to_idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx_to_char = {0: '<PAD>', 1: '<UNK>'}
        
        # Malayalam characters (U+0D00 to U+0D7F)
        idx = 2
        for codepoint in range(0x0D00, 0x0D80):
            char = chr(codepoint)
            self.char_to_idx[char] = idx
            self.idx_to_char[idx] = char
            idx += 1
        
        # Common ASCII
        for char in ' 0123456789.,!?;:\'"-()[]{}':
            if char not in self.char_to_idx:
                self.char_to_idx[char] = idx
                self.idx_to_char[idx] = char
                idx += 1
        
        self.vocab_size = len(self.char_to_idx)
        print(f"✓ Vocabulary: {self.vocab_size} characters")
    
    def get_phoneme_features(self, char: str, context: str = None) -> List[float]:
        """
        Get phoneme feature vector for a character.
        
        Args:
            char: The character to encode
            context: Surrounding context for conjunct detection
        
        Returns:
            10-dimensional feature vector
        """
        features = [
            # 0: Is_Vowel
            float(char in self.VOWELS),
            # 1: Is_VowelSign
            float(char in self.VOWEL_SIGNS),
            # 2: Is_Consonant
            float(char in self.CONSONANTS),
            # 3: Is_Virama (CRITICAL for sandhi!)
            float(char == self.VIRAMA),
            # 4: Is_AnuSvara (transforms in sandhi)
            float(char == self.ANUSVARA),
            # 5: Is_Chillu
            float(char in self.CHILLU),
            # 6: Is_Conjunct (context-dependent)
            float(self._is_in_conjunct(char, context) if context else False),
            # 7: Is_Digit
            float(char.isdigit() or char in '൦൧൨൩൪൫൬൭൮൯'),
            # 8: Is_Punctuation
            float(char in '.,!?;:\'"-()[]{}।॥'),
            # 9: Is_Other
            float(char not in self.VOWELS and char not in self.VOWEL_SIGNS and 
                  char not in self.CONSONANTS and char != self.VIRAMA and 
                  char != self.ANUSVARA and char not in self.CHILLU and
                  not char.isdigit() and char not in '.,!?;:\'"-()[]{}।॥൦൧൨൩൪൫൬൭൮൯')
        ]
        return features
    
    def _is_in_conjunct(self, char: str, context: str) -> bool:
        """Check if character is part of a conjunct."""
        if not context:
            return False
        for conjunct in self.CONJUNCTS:
            if char in conjunct and conjunct in context:
                return True
        return False
    
    def encode(self, text: str) -> Tuple[List[int], List[List[float]]]:
        """
        Encode text to character indices and phoneme features.
        
        Returns:
            char_ids: List of character indices
            phoneme_features: List of phoneme feature vectors
        """
        char_ids = []
        phoneme_features = []
        
        for i, char in enumerate(text):
            char_ids.append(self.char_to_idx.get(char, 1))
            
            # Get context for conjunct detection
            start = max(0, i - 2)
            end = min(len(text), i + 3)
            context = text[start:end]
            
            phoneme_features.append(self.get_phoneme_features(char, context))
        
        return char_ids, phoneme_features
    
    def decode(self, indices: List[int]) -> str:
        """Decode indices back to text."""
        return ''.join(self.idx_to_char.get(i, '<UNK>') for i in indices)


class PhonemeEnhancedDataset(Dataset):
    """Dataset with phoneme features for sandhi split prediction."""
    
    def __init__(
        self,
        words: List[str],
        labels: List[List[int]],
        encoder: MalayalamPhonemeEncoder,
        max_len: int = 30
    ):
        self.words = words
        self.labels = labels
        self.encoder = encoder
        self.max_len = max_len
    
    def __len__(self):
        return len(self.words)
    
    def __getitem__(self, idx):
        word = self.words[idx]
        labels = self.labels[idx]
        
        # Encode with phoneme features
        char_ids, phoneme_feats = self.encoder.encode(word)
        
        # Pad/truncate
        seq_len = min(len(char_ids), self.max_len)
        
        padded_chars = char_ids[:self.max_len] + [0] * (self.max_len - len(char_ids))
        padded_labels = labels[:self.max_len] + [0] * (self.max_len - len(labels))
        
        # Pad phoneme features
        padded_phonemes = phoneme_feats[:self.max_len] + [[0] * 10] * (self.max_len - len(phoneme_feats))
        
        # Mask
        mask = [1] * seq_len + [0] * (self.max_len - seq_len)
        
        return {
            'char_ids': torch.tensor(padded_chars, dtype=torch.long),
            'phoneme_feats': torch.tensor(padded_phonemes, dtype=torch.float),
            'labels': torch.tensor(padded_labels, dtype=torch.float),
            'mask': torch.tensor(mask, dtype=torch.float),
            'seq_len': seq_len,
            'word': word
        }


class PhonemeBiLSTM(nn.Module):
    """
    Bi-LSTM model with phoneme feature enhancement.
    
    Architecture:
        Input: Character IDs + Phoneme Features
            ↓
        Char Embedding (32-dim) ──┐
                                  ├─→ Concat (32 + 10 = 42-dim)
        Phoneme Features (10-dim) ┘
            ↓
        Bi-LSTM (2 layers, 96 hidden)
            ↓
        Dense Layers
            ↓
        Sigmoid → Split probabilities
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 32,
        hidden_dim: int = 96,
        phoneme_dim: int = 10,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.phoneme_dim = phoneme_dim
        
        # Character embedding
        self.char_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Combined input dimension
        input_dim = embed_dim + phoneme_dim
        
        # Bi-LSTM
        self.lstm = nn.LSTM(
            input_dim,
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
    
    def forward(self, char_ids, phoneme_feats, mask=None):
        """
        Forward pass.
        
        Args:
            char_ids: [batch, seq_len] character indices
            phoneme_feats: [batch, seq_len, 10] phoneme features
            mask: [batch, seq_len] mask for valid positions
        """
        # Embed characters
        char_embeds = self.char_embed(char_ids)  # [batch, seq_len, embed_dim]
        
        # Concatenate with phoneme features
        combined = torch.cat([char_embeds, phoneme_feats], dim=-1)  # [batch, seq_len, embed_dim + phoneme_dim]
        
        # Bi-LSTM
        lstm_out, _ = self.lstm(combined)  # [batch, seq_len, hidden_dim * 2]
        
        # Output
        out = self.dropout(lstm_out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out).squeeze(-1)  # [batch, seq_len]
        out = self.sigmoid(out)
        
        if mask is not None:
            out = out * mask
        
        return out
    
    def predict_splits(self, char_ids, phoneme_feats, mask, threshold=0.3):
        """Predict split points."""
        self.eval()
        with torch.no_grad():
            probs = self.forward(char_ids, phoneme_feats, mask)
            predictions = (probs > threshold).int()
        return predictions


class PhonemeSandhiTrainer:
    """Trainer for phoneme-enhanced model."""
    
    def __init__(self, model, encoder, learning_rate=0.003):
        self.model = model.to(DEVICE)
        self.encoder = encoder
        self.criterion = nn.BCELoss(reduction='none')
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    def train_epoch(self, dataloader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_samples = 0
        
        for batch in dataloader:
            char_ids = batch['char_ids'].to(DEVICE)
            phoneme_feats = batch['phoneme_feats'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            mask = batch['mask'].to(DEVICE)
            
            self.optimizer.zero_grad()
            outputs = self.model(char_ids, phoneme_feats, mask)
            
            loss = self.criterion(outputs, labels)
            loss = (loss * mask).sum() / mask.sum()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item() * char_ids.size(0)
            total_samples += char_ids.size(0)
        
        return total_loss / total_samples
    
    def evaluate(self, dataloader, threshold=0.3):
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0
        total_samples = 0
        correct = 0
        total_positions = 0
        
        with torch.no_grad():
            for batch in dataloader:
                char_ids = batch['char_ids'].to(DEVICE)
                phoneme_feats = batch['phoneme_feats'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)
                mask = batch['mask'].to(DEVICE)
                
                outputs = self.model(char_ids, phoneme_feats, mask)
                
                loss = self.criterion(outputs, labels)
                loss = (loss * mask).sum() / mask.sum()
                total_loss += loss.item() * char_ids.size(0)
                total_samples += char_ids.size(0)
                
                predictions = (outputs > threshold).float()
                correct += ((predictions == labels) * mask).sum().item()
                total_positions += mask.sum().item()
        
        avg_loss = total_loss / total_samples
        accuracy = correct / total_positions if total_positions > 0 else 0
        
        return avg_loss, accuracy


def split_word_with_phoneme_model(model, encoder, word, threshold=0.3):
    """Split a word using the phoneme-enhanced model."""
    model.eval()
    
    char_ids, phoneme_feats = encoder.encode(word)
    max_len = 30
    
    padded_chars = char_ids[:max_len] + [0] * (max_len - len(char_ids))
    padded_phonemes = phoneme_feats[:max_len] + [[0] * 10] * (max_len - len(phoneme_feats))
    mask = [1] * min(len(char_ids), max_len) + [0] * (max_len - min(len(char_ids), max_len))
    
    char_tensor = torch.tensor([padded_chars], dtype=torch.long).to(DEVICE)
    phoneme_tensor = torch.tensor([padded_phonemes], dtype=torch.float).to(DEVICE)
    mask_tensor = torch.tensor([mask], dtype=torch.float).to(DEVICE)
    
    with torch.no_grad():
        probs = model(char_tensor, phoneme_tensor, mask_tensor)[0][:len(word)]
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


def create_training_data_with_phonemes(output_path: str = None):
    """Create expanded training data with phonetically accurate split points."""
    
    training_data = [
        # === VERB FORMS - Present Tense (ഉന്നു) ===
        # Split AFTER virama before ുന്നു (stem + suffix)
        ('പഠിക്കുന്നു', [0,0,0,0,1,0,0,0,0,0,0]),  # പഠിക്ക് + ുന്നു
        ('വരുന്നു', [0,0,0,1,0,0,0]),              # വര് + ുന്നു
        ('ചെയ്യുന്നു', [0,0,0,0,1,0,0,0,0]),      # ചെയ്യ് + ുന്നു
        ('നടക്കുന്നു', [0,0,0,0,0,1,0,0,0,0]),    # നടക്ക് + ുന്നു
        ('എഴുതുന്നു', [0,0,0,0,1,0,0,0,0]),      # എഴുത് + ുന്നു
        ('കാണുന്നു', [0,0,0,1,0,0,0]),            # കാണ് + ുന്നു
        ('പറയുന്നു', [0,0,0,1,0,0,0]),            # പറ + യുന്നു
        ('ഓടുന്നു', [0,0,0,1,0,0,0]),              # ഓട് + ുന്നു
        ('പാടുന്നു', [0,0,0,1,0,0,0]),            # പാട് + ുന്നു
        ('ആടുന്നു', [0,0,0,1,0,0,0]),              # ആട് + ുന്നു
        ('കേൾക്കുന്നു', [0,0,0,0,0,1,0,0,0,0]),  # കേൾക്ക് + ുന്നു
        ('നടത്തുന്നു', [0,0,0,0,1,0,0,0,0]),      # നടത്ത് + ുന്നു
        ('പഠിപ്പിക്കുന്നു', [0,0,0,0,0,0,1,0,0,0,0,0]),  # പഠിപ്പിക്ക് + ുന്നു
        ('തിരുത്തുന്നു', [0,0,0,0,0,1,0,0,0,0]),  # തിരുത്ത് + ുന്നു
        ('തുടരുന്നു', [0,0,0,0,1,0,0,0,0]),      # തുടര് + ുന്നു
        
        # === VERB FORMS - Past Tense ===
        ('പഠിച്ചു', [0,0,0,1,0,0,0]),            # പഠി + ച്ചു
        ('വന്നു', [0,0,0,0,0,0]),                # Single
        ('പോയി', [0,0,0,0,0]),                  # Single
        ('ചെയ്തു', [0,0,0,0,0,0,0]),            # Single
        ('എഴുതി', [0,0,0,0,0,0,0]),              # Single
        ('കണ്ടു', [0,0,0,0,0,0]),                # Single
        ('കേട്ടു', [0,0,0,0,0,0,0]),            # Single
        ('ഓടി', [0,0,0,0,0]),                    # Single
        ('ആടി', [0,0,0,0,0]),                    # Single
        ('പാടി', [0,0,0,0,0]),                    # Single
        ('നടന്നു', [0,0,0,0,0,0,0,0]),          # Single
        ('വളർന്നു', [0,0,0,0,0,0,0,0]),        # Single
        
        # === VERB FORMS - Future/Necessity ===
        ('പഠിക്കണം', [0,0,0,0,1,0,0]),          # പഠിക്ക് + ണം
        ('പഠിക്കാൻ', [0,0,0,0,1,0,0]),          # പഠിക്ക് + ാൻ
        ('വരണം', [0,0,0,1,0,0]),                # വര് + ണം
        ('വരാൻ', [0,0,0,1,0,0]),                # വര് + ാൻ
        ('ചെയ്യണം', [0,0,0,0,1,0,0]),            # ചെയ്യ് + ണം
        ('പോകണം', [0,0,0,1,0,0]),                # പോക് + ണം
        ('നടക്കണം', [0,0,0,0,0,1,0,0]),          # നടക്ക് + ണം
        ('കാണണം', [0,0,0,1,0,0]),                # കാണ് + ണം
        ('എഴുതണം', [0,0,0,0,1,0,0]),            # എഴുത് + ണം
        
        # === CASE-MARKED NOUNS (ം → ത്ത് transformation) ===
        # Key insight: Anusvara (ം) + case marker creates ത്ത് infix
        ('വിദ്യാലയത്തിൽ', [0,0,0,0,0,0,0,0,0,1,0,0]),    # വിദ്യാലയം + ത്തിൽ
        ('വിദ്യാലയത്തിന്റെ', [0,0,0,0,0,0,0,0,0,1,0,0,0,0]),
        ('വിദ്യാലയത്തിലെ', [0,0,0,0,0,0,0,0,0,1,0,0,0]),
        ('കേരളത്തിൽ', [0,0,0,0,0,1,0,0]),                # കേരളം + ത്തിൽ
        ('കേരളത്തിന്റെ', [0,0,0,0,0,1,0,0,0,0]),
        ('കേരളത്തിലെ', [0,0,0,0,0,1,0,0,0]),
        ('വീട്ടിൽ', [0,0,0,1,0,0]),                    # വീട് + ടിൽ
        ('വീട്ടിന്റെ', [0,0,0,1,0,0,0,0]),
        ('പാലത്തിൽ', [0,0,0,0,1,0,0]),                # പാലം + ത്തിൽ
        ('പുസ്തകത്തിൽ', [0,0,0,0,0,0,1,0,0]),          # പുസ്തകം + ത്തിൽ
        ('പുസ്തകത്തിന്റെ', [0,0,0,0,0,0,1,0,0,0,0]),
        ('മനസ്സിൽ', [0,0,0,0,0,1,0,0]),              # മനസ്സ് + സിൽ
        ('നാട്ടിൽ', [0,0,0,1,0,0]),                  # നാട് + ടിൽ
        ('സ്കൂളിൽ', [0,0,0,0,1,0,0]),                # സ്കൂൾ + ലിൽ
        ('കാറിൽ', [0,0,0,1,0,0]),                    # കാർ + റിൽ
        ('ബസ്സിൽ', [0,0,0,0,1,0,0]),                  # ബസ് + സിൽ
        ('വിമാനത്തിൽ', [0,0,0,0,0,0,1,0,0]),          # വിമാനം + ത്തിൽ
        
        # === OTHER NOUN FORMS (vowel-initial suffixes) ===
        ('അധ്യാപികയുടെ', [0,0,0,0,0,0,0,1,0,0,0]),  # അധ്യാപിക + യുടെ
        ('വിദ്യാർത്ഥിയുടെ', [0,0,0,0,0,0,0,0,1,0,0,0,0]),
        ('അമ്മയുടെ', [0,0,0,0,1,0,0,0]),            # അമ്മ + യുടെ
        ('സഹോദരിയുടെ', [0,0,0,0,0,0,0,1,0,0,0]),
        ('മകന്റെ', [0,0,0,0,1,0,0,0]),              # മകൻ + ൻ്റെ
        ('അച്ഛന്റെ', [0,0,0,0,1,0,0,0]),            # അച്ഛൻ + ൻ്റെ
        
        # === COMPOUND WORDS ===
        ('ഭാരതനാട്യം', [0,0,0,0,0,1,0,0,0,0]),      # ഭാരത + നാട്യം
        ('രമാവൈദ്യനാഥൻ', [0,0,0,1,0,0,0,0,0,0,0,0]),
        ('പ്രധാനമന്ത്രി', [0,0,0,0,0,0,1,0,0,0,0]),
        ('രക്തസമ്മർദ്ദം', [0,0,0,0,1,0,0,0,0,0,0]),
        ('പാലക്കാട്', [0,0,0,0,0,1,0,0]),            # പാല + ക്കാട്
        ('തിരുവനന്തപുരം', [0,0,0,0,1,0,0,0,0,0,0,0,0]),
        ('കോഴിക്കോട്', [0,0,0,0,0,1,0,0]),          # കോഴി + ക്കോട്
        ('സ്വാതന്ത്ര്യം', [0,0,0,0,0,1,0,0,0,0]),
        ('പ്രവൃത്തി', [0,0,0,0,0,1,0,0]),
        ('വിദ്യാഭ്യാസം', [0,0,0,0,0,0,1,0,0,0]),
        
        # === SINGLE MORPHEME WORDS ===
        ('മലയാളം', [0,0,0,0,0,0,0,0]),
        ('ഇന്ത്യ', [0,0,0,0,0]),
        ('കേരളം', [0,0,0,0,0,0,0]),
        ('വീട്', [0,0,0,0]),
        ('പാഠം', [0,0,0,0,0]),
        ('പുസ്തകം', [0,0,0,0,0,0,0,0]),
        ('അച്ഛൻ', [0,0,0,0,0,0]),
        ('അമ്മ', [0,0,0,0]),
        ('സ്കൂൾ', [0,0,0,0,0]),
        ('കോളേജ്', [0,0,0,0,0,0,0]),
        
        # === ADJECTIVES ===
        ('നല്ലതിൽ', [0,0,0,0,0,1,0,0]),
        ('വലിയതിൽ', [0,0,0,0,0,1,0,0]),
        ('ചെറിയതിൽ', [0,0,0,0,0,1,0,0]),
        ('സുന്ദരമായ', [0,0,0,0,0,0,1,0,0]),
        ('പുതിയത്', [0,0,0,0,0,1,0,0]),
        ('പഴയത്', [0,0,0,0,1,0,0]),
        
        # === MORE VERB VARIATIONS ===
        ('പഠിക്കുന്ന', [0,0,0,0,1,0,0,0,0]),        # പഠിക്ക് + ുന്ന
        ('പഠിപ്പിക്കുന്ന', [0,0,0,0,0,0,1,0,0,0,0]),
        ('എഴുതുന്ന', [0,0,0,0,1,0,0,0]),          # എഴുത് + ുന്ന
        ('വായിക്കുന്നു', [0,0,0,0,1,0,0,0,0,0]),    # വായിക്ക് + ുന്നു
        ('കളിക്കുന്നു', [0,0,0,0,0,1,0,0,0,0]),    # കളിക്ക് + ുന്നു
        
        # === MORE NOUNS WITH CASE ===
        ('ജില്ലയിൽ', [0,0,0,0,0,1,0,0]),          # ജില്ല + യിൽ
        ('രാജ്യത്തിൽ', [0,0,0,0,0,0,1,0,0]),      # രാജ്യം + ത്തിൽ
        ('ഗ്രാമത്തിൽ', [0,0,0,0,0,0,1,0,0]),      # ഗ്രാമം + ത്തിൽ
        ('പഞ്ചായത്തിൽ', [0,0,0,0,0,0,0,1,0,0]),  # പഞ്ചായത്ത് + തിൽ
        ('നഗരത്തിൽ', [0,0,0,0,0,0,1,0,0]),        # നഗരം + ത്തിൽ
    ]
    
    print(f"Created {len(training_data)} training examples")
    
    words = [t[0] for t in training_data]
    labels = [t[1] for t in training_data]
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({'words': words, 'labels': labels}, f, ensure_ascii=False, indent=2)
        print(f"Saved to {output_path}")
    
    multi = sum(1 for _, l in training_data if sum(l) > 0)
    print(f"Multi-component: {multi} ({100*multi/len(training_data):.1f}%)")
    
    return words, labels


def train_phoneme_model(epochs=100, batch_size=16):
    """Train the phoneme-enhanced model."""
    
    print("="*60)
    print("Phoneme-Enhanced Neural Sandhi Splitter")
    print("="*60)
    
    # Create training data
    print("\n[1/5] Creating training data...")
    words, labels = create_training_data_with_phonemes('data/phoneme_training_data.json')
    
    # Initialize encoder
    print("\n[2/5] Initializing phoneme encoder...")
    encoder = MalayalamPhonemeEncoder()
    
    # Create datasets
    print("\n[3/5] Creating datasets...")
    split_idx = int(0.9 * len(words))
    train_dataset = PhonemeEnhancedDataset(words[:split_idx], labels[:split_idx], encoder)
    val_dataset = PhonemeEnhancedDataset(words[split_idx:], labels[split_idx:], encoder)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    print(f"   Training: {len(train_dataset)}, Validation: {len(val_dataset)}")
    
    # Initialize model
    print("\n[4/5] Initializing model...")
    model = PhonemeBiLSTM(
        vocab_size=encoder.vocab_size,
        embed_dim=32,
        hidden_dim=96,
        phoneme_dim=10
    )
    params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {params:,}")
    
    # Train
    print("\n[5/5] Training...")
    trainer = PhonemeSandhiTrainer(model, encoder)
    
    best_val_loss = float('inf')
    for epoch in range(epochs):
        train_loss = trainer.train_epoch(train_loader)
        val_loss, val_acc = trainer.evaluate(val_loader)
        
        trainer.history['train_loss'].append(train_loss)
        trainer.history['val_loss'].append(val_loss)
        trainer.history['val_acc'].append(val_acc)
        
        trainer.scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs('models', exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'char2idx': encoder.char_to_idx,
                'config': {
                    'embed_dim': 32,
                    'hidden_dim': 96,
                    'phoneme_dim': 10,
                    'vocab_size': encoder.vocab_size
                }
            }, 'models/phoneme_sandhi_model.pt')
        
        if (epoch + 1) % 20 == 0:
            print(f"   Epoch {epoch+1}: Loss={train_loss:.4f}, ValLoss={val_loss:.4f}, Acc={val_acc:.4f}")
    
    # Test
    print("\n" + "="*60)
    print("Testing Phoneme-Enhanced Model")
    print("="*60)
    
    test_words = [
        'പഠിക്കുന്നു', 'വിദ്യാലയത്തിൽ', 'കേരളത്തിൽ', 'ഭാരതനാട്യം',
        'അധ്യാപികയുടെ', 'പ്രധാനമന്ത്രി', 'വീട്ടിൽ', 'പാലക്കാട്',
        'എഴുതുന്നു', 'പുസ്തകത്തിൽ', 'വരുന്നു', 'രമാവൈദ്യനാഥൻ'
    ]
    
    for word in test_words:
        components = split_word_with_phoneme_model(model, encoder, word)
        print(f"   {word} → {' + '.join(components)}")
    
    print("\n✓ Training complete!")
    
    return model, encoder, trainer.history


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--test', action='store_true', help='Test the model')
    
    args = parser.parse_args()
    
    if args.train:
        train_phoneme_model(epochs=args.epochs)
    else:
        # Default: train
        train_phoneme_model(epochs=100)
