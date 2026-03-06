"""
BIO-Tagging Enhanced Neural Sandhi Splitter

Uses BIO (Begin-Inside-Outside) tagging for morpheme boundary detection:
- B-MORPH: Beginning of a new morpheme
- I-MORPH: Inside a morpheme (continuation)
- O: Outside (rare, mainly for punctuation)

This approach:
1. Provides better structural coherence for multi-component words
2. Handles complex words with multiple splits naturally
3. Allows the model to learn morpheme-level patterns

Architecture:
    Input: Character + Phoneme Features
        ↓
    Bi-LSTM Encoder
        ↓
    Dense Layer + Softmax
        ↓
    BIO Tags: [B, I, I, I, B, I, I, ...]
        ↓
    Components extracted from tag sequence
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import os
from typing import List, Tuple, Dict
from pathlib import Path

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# BIO Tags
BIO_TAGS = {
    'B': 0,  # Begin morpheme
    'I': 1,  # Inside morpheme
    'PAD': 2  # Padding (ignored in loss)
}
NUM_TAGS = 2  # Only B and I for actual prediction


class MalayalamPhonemeEncoder:
    """Phoneme-aware character encoder."""
    
    VOWELS = set('അആഇഈഉഊഋൠഎഏഐഒഓഔ')
    VOWEL_SIGNS = set('ാിീുൂൃെേൈൊോൌൗ')
    CONSONANTS = set('കഖഗഘങചഛജഝഞടഠഡഢണതഥദധനപഫബഭമയരലവശഷസഹളഴറഩ')
    CHILLU = set('ൽർൻൺൿ')
    VIRAMA = '്'
    ANUSVARA = 'ം'
    
    NUM_FEATURES = 10
    
    def __init__(self):
        self.char_to_idx = {'<PAD>': 0, '<UNK>': 1}
        idx = 2
        for c in range(0x0D00, 0x0D80):
            self.char_to_idx[chr(c)] = idx
            idx += 1
        self.vocab_size = len(self.char_to_idx)
    
    def get_phoneme_features(self, char: str) -> List[float]:
        """Get 10-dimensional phoneme feature vector."""
        return [
            float(char in self.VOWELS),
            float(char in self.VOWEL_SIGNS),
            float(char in self.CONSONANTS),
            float(char == self.VIRAMA),
            float(char == self.ANUSVARA),
            float(char in self.CHILLU),
            float(char in '൦൧൨൩൪൫൬൭൮൯'),  # Malayalam digits
            float(char.isdigit()),
            float(char in '.,!?;:\'"-()[]{}'),
            float(char not in self.VOWELS and char not in self.VOWEL_SIGNS and 
                  char not in self.CONSONANTS and char != self.VIRAMA and 
                  char != self.ANUSVARA and char not in self.CHILLU)
        ]
    
    def encode(self, text: str) -> Tuple[List[int], List[List[float]]]:
        """Encode text to char indices and phoneme features."""
        char_ids = [self.char_to_idx.get(c, 1) for c in text]
        phoneme_feats = [self.get_phoneme_features(c) for c in text]
        return char_ids, phoneme_feats


def split_to_bio(split_positions: List[int], word_len: int) -> List[int]:
    """
    Convert split positions to BIO tags.
    
    Args:
        split_positions: Positions AFTER which a split occurs
        word_len: Length of the word
    
    Returns:
        BIO tag sequence
    """
    bio_tags = []
    for i in range(word_len):
        if i == 0:
            bio_tags.append(BIO_TAGS['B'])  # First char always begins a morpheme
        elif i in split_positions:
            bio_tags.append(BIO_TAGS['B'])  # New morpheme starts
        else:
            bio_tags.append(BIO_TAGS['I'])  # Continue morpheme
    return bio_tags


def bio_to_components(word: str, bio_tags: List[int]) -> List[str]:
    """
    Extract morpheme components from BIO tags.
    
    Args:
        word: The word
        bio_tags: Predicted BIO tags
    
    Returns:
        List of morpheme components
    """
    components = []
    current = ""
    
    for i, (char, tag) in enumerate(zip(word, bio_tags)):
        if tag == BIO_TAGS['B']:  # Begin new morpheme
            if current:
                components.append(current)
            current = char
        else:  # Inside morpheme
            current += char
    
    if current:
        components.append(current)
    
    return components


class BIODataset(Dataset):
    """Dataset with BIO tags for sandhi prediction."""
    
    def __init__(self, words, split_positions_list, encoder, max_len=30):
        self.words = words
        self.split_positions_list = split_positions_list
        self.encoder = encoder
        self.max_len = max_len
    
    def __len__(self):
        return len(self.words)
    
    def __getitem__(self, idx):
        word = self.words[idx]
        split_positions = self.split_positions_list[idx]
        
        # Encode
        char_ids, phoneme_feats = self.encoder.encode(word)
        
        # Convert to BIO tags
        bio_tags = split_to_bio(split_positions, len(word))
        
        # Pad
        seq_len = min(len(char_ids), self.max_len)
        padded_chars = char_ids[:self.max_len] + [0] * (self.max_len - len(char_ids))
        padded_phonemes = phoneme_feats[:self.max_len] + [[0] * 10] * (self.max_len - len(phoneme_feats))
        padded_bio = bio_tags[:self.max_len] + [BIO_TAGS['PAD']] * (self.max_len - len(bio_tags))
        mask = [1] * seq_len + [0] * (self.max_len - seq_len)
        
        return {
            'char_ids': torch.tensor(padded_chars, dtype=torch.long),
            'phoneme_feats': torch.tensor(padded_phonemes, dtype=torch.float),
            'bio_tags': torch.tensor(padded_bio, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.float),
            'seq_len': seq_len,
            'word': word
        }


class BIOBiLSTM(nn.Module):
    """
    Bi-LSTM model with BIO tagging output.
    
    Uses CrossEntropyLoss for multi-class classification (B vs I).
    """
    
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=96, phoneme_dim=10):
        super().__init__()
        
        self.char_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        input_dim = embed_dim + phoneme_dim
        
        self.lstm = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=2, batch_first=True,
            bidirectional=True, dropout=0.2
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, NUM_TAGS)  # B or I
        )
    
    def forward(self, char_ids, phoneme_feats, mask=None):
        char_embeds = self.char_embed(char_ids)
        combined = torch.cat([char_embeds, phoneme_feats], dim=-1)
        
        lstm_out, _ = self.lstm(combined)
        
        logits = self.fc(lstm_out)  # [batch, seq_len, 2]
        
        return logits
    
    def predict_bio(self, char_ids, phoneme_feats, mask):
        """Predict BIO tags."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(char_ids, phoneme_feats, mask)
            predictions = logits.argmax(dim=-1)  # [batch, seq_len]
        return predictions


def create_bio_training_data():
    """Create training data with split positions for BIO tagging."""
    
    # Format: (word, split_positions)
    # split_positions: list of character positions AFTER which a split occurs
    # Example: 'പഠിക്കുന്നു' with splits after position 4 → [4]
    # This means: പഠിക്ക് | ുന്നു
    
    training_data = [
        # Present tense verbs - split after virama before suffix
        ('പഠിക്കുന്നു', [4]),      # പഠിക്ക് | ുന്നു
        ('വരുന്നു', [3]),          # വര് | ുന്നു
        ('ചെയ്യുന്നു', [4]),        # ചെയ്യ് | ുന്നു
        ('നടക്കുന്നു', [5]),        # നടക്ക് | ുന്നു
        ('എഴുതുന്നു', [4]),        # എഴുത് | ുന്നു
        ('കാണുന്നു', [3]),          # കാണ് | ുന്നു
        ('പറയുന്നു', [3]),          # പറ | യുന്നു
        ('ഓടുന്നു', [3]),            # ഓട് | ുന്നു
        ('പാടുന്നു', [3]),          # പാട് | ുന്നു
        ('ആടുന്നു', [3]),            # ആട് | ുന്നു
        ('കേൾക്കുന്നു', [5]),      # കേൾക്ക് | ുന്നു
        ('പഠിപ്പിക്കുന്നു', [6]),  # പഠിപ്പിക്ക് | ുന്നു
        ('തിരുത്തുന്നു', [5]),      # തിരുത്ത് | ുന്നു
        ('വായിക്കുന്നു', [5]),      # വായിക്ക് | ുന്നു
        ('കളിക്കുന്നു', [5]),        # കളിക്ക് | ുന്നു
        
        # Past tense
        ('പഠിച്ചു', [3]),          # പഠി | ച്ചു
        ('വന്നു', []),              # Single
        ('പോയി', []),              # Single
        ('ചെയ്തു', []),            # Single
        ('എഴുതി', []),              # Single
        ('കണ്ടു', []),              # Single
        ('ഓടി', []),                # Single
        
        # Future/Necessity
        ('പഠിക്കണം', [4]),          # പഠിക്ക് | ണം
        ('പഠിക്കാൻ', [4]),          # പഠിക്ക് | ാൻ
        ('വരണം', [3]),              # വര് | ണം
        ('ചെയ്യണം', [4]),            # ചെയ്യ് | ണം
        ('നടക്കണം', [5]),            # നടക്ക് | ണം
        
        # Case-marked nouns (ം → ത്ത്)
        ('വിദ്യാലയത്തിൽ', [9]),    # വിദ്യാലയം → ത്തിൽ
        ('കേരളത്തിൽ', [5]),          # കേരളം → ത്തിൽ
        ('വീട്ടിൽ', [3]),            # വീട് | ടിൽ
        ('പാലത്തിൽ', [4]),            # പാലം → ത്തിൽ
        ('പുസ്തകത്തിൽ', [6]),        # പുസ്തകം → ത്തിൽ
        ('മനസ്സിൽ', [5]),            # മനസ്സ് | സിൽ
        ('നാട്ടിൽ', [3]),            # നാട് | ടിൽ
        ('സ്കൂളിൽ', [4]),            # സ്കൂൾ | ലിൽ
        ('വിമാനത്തിൽ', [6]),          # വിമാനം → ത്തിൽ
        
        # More case markers
        ('അധ്യാപികയുടെ', [7]),      # അധ്യാപിക | യുടെ
        ('വിദ്യാർത്ഥിയുടെ', [8]),    # വിദ്യാർത്ഥി | യുടെ
        ('അമ്മയുടെ', [4]),          # അമ്മ | യുടെ
        ('മകന്റെ', [4]),            # മകൻ | ൻ്റെ
        
        # Compound words
        ('ഭാരതനാട്യം', [5]),          # ഭാരത | നാട്യം
        ('രമാവൈദ്യനാഥൻ', [3]),      # രമാ | വൈദ്യനാഥൻ
        ('പ്രധാനമന്ത്രി', [6]),      # പ്രധാന | മന്ത്രി
        ('രക്തസമ്മർദ്ദം', [4]),      # രക്ത | സമ്മർദ്ദം
        ('പാലക്കാട്', [5]),          # പാല | ക്കാട്
        ('തിരുവനന്തപുരം', [4]),      # തിരു | അനന്തപുരം
        ('കോഴിക്കോട്', [5]),          # കോഴി | ക്കോട്
        ('സ്വാതന്ത്ര്യം', [5]),        # സ്വ | തന്ത്ര്യം
        ('വിദ്യാഭ്യാസം', [6]),        # വിദ്യാ | ഭ്യാസം
        
        # Single morpheme words
        ('മലയാളം', []),
        ('ഇന്ത്യ', []),
        ('കേരളം', []),
        ('വീട്', []),
        ('പാഠം', []),
        ('പുസ്തകം', []),
        ('അച്ഛൻ', []),
        ('അമ്മ', []),
        ('സ്കൂൾ', []),
        ('കോളേജ്', []),
        
        # Adjectives
        ('നല്ലതിൽ', [5]),            # നല്ല | തിൽ
        ('വലിയതിൽ', [5]),            # വലിയ | തിൽ
        ('ചെറിയതിൽ', [5]),          # ചെറിയ | തിൽ
        ('സുന്ദരമായ', [6]),          # സുന്ദര | മായ
        
        # More nouns
        ('ജില്ലയിൽ', [5]),            # ജില്ല | യിൽ
        ('രാജ്യത്തിൽ', [6]),          # രാജ്യം → ത്തിൽ
        ('ഗ്രാമത്തിൽ', [6]),          # ഗ്രാമം → ത്തിൽ
        ('നഗരത്തിൽ', [6]),            # നഗരം → ത്തിൽ
    ]
    
    words = [t[0] for t in training_data]
    split_positions = [t[1] for t in training_data]
    
    multi = sum(1 for s in split_positions if len(s) > 0)
    print(f"Created {len(words)} training examples, {multi} ({100*multi/len(words):.1f}%) multi-component")
    
    return words, split_positions


def train_bio_model(epochs=100):
    """Train BIO-tagging model."""
    
    print("="*60)
    print("BIO-Tagging Neural Sandhi Splitter")
    print("="*60)
    
    # Create data
    print("\n[1/4] Creating BIO training data...")
    words, split_positions = create_bio_training_data()
    
    # Encoder
    print("\n[2/4] Initializing encoder...")
    encoder = MalayalamPhonemeEncoder()
    
    # Show BIO tag example
    example_word = 'പഠിക്കുന്നു'
    example_splits = [4]
    example_bio = split_to_bio(example_splits, len(example_word))
    print(f"\n   Example BIO tags:")
    print(f"   Word: {example_word}")
    print(f"   Chars: {'  '.join(example_word)}")
    print(f"   BIO:   {'  '.join(['B' if t==0 else 'I' for t in example_bio])}")
    
    # Dataset
    print("\n[3/4] Creating datasets...")
    split_idx = int(0.9 * len(words))
    train_dataset = BIODataset(words[:split_idx], split_positions[:split_idx], encoder)
    val_dataset = BIODataset(words[split_idx:], split_positions[split_idx:], encoder)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    
    # Model
    print("\n[4/4] Training...")
    model = BIOBiLSTM(encoder.vocab_size)
    model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss(ignore_index=BIO_TAGS['PAD'])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    
    for epoch in range(epochs):
        # Train
        model.train()
        total_loss = 0
        for batch in train_loader:
            char_ids = batch['char_ids'].to(DEVICE)
            phoneme_feats = batch['phoneme_feats'].to(DEVICE)
            bio_tags = batch['bio_tags'].to(DEVICE)
            mask = batch['mask'].to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(char_ids, phoneme_feats, mask)  # [batch, seq, 2]
            
            # CrossEntropyLoss expects [batch, classes, seq]
            loss = criterion(logits.transpose(1, 2), bio_tags)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 25 == 0:
            # Evaluate
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in val_loader:
                    char_ids = batch['char_ids'].to(DEVICE)
                    phoneme_feats = batch['phoneme_feats'].to(DEVICE)
                    bio_tags = batch['bio_tags'].to(DEVICE)
                    mask = batch['mask'].to(DEVICE)
                    
                    preds = model.predict_bio(char_ids, phoneme_feats, mask)
                    
                    # Count correct (excluding PAD)
                    valid = (bio_tags != BIO_TAGS['PAD'])
                    correct += ((preds == bio_tags) & valid).sum().item()
                    total += valid.sum().item()
            
            acc = correct / total if total > 0 else 0
            print(f"   Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, ValAcc={acc:.4f}")
    
    # Save
    os.makedirs('models', exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'char2idx': encoder.char_to_idx,
        'config': {
            'embed_dim': 32,
            'hidden_dim': 96,
            'vocab_size': encoder.vocab_size
        }
    }, 'models/bio_sandhi_model.pt')
    
    # Test
    print("\n" + "="*60)
    print("Testing BIO-Tagging Model")
    print("="*60)
    
    test_words = [
        'പഠിക്കുന്നു', 'വിദ്യാലയത്തിൽ', 'കേരളത്തിൽ', 'ഭാരതനാട്യം',
        'അധ്യാപികയുടെ', 'പ്രധാനമന്ത്രി', 'വീട്ടിൽ', 'പാലക്കാട്'
    ]
    
    model.eval()
    for word in test_words:
        char_ids, phoneme_feats = encoder.encode(word)
        max_len = 30
        padded_chars = torch.tensor([char_ids[:max_len] + [0]*(max_len-len(char_ids))], dtype=torch.long).to(DEVICE)
        padded_phonemes = torch.tensor([phoneme_feats[:max_len] + [[0]*10]*(max_len-len(phoneme_feats))], dtype=torch.float).to(DEVICE)
        mask = torch.tensor([[1]*min(len(char_ids),max_len) + [0]*(max_len-min(len(char_ids),max_len))], dtype=torch.float).to(DEVICE)
        
        with torch.no_grad():
            preds = model.predict_bio(padded_chars, padded_phonemes, mask)[0][:len(word)].tolist()
        
        components = bio_to_components(word, preds)
        print(f"   {word} → {' + '.join(components)}")
    
    print("\n✓ BIO model trained!")
    
    return model, encoder


if __name__ == "__main__":
    train_bio_model(epochs=100)
