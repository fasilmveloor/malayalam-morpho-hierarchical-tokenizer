"""
Bi-LSTM-CRF Model for Malayalam Sandhi Splitting

CRF (Conditional Random Field) layer ensures valid BIO sequences:
- B (Begin) can only follow B or I
- I (Inside) must follow B or I of same morpheme
- Prevents invalid transitions like I at start

Architecture:
    Input: Character IDs + Phoneme Features
        ↓
    Bi-LSTM Encoder
        ↓
    CRF Layer (learns transition probabilities)
        ↓
    Viterbi Decoding → Optimal BIO sequence
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import os
from typing import List, Tuple, Dict, Optional
from collections import Counter

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MalayalamPhonemeEncoder:
    """Phoneme-aware character encoder."""
    
    VOWELS = set('അആഇഈഉഊഋൠഎഏഐഒഓഔ')
    VOWEL_SIGNS = set('ാിീുൂൃെേൈൊോൌൗ')
    CONSONANTS = set('കഖഗഘങചഛജഝഞടഠഡഢണതഥദധനപഫബഭമയരലവശഷസഹളഴറ')
    CHILLU = set('ൽർൻൺൿ')
    VIRAMA = '്'
    ANUSVARA = 'ം'
    
    def __init__(self):
        self.char_to_idx = {'<PAD>': 0, '<UNK>': 1}
        idx = 2
        for c in range(0x0D00, 0x0D80):
            self.char_to_idx[chr(c)] = idx
            idx += 1
        self.vocab_size = len(self.char_to_idx)
    
    def get_phoneme_features(self, char: str) -> List[float]:
        return [
            float(char in self.VOWELS),
            float(char in self.VOWEL_SIGNS),
            float(char in self.CONSONANTS),
            float(char == self.VIRAMA),
            float(char == self.ANUSVARA),
            float(char in self.CHILLU),
            float(char in '൦൧൨൩൪൫൬൭൮൯'),
            float(char.isdigit()),
            float(char in '.,!?;:\'"-()[]{}'),
            0.0  # padding
        ]
    
    def encode(self, text: str):
        char_ids = [self.char_to_idx.get(c, 1) for c in text]
        phonemes = [self.get_phoneme_features(c) for c in text]
        return char_ids, phonemes


class CRF(nn.Module):
    """
    Conditional Random Field for BIO tagging.
    
    Learns transition probabilities between tags:
        P(tag_i | tag_{i-1}) for all tag pairs
    
    Uses Viterbi algorithm for optimal sequence decoding.
    """
    
    def __init__(self, num_tags: int):
        super().__init__()
        self.num_tags = num_tags
        
        # Transition matrix: transitions[i,j] = log P(tag_j | tag_i)
        # Index 0 = B, Index 1 = I
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        
        # Start and end transitions
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))
    
    def forward(self, emissions, mask=None):
        """
        Compute log partition function using forward algorithm.
        
        Args:
            emissions: [batch, seq_len, num_tags] - emission scores
            mask: [batch, seq_len] - valid positions
        
        Returns:
            log_partition: [batch] - log of partition function
        """
        if mask is None:
            mask = torch.ones(emissions.size(0), emissions.size(1), device=emissions.device)
        
        return self._compute_log_partition(emissions, mask)
    
    def _compute_log_partition(self, emissions, mask):
        """Forward algorithm to compute partition function."""
        batch_size, seq_len, num_tags = emissions.size()
        
        # Initialize with start transitions
        score = self.start_transitions + emissions[:, 0]  # [batch, num_tags]
        
        for i in range(1, seq_len):
            # score[j] = log sum over all paths ending in tag j
            broadcast_score = score.unsqueeze(2)  # [batch, num_tags, 1]
            broadcast_emissions = emissions[:, i].unsqueeze(1)  # [batch, 1, num_tags]
            
            # Compute next scores
            next_score = broadcast_score + self.transitions + broadcast_emissions
            next_score = torch.logsumexp(next_score, dim=1)  # [batch, num_tags]
            
            # Apply mask
            mask_i = mask[:, i].unsqueeze(1)
            score = torch.where(mask_i.bool(), next_score, score)
        
        # Add end transitions
        score = score + self.end_transitions
        
        return torch.logsumexp(score, dim=1)
    
    def decode(self, emissions, mask=None):
        """
        Find optimal tag sequence using Viterbi algorithm.
        
        Args:
            emissions: [batch, seq_len, num_tags]
            mask: [batch, seq_len]
        
        Returns:
            best_paths: List[List[int]] - optimal tag sequences
        """
        if mask is None:
            mask = torch.ones(emissions.size(0), emissions.size(1), device=emissions.device)
        
        return self._viterbi_decode(emissions, mask)
    
    def _viterbi_decode(self, emissions, mask):
        """Viterbi algorithm for optimal sequence."""
        batch_size, seq_len, num_tags = emissions.size()
        
        # Initialize
        score = self.start_transitions + emissions[:, 0]
        history = []
        
        for i in range(1, seq_len):
            broadcast_score = score.unsqueeze(2)
            broadcast_emissions = emissions[:, i].unsqueeze(1)
            
            next_score = broadcast_score + self.transitions + broadcast_emissions
            
            # Find best previous tag for each current tag
            next_score, indices = next_score.max(dim=1)
            
            # Apply mask
            mask_i = mask[:, i].unsqueeze(1)
            score = torch.where(mask_i.bool(), next_score, score)
            history.append(indices)
        
        # Add end transitions
        score = score + self.end_transitions
        
        # Backtrack
        best_tags_list = []
        for b in range(batch_size):
            seq_len_b = int(mask[b].sum().item())
            
            # Find best final tag
            _, best_last_tag = score[b].max(dim=0)
            best_tags = [best_last_tag.item()]
            
            # Backtrack
            for indices in reversed(history[:seq_len_b - 1]):
                best_last_tag = indices[b, best_tags[-1]]
                best_tags.append(best_last_tag.item())
            
            best_tags.reverse()
            best_tags_list.append(best_tags)
        
        return best_tags_list


class BiLSTMCRF(nn.Module):
    """
    Bi-LSTM with CRF for morpheme boundary detection.
    
    Tags: B (0) = Begin morpheme, I (1) = Inside morpheme
    """
    
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=96, phoneme_dim=10, num_tags=2):
        super().__init__()
        
        # Character embedding
        self.char_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Bi-LSTM
        self.lstm = nn.LSTM(
            embed_dim + phoneme_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        
        # Emission layer
        self.emission = nn.Linear(hidden_dim * 2, num_tags)
        
        # CRF layer
        self.crf = CRF(num_tags)
        
        self.num_tags = num_tags
    
    def forward(self, char_ids, phoneme_feats, tags=None, mask=None):
        """
        Forward pass.
        
        If tags provided: compute negative log-likelihood
        Otherwise: return emissions
        """
        # Embed
        char_embeds = self.char_embed(char_ids)
        combined = torch.cat([char_embeds, phoneme_feats], dim=-1)
        
        # Bi-LSTM
        lstm_out, _ = self.lstm(combined)
        
        # Emissions
        emissions = self.emission(lstm_out)
        
        if tags is not None:
            # Compute CRF loss
            log_partition = self.crf(emissions, mask)
            
            # Compute score of gold sequence
            score = self._score_sequence(emissions, tags, mask)
            
            # Negative log-likelihood
            nll = log_partition - score
            return nll.mean()
        
        return emissions
    
    def _score_sequence(self, emissions, tags, mask):
        """Compute score of a tag sequence."""
        batch_size, seq_len, _ = emissions.size()
        
        score = self.crf.start_transitions[tags[:, 0]]
        score = score + emissions[torch.arange(batch_size), 0, tags[:, 0]]
        
        for i in range(1, seq_len):
            # Transition score
            score = score + self.crf.transitions[tags[:, i-1], tags[:, i]]
            # Emission score
            score = score + emissions[torch.arange(batch_size), i, tags[:, i]]
            
            # Apply mask
            score = torch.where(mask[:, i].bool(), score, score)
        
        # End transition
        last_tag_indices = mask.sum(dim=1).long() - 1
        score = score + self.crf.end_transitions[tags[torch.arange(batch_size), last_tag_indices]]
        
        return score
    
    def decode(self, char_ids, phoneme_feats, mask=None):
        """Decode optimal BIO sequence."""
        emissions = self.forward(char_ids, phoneme_feats, mask=mask)
        return self.crf.decode(emissions, mask)


class SandhiDataset(Dataset):
    """Dataset for sandhi split training with BIO tags."""
    
    def __init__(self, data_path: str, encoder: MalayalamPhonemeEncoder, max_len: int = 30):
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.words = data['words'] if isinstance(data, dict) else [d['word'] for d in data]
        self.splits = data['splits'] if isinstance(data, dict) else [d['splits'] for d in data]
        self.encoder = encoder
        self.max_len = max_len
    
    def __len__(self):
        return len(self.words)
    
    def __getitem__(self, idx):
        word = self.words[idx]
        splits = self.splits[idx] if isinstance(self.splits[idx], list) else self.splits[idx]['splits'] if isinstance(self.splits[idx], dict) else []
        
        # Encode
        char_ids, phonemes = self.encoder.encode(word)
        
        # Convert splits to BIO tags
        # B = 0 (begin), I = 1 (inside)
        bio_tags = []
        split_set = set(splits)
        
        for i in range(len(word)):
            if i == 0 or i in split_set:
                bio_tags.append(0)  # B
            else:
                bio_tags.append(1)  # I
        
        # Pad
        seq_len = min(len(char_ids), self.max_len)
        padded_chars = char_ids[:self.max_len] + [0] * (self.max_len - len(char_ids))
        padded_phones = phonemes[:self.max_len] + [[0]*10] * (self.max_len - len(phonemes))
        padded_tags = bio_tags[:self.max_len] + [0] * (self.max_len - len(bio_tags))
        mask = [1] * seq_len + [0] * (self.max_len - seq_len)
        
        return {
            'char_ids': torch.tensor(padded_chars, dtype=torch.long),
            'phoneme_feats': torch.tensor(padded_phones, dtype=torch.float),
            'tags': torch.tensor(padded_tags, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.float),
            'word': word,
            'seq_len': seq_len
        }


def train_bilstm_crf(epochs=50, batch_size=32, lr=0.001):
    """Train Bi-LSTM-CRF model."""
    
    print("="*60)
    print("TRAINING Bi-LSTM-CRF MODEL")
    print("="*60)
    
    # Initialize encoder
    encoder = MalayalamPhonemeEncoder()
    print(f"Vocabulary size: {encoder.vocab_size}")
    
    # Load dataset
    dataset = SandhiDataset('data/comprehensive_training_data.json', encoder)
    print(f"Dataset size: {len(dataset)}")
    
    # Split
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    print(f"Train: {train_size}, Val: {val_size}")
    
    # Initialize model
    model = BiLSTMCRF(encoder.vocab_size).to(DEVICE)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {params:,}")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Train
        model.train()
        total_loss = 0
        for batch in train_loader:
            char_ids = batch['char_ids'].to(DEVICE)
            phonemes = batch['phoneme_feats'].to(DEVICE)
            tags = batch['tags'].to(DEVICE)
            mask = batch['mask'].to(DEVICE)
            
            optimizer.zero_grad()
            loss = model(char_ids, phonemes, tags, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        train_loss = total_loss / len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                char_ids = batch['char_ids'].to(DEVICE)
                phonemes = batch['phoneme_feats'].to(DEVICE)
                tags = batch['tags'].to(DEVICE)
                mask = batch['mask'].to(DEVICE)
                
                # Loss
                loss = model(char_ids, phonemes, tags, mask)
                val_loss += loss.item()
                
                # Decode and compute accuracy
                predictions = model.decode(char_ids, phonemes, mask)
                
                for i, pred in enumerate(predictions):
                    seq_len = int(mask[i].sum().item())
                    gold = tags[i, :seq_len].tolist()
                    
                    for j, (p, g) in enumerate(zip(pred[:seq_len], gold)):
                        if p == g:
                            correct += 1
                        total += 1
        
        val_loss = val_loss / len(val_loader)
        val_acc = correct / total if total > 0 else 0
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs('models', exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'char2idx': encoder.char_to_idx,
                'config': {
                    'embed_dim': 32,
                    'hidden_dim': 96,
                    'vocab_size': encoder.vocab_size
                }
            }, 'models/bilstm_crf_model.pt')
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
    
    print("\nTraining complete!")
    return model, encoder


def evaluate_model(model, encoder, test_words):
    """Evaluate model on test words."""
    model.eval()
    
    print("\n" + "="*60)
    print("MODEL OUTPUT")
    print("="*60)
    
    for word in test_words:
        char_ids, phonemes = encoder.encode(word)
        max_len = 30
        
        padded_chars = torch.tensor([char_ids[:max_len] + [0]*(max_len-len(char_ids))]).to(DEVICE)
        padded_phones = torch.tensor([phonemes[:max_len] + [[0]*10]*(max_len-len(phonemes))], dtype=torch.float).to(DEVICE)
        mask = torch.tensor([[1]*min(len(char_ids), max_len) + [0]*(max_len-min(len(char_ids), max_len))]).to(DEVICE)
        
        with torch.no_grad():
            predictions = model.decode(padded_chars, padded_phones, mask)[0][:len(word)]
        
        # Extract components
        components = []
        start = 0
        for i, tag in enumerate(predictions):
            if tag == 0 and i > 0:  # B tag means new morpheme
                components.append(word[start:i])
                start = i
        components.append(word[start:])
        
        print(f"  {word} → {' + '.join(components)}")


if __name__ == "__main__":
    # Train
    model, encoder = train_bilstm_crf(epochs=50)
    
    # Test
    test_words = [
        'പഠിക്കുന്നു', 'വിദ്യാലയത്തിൽ', 'കേരളത്തിൽ', 'ഭാരതനാട്യം',
        'അധ്യാപികയുടെ', 'പ്രധാനമന്ത്രി', 'വീട്ടിൽ', 'പാലക്കാട്'
    ]
    evaluate_model(model, encoder, test_words)
