"""
Assignment 4 - Part 2.2: Abstractive Summarization using Transformer Encoder-Decoder

This implements an article-to-headline generation model using PyTorch's nn.Transformer.
The model learns to generate headlines from article content.

Architectural Choices:
- Positional Embedding: Sinusoidal (classic choice, no additional parameters)
- Encoder: 2 heads, 256 FFN dim, 2 layers, Pre-LN
- Decoder: 2 heads, 256 FFN dim, 2 layers, Pre-LN, with autoregressive masking
- Generation: Greedy decoding (simple, deterministic)
"""

import os
import re
import math
from typing import List, Tuple, Dict
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


# =============================================================================
# Tokenization and Vocabulary
# =============================================================================

class SimpleTokenizer:
    """Simple word-level tokenizer with special tokens."""
    
    PAD_TOKEN = "<PAD>"
    SOS_TOKEN = "<SOS>"
    EOS_TOKEN = "<EOS>"
    UNK_TOKEN = "<UNK>"
    
    def __init__(self, min_freq: int = 1):
        self.min_freq = min_freq
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 0
        
    def build_vocab(self, texts: List[str]):
        """Build vocabulary from list of texts."""
        # Count word frequencies
        word_counts = Counter()
        for text in texts:
            tokens = self._tokenize(text)
            word_counts.update(tokens)
        
        # Add special tokens
        special_tokens = [self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN]
        self.word2idx = {token: idx for idx, token in enumerate(special_tokens)}
        
        # Add words meeting frequency threshold
        for word, count in word_counts.items():
            if count >= self.min_freq and word not in self.word2idx:
                self.word2idx[word] = len(self.word2idx)
        
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)
        
        print(f"Vocabulary size: {self.vocab_size}")
        
    def _tokenize(self, text: str) -> List[str]:
        """Basic tokenization."""
        text = text.lower()
        # Keep alphanumeric and some punctuation
        tokens = re.findall(r'\b\w+\b|[.,!?;]', text)
        return tokens
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Convert text to token indices."""
        tokens = self._tokenize(text)
        indices = [self.word2idx.get(t, self.word2idx[self.UNK_TOKEN]) for t in tokens]
        
        if add_special_tokens:
            indices = [self.word2idx[self.SOS_TOKEN]] + indices + [self.word2idx[self.EOS_TOKEN]]
        
        return indices
    
    def decode(self, indices: List[int], skip_special_tokens: bool = True) -> str:
        """Convert token indices back to text."""
        special = {self.word2idx[self.PAD_TOKEN], 
                   self.word2idx[self.SOS_TOKEN], 
                   self.word2idx[self.EOS_TOKEN]}
        
        tokens = []
        for idx in indices:
            if skip_special_tokens and idx in special:
                continue
            tokens.append(self.idx2word.get(idx, self.UNK_TOKEN))
        
        return " ".join(tokens)
    
    @property
    def pad_idx(self) -> int:
        return self.word2idx[self.PAD_TOKEN]
    
    @property
    def sos_idx(self) -> int:
        return self.word2idx[self.SOS_TOKEN]
    
    @property
    def eos_idx(self) -> int:
        return self.word2idx[self.EOS_TOKEN]


# =============================================================================
# Dataset
# =============================================================================

class SummarizationDataset(Dataset):
    """Dataset for article-headline pairs."""
    
    def __init__(self, articles: List[str], headlines: List[str], tokenizer: SimpleTokenizer,
                 max_article_len: int = 128, max_headline_len: int = 32):
        self.articles = articles
        self.headlines = headlines
        self.tokenizer = tokenizer
        self.max_article_len = max_article_len
        self.max_headline_len = max_headline_len
        
    def __len__(self):
        return len(self.articles)
    
    def __getitem__(self, idx):
        article = self.articles[idx]
        headline = self.headlines[idx]
        
        # Encode
        src = self.tokenizer.encode(article, add_special_tokens=True)
        tgt = self.tokenizer.encode(headline, add_special_tokens=True)
        
        # Truncate if necessary
        src = src[:self.max_article_len]
        tgt = tgt[:self.max_headline_len]
        
        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)


def collate_fn(batch, pad_idx):
    """Collate function for DataLoader with padding."""
    src_batch, tgt_batch = zip(*batch)
    
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=pad_idx)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=pad_idx)
    
    return src_padded, tgt_padded


# =============================================================================
# Positional Encoding (Sinusoidal)
# =============================================================================

class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding (Vaswani et al., 2017).
    
    Uses sine and cosine functions of different frequencies.
    Advantages:
    - No learnable parameters
    - Can extrapolate to longer sequences
    - Classic, well-understood approach
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# =============================================================================
# Transformer Encoder-Decoder for Summarization
# =============================================================================

class AbstractiveSummarizer(nn.Module):
    """
    Transformer Encoder-Decoder for Abstractive Summarization.
    
    Architecture:
    - Embedding layer (shared between encoder and decoder)
    - Sinusoidal positional encoding
    - PyTorch nn.Transformer (encoder-decoder)
    - Linear output projection
    
    Key Design Choices:
    - Positional Encoding: Sinusoidal (no extra parameters)
    - Encoder: 2 heads, 256 FFN, 2 layers
    - Decoder: 2 heads, 256 FFN, 2 layers with causal masking
    - Generation: Greedy decoding
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        nhead: int = 2,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_len: int = 256,
        pad_idx: int = 0
    ):
        super().__init__()
        
        self.d_model = d_model
        self.pad_idx = pad_idx
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        
        # Positional encoding
        self.pos_encoder = SinusoidalPositionalEncoding(d_model, max_len, dropout)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # Use batch_first=True for easier handling
        )
        
        # Output projection
        self.fc_out = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """
        Generate causal mask for autoregressive decoding.
        
        Autoregressive Masking Explanation:
        - The decoder must not attend to future tokens during training
        - This mask sets future positions to -inf, making their attention weights 0
        - Ensures the model only uses past context when predicting each token
        """
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def create_padding_mask(self, seq: torch.Tensor) -> torch.Tensor:
        """Create mask for padding tokens."""
        return (seq == self.pad_idx)
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for training.
        
        Args:
            src: Source sequence (batch_size, src_len)
            tgt: Target sequence (batch_size, tgt_len)
        
        Returns:
            Output logits (batch_size, tgt_len, vocab_size)
        """
        # Create masks
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1), tgt.device)
        src_padding_mask = self.create_padding_mask(src)
        tgt_padding_mask = self.create_padding_mask(tgt)
        
        # Embed and add positional encoding
        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)
        
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb)
        
        # Transformer forward
        output = self.transformer(
            src_emb, tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )
        
        # Project to vocabulary
        logits = self.fc_out(output)
        
        return logits
    
    def generate(
        self, 
        src: torch.Tensor, 
        max_len: int = 32,
        sos_idx: int = 1,
        eos_idx: int = 2
    ) -> torch.Tensor:
        """
        Generate headline using greedy decoding.
        
        Generation Strategy: Greedy Decoding
        - At each step, select the token with highest probability
        - Simple and deterministic
        - Fast inference
        
        Alternative strategies (not implemented):
        - Beam Search: Keep top-k candidates, better quality
        - Top-k/Top-p Sampling: More diverse outputs
        """
        self.eval()
        device = src.device
        batch_size = src.size(0)
        
        # Encode source
        src_padding_mask = self.create_padding_mask(src)
        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)
        
        # Get encoder output
        memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_padding_mask)
        
        # Initialize decoder input with SOS token
        tgt = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=device)
        
        for _ in range(max_len - 1):
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1), device)
            tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
            tgt_emb = self.pos_encoder(tgt_emb)
            
            # Decode
            output = self.transformer.decoder(
                tgt_emb, memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_padding_mask
            )
            
            # Get next token (greedy)
            logits = self.fc_out(output[:, -1, :])
            next_token = logits.argmax(dim=-1, keepdim=True)
            
            tgt = torch.cat([tgt, next_token], dim=1)
            
            # Stop if all sequences have generated EOS
            if (next_token == eos_idx).all():
                break
        
        return tgt


# =============================================================================
# Training
# =============================================================================

def train_summarizer(
    model: AbstractiveSummarizer,
    train_loader: DataLoader,
    num_epochs: int = 30,
    learning_rate: float = 1e-3,
    device: str = "cpu",
    verbose: bool = True
) -> List[float]:
    """Train the abstractive summarization model."""
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=model.pad_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    losses = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for src, tgt in train_loader:
            src = src.to(device)
            tgt = tgt.to(device)
            
            # Teacher forcing: use target as input, predict next token
            tgt_input = tgt[:, :-1]  # Remove last token
            tgt_output = tgt[:, 1:]  # Remove first token (SOS)
            
            optimizer.zero_grad()
            
            logits = model(src, tgt_input)
            
            # Reshape for loss computation
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt_output.reshape(-1)
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        
        if verbose and (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    return losses


# =============================================================================
# Main Function
# =============================================================================

def create_article_headline_pairs(df: pd.DataFrame, num_pairs: int = 50) -> Tuple[List[str], List[str]]:
    """
    Create article-headline pairs from the dataset.
    
    For this assignment, we treat the concatenated news as "articles"
    and extract a shorter version as the "headline".
    """
    articles = []
    headlines = []
    
    for _, row in df.iterrows():
        news = row['news'] if pd.notna(row['news']) else ""
        
        if len(news) < 20:
            continue
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', news)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) >= 2:
            # Use full text as article, first sentence as headline
            article = news
            headline = sentences[0][:100]  # Limit headline length
            
            articles.append(article)
            headlines.append(headline)
        
        if len(articles) >= num_pairs:
            break
    
    return articles, headlines


def main():
    """Main function for abstractive summarization."""
    
    print("="*60)
    print("ABSTRACTIVE SUMMARIZATION")
    print("Using Transformer Encoder-Decoder (nn.Transformer)")
    print("="*60)
    print()
    
    # Architecture description
    print("MODEL ARCHITECTURE:")
    print("-" * 40)
    print("""
    Positional Encoding: Sinusoidal
    - Uses sin/cos functions of different frequencies
    - No learnable parameters
    - Classic approach from "Attention Is All You Need"

    Encoder Block:
    - Number of heads: 2
    - Feedforward dimension: 256
    - Number of layers: 2
    - Layer norm: Pre-LN (applied before attention/FFN)

    Decoder Block:
    - Number of heads: 2
    - Feedforward dimension: 256
    - Number of layers: 2
    - Layer norm: Pre-LN
    - Autoregressive masking: Causal mask prevents attending to future

    Generation Strategy: Greedy Decoding
    - Select highest probability token at each step
    - Simple and deterministic
    """)
    print()
    
    # Load dataset
    dataset_path = "datasets/aggregated_news.csv"
    
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        return
    
    df = pd.read_csv(dataset_path)
    print(f"Loaded {len(df)} records")
    
    # Create article-headline pairs
    print("\nCreating article-headline pairs...")
    articles, headlines = create_article_headline_pairs(df, num_pairs=50)
    print(f"Created {len(articles)} pairs")
    
    if len(articles) < 10:
        print("Warning: Not enough data for meaningful training")
        return
    
    # Build tokenizer
    print("\nBuilding vocabulary...")
    tokenizer = SimpleTokenizer(min_freq=1)
    tokenizer.build_vocab(articles + headlines)
    
    # Create dataset and dataloader
    dataset = SummarizationDataset(
        articles, headlines, tokenizer,
        max_article_len=128,
        max_headline_len=32
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_idx)
    )
    
    # Initialize model
    print("\nInitializing model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    model = AbstractiveSummarizer(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        nhead=2,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        max_len=256,
        pad_idx=tokenizer.pad_idx
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Train model
    print("\nTraining model...")
    print("-" * 40)
    losses = train_summarizer(
        model, dataloader,
        num_epochs=30,
        learning_rate=1e-3,
        device=device,
        verbose=True
    )
    
    # Save model
    model_path = "models/abstractive_summarizer.pth"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': tokenizer.vocab_size,
        'word2idx': tokenizer.word2idx,
        'idx2word': tokenizer.idx2word
    }, model_path)
    print(f"\nModel saved to {model_path}")
    
    # Generate examples
    print("\n" + "="*60)
    print("GENERATION EXAMPLES")
    print("="*60)
    
    model.eval()
    
    for i in range(min(5, len(articles))):
        article = articles[i]
        actual_headline = headlines[i]
        
        # Encode article
        src = torch.tensor([tokenizer.encode(article)], dtype=torch.long, device=device)
        
        # Generate headline
        with torch.no_grad():
            generated = model.generate(
                src, 
                max_len=32,
                sos_idx=tokenizer.sos_idx,
                eos_idx=tokenizer.eos_idx
            )
        
        generated_headline = tokenizer.decode(generated[0].tolist())
        
        print(f"\n--- Example {i+1} ---")
        print(f"Article: {article[:150]}...")
        print(f"Actual headline: {actual_headline}")
        print(f"Generated headline: {generated_headline}")
    
    # Comparison with extractive summary
    print("\n" + "="*60)
    print("COMPARISON: Extractive vs Abstractive")
    print("="*60)
    print("""
    For a fair comparison, we compare both methods on the same article.
    
    Extractive Summary:
    - Selects existing sentences from the article
    - Preserves original wording
    - Limited to content in the source
    
    Abstractive Summary:
    - Generates new text
    - Can paraphrase and condense
    - May introduce novel phrasing
    
    Evaluation Criteria:
    1. Similarity to headline: How close is the summary to the actual headline?
    2. Accuracy: Does it capture the main point?
    3. Fluency: Is the generated text grammatical?
    """)
    
    return model, tokenizer, losses


if __name__ == "__main__":
    main()