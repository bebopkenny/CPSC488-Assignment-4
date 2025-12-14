import os
import ast
import math
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Transformer Components
class PositionalEncoding(nn.Module):
    """
    Learnable positional embeddings.
    We use learnable embeddings instead of sinusoidal because:
    1. Simpler to implement
    2. Works well for fixed-length sequences
    3. Can learn task-specific position patterns
    """
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Learnable positional embeddings
        self.pos_embedding = nn.Embedding(max_len, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor with positional encoding added
        """
        batch_size, seq_len, _ = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        x = x + self.pos_embedding(positions)
        return self.dropout(x)


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism built from scratch.
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Linear projections
        Q = self.W_q(x)  # (batch_size, seq_len, d_model)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Shape: (batch_size, num_heads, seq_len, head_dim)
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # Shape: (batch_size, num_heads, seq_len, seq_len)
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attention_output = torch.matmul(attention_weights, V)
        # Shape: (batch_size, num_heads, seq_len, head_dim)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, seq_len, self.d_model)
        
        # Final linear projection
        output = self.W_o(attention_output)
        
        return output


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()  # GELU activation (modern choice)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Pre-LN: Apply LayerNorm before the sublayer
        # Self-attention with residual connection
        normalized = self.norm1(x)
        attention_output = self.self_attention(normalized, mask)
        x = x + self.dropout(attention_output)
        
        # Feed-forward with residual connection
        normalized = self.norm2(x)
        ff_output = self.feed_forward(normalized)
        x = x + self.dropout(ff_output)
        
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)  # Final layer norm
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int = 100,      # Dimension of input embeddings (Skip-gram)
        d_model: int = 64,         # Transformer hidden dimension
        num_heads: int = 2,        # Number of attention heads
        d_ff: int = 256,           # Feed-forward dimension
        num_layers: int = 2,       # Number of encoder layers
        num_classes: int = 1,      # Binary classification (sigmoid output)
        max_len: int = 10,         # Maximum sequence length
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Project input embeddings to transformer dimension
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # Transformer encoder
        self.encoder = TransformerEncoder(num_layers, d_model, num_heads, d_ff, dropout)
        
        # Classification head (mean pooling + linear)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Project to transformer dimension
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Pass through transformer encoder
        x = self.encoder(x)  # (batch_size, seq_len, d_model)
        
        # Mean pooling over sequence dimension
        x = x.mean(dim=1)  # (batch_size, d_model)
        
        # Classification
        logits = self.classifier(x)  # (batch_size, num_classes)
        
        return logits


# Data Loading and Training
def load_vectorized_dataset(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    
    df = pd.read_csv(csv_path, converters={"news_vector": ast.literal_eval})
    
    X = np.array(df["news_vector"].tolist(), dtype=np.float32)
    
    # Binary classification: positive impact vs non-positive
    impact_scores = df["impact_score"].astype(float).values
    y = (impact_scores > 0).astype(np.float32)
    
    return X, y


def train_transformer_classifier(
    X: np.ndarray,
    y: np.ndarray,
    num_epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    device: str = None,
    verbose: bool = True
) -> Tuple[TransformerClassifier, float, str]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if verbose:
        print(f"Training on device: {device}")
        print(f"Dataset size: {len(X)} samples")
        print(f"Positive class ratio: {y.mean():.2%}")
    
    # Reshape X to (batch, seq_len=1, features) for transformer
    X_seq = X.reshape(X.shape[0], 1, X.shape[1])
    
    # Train/test split with stratification
    unique_labels = np.unique(y)
    stratify = y if len(unique_labels) > 1 else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y, test_size=0.2, random_state=42, stratify=stratify
    )
    
    if verbose:
        print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Create data loaders
    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train).unsqueeze(1)
    )
    test_dataset = TensorDataset(
        torch.from_numpy(X_test),
        torch.from_numpy(y_test).unsqueeze(1)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = TransformerClassifier(
        input_dim=X.shape[1],
        d_model=64,
        num_heads=2,
        d_ff=256,
        num_layers=2,
        num_classes=1,
        max_len=10,
        dropout=0.1
    ).to(device)
    
    if verbose:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Training loop
    best_acc = 0.0
    best_model_state = None
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            running_loss += loss.item() * batch_X.size(0)
        
        scheduler.step()
        epoch_loss = running_loss / len(train_dataset)
        
        # Evaluate on test set
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                logits = model(batch_X)
                predictions = (torch.sigmoid(logits) >= 0.5).float()
                correct += (predictions == batch_y).sum().item()
                total += batch_y.size(0)
        
        epoch_acc = correct / total
        
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_state = model.state_dict().copy()
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}, Test Acc: {epoch_acc:.4f}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Final evaluation
    model.eval()
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            logits = model(batch_X)
            all_logits.append(logits.cpu())
            all_labels.append(batch_y)
    
    logits = torch.cat(all_logits, dim=0).squeeze(1)
    y_true = torch.cat(all_labels, dim=0).squeeze(1).numpy()
    y_prob = torch.sigmoid(logits).numpy()
    y_pred = (y_prob >= 0.5).astype(int)
    
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4, zero_division=0)
    
    if verbose:
        print("\n" + "="*60)
        print("FINAL EVALUATION")
        print("="*60)
        print(f"Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
        print("\nClassification Report:")
        print(report)
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_true, y_pred))
    
    return model, acc, report


def compare_with_mlp(transformer_acc: float, mlp_acc: float = None):
    print("\n" + "="*60)
    print("COMPARISON: Transformer vs MLP (Assignment 3)")
    print("="*60)
    
    print(f"\nTransformer Accuracy: {transformer_acc:.4f} ({transformer_acc*100:.2f}%)")
    
    if mlp_acc is not None:
        print(f"MLP Accuracy (Assignment 3): {mlp_acc:.4f} ({mlp_acc*100:.2f}%)")
        
        diff = transformer_acc - mlp_acc
        if diff > 0:
            print(f"\n✓ Transformer improves by {diff*100:.2f}%")
        elif diff < 0:
            print(f"\n✗ MLP outperforms by {abs(diff)*100:.2f}%")
        else:
            print("\n= Performance is identical")
    else:
        print("MLP Accuracy: Not provided")
    
    print("\nAnalysis:")
    print("-" * 40)
    print("""
The Transformer and MLP may show similar performance because:

1. SMALL DATASET: With only ~108 samples, the Transformer's 
   capacity to learn complex patterns is limited by data scarcity.

2. PRE-AGGREGATED EMBEDDINGS: The input is already a single 
   100-dim vector per document (from Skip-gram averaging). 
   This means sequence modeling capabilities are underutilized.

3. BAG-OF-WORDS LIMITATION: Skip-gram embeddings via mean 
   pooling lose word order information, negating Transformer's 
   key advantage (modeling sequential dependencies).

For improved Transformer performance, consider:
- Using token-level embeddings (not averaged)
- Training on raw text with learned embeddings
- Collecting more training data
""")


def main():    
    print("="*60)
    print("TRANSFORMER CLASSIFIER FOR SENTIMENT ANALYSIS")
    print("="*60)
    print()
    
    # Paths
    dataset_path = "datasets/vectorized_news_skipgram_embeddings.csv"
    model_save_path = "models/transformer_classifier.pth"
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        print("Please ensure vectorized_news_skipgram_embeddings.csv is in the datasets folder.")
        return
    
    # Load data
    print("Loading dataset...")
    X, y = load_vectorized_dataset(dataset_path)
    print(f"Loaded {len(X)} samples with {X.shape[1]}-dimensional embeddings")
    print(f"Label distribution: Positive={y.sum():.0f}, Negative={len(y)-y.sum():.0f}")
    print()
    
    # Train Transformer
    print("Training Transformer Classifier...")
    print("-" * 40)
    model, transformer_acc, report = train_transformer_classifier(
        X, y,
        num_epochs=50,
        batch_size=16,
        learning_rate=1e-3,
        verbose=True
    )
    
    # Save model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': X.shape[1],
        'accuracy': transformer_acc
    }, model_save_path)
    print(f"\nModel saved to {model_save_path}")
    
    # Compare with MLP baseline
    mlp_acc = None  # Set to your Assignment 3 accuracy if known
    compare_with_mlp(transformer_acc, mlp_acc)
    
    return model, transformer_acc, report


if __name__ == "__main__":
    main()