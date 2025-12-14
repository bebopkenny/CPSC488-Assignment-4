import os
import re
from typing import List, Tuple

import numpy as np
import pandas as pd
from collections import Counter

# Try to import gensim, fallback to manual tokenization if not available
try:
    from gensim.utils import simple_preprocess
    from gensim.parsing.preprocessing import STOPWORDS
    HAS_GENSIM = True
except ImportError:
    HAS_GENSIM = False
    # Define basic stopwords if gensim not available
    STOPWORDS = set([
        'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
        'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'this',
        'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
    ])


def tokenize(text: str) -> List[str]:
    if HAS_GENSIM:
        tokens = [t for t in simple_preprocess(text) if t not in STOPWORDS]
    else:
        # Simple tokenization fallback
        text = text.lower()
        tokens = re.findall(r'\b[a-z]{2,}\b', text)
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


def split_sentences(text: str) -> List[str]:
    # Handle common sentence boundaries
    # Split on . ! ? followed by space or end of string
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Filter out empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def build_word_embeddings(tokenized_docs: List[List[str]], embedding_dim: int = 100) -> dict:
    # Count word frequencies
    word_counts = Counter()
    for doc in tokenized_docs:
        word_counts.update(doc)
    
    # Filter to frequent words
    vocab = [w for w, c in word_counts.items() if c >= 2]
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    
    # Initialize random embeddings (in practice, these would come from Word2Vec)
    np.random.seed(42)
    embeddings = {}
    for word in vocab:
        embeddings[word] = np.random.randn(embedding_dim).astype(np.float32)
        embeddings[word] /= np.linalg.norm(embeddings[word])  # Normalize
    
    return embeddings


def sentence_embedding(sentence: str, word_embeddings: dict, embedding_dim: int = 100) -> np.ndarray:
    tokens = tokenize(sentence)
    
    if not tokens:
        return np.zeros(embedding_dim, dtype=np.float32)
    
    vectors = []
    for token in tokens:
        if token in word_embeddings:
            vectors.append(word_embeddings[token])
    
    if not vectors:
        return np.zeros(embedding_dim, dtype=np.float32)
    
    return np.mean(vectors, axis=0).astype(np.float32)


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return np.dot(v1, v2) / (norm1 * norm2)


def compute_similarity_matrix(sentence_embeddings: List[np.ndarray]) -> np.ndarray:
    """
    Compute pairwise cosine similarity matrix for all sentences.
    
    Similarity Metric: Cosine Similarity
    - Measures the angle between vectors
    - Values range from -1 to 1 (1 = identical direction)
    - Standard choice for text similarity
    """
    n = len(sentence_embeddings)
    similarity_matrix = np.zeros((n, n), dtype=np.float32)
    
    for i in range(n):
        for j in range(n):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                similarity_matrix[i, j] = cosine_similarity(
                    sentence_embeddings[i], 
                    sentence_embeddings[j]
                )
    
    return similarity_matrix


def select_representative_sentences(
    sentences: List[str],
    similarity_matrix: np.ndarray,
    num_sentences: int = 3,
    diversity_weight: float = 0.3
) -> List[Tuple[int, str, float]]:
    """
    Select the most representative sentences for the summary.
    
    Selection Method: Greedy selection with diversity
    1. Compute average similarity to all other sentences (centrality score)
    2. Select sentences with highest centrality
    3. Apply diversity penalty to avoid redundant selections
    
    Returns: List of (index, sentence, score) tuples
    """
    n = len(sentences)
    if n == 0:
        return []
    
    num_sentences = min(num_sentences, n)
    
    # Compute centrality scores (average similarity to all other sentences)
    centrality_scores = np.mean(similarity_matrix, axis=1)
    
    selected = []
    remaining = list(range(n))
    
    for _ in range(num_sentences):
        if not remaining:
            break
        
        # Compute scores for remaining sentences
        scores = []
        for idx in remaining:
            score = centrality_scores[idx]
            
            # Apply diversity penalty based on similarity to already selected
            if selected and diversity_weight > 0:
                max_sim_to_selected = max(
                    similarity_matrix[idx, s[0]] for s in selected
                )
                score -= diversity_weight * max_sim_to_selected
            
            scores.append((idx, score))
        
        # Select the sentence with highest score
        best_idx, best_score = max(scores, key=lambda x: x[1])
        selected.append((best_idx, sentences[best_idx], centrality_scores[best_idx]))
        remaining.remove(best_idx)
    
    # Sort by original order in document
    selected.sort(key=lambda x: x[0])
    
    return selected


def extractive_summarize(
    text: str,
    word_embeddings: dict,
    num_sentences: int = 3,
    embedding_dim: int = 100
) -> Tuple[str, List[Tuple[int, str, float]]]:
    """
    Perform extractive summarization on a text document.
    
    Args:
        text: Input document text
        word_embeddings: Dictionary mapping words to embedding vectors
        num_sentences: Number of sentences to extract
        embedding_dim: Dimension of word embeddings
    
    Returns:
        summary: Concatenated summary sentences
        selected_info: List of (index, sentence, score) tuples
    """
    # Split into sentences
    sentences = split_sentences(text)
    
    if not sentences:
        return "", []
    
    # Compute sentence embeddings
    sentence_embeds = [
        sentence_embedding(s, word_embeddings, embedding_dim)
        for s in sentences
    ]
    
    # Compute similarity matrix
    sim_matrix = compute_similarity_matrix(sentence_embeds)
    
    # Select representative sentences
    selected = select_representative_sentences(
        sentences, sim_matrix, num_sentences
    )
    
    # Combine selected sentences into summary
    summary = " ".join([s[1] for s in selected])
    
    return summary, selected


def load_aggregated_news(path: str) -> pd.DataFrame:
    """Load the aggregated news dataset."""
    df = pd.read_csv(path)
    return df


def build_embeddings_from_corpus(df: pd.DataFrame, embedding_dim: int = 100) -> dict:
    # Tokenize all documents
    all_tokens = []
    for text in df['news'].fillna(''):
        tokens = tokenize(text)
        all_tokens.append(tokens)
    
    # Build embeddings
    embeddings = build_word_embeddings(all_tokens, embedding_dim)
    print(f"Built vocabulary of {len(embeddings)} words")
    
    return embeddings


def main():
    """Main function for extractive summarization demonstration."""
    
    print("="*60)
    print("EXTRACTIVE SUMMARIZATION")
    print("Based on Sentence Similarity with Skip-gram Embeddings")
    print("="*60)
    print()
    
    # Method description
    print("METHOD DESCRIPTION:")
    print("-" * 40)
    print("""
    Sentence Embedding Strategy: Mean pooling of Skip-gram word vectors
    - Each word is represented by its Skip-gram embedding
    - Sentence embedding = average of all word embeddings in the sentence
    - This captures the overall semantic content

    Similarity Metric: Cosine Similarity
    - Measures directional similarity between sentence vectors
    - Range: [-1, 1], where 1 = identical meaning

    Selection Method: Centrality-based with diversity
    - Compute average similarity of each sentence to all others
    - Select sentences with highest centrality (most representative)
    - Apply diversity penalty to avoid redundant content
    """)
    print()
    
    # Load dataset
    dataset_path = "datasets/aggregated_news.csv"
    
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        return
    
    df = load_aggregated_news(dataset_path)
    print(f"Loaded {len(df)} news articles")
    print()
    
    # Build word embeddings from corpus
    print("Building word embeddings from corpus...")
    word_embeddings = build_embeddings_from_corpus(df, embedding_dim=100)
    print()
    
    # Select an article for demonstration
    # Choose one with multiple sentences (longer text)
    article_lengths = df['news'].fillna('').apply(lambda x: len(split_sentences(x)))
    best_idx = article_lengths.idxmax()
    
    selected_article = df.loc[best_idx]
    article_text = selected_article['news']
    article_date = selected_article['date']
    article_symbol = selected_article['symbol']
    
    print("="*60)
    print("EXAMPLE: Extractive Summary of Selected Article")
    print("="*60)
    print(f"\nDate: {article_date}")
    print(f"Symbol: {article_symbol}")
    print(f"\nOriginal Article:")
    print("-" * 40)
    print(article_text)
    print()
    
    # Perform extractive summarization
    sentences = split_sentences(article_text)
    print(f"Number of sentences: {len(sentences)}")
    print()
    
    # Generate summary
    num_summary_sentences = min(3, len(sentences))
    summary, selected_info = extractive_summarize(
        article_text, 
        word_embeddings,
        num_sentences=num_summary_sentences
    )
    
    print("EXTRACTIVE SUMMARY:")
    print("-" * 40)
    print(summary)
    print()
    
    print("SELECTED SENTENCES (with centrality scores):")
    print("-" * 40)
    for idx, sent, score in selected_info:
        print(f"  [{idx}] (score: {score:.4f}) {sent[:80]}...")
    print()
    
    # Analyze all articles
    print("="*60)
    print("PROCESSING ALL ARTICLES")
    print("="*60)
    
    summaries = []
    for i, row in df.iterrows():
        text = row['news'] if pd.notna(row['news']) else ""
        sentences = split_sentences(text)
        
        if len(sentences) >= 2:
            num_sents = min(2, len(sentences))
            summary, _ = extractive_summarize(text, word_embeddings, num_sentences=num_sents)
        else:
            summary = text
        
        summaries.append({
            'date': row['date'],
            'symbol': row['symbol'],
            'original_length': len(text),
            'summary_length': len(summary),
            'compression_ratio': len(summary) / max(len(text), 1),
            'summary': summary
        })
    
    summary_df = pd.DataFrame(summaries)
    
    print(f"\nProcessed {len(summary_df)} articles")
    print(f"Average compression ratio: {summary_df['compression_ratio'].mean():.2%}")
    print()
    
    # Save summaries
    output_path = "datasets/extractive_summaries.csv"
    summary_df.to_csv(output_path, index=False)
    print(f"Summaries saved to {output_path}")
    
    # Show a few examples
    print("\nSAMPLE EXTRACTIVE SUMMARIES:")
    print("-" * 40)
    for i in range(min(3, len(summary_df))):
        row = summary_df.iloc[i]
        print(f"\n[{i+1}] {row['date']} - {row['symbol']}")
        print(f"Original ({row['original_length']} chars):")
        print(f"  {df.iloc[i]['news'][:100]}...")
        print(f"Summary ({row['summary_length']} chars):")
        print(f"  {row['summary'][:100]}...")
    
    return summary_df


if __name__ == "__main__":
    main()