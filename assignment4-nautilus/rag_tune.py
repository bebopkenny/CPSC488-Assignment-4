"""
Assignment 4 - Part 4: Retrieval-Augmented Generation (RAG) with TinyLlama
For running on Nautilus HyperCluster with GPU

This script implements a RAG pipeline:
1. Chunk the CS handbook
2. Embed chunks using sentence transformers
3. Store in FAISS vector database
4. Retrieve relevant chunks for queries
5. Generate answers using TinyLlama

Usage:
    python rag_tune.py --build          # Build the RAG index
    python rag_tune.py --query "..."    # Query the system
    python rag_tune.py --evaluate       # Evaluate on test questions
    python rag_tune.py --info           # Show configuration info
"""

import os
import json
import argparse
from typing import List, Dict, Tuple
from dataclasses import dataclass
import numpy as np

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("Warning: sentence-transformers not installed, using fallback embeddings")

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    print("Warning: faiss not installed, using fallback vector store")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class RAGConfig:
    """
    RAG System Configuration.
    
    Chunk Size: 500 characters
    - Large enough to contain complete information
    - Small enough for precise retrieval
    
    Chunk Overlap: 100 characters
    - Prevents information loss at boundaries
    - Helps with cross-chunk queries
    
    Vector Database: FAISS
    - Efficient similarity search
    - Scales to large document collections
    - Supports approximate nearest neighbor search
    
    Retrieval Strategy:
    - K value: 3 (retrieve top 3 chunks)
    - Scoring: cosine similarity
    - Returns chunks most similar to query
    """
    chunk_size: int = 500
    chunk_overlap: int = 100
    retrieval_k: int = 3
    embedding_model: str = "all-MiniLM-L6-v2"
    llm_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    use_4bit: bool = True
    temperature: float = 0.3
    top_k: int = 50
    top_p: float = 0.9
    max_new_tokens: int = 256

OUTPUT_DIR = "models/rag"


# =============================================================================
# Handbook Content
# =============================================================================

def get_handbook_content() -> str:
    """
    Get the CS handbook content.
    In a real implementation, this would read from cpsc-handbook-2022.pdf
    """
    return """
The requirements for the B.S. in Computer Science are detailed in the University Catalog. The requirements fit into 6 categories:

1. Lower-Division Core: 100/200-level CPSC courses covering computer programming, data structures, and cybersecurity fundamentals. The first three courses are CPSC 120 (Introduction to Programming), CPSC 121 (Object-Oriented Programming), and CPSC 131 (Data Structures). These courses must be taken in sequence. Additional lower-division courses include CPSC 223J/P/W (Programming in a specific language), CPSC 240 (Computer Organization and Assembly Language), and CPSC 254 (Software Development with Open Source Systems).

2. Mathematics Requirements: MATH courses including MATH 150A (Calculus I), MATH 150B (Calculus II), MATH 270A (Linear Algebra), MATH 270B (Discrete Mathematics), and MATH 338 (Statistics for Science and Engineering). These lay the foundation for CS theory and practice. A grade of C or better is required in MATH 150A and MATH 338.

3. Science and Mathematics Electives: At least 12 units of natural science and/or mathematics courses. PHYS 225, 225L, 226, 226L, and MATH 250A provide a strong foundation for students interested in graduate school.

4. Upper-Division Core: 300/400-level CPSC courses including CPSC 315 (Professional Ethics in Computing), CPSC 323 (Compilers and Languages), CPSC 332 (File Structures and Database Systems), CPSC 335 (Algorithm Engineering), CPSC 351 (Operating Systems Concepts), CPSC 362 (Foundations of Software Engineering), CPSC 471 (Computer Communications), CPSC 481 (Artificial Intelligence), CPSC 490 (Undergraduate Seminar), and CPSC 491 (Senior Capstone Project).

5. Major Electives: 15 units of elective courses (typically 5 courses) supporting your interests and career goals. Options include CPSC 349 (Web Front-End Engineering), CPSC 375 (Introduction to Data Science and Big Data), CPSC 386 (Introduction to Game Design and Production), CPSC 411 (Mobile Device Application Programming), CPSC 431 (Database and Applications), CPSC 449 (Web Back-End Engineering), CPSC 452 (Cryptography), CPSC 454 (Cloud Computing and Security), CPSC 455 (Web Security), CPSC 456 (Network Security Fundamentals), CPSC 462 (Software Design), CPSC 463 (Software Testing), CPSC 464 (Software Architecture), CPSC 474 (Parallel and Distributed Computing), CPSC 479 (Introduction to High Performance Computing), CPSC 483 (Introduction to Machine Learning), CPSC 484 (Principles of Computer Graphics), CPSC 485 (Computational Bioinformatics), and CPSC 486 (Game Programming).

6. General Education (GE): Approximately 48 units including oral and written communication, critical thinking, mathematics and quantitative reasoning, physical science, life science, arts, humanities, social sciences, and lifelong learning. A grade of C or better is required in all GE courses.

Senior Capstone Project (CPSC 491):
CPSC 491 has the longest prerequisite chain in the major: CPSC 120 (Introduction to Programming) → CPSC 121 (Object-Oriented Programming) → CPSC 131 (Data Structures) → CPSC 362 (Foundations of Software Engineering) → CPSC 490 (Undergraduate Seminar) → CPSC 491 (Senior Capstone Project). Prerequisites include CPSC 362 and CPSC 490. Students work on a significant software project demonstrating comprehensive skills acquired in the program. Must complete all prerequisite chains before enrollment. The project is typically completed in teams and students must present their project at the end of the semester.

CPSC 362 Foundations of Software Engineering:
CPSC 362 covers software development lifecycle, requirements analysis, design patterns and principles, testing methodologies, team collaboration, version control (Git), and Agile development practices. It is a required prerequisite for CPSC 491 (Senior Capstone Project) and should be taken in junior year. The prerequisite is CPSC 131 (Data Structures).

Graduation Requirements:
To graduate with a B.S. in Computer Science from CSUF, students must complete all 6 requirement categories, maintain a minimum 2.0 GPA in major courses, achieve a minimum C grade in GE courses including MATH 150A and MATH 338, and complete at least 120 total units. Students should apply for graduation after completing 90 units via Titan Online. Grade forgiveness is available for the first 16 units of repeated courses.

The Bachelor of Science in Computer Science at CSUF is accredited by ABET (Accreditation Board for Engineering and Technology). ABET accreditation ensures the program meets quality standards, is required for some graduate programs, is important for federal jobs requiring CS degrees, and is recognized by employers nationally and internationally.

Cybersecurity Concentration:
The optional Cybersecurity Concentration requires CPSC 456 (Network Security Fundamentals) plus 9 additional units from: CPSC 452 (Cryptography), CPSC 454 (Cloud Computing and Security), CPSC 455 (Web Security), CPSC 458 (Malware Analysis), CPSC 459 (Blockchain Technologies), or CPSC 483 (Machine Learning).

Programming Languages in the Curriculum:
The CS curriculum uses C++ as the primary language (CPSC 120, 121, 131, and most upper-division courses), Assembly Language (CPSC 240), and offers language electives including Java (CPSC 223J), Python (CPSC 223P), and JavaScript (CPSC 223W). SQL is covered in database courses (CPSC 332, 431), and web technologies (HTML, CSS, JavaScript) are covered in web development electives (CPSC 349, 449).

Transfer Students:
Transfer students should complete CPSC 120, 121, 131 equivalents and the calculus sequence (MATH 150A, 150B equivalents) before transferring. Check ASSIST.org for official course equivalencies. After transfer, meet with a CS advisor, complete upper-division courses at CSUF (minimum residency requirement), and apply for graduation at 90 units.

How to Apply for Graduation:
Complete at least 90 units, log into Titan Online (Student Portal), navigate to Student Records, select 'Apply for Graduation', submit the application, and pay the graduation fee. Apply early in your final semester and check the academic calendar for deadlines. Your degree audit will be reviewed and any issues will be communicated via email.
"""


# =============================================================================
# Document Chunking
# =============================================================================

def chunk_document(text: str, chunk_size: int = 500, overlap: int = 100) -> List[Dict]:
    """
    Split document into overlapping chunks.
    
    Chunk Size: 500 characters
    - Balances context length and specificity
    - Large enough for coherent information
    - Small enough for targeted retrieval
    
    Chunk Overlap: 100 characters
    - Prevents information loss at boundaries
    - Helps with cross-chunk queries
    """
    import re
    text = re.sub(r'\s+', ' ', text).strip()
    
    chunks = []
    start = 0
    chunk_id = 0
    
    while start < len(text):
        end = min(start + chunk_size, len(text))
        
        # Try to end at sentence boundary
        if end < len(text):
            for punct in ['. ', '! ', '? ', '\n']:
                last_punct = text.rfind(punct, start + chunk_size // 2, end)
                if last_punct > start:
                    end = last_punct + 1
                    break
        
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append({
                "id": chunk_id,
                "text": chunk_text,
                "start": start,
                "end": end
            })
            chunk_id += 1
        
        start = end - overlap if end < len(text) else len(text)
    
    return chunks


# =============================================================================
# Embedder
# =============================================================================

class Embedder:
    """Embedding model using SentenceTransformers or fallback TF-IDF."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.dim = 384
        
        if HAS_SENTENCE_TRANSFORMERS:
            print(f"Loading embedding model: {model_name}")
            self.model = SentenceTransformer(model_name)
            self.dim = self.model.get_sentence_embedding_dimension()
            print(f"Embedding dimension: {self.dim}")
        else:
            print("Using fallback TF-IDF embeddings")
            self.vocab = {}
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts."""
        if self.model is not None:
            return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        else:
            return self._fallback_embed(texts)
    
    def _fallback_embed(self, texts: List[str]) -> np.ndarray:
        """Fallback embedding using word frequencies."""
        # Build vocabulary
        for text in texts:
            for word in text.lower().split():
                word = ''.join(c for c in word if c.isalnum())
                if word and word not in self.vocab:
                    self.vocab[word] = len(self.vocab) % self.dim
        
        embeddings = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in text.lower().split():
                word = ''.join(c for c in word if c.isalnum())
                if word in self.vocab:
                    embeddings[i, self.vocab[word]] += 1
            norm = np.linalg.norm(embeddings[i])
            if norm > 0:
                embeddings[i] /= norm
        
        return embeddings


# =============================================================================
# Vector Store with FAISS
# =============================================================================

class VectorStore:
    """
    Vector Database: FAISS (Facebook AI Similarity Search)
    - Efficient similarity search
    - Scales to large document collections
    - Supports approximate nearest neighbor search
    """
    
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.index = None
        self.documents = []
        self.embeddings = None
        
        if HAS_FAISS:
            self.index = faiss.IndexFlatIP(embedding_dim)
            print("Using FAISS vector store")
        else:
            print("Using fallback numpy vector store")
    
    def add(self, embeddings: np.ndarray, documents: List[Dict]):
        """Add documents to the index."""
        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-9)
        normalized = normalized.astype(np.float32)
        
        if HAS_FAISS and self.index is not None:
            self.index.add(normalized)
        
        self.embeddings = normalized
        self.documents.extend(documents)
        print(f"Indexed {len(documents)} documents")
    
    def search(self, query_embedding: np.ndarray, k: int = 3) -> List[Tuple[Dict, float]]:
        """
        Search for k most similar documents.
        
        Retrieval Strategy:
        - K value: 3 (retrieve top 3 chunks)
        - Scoring: Cosine similarity
        """
        # Normalize query
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
        query_norm = query_norm.reshape(1, -1).astype(np.float32)
        
        if HAS_FAISS and self.index is not None:
            scores, indices = self.index.search(query_norm, k)
            results = [(self.documents[i], float(scores[0][j])) for j, i in enumerate(indices[0]) if i < len(self.documents)]
        else:
            # Fallback: manual similarity
            similarities = np.dot(self.embeddings, query_norm.T).flatten()
            top_k_idx = np.argsort(similarities)[::-1][:k]
            results = [(self.documents[i], float(similarities[i])) for i in top_k_idx]
        
        return results
    
    def save(self, path: str):
        """Save the index."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        if HAS_FAISS and self.index is not None:
            faiss.write_index(self.index, f"{path}.faiss")
        np.save(f"{path}_embeddings.npy", self.embeddings)
        with open(f"{path}_docs.json", "w") as f:
            json.dump(self.documents, f)
        print(f"Index saved to {path}")
    
    def load(self, path: str):
        """Load the index."""
        if HAS_FAISS and os.path.exists(f"{path}.faiss"):
            self.index = faiss.read_index(f"{path}.faiss")
        self.embeddings = np.load(f"{path}_embeddings.npy")
        with open(f"{path}_docs.json") as f:
            self.documents = json.load(f)
        print(f"Loaded index with {len(self.documents)} documents")


# =============================================================================
# LLM Generator
# =============================================================================

class LLMGenerator:
    """TinyLlama generator for RAG."""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
    
    def load(self):
        """Load the LLM."""
        print(f"\nLoading LLM: {self.config.llm_model}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        if self.config.use_4bit and torch.cuda.is_available():
            print("Using 4-bit quantization...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.llm_model,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.llm_model,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
            )
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.llm_model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        print("LLM loaded successfully!")
    
    def generate(self, prompt: str) -> str:
        """Generate response from prompt."""
        if self.model is None:
            return "[LLM not loaded]"
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_k=self.config.top_k,
                top_p=self.config.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract answer part
        if "Answer:" in response:
            response = response.split("Answer:")[-1].strip()
        elif prompt in response:
            response = response[len(prompt):].strip()
        
        return response


# =============================================================================
# RAG Pipeline
# =============================================================================

class RAGPipeline:
    """
    Complete RAG Pipeline.
    
    Prompt Template Structure:
    1. System Instruction:
       "You are a helpful assistant that answers questions about 
        the CSUF Computer Science undergraduate program..."
    
    2. Retrieved Context:
       "Context from the CS Handbook:
        ---
        [Retrieved chunk 1]
        [Retrieved chunk 2]
        [Retrieved chunk 3]
        ---"
    
    3. User Query:
       "Question: {user_question}"
    
    4. Answer Format:
       "Answer:"
    """
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.embedder = Embedder(config.embedding_model)
        self.vector_store = VectorStore(self.embedder.dim)
        self.generator = LLMGenerator(config)
        self.chunks = []
    
    def build_index(self):
        """Build the RAG index from handbook."""
        print("\n" + "="*50)
        print("Building RAG Index")
        print("="*50)
        
        # Get and chunk content
        content = get_handbook_content()
        self.chunks = chunk_document(
            content, 
            chunk_size=self.config.chunk_size, 
            overlap=self.config.chunk_overlap
        )
        print(f"Created {len(self.chunks)} chunks")
        
        # Embed chunks
        print("Embedding chunks...")
        texts = [c["text"] for c in self.chunks]
        embeddings = self.embedder.embed(texts)
        
        # Add to vector store
        self.vector_store.add(embeddings, self.chunks)
        
        # Save index
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        self.vector_store.save(f"{OUTPUT_DIR}/index")
        
        print("Index built and saved successfully!")
    
    def load_index(self):
        """Load existing index."""
        self.vector_store.load(f"{OUTPUT_DIR}/index")
    
    def retrieve(self, query: str) -> List[Tuple[Dict, float]]:
        """Retrieve relevant chunks for a query."""
        query_embedding = self.embedder.embed([query])[0]
        return self.vector_store.search(query_embedding, self.config.retrieval_k)
    
    def generate_prompt(self, query: str, contexts: List[str]) -> str:
        """
        Generate prompt with retrieved context.
        
        This is the prompt template used for RAG.
        """
        context_text = "\n\n".join([f"[{i+1}] {ctx}" for i, ctx in enumerate(contexts)])
        
        prompt = f"""You are a helpful assistant that answers questions about the CSUF Computer Science undergraduate program based on the provided handbook content.

Context from the CS Handbook:
---
{context_text}
---

Based on the above context, please answer the following question accurately and completely. If the answer is not fully covered in the context, say so.

Question: {query}

Answer:"""
        
        return prompt
    
    def query(self, question: str, verbose: bool = True) -> Dict:
        """Full RAG pipeline: retrieve and generate."""
        # Retrieve
        results = self.retrieve(question)
        contexts = [r[0]["text"] for r in results]
        scores = [r[1] for r in results]
        
        if verbose:
            print(f"\nRetrieved {len(contexts)} chunks:")
            for i, (ctx, score) in enumerate(zip(contexts, scores)):
                print(f"  [{i+1}] (score: {score:.3f}) {ctx[:80]}...")
        
        # Generate prompt
        prompt = self.generate_prompt(question, contexts)
        
        # Generate answer
        answer = self.generator.generate(prompt)
        
        return {
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "scores": scores,
            "prompt": prompt
        }


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_rag(config: RAGConfig):
    """Evaluate RAG system on the three test questions."""
    
    # Same evaluation questions as Part 3
    questions = [
        "What are the core courses required for a computer science undergraduate degree?",
        "Describe the rules for completing a senior project, including prerequisites.",
        "What are the degree requirements for graduation?"
    ]
    
    # Initialize RAG
    rag = RAGPipeline(config)
    
    # Load or build index
    index_path = f"{OUTPUT_DIR}/index_docs.json"
    if os.path.exists(index_path):
        rag.load_index()
    else:
        rag.build_index()
    
    # Load LLM
    rag.generator.load()
    
    results = []
    
    print("\n" + "="*60)
    print("RAG SYSTEM EVALUATION")
    print("="*60)
    
    for i, question in enumerate(questions, 1):
        print(f"\n{'='*60}")
        print(f"Question {i}: {question}")
        print('='*60)
        
        result = rag.query(question)
        results.append(result)
        
        print(f"\nGenerated Answer:")
        print(f"{result['answer'][:700]}{'...' if len(result['answer']) > 700 else ''}")
    
    # Save results
    output_results = []
    for r in results:
        output_results.append({
            "question": r["question"],
            "answer": r["answer"],
            "num_contexts": len(r["contexts"]),
            "top_score": r["scores"][0] if r["scores"] else 0
        })
    
    with open("rag_results.json", "w") as f:
        json.dump(output_results, f, indent=2)
    
    print(f"\nResults saved to rag_results.json")
    
    # Print comparison discussion
    print("\n" + "="*60)
    print("COMPARISON: RAG vs FINE-TUNED MODEL")
    print("="*60)
    print("""
Similarities:
- Both can answer factual questions about the handbook
- Both use TinyLlama as the base model
- Both require careful prompt engineering

Differences:
- RAG provides source attribution (retrieved chunks)
- RAG can handle documents not seen during training
- Fine-tuned model may be more fluent
- Fine-tuned model doesn't need retrieval at inference

Accuracy:
- RAG: High accuracy when relevant chunks are retrieved
- Fine-tuned: Depends on training data coverage

Trade-offs:

RAG Advantages:
+ No retraining needed for document updates
+ Provides source attribution (explainability)
+ Works with any document collection
+ Lower risk of hallucination (grounded in retrieved text)

RAG Disadvantages:
- Requires vector database infrastructure
- Retrieval latency added to inference
- Answer quality depends on retrieval quality
- May struggle with multi-hop reasoning

Use Cases for RAG:
1. Document QA with frequently updated content
2. Multi-document synthesis
3. When source attribution is important
4. Enterprise knowledge bases
5. Customer support with product documentation
""")
    
    return results


# =============================================================================
# Print Information
# =============================================================================

def print_rag_info():
    """Print RAG configuration information."""
    print("\n" + "="*60)
    print("RAG SYSTEM CONFIGURATION")
    print("="*60)
    print("""
Chunk Size: 500 characters
- Large enough to contain complete information
- Small enough for precise retrieval

Chunk Overlap: 100 characters
- Prevents information loss at boundaries
- Helps with cross-chunk queries

Vector Database: FAISS
- Efficient similarity search
- Scales to large document collections
- Supports approximate nearest neighbor search

Retrieval Strategy:
- K value: 3 (retrieve top 3 chunks)
- Scoring: cosine similarity
- Returns chunks most similar to query
""")
    
    print("="*60)
    print("PROMPT TEMPLATE")
    print("="*60)
    print("""
Template Structure:

1. System Instruction:
   "You are a helpful assistant that answers questions about 
    the CSUF Computer Science undergraduate program..."

2. Retrieved Context:
   "Context from the CS Handbook:
    ---
    [1] {Retrieved chunk 1}
    [2] {Retrieved chunk 2}
    [3] {Retrieved chunk 3}
    ---"

3. User Query:
   "Question: {user_question}"

4. Answer Format:
   "Answer:"

Example prompt:
---
You are a helpful assistant that answers questions about the 
CSUF Computer Science undergraduate program based on the 
provided handbook content.

Context from the CS Handbook:
---
[1] The requirements for the B.S. in Computer Science are 
    detailed in the University Catalog...
[2] Senior Capstone Project (CPSC 491): CPSC 491 has the 
    longest prerequisite chain...
[3] Graduation Requirements: To graduate with a B.S. in 
    Computer Science from CSUF...
---

Based on the above context, please answer the following 
question accurately and completely.

Question: What are the core courses required for graduation?

Answer:
---
""")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="RAG with TinyLlama on CS Handbook")
    parser.add_argument("--build", action="store_true", help="Build the RAG index")
    parser.add_argument("--query", type=str, help="Query the RAG system")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate on test questions")
    parser.add_argument("--no_4bit", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument("--info", action="store_true", help="Print configuration info")
    
    args = parser.parse_args()
    
    print("="*60)
    print("RETRIEVAL-AUGMENTED GENERATION (RAG) SYSTEM")
    print("="*60)
    
    if args.info:
        print_rag_info()
        return
    
    config = RAGConfig(use_4bit=not args.no_4bit)
    
    print(f"\nConfiguration:")
    print(f"  Chunk size: {config.chunk_size}")
    print(f"  Chunk overlap: {config.chunk_overlap}")
    print(f"  Retrieval K: {config.retrieval_k}")
    print(f"  Embedding model: {config.embedding_model}")
    print(f"  LLM: {config.llm_model}")
    print(f"  4-bit quantization: {config.use_4bit}")
    print(f"  FAISS available: {HAS_FAISS}")
    print(f"  SentenceTransformers available: {HAS_SENTENCE_TRANSFORMERS}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if args.build:
        rag = RAGPipeline(config)
        rag.build_index()
    
    elif args.query:
        rag = RAGPipeline(config)
        rag.load_index()
        rag.generator.load()
        
        result = rag.query(args.query)
        print(f"\nQuestion: {result['question']}")
        print(f"\nAnswer: {result['answer']}")
    
    elif args.evaluate:
        evaluate_rag(config)
    
    else:
        print("\nNo action specified. Use --build, --query, --evaluate, or --info")
        print("\nExamples:")
        print("  python rag_tune.py --build")
        print("  python rag_tune.py --query 'What are the core courses?'")
        print("  python rag_tune.py --evaluate")
        print("  python rag_tune.py --info")
        print_rag_info()


if __name__ == "__main__":
    main()