from sentence_transformers import SentenceTransformer, util
import numpy as np
import time
import hashlib
from typing import List, Dict, Tuple
import torch


class SemanticSearchEngine:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the search engine with a specific embedding model.

        Args:
            model_name: The name of the SentenceTransformer model to use
        """
        # Initialize the model
        self.model = SentenceTransformer(model_name)

        # Storage for documents and their embeddings
        self.document_embeddings = []
        self.document_metadata = []

        # Setup embedding cache
        self.embedding_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

    def add_documents(self, documents: List[Dict[str, str]], batch_size: int = 32) -> None:
        """
        Process and add documents to the search engine.

        Args:
            documents: List of document dictionaries with 'id' and 'content' keys
            batch_size: Batch size for efficient embedding generation
        """
        # Extract document content
        doc_contents = [doc['content'] for doc in documents]

        # Process in batches for efficiency
        for i in range(0, len(doc_contents), batch_size):
            batch = doc_contents[i:i+batch_size]

            # Generate embeddings for batch
            batch_embeddings = self.model.encode(batch, convert_to_tensor=True)

            # Store embeddings and metadata
            for j, embedding in enumerate(batch_embeddings):
                doc_idx = i + j
                if doc_idx < len(documents):
                    self.document_embeddings.append(embedding)
                    self.document_metadata.append({
                        'id': documents[doc_idx]['id'],
                        'content': documents[doc_idx]['content']
                    })

        # Convert list of embeddings to a tensor for efficient similarity computation
        if self.document_embeddings:
            if isinstance(self.document_embeddings[0], torch.Tensor):
                self.document_embeddings = torch.stack(
                    self.document_embeddings)
            else:
                self.document_embeddings = np.array(self.document_embeddings)

    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for a text, using cache if available.

        Args:
            text: The text to embed

        Returns:
            The embedding vector for the text
        """
        # Create a hash of the input text to use as cache key
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()

        # Check if embedding exists in cache
        if text_hash in self.embedding_cache:
            self.cache_hits += 1
            return self.embedding_cache[text_hash]

        # If not in cache, generate the embedding
        self.cache_misses += 1
        embedding = self.model.encode(text, convert_to_tensor=True)

        # Cache the embedding
        self.embedding_cache[text_hash] = embedding

        return embedding

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, any]]:
        """
        Search for documents most similar to the query.

        Args:
            query: The search query text
            top_k: Number of top results to return

        Returns:
            List of top_k documents with their similarity scores
        """
        # Get embedding for the query (using cache if possible)
        query_embedding = self._get_embedding(query)

        # Calculate cosine similarity between query and all documents
        cos_scores = util.cos_sim(query_embedding, self.document_embeddings)[0]

        # Get top_k results
        top_results = []
        top_indices = torch.topk(cos_scores, k=min(top_k, len(cos_scores)))[1]

        for idx in top_indices:
            result = {
                'id': self.document_metadata[idx]['id'],
                'content': self.document_metadata[idx]['content'],
                'score': cos_scores[idx].item()
            }
            top_results.append(result)

        return top_results

    def get_cache_stats(self) -> Dict[str, int]:
        """
        Return statistics about the cache performance.

        Returns:
            Dictionary with cache hit/miss statistics
        """
        total = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total * 100) if total > 0 else 0

        return {
            'hits': self.cache_hits,
            'misses': self.cache_misses,
            'total': total,
            'hit_rate_percent': hit_rate
        }


# Example usage
if __name__ == "__main__":
    # Sample documents
    documents = [
        {"id": "doc1", "content": "How to reset your password in our application"},
        {"id": "doc2", "content": "Troubleshooting login issues and account access problems"},
        {"id": "doc3", "content": "Understanding your monthly billing statement"},
        {"id": "doc4", "content": "How to upgrade your subscription plan"},
        {"id": "doc5", "content": "Setting up two-factor authentication for security"},
        # Add more documents as needed
    ]

    # Sample queries
    queries = [
        "I forgot my password",
        "Can't log into my account",
        "How do I understand my bill",
        "I want to upgrade my account",
        "password reset",
        "I forgot my password",  # Repeated query to test caching
    ]

    # Initialize and use the search engine
    search_engine = SemanticSearchEngine()

    # Add documents
    start_time = time.time()
    search_engine.add_documents(documents)
    print(f"Document processing time: {time.time() - start_time:.4f}s")

    # Search with each query
    for query in queries:
        start_time = time.time()
        results = search_engine.search(query)
        print(f"\nQuery: '{query}'")
        print(f"Search time: {time.time() - start_time:.4f}s")

        for result in results:
            print(
                f"  - {result['id']} (Score: {result['score']:.4f}): {result['content']}")

    # Print cache statistics
    print("\nCache statistics:")
    stats = search_engine.get_cache_stats()
    print(f"Cache hits: {stats['hits']}")
    print(f"Cache misses: {stats['misses']}")
    print(f"Total cache accesses: {stats['total']}")
    print(f"Hit rate: {stats['hit_rate_percent']:.2f}%")
