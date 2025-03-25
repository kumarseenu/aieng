from sentence_transformers import SentenceTransformer, util
import numpy as np
import time
import hashlib
from typing import List, Dict, Tuple


class SemanticSearchEngine:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the search engine with a specific embedding model.

        Args:
            model_name: The name of the SentenceTransformer model to use
        """
        # TODO: Initialize the model and necessary data structures
        # - Initialize the embedding model
        # - Create empty data structures for document storage
        # - Initialize a cache for query embeddings
        pass

    def add_documents(self, documents: List[Dict[str, str]], batch_size: int = 32) -> None:
        """
        Process and add documents to the search engine.

        Args:
            documents: List of document dictionaries with 'id' and 'content' keys
            batch_size: Batch size for efficient embedding generation
        """
        # TODO: Process documents in batches and store their embeddings
        # - Process documents in batches of the specified size
        # - Generate embeddings for each batch
        # - Store document embeddings with their metadata
        pass

    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for a text, using cache if available.

        Args:
            text: The text to embed

        Returns:
            The embedding vector for the text
        """
        # TODO: Implement caching for embeddings
        # - Create a hash of the input text
        # - Check if embedding exists in cache
        # - If not, generate and cache the embedding
        # - Return the embedding
        pass

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, any]]:
        """
        Search for documents most similar to the query.

        Args:
            query: The search query text
            top_k: Number of top results to return

        Returns:
            List of top_k documents with their similarity scores
        """
        # TODO: Implement semantic search functionality
        # - Get embedding for the query (using cache if possible)
        # - Calculate similarity between query and all documents
        # - Return top_k results with their scores and metadata
        pass

    def get_cache_stats(self) -> Dict[str, int]:
        """
        Return statistics about the cache performance.

        Returns:
            Dictionary with cache hit/miss statistics
        """
        # TODO: Return cache statistics
        pass


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
    print(search_engine.get_cache_stats())
