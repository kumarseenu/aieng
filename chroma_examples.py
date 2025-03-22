import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd
from chromadb.utils import embedding_functions
import chromadb

# client = chromadb.Client()
client = chromadb.PersistentClient()
client.heartbeat()


# Vector DB Fundamentals with Chroma
# LinkedIn Learning: Fundamentals of AI Engineering

# Setup and Installation
#pip install chromadb sentence-transformers

# Import dependencies

# Basic client initialization


def initialize_chroma():
    """Initialize a basic in-memory Chroma client"""
    print("Initializing Chroma client...")
    client = chromadb.Client()

    # Create embedding function
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    return client, embedding_function

# Create a collection


def create_collection(client, embedding_function, name="documents"):
    """Create a collection with the specified name and embedding function"""
    print(f"Creating collection: {name}")
    # Delete if exists
    try:
        client.delete_collection(name)
    except:
        pass

    # Create new collection
    collection = client.create_collection(
        name=name,
        embedding_function=embedding_function
    )

    return collection

# Demo 1: Basic Vector Operations


def basic_vector_operations(collection):
    """Demo of basic vector database operations"""
    print("\n=== BASIC VECTOR OPERATIONS ===")

    # Sample documents
    documents = [
        "The quick brown fox jumps over the lazy dog",
        "A man is walking his dog in the park",
        "The weather is sunny and warm today",
        "Artificial intelligence is transforming the technology landscape",
        "Vector databases are essential for semantic search applications"
    ]
    ids = ["doc1", "doc2", "doc3", "doc4", "doc5"]

    # Add documents to collection
    print("Adding documents to collection...")
    collection.add(
        documents=documents,
        ids=ids
    )

    # Get collection count
    count = collection.count()
    print(f"Collection now contains {count} documents")

    # Query the collection
    query_text = "AI and technology trends"
    print(f"\nPerforming similarity search for: '{query_text}'")

    results = collection.query(
        query_texts=[query_text],
        n_results=3
    )

    # Display results
    print("\nResults:")
    for i, (doc, doc_id, distance) in enumerate(zip(
        results['documents'][0],
        results['ids'][0],
        results['distances'][0]
    )):
        print(f"{i+1}. Document: {doc}")
        print(f"   ID: {doc_id}")
        print(f"   Distance: {distance}")
        print()

    # Get a specific item
    print("Retrieving document by ID...")
    get_result = collection.get(ids=["doc1"])
    print(f"Retrieved: {get_result['documents'][0]}")

    return documents, ids

# Demo 2: Metadata and Filtering


def metadata_filtering(client, embedding_function):
    """Demo of metadata filtering capabilities"""
    print("\n=== METADATA AND FILTERING ===")

    # Create a new collection
    collection = create_collection(
        client, embedding_function, "filtered_documents")

    # Sample documents with metadata
    documents = [
        "The quick brown fox jumps over the lazy dog",
        "A man is walking his dog in the park",
        "The weather is sunny and warm today",
        "Artificial intelligence is transforming the technology landscape",
        "Vector databases are essential for semantic search applications",
        "Deep learning models require substantial computational resources",
        "The city skyline looks beautiful at sunset",
        "Machine learning algorithms find patterns in data"
    ]

    ids = [f"doc{i+1}" for i in range(len(documents))]

    metadatas = [
        {"category": "animal", "length": "short", "year": 2021},
        {"category": "lifestyle", "length": "short", "year": 2022},
        {"category": "weather", "length": "short", "year": 2023},
        {"category": "technology", "length": "medium", "year": 2023},
        {"category": "technology", "length": "medium", "year": 2024},
        {"category": "technology", "length": "long", "year": 2024},
        {"category": "travel", "length": "short", "year": 2023},
        {"category": "technology", "length": "medium", "year": 2024}
    ]

    # Add documents with metadata
    print("Adding documents with metadata...")
    collection.add(
        documents=documents,
        ids=ids,
        metadatas=metadatas
    )

    # Simple metadata filtering
    print("\nFiltering by category 'technology':")
    results = collection.query(
        query_texts=["AI advancements"],
        n_results=3,
        where={"category": "technology"}
    )

    # Display results
    for i, (doc, doc_id, metadata) in enumerate(zip(
        results['documents'][0],
        results['ids'][0],
        results['metadatas'][0]
    )):
        print(f"{i+1}. Document: {doc}")
        print(f"   ID: {doc_id}")
        print(f"   Metadata: {metadata}")
        print()

    # Complex filtering
    print("\nComplex filtering (technology documents from 2024):")
    results = collection.query(
        query_texts=["AI advancements"],
        n_results=3,
        where={"category": "technology", "year": 2024}
    )

    # Display results
    for i, (doc, doc_id, metadata) in enumerate(zip(
        results['documents'][0],
        results['ids'][0],
        results['metadatas'][0]
    )):
        print(f"{i+1}. Document: {doc}")
        print(f"   ID: {doc_id}")
        print(f"   Metadata: {metadata}")
        print()

    # Using where_document
    print("\nFiltering documents containing 'machine learning':")
    results = collection.query(
        query_texts=["AI advancements"],
        n_results=3,
        where_document={"$contains": "machine learning"}
    )

    # Display results
    for i, (doc, doc_id, metadata) in enumerate(zip(
        results['documents'][0],
        results['ids'][0],
        results['metadatas'][0] if 'metadatas' in results and results['metadatas'] else [
            {}] * len(results['documents'][0])
    )):
        print(f"{i+1}. Document: {doc}")
        print(f"   ID: {doc_id}")
        print(f"   Metadata: {metadata}")
        print()

    return collection

# Demo 3: Understanding Embeddings


def explore_embeddings(embedding_function):
    """Explore and visualize embeddings"""
    print("\n=== UNDERSTANDING EMBEDDINGS ===")

    # Sample texts
    texts = [
        "The quick brown fox jumps over the lazy dog",
        "A fox quickly jumped over a lazy dog",
        "Artificial intelligence is changing the world",
        "Machine learning transforms how we work",
        "The weather is sunny and warm today",
        "Today's forecast shows warm temperatures"
    ]

    # Generate embeddings
    print("Generating embeddings for sample texts...")
    embeddings = embedding_function(texts)

    # Print embedding information
    embedding_dim = len(embeddings[0])
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Number of embeddings: {len(embeddings)}")

    # Show a sample of the first embedding vector
    print("\nSample of first embedding vector (first 10 dimensions):")
    print(embeddings[0][:10])

    # Calculate cosine similarity between pairs
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    print("\nCosine similarity between texts:")
    for i in range(len(texts)):
        for j in range(i+1, len(texts)):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            print(
                f"Similarity between \"{texts[i][:30]}...\" and \"{texts[j][:30]}...\": {sim:.4f}")

    return embeddings, texts

# Demo 4: Persistence


def persistent_storage():
    """Demo of persistent storage in Chroma"""
    print("\n=== PERSISTENT STORAGE ===")

    # Initialize a persistent client
    print("Creating a persistent Chroma client...")
    persistent_client = chromadb.PersistentClient(path="./chroma_db")

    # Create embedding function
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    # Check if collection exists and delete
    try:
        persistent_client.delete_collection("persistent_docs")
    except:
        pass

    # Create a new collection
    print("Creating a persistent collection...")
    persistent_collection = persistent_client.create_collection(
        name="persistent_docs",
        embedding_function=embedding_function
    )

    # Add some documents
    documents = [
        "This is a document that will be stored persistently",
        "Vector databases need to persist data for production use",
        "Data persistence ensures your embeddings survive restarts"
    ]

    ids = ["pdoc1", "pdoc2", "pdoc3"]

    persistent_collection.add(
        documents=documents,
        ids=ids
    )

    print(f"Added {len(documents)} documents to persistent storage")

    # Show that we can query the persistent collection
    results = persistent_collection.query(
        query_texts=["persistent data storage"],
        n_results=1
    )

    print(
        f"\nQuery result from persistent storage: {results['documents'][0][0]}")

    return persistent_client, persistent_collection

# Demo 5: Performance Considerations


def performance_demo(client, embedding_function):
    """Demo of performance considerations"""
    print("\n=== PERFORMANCE CONSIDERATIONS ===")

    # Create a larger collection for performance testing
    collection = create_collection(
        client, embedding_function, "performance_test")

    # Generate synthetic documents
    print("Generating synthetic documents for performance testing...")
    words = ["AI", "machine", "learning", "vector", "database", "embedding", "neural",
             "network", "transformer", "data", "science", "engineering", "model",
             "algorithm", "optimization", "natural", "language", "processing"]

    num_docs = 1000
    documents = []

    for i in range(num_docs):
        # Create a random document of 10-20 words
        doc_len = np.random.randint(10, 20)
        doc = " ".join(np.random.choice(words, size=doc_len))
        documents.append(doc)

    ids = [f"perf_doc_{i}" for i in range(num_docs)]

    # Time the addition of documents
    print(f"Adding {num_docs} documents to collection...")
    start_time = time.time()

    # Add in batches
    batch_size = 100
    for i in range(0, num_docs, batch_size):
        end_idx = min(i + batch_size, num_docs)
        collection.add(
            documents=documents[i:end_idx],
            ids=ids[i:end_idx]
        )

    add_time = time.time() - start_time
    print(f"Time to add {num_docs} documents: {add_time:.2f} seconds")

    # Time query performance
    print("\nTesting query performance...")
    query_times = []
    num_queries = 5

    for i in range(num_queries):
        query = " ".join(np.random.choice(words, size=5))
        start_time = time.time()
        collection.query(
            query_texts=[query],
            n_results=10
        )
        query_time = time.time() - start_time
        query_times.append(query_time)
        print(f"Query {i+1}: {query_time:.4f} seconds")

    print(f"Average query time: {np.mean(query_times):.4f} seconds")

    return collection, query_times

# Main demo function


def run_vector_db_fundamentals_demo():
    """Run the full vector database fundamentals demo"""
    print("===== VECTOR DATABASE FUNDAMENTALS DEMO =====")

    # Initialize client
    client, embedding_function = initialize_chroma()

    # Create initial collection
    collection = create_collection(client, embedding_function)

    # Run basic operations demo
    docs, ids = basic_vector_operations(collection)

    # Run metadata filtering demo
    filtered_collection = metadata_filtering(client, embedding_function)

    # Explore embeddings
    embeddings, texts = explore_embeddings(embedding_function)

    # Test persistence
    persistent_client, persistent_collection = persistent_storage()

    # Performance considerations
    perf_collection, query_times = performance_demo(client, embedding_function)

    print("\n===== DEMO COMPLETE =====")
    print("You've now learned the fundamentals of vector databases with Chroma!")


# Run the demo
if __name__ == "__main__":
    run_vector_db_fundamentals_demo()
