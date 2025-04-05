import chromadb
from chromadb.utils import embedding_functions
import pandas as pd
import numpy as np
import json
import time
import random
from typing import List, Dict, Any, Optional, Tuple

# Sample product data
product_data = [
    {"id": "p001", "name": "Premium Bluetooth Headphones",
        "description": "Wireless over-ear headphones with noise cancellation and 20-hour battery life.", "category": "electronics", "price": 199.99},
    {"id": "p002", "name": "Ergonomic Office Chair",
        "description": "Adjustable office chair with lumbar support and breathable mesh back.", "category": "furniture", "price": 249.99},
    {"id": "p003", "name": "Stainless Steel Water Bottle",
        "description": "Vacuum insulated water bottle that keeps drinks cold for 24 hours or hot for 12 hours.", "category": "kitchenware", "price": 29.99},
    {"id": "p004", "name": "Wireless Charging Pad",
        "description": "Fast-charging wireless pad compatible with all Qi-enabled devices.", "category": "electronics", "price": 39.99},
    {"id": "p005", "name": "HD Webcam", "description": "1080p webcam with auto light correction and noise-cancelling microphone.",
        "category": "electronics", "price": 79.99},
    {"id": "p006", "name": "Yoga Mat", "description": "Non-slip exercise mat ideal for yoga, pilates, and home workouts.",
        "category": "fitness", "price": 24.99},
    {"id": "p007", "name": "Smart LED Desk Lamp",
        "description": "Adjustable desk lamp with multiple brightness levels and color temperatures.", "category": "furniture", "price": 59.99},
    {"id": "p008", "name": "Bluetooth Speaker", "description": "Portable wireless speaker with 360-degree sound and waterproof design.",
        "category": "electronics", "price": 129.99},
    {"id": "p009", "name": "French Press Coffee Maker",
        "description": "Glass and stainless steel coffee press that makes rich, flavorful coffee.", "category": "kitchenware", "price": 34.99},
    {"id": "p010", "name": "Adjustable Dumbbell Set",
        "description": "Space-saving dumbbell set with adjustable weights from 5 to 52.5 pounds.", "category": "fitness", "price": 299.99},
    {"id": "p011", "name": "Noise-Cancelling Earbuds",
        "description": "Wireless earbuds with active noise cancellation and touch controls.", "category": "electronics", "price": 149.99},
    {"id": "p012", "name": "Standing Desk Converter",
        "description": "Adjustable desktop riser that transforms any desk into a standing workstation.", "category": "furniture", "price": 189.99},
    {"id": "p013", "name": "Cast Iron Skillet", "description": "Pre-seasoned 10-inch cast iron pan perfect for stovetop, oven, and campfire cooking.",
        "category": "kitchenware", "price": 39.99},
    {"id": "p014", "name": "Fitness Tracker",
        "description": "Waterproof activity tracker with heart rate monitor and sleep tracking.", "category": "fitness", "price": 89.99},
    {"id": "p015", "name": "Mechanical Keyboard",
        "description": "Programmable mechanical keyboard with customizable RGB backlighting and tactile switches.", "category": "electronics", "price": 129.99}
]

# Implement the LRU cache


class LRUCache:
    def __init__(self, capacity=100):
        """Initialize an LRU cache with the given capacity"""
        self.capacity = capacity
        self.cache = {}  # Dictionary to store key-value pairs
        self.usage_order = []  # List to track access order

    def get(self, key):
        """Get an item from the cache and update its position in the LRU order"""
        if key in self.cache:
            # Update usage order (move to end = most recently used)
            self.usage_order.remove(key)
            self.usage_order.append(key)
            return self.cache[key]
        return None  # Cache miss

    def put(self, key, value):
        """Add an item to the cache, evicting the least recently used item if necessary"""
        if key in self.cache:
            # Update existing entry
            self.cache[key] = value
            # Update usage order
            self.usage_order.remove(key)
            self.usage_order.append(key)
        else:
            # Add new entry
            if len(self.cache) >= self.capacity:
                # Cache is full, evict least recently used item
                lru_key = self.usage_order.pop(0)
                del self.cache[lru_key]

            self.cache[key] = value
            self.usage_order.append(key)

    def clear(self):
        """Clear all items from the cache"""
        self.cache = {}
        self.usage_order = []

    def __len__(self):
        """Return the number of items in the cache"""
        return len(self.cache)


class ProductSearch:
    def __init__(self, cache_capacity=50):
        """Initialize the product search system with Chroma and caching"""
        # Initialize Chroma client
        self.client = chromadb.Client()

        # Create embedding function
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        # Create collection (or reset it if it already exists)
        try:
            self.client.delete_collection("products")
        except:
            pass

        self.collection = self.client.create_collection(
            name="products",
            embedding_function=self.embedding_function
        )

        # Initialize the query cache
        self.query_cache = LRUCache(capacity=cache_capacity)

        # Store product data for reference
        self.products = {}

        # Cache statistics
        self.cache_hits = 0
        self.cache_misses = 0

    def add_products(self, products: List[Dict[str, Any]]):
        """Add products to the vector database"""
        # Extract product information
        ids = []
        documents = []
        metadatas = []

        for product in products:
            # Store product in our dictionary for quick lookup
            self.products[product["id"]] = product

            # Prepare data for collection
            ids.append(product["id"])

            # Create search document by combining name and description
            document = f"{product['name']}. {product['description']}"
            documents.append(document)

            # Prepare metadata for filtering
            metadata = {
                "name": product["name"],
                "category": product["category"],
                "price": product["price"]
            }
            metadatas.append(metadata)

        # Add to collection
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )

        print(f"Added {len(products)} products to the vector database")

    def search(self, query: str, n_results: int = 3, category: Optional[str] = None,
               max_price: Optional[float] = None, use_cache: bool = True) -> Tuple[List[Dict[str, Any]], bool]:
        """
        Search for products similar to the query, with optional filters

        Returns a tuple of (results, is_cache_hit)
        """
        # Create a cache key that includes query and filters
        filter_str = ""
        if category:
            filter_str += f"_cat:{category}"
        if max_price:
            filter_str += f"_price:<={max_price}"

        cache_key = f"{query}_{n_results}{filter_str}"

        # Check cache if enabled
        if use_cache:
            cached_result = self.query_cache.get(cache_key)
            if cached_result is not None:
                self.cache_hits += 1
                return cached_result, True  # Cache hit

        # Cache miss or cache disabled, perform the actual search
        self.cache_misses += 1

        # Prepare where clause for filtering
        where_clause = {}

        if category:
            where_clause["category"] = category

        if max_price:
            where_clause["price"] = {"$lte": max_price}

        # Execute search
        try:
            if where_clause:
                results = self.collection.query(
                    query_texts=[query],
                    n_results=n_results,
                    where=where_clause
                )
            else:
                results = self.collection.query(
                    query_texts=[query],
                    n_results=n_results
                )
        except Exception as e:
            print(f"Search error: {e}")
            return [], False

        # Format results
        formatted_results = []

        if len(results["ids"][0]) == 0:
            return [], False

        for i, (doc_id, distance) in enumerate(zip(results["ids"][0], results["distances"][0])):
            product = self.products[doc_id]
            formatted_results.append({
                "id": doc_id,
                "name": product["name"],
                "description": product["description"],
                "category": product["category"],
                "price": product["price"],
                # Convert distance to a similarity score (0-1)
                "score": 1.0 - distance/2
            })

        # Update cache if enabled
        if use_cache:
            self.query_cache.put(cache_key, formatted_results)

        return formatted_results, False  # Cache miss

    def benchmark_cache_performance(self, queries: List[str], n_trials: int = 5) -> Dict[str, Any]:
        """
        Benchmark the performance improvement from caching

        - Run the same set of queries with and without caching
        - Measure execution time and cache hit rate
        - Return performance metrics
        """
        results = {
            "without_cache": {
                "total_time": 0,
                "queries_per_second": 0
            },
            "with_cache": {
                "total_time": 0,
                "queries_per_second": 0,
                "hit_rate": 0
            },
            "speedup": 0
        }

        total_queries = len(queries) * n_trials

        # Reset cache stats
        self.query_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0

        # First run: without cache
        print("Benchmarking searches without cache...")
        start_time = time.time()

        for _ in range(n_trials):
            for query in queries:
                _, _ = self.search(query, use_cache=False)

        no_cache_time = time.time() - start_time

        # Second run: with cache
        print("Benchmarking searches with cache...")
        self.query_cache.clear()  # Clear the cache before starting

        start_time = time.time()

        for _ in range(n_trials):
            for query in queries:
                _, _ = self.search(query, use_cache=True)

        with_cache_time = time.time() - start_time
        hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses)

        # Calculate metrics
        results["without_cache"]["total_time"] = no_cache_time
        results["without_cache"]["queries_per_second"] = total_queries / no_cache_time

        results["with_cache"]["total_time"] = with_cache_time
        results["with_cache"]["queries_per_second"] = total_queries / \
            with_cache_time
        results["with_cache"]["hit_rate"] = hit_rate

        # Calculate speedup
        results["speedup"] = no_cache_time / with_cache_time

        # Print results
        print("\nCache Performance Results:")
        print(f"Total queries: {total_queries}")
        print(
            f"Without cache: {no_cache_time:.4f} seconds ({results['without_cache']['queries_per_second']:.2f} queries/sec)")
        print(
            f"With cache: {with_cache_time:.4f} seconds ({results['with_cache']['queries_per_second']:.2f} queries/sec)")
        print(f"Speedup: {results['speedup']:.2f}x faster with caching")
        print(f"Cache hit rate: {hit_rate:.1%}")
        print(f"Cache size: {len(self.query_cache)}")

        return results

    def display_search_results(self, query: str, results: List[Dict[str, Any]], cache_hit: bool):
        """Pretty-print search results"""
        source = "CACHE" if cache_hit else "DATABASE"
        print(f"\nSearch Results for: '{query}' (from {source})")
        print("-" * 50)

        if not results:
            print("No results found.")
            return

        for i, result in enumerate(results):
            print(f"{i+1}. {result['name']} (${result['price']:.2f})")
            print(f"   Category: {result['category']}")
            print(f"   Score: {result['score']:.2f}")
            print(f"   {result['description']}")
            print()


def main():
    # 1. Create ProductSearch instance
    product_searcher = ProductSearch(cache_capacity=50)

    # 2. Add products
    product_searcher.add_products(product_data)

    # 3. Perform sample searches with and without cache
    print("\n--- Sample Search with Caching ---")

    # First search (cache miss)
    query = "wireless audio devices"
    results, cache_hit = product_searcher.search(query, n_results=3)
    product_searcher.display_search_results(query, results, cache_hit)

    # Repeat the same search (should be a cache hit)
    results, cache_hit = product_searcher.search(query, n_results=3)
    product_searcher.display_search_results(query, results, cache_hit)

    # 4. Benchmark cache performance
    print("\n--- Cache Benchmark ---")

    # Create a mix of queries (common and unique)
    common_queries = [
        "wireless headphones",
        "office furniture",
        "kitchen appliances",
        "fitness equipment",
        "charging devices"
    ]

    # Add some variations
    unique_queries = [f"unique query {i}" for i in range(10)]

    # Create a realistic query mix (some repeated, some unique)
    mixed_queries = []
    for _ in range(5):
        # Add common queries (higher probability)
        mixed_queries.extend(common_queries)

        # Add some unique queries
        mixed_queries.extend(random.sample(unique_queries, 2))

    random.shuffle(mixed_queries)

    # Run benchmark
    metrics = product_searcher.benchmark_cache_performance(mixed_queries)

    return product_searcher


if __name__ == "__main__":
    searcher = main()
