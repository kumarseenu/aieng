'''
In this challenge, you'll build a semantic document search system using Chroma as your vector database. 

You'll work with a collection of product descriptions and implement functionality to:
1. Load and embed product descriptions
2. Create an efficient vector search
3. Implement a caching layer to improve performance
4. Measure the impact of caching on search latency

Requirements:
1. Use Chroma as your vector database
2. Use the SentenceTransformer model "all-MiniLM-L6-v2" for embeddings
3. Implement a Least Recently Used (LRU) cache for query results
4. Create search functionality with optional filtering
5. Measure and compare performance with and without caching
'''

import chromadb
from chromadb.utils import embedding_functions
import pandas as pd
import numpy as np
import json
import time
import random

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

# TODO: Implement the LRUCache class


class LRUCache:
    def __init__(self, capacity=100):
        """Initialize an LRU cache with the given capacity"""
        # TODO: Initialize the cache structure
        pass

    def get(self, key):
        """Get an item from the cache and update its position in the LRU order"""
        # TODO: Implement cache lookup with LRU update
        pass

    def put(self, key, value):
        """Add an item to the cache, evicting the least recently used item if necessary"""
        # TODO: Implement item insertion with LRU eviction policy
        pass

    def clear(self):
        """Clear all items from the cache"""
        # TODO: Implement cache clearing
        pass

    def __len__(self):
        """Return the number of items in the cache"""
        # TODO: Implement cache size calculation
        pass

# TODO: Implement the ProductSearch class


class ProductSearch:
    def __init__(self, cache_capacity=50):
        """Initialize the product search system with Chroma and caching"""
        # TODO: Initialize Chroma client
        # TODO: Create embedding function
        # TODO: Create collection
        # TODO: Initialize the cache with the given capacity
        pass

    def add_products(self, products):
        """Add products to the vector database"""
        # TODO: Extract product information
        # TODO: Add to collection with proper metadata
        pass

    def search(self, query, n_results=3, category=None, max_price=None, use_cache=True):
        """
        Search for products similar to the query, with optional filters

        If use_cache is True, check the cache before performing a vector search
        """
        # TODO: Implement cached search functionality
        # TODO: Create a cache key that includes query and filters
        # TODO: Check cache for the key if use_cache is True
        # TODO: If cache miss or use_cache is False, perform the actual search
        # TODO: Update cache with the results if use_cache is True
        # TODO: Return results and whether it was a cache hit
        pass

    def benchmark_cache_performance(self, queries, n_trials=5):
        """
        Benchmark the performance improvement from caching

        - Run the same set of queries with and without caching
        - Measure execution time and cache hit rate
        - Return performance metrics
        """
        # TODO: Implement benchmarking to compare cached vs. non-cached performance
        pass

# TODO: Implement your solution
# 1. Create ProductSearch instance
# 2. Add products
# 3. Perform sample searches with and without cache
# 4. Benchmark cache performance
