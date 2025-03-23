import chromadb
from chromadb.utils import embedding_functions
import pandas as pd
import numpy as np
import json
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

# List of test queries to evaluate your system
test_queries = [
    {"query": "wireless headphones for music",
        "relevant_ids": ["p001", "p011"]},
    {"query": "equipment for working out at home",
        "relevant_ids": ["p006", "p010", "p014"]},
    {"query": "office furniture for better posture",
        "relevant_ids": ["p002", "p012"]},
    {"query": "kitchen tools for making beverages",
        "relevant_ids": ["p003", "p009"]},
    {"query": "electronic devices for video conferencing",
        "relevant_ids": ["p005"]}
]


class ProductSearch:
    def __init__(self):
        """Initialize the product search system with Chroma and embedding function"""
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

        # Store product data for reference
        self.products = {}

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
               max_price: Optional[float] = None) -> List[Dict[str, Any]]:
        """Search for products similar to the query, with optional filters"""
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
            return []

        # Format results
        formatted_results = []

        if len(results["ids"][0]) == 0:
            return []

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

        return formatted_results

    def evaluate(self, test_queries: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate search quality using the test queries"""
        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0
        num_queries = len(test_queries)

        for query_item in test_queries:
            query = query_item["query"]
            relevant_ids = set(query_item["relevant_ids"])

            # Get more results than relevant_ids to ensure we find all
            k = max(len(relevant_ids) * 2, 5)
            results = self.search(query, n_results=k)
            retrieved_ids = set([r["id"] for r in results])

            # Calculate metrics
            true_positives = len(relevant_ids.intersection(retrieved_ids))

            # Precision: what fraction of retrieved items are relevant
            precision = true_positives / \
                len(retrieved_ids) if retrieved_ids else 0.0

            # Recall: what fraction of relevant items are retrieved
            recall = true_positives / \
                len(relevant_ids) if relevant_ids else 0.0

            # F1 score: harmonic mean of precision and recall
            f1 = 2 * precision * recall / \
                (precision + recall) if (precision + recall) > 0 else 0.0

            total_precision += precision
            total_recall += recall
            total_f1 += f1

            # Print per-query results
            print(f"Query: '{query}'")
            print(
                f"  Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")
            print(f"  Relevant IDs: {relevant_ids}")
            print(f"  Retrieved IDs: {retrieved_ids}")
            print()

        # Calculate averages
        avg_precision = total_precision / num_queries
        avg_recall = total_recall / num_queries
        avg_f1 = total_f1 / num_queries

        print(f"Overall Evaluation:")
        print(f"  Average Precision: {avg_precision:.2f}")
        print(f"  Average Recall: {avg_recall:.2f}")
        print(f"  Average F1 Score: {avg_f1:.2f}")

        return {
            "precision": avg_precision,
            "recall": avg_recall,
            "f1": avg_f1
        }

    def display_search_results(self, query: str, results: List[Dict[str, Any]]):
        """Pretty-print search results"""
        print(f"\nSearch Results for: '{query}'")
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

# Main execution - demonstrates the solution


def main():
    # 1. Create ProductSearch instance
    product_searcher = ProductSearch()

    # 2. Add products
    product_searcher.add_products(product_data)

    # 3. Perform sample searches
    print("\n--- Sample Searches ---")

    # Basic search
    query = "wireless audio devices"
    results = product_searcher.search(query, n_results=3)
    product_searcher.display_search_results(query, results)

    # Search with category filter
    query = "comfortable seating"
    results = product_searcher.search(query, n_results=3, category="furniture")
    product_searcher.display_search_results(
        f"{query} (furniture category only)", results)

    # Search with price filter
    query = "portable electronics"
    results = product_searcher.search(query, n_results=3, max_price=100.0)
    product_searcher.display_search_results(f"{query} (under $100)", results)

    # 4. Evaluate the system
    print("\n--- Evaluation ---")
    metrics = product_searcher.evaluate(test_queries)

    return product_searcher


if __name__ == "__main__":
    searcher = main()
