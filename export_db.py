#!/usr/bin/env python
"""
Export Qdrant database to a serialized file for cloud deployment.
This script exports the vectors from the local Qdrant database to a pickle file
that can be loaded in the cloud environment.
"""

import os
import pickle
import numpy as np
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_openai import AzureOpenAIEmbeddings

# Load environment variables
load_dotenv()

def get_embeddings():
    """Initialize Azure OpenAI embeddings."""
    return AzureOpenAIEmbeddings(
        deployment=os.getenv("AZURE_EMBEDDING_DEPLOYMENT"),
        api_key=os.getenv("AZURE_EMBEDDING_KEY"),
        azure_endpoint=os.getenv("AZURE_EMBEDDING_ENDPOINT"),
        api_version=os.getenv("AZURE_EMBEDDING_API_VERSION")
    )

def export_qdrant_db():
    """Export Qdrant database to a pickle file."""
    print("Exporting Qdrant database...")
    
    # Initialize local Qdrant client
    client = QdrantClient(path="./local_qdrant_db")
    
    # Get collection info
    collection_info = client.get_collection(collection_name="hr-policies")
    print(f"Collection info: {collection_info}")
    print(f"Vector count: {collection_info.vectors_count}")
    print(f"Points count: {collection_info.points_count}")
    
    # Use points_count instead of vectors_count if vectors_count is None
    points_limit = collection_info.points_count or 1000  # Use a reasonable default if both are None
    
    if points_limit == 0:
        print("Error: Vector database is empty. Please run ingest.py first.")
        return
    
    # Get all points from the collection
    points = client.scroll(
        collection_name="hr-policies",
        limit=points_limit
    )[0]
    
    print(f"Retrieved {len(points)} points from the database")
    
    # Extract vectors and payloads
    vectors = []
    metadatas = []
    texts = []
    
    for point in points:
        vectors.append(point.vector)
        metadata = point.payload.get("metadata", {})
        metadatas.append(metadata)
        texts.append(point.payload.get("text", ""))
    
    # Create export data
    export_data = {
        "vectors": vectors,
        "metadatas": metadatas,
        "texts": texts
    }
    
    # Save to pickle file
    with open("qdrant_export.pkl", "wb") as f:
        pickle.dump(export_data, f)
    
    print(f"Successfully exported {len(vectors)} vectors to qdrant_export.pkl")
    print("File size:", os.path.getsize("qdrant_export.pkl") / (1024 * 1024), "MB")

if __name__ == "__main__":
    export_qdrant_db()
