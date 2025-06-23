#!/usr/bin/env python
"""
Cloud database loader for Tesla HR Policy Assistant.
This script loads the exported Qdrant database from a pickle file
and creates an in-memory Qdrant database for cloud deployment.
"""

import os
import pickle
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

def load_exported_db():
    """Load exported database from pickle file."""
    try:
        # Check if export file exists
        if not os.path.exists("qdrant_export.pkl"):
            print("Error: qdrant_export.pkl not found")
            return False
        
        print("Loading exported database...")
        
        # Load export data
        with open("qdrant_export.pkl", "rb") as f:
            export_data = pickle.load(f)
        
        vectors = export_data["vectors"]
        metadatas = export_data["metadatas"]
        texts = export_data["texts"]
        
        print(f"Loaded {len(vectors)} vectors from export file")
        
        # Initialize in-memory Qdrant client
        client = QdrantClient(":memory:")
        
        # Create collection
        client.create_collection(
            collection_name="hr-policies",
            vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE)
        )
        
        # Prepare points for batch upload
        points = []
        for i, (vector, metadata, text) in enumerate(zip(vectors, metadatas, texts)):
            points.append(
                PointStruct(
                    id=i,
                    vector=vector,
                    payload={"metadata": metadata, "text": text}
                )
            )
        
        # Upload points in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i+batch_size]
            client.upsert(
                collection_name="hr-policies",
                points=batch
            )
            print(f"Uploaded batch {i//batch_size + 1}/{(len(points) + batch_size - 1)//batch_size}")
        
        # Verify upload
        collection_info = client.get_collection(collection_name="hr-policies")
        print(f"Vector count after upload: {collection_info.vectors_count}")
        
        return client
    
    except Exception as e:
        print(f"Error loading exported database: {str(e)}")
        return False

if __name__ == "__main__":
    # Test loading the database
    client = load_exported_db()
    if client:
        print("Successfully loaded database")
    else:
        print("Failed to load database")
