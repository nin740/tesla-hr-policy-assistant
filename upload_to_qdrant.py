#!/usr/bin/env python
"""
Upload vectors from pickle file to Qdrant Cloud.
This script loads the vectors from the qdrant_export.pkl file
and uploads them to your Qdrant Cloud collection.
"""

import os
import pickle
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# Load environment variables
load_dotenv()

def main():
    # Check if pickle file exists
    if not os.path.exists("qdrant_export.pkl"):
        print("❌ Error: qdrant_export.pkl not found")
        return
    
    print("Loading vectors from pickle file...")
    
    # Load export data
    with open("qdrant_export.pkl", "rb") as f:
        export_data = pickle.load(f)
    
    vectors = export_data["vectors"]
    metadatas = export_data["metadatas"]
    texts = export_data["texts"]
    
    # Print debug information
    print(f"Vectors type: {type(vectors)}")
    print(f"First vector type: {type(vectors[0]) if vectors and len(vectors) > 0 else 'No vectors'}")
    print(f"Vector count: {len(vectors) if vectors else 0}")
    print(f"Metadata count: {len(metadatas) if metadatas else 0}")
    print(f"Texts count: {len(texts) if texts else 0}")
    
    # Check if vectors are valid
    if not vectors or len(vectors) == 0:
        print("❌ No vectors found in the export file")
        return
        
    # Check if first vector is None
    if vectors[0] is None:
        print("❌ First vector is None, cannot determine vector size")
        return
    
    print(f"✅ Loaded {len(vectors)} vectors from export file")
    
    # Get Qdrant Cloud credentials
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    
    if not qdrant_url or not qdrant_api_key:
        print("❌ Missing QDRANT_URL or QDRANT_API_KEY environment variables")
        return
    
    print(f"Connecting to Qdrant Cloud at {qdrant_url}...")
    
    try:
        # Initialize Qdrant client
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        print("✅ Connected to Qdrant Cloud")
        
        # Create collection if it doesn't exist
        try:
            client.get_collection(collection_name="hr-policies")
            print("Collection 'hr-policies' already exists")
            
            # Ask if we should recreate the collection
            response = input("Do you want to recreate the collection? (y/n): ")
            if response.lower() == 'y':
                client.delete_collection(collection_name="hr-policies")
                client.create_collection(
                    collection_name="hr-policies",
                    vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE)
                )
                print("✅ Recreated collection 'hr-policies'")
            else:
                print("Using existing collection")
        except Exception:
            # Collection doesn't exist, create it
            client.create_collection(
                collection_name="hr-policies",
                vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE)
            )
            print("✅ Created new collection 'hr-policies'")
        
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
        print(f"✅ Collection now contains {collection_info.vectors_count or collection_info.points_count} vectors")
        
        print("\n🔥 IMPORTANT: Your Qdrant Cloud collection is now ready to use with Streamlit Cloud.")
        print("Make sure to set these environment variables in your Streamlit Cloud app:")
        print("  - QDRANT_URL = your_qdrant_url")
        print("  - QDRANT_API_KEY = your_qdrant_api_key")
        print("  - IS_STREAMLIT_CLOUD = true")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
