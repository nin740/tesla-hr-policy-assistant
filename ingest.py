#!/usr/bin/env python

"""
Ingest script for Tesla HR Policy Assistant.
This script loads the Tesla Employee Handbook PDF, splits it into chunks,
embeds them using Azure OpenAI, and stores them in Qdrant (local or cloud).

Usage:
  python ingest.py         # Use local Qdrant database
  python ingest.py cloud   # Use Qdrant cloud service
"""

import sys
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient

# Load environment variables
load_dotenv()

def main():
    # Check if we should use cloud or local
    use_cloud = len(sys.argv) > 1 and sys.argv[1].lower() == "cloud"
    
    # Initialize Azure OpenAI embeddings
    try:
        embeddings = AzureOpenAIEmbeddings(
            deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION")
        )
        print("✅ Successfully initialized Azure OpenAI embeddings.")
    except Exception as e:
        print(f"❌ Error initializing embeddings: {str(e)}")
        print("Please check your Azure OpenAI environment variables.")
        return
    
    # Load and split the document
    try:
        print("Loading PDF document...")
        loader = PyPDFLoader("data/Tesla_Employee_Handbook.pdf")
        documents = loader.load()
        print(f"✅ Loaded {len(documents)} pages from PDF.")
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)
        print(f"✅ Split into {len(texts)} text chunks.")
    except Exception as e:
        print(f"❌ Error loading or splitting document: {str(e)}")
        return
    
    # Initialize Qdrant client
    try:
        if use_cloud:
            # Use Qdrant cloud
            qdrant_url = os.getenv("QDRANT_URL")
            qdrant_api_key = os.getenv("QDRANT_API_KEY")
            
            if not qdrant_url or not qdrant_api_key:
                print("❌ Missing QDRANT_URL or QDRANT_API_KEY environment variables.")
                return
                
            print(f"Connecting to Qdrant cloud at {qdrant_url}...")
            client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
            print("✅ Connected to Qdrant cloud.")
        else:
            # Use local storage
            print("Using local Qdrant database...")
            client = QdrantClient(path="./local_qdrant_db")
            print("✅ Initialized local Qdrant database.")
    except Exception as e:
        print(f"❌ Error initializing Qdrant client: {str(e)}")
        return
    
    # Create a vector store from the documents
    try:
        print("Creating vector store from documents...")
        Qdrant.from_documents(
            documents=texts,
            embedding=embeddings,
            url=None,  # Use the client we already initialized
            collection_name="hr-policies",
            client=client
        )
        print("✅ Successfully created vector store.")
        
        # Verify the collection was created
        collection_info = client.get_collection(collection_name="hr-policies")
        points_count = collection_info.points_count or 0
        print(f"✅ Collection 'hr-policies' contains {points_count} points.")
        
        if use_cloud:
            print("\n🔥 IMPORTANT: Your Qdrant cloud collection is now ready to use with Streamlit Cloud.")
            print("Make sure to set these environment variables in your Streamlit Cloud app:")
            print("  - QDRANT_URL = your_qdrant_url")
            print("  - QDRANT_API_KEY = your_qdrant_api_key")
            print("  - IS_STREAMLIT_CLOUD = true")
        else:
            print("\n🔥 IMPORTANT: Your local Qdrant database is now ready to use.")
            print("This database is stored in the ./local_qdrant_db directory.")
    except Exception as e:
        print(f"❌ Error creating vector store: {str(e)}")

if __name__ == "__main__":
    main()