#!/usr/bin/env python
"""
Debug script for Streamlit Cloud deployment.
This script helps diagnose issues with file paths and environment variables.
"""

import os
import sys
import streamlit as st

def debug_info():
    """Display debug information."""
    st.title("Debug Information")
    
    # System info
    st.header("System Information")
    st.write(f"Python version: {sys.version}")
    st.write(f"Current working directory: {os.getcwd()}")
    
    # Environment variables
    st.header("Environment Variables")
    env_vars = {
        "IS_STREAMLIT_CLOUD": os.environ.get("IS_STREAMLIT_CLOUD"),
        "AZURE_OPENAI_API_KEY": "***" if os.environ.get("AZURE_OPENAI_API_KEY") else None,
        "AZURE_OPENAI_ENDPOINT": os.environ.get("AZURE_OPENAI_ENDPOINT"),
        "AZURE_OPENAI_API_VERSION": os.environ.get("AZURE_OPENAI_API_VERSION"),
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        "AZURE_OPENAI_CHAT_DEPLOYMENT": os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT"),
    }
    st.json(env_vars)
    
    # File system
    st.header("File System")
    st.subheader("Current Directory Files")
    files = os.listdir(".")
    st.write(files)
    
    # Check for export file
    st.subheader("Export File Check")
    possible_paths = [
        "qdrant_export.pkl",  # Current directory
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "qdrant_export.pkl"),  # Script directory
        "/app/qdrant_export.pkl",  # Streamlit Cloud app directory
        "/mount/src/tesla-hr-policy-assistant/qdrant_export.pkl"  # Another possible Streamlit Cloud path
    ]
    
    for path in possible_paths:
        exists = os.path.exists(path)
        st.write(f"{path}: {'✅ Exists' if exists else '❌ Not found'}")
        if exists:
            st.write(f"File size: {os.path.getsize(path) / 1024:.2f} KB")

if __name__ == "__main__":
    debug_info()
