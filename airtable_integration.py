"""
Airtable integration for the HR FAQ chatbot.
This module provides functions to interact with Airtable for storing chat history.
"""

import os
from dotenv import load_dotenv
from airtable_client import (
    initialize_airtable, 
    save_chat_interaction, 
    get_chat_history, 
    delete_chat_history,
    get_unique_sessions
)

# Load environment variables
load_dotenv()

def initialize_db():
    """Initialize the Airtable database connection"""
    airtable_connected, status = initialize_airtable()
    return airtable_connected, status

def save_chat_message(session_id, role, content, sources=None):
    """
    Save a chat message to Airtable
    
    Args:
        session_id: The session ID
        role: Either 'user' or 'assistant'
        content: The message content
        sources: Optional sources for assistant responses
    
    Returns:
        True if successful, False otherwise
    """
    # For Airtable, we only save complete interactions (question-answer pairs)
    # We'll handle this in the main application logic
    if role == "user":
        # Store temporarily until we have the assistant's response
        return True
    elif role == "assistant":
        # Find the last user message in the chat history
        # This assumes the chat history is maintained in the application
        # and this function is called right after adding messages to it
        from streamlit import session_state
        
        # Find the last user message
        last_user_message = ""
        if hasattr(session_state, "chat_history") and session_state.chat_history:
            for msg in reversed(session_state.chat_history):
                if msg["role"] == "user":
                    last_user_message = msg["content"]
                    break
        
        # Save the complete interaction
        record_id = save_chat_interaction(session_id, last_user_message, content)
        return record_id is not None
    
    return False

def get_session_history(session_id):
    """
    Get chat history for a specific session
    
    Args:
        session_id: The session ID
    
    Returns:
        List of chat messages
    """
    return get_chat_history(session_id)

def delete_session_history(session_id):
    """
    Delete chat history for a specific session
    
    Args:
        session_id: The session ID
    
    Returns:
        True if successful, False otherwise
    """
    return delete_chat_history(session_id)

def get_all_sessions():
    """
    Get all unique chat sessions
    
    Returns:
        List of session data
    """
    return get_unique_sessions()
