import os
import requests
import json
from dotenv import load_dotenv
from datetime import datetime
import uuid

# Load environment variables
load_dotenv()

# Initialize Airtable client
AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY")
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_TABLE_NAME = os.getenv("AIRTABLE_TABLE_NAME")

def initialize_airtable():
    """Verify connection to Airtable and return status"""
    
    if not AIRTABLE_API_KEY or not AIRTABLE_BASE_ID or not AIRTABLE_TABLE_NAME:
        return False, "Missing credentials"
        
    try:
        # Attempt to connect to Airtable
        
        # Test connection by listing records
        headers = {
            "Authorization": f"Bearer {AIRTABLE_API_KEY}",
            "Content-Type": "application/json"
        }
        
        url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE_NAME}"
        response = requests.get(url, headers=headers, params={"maxRecords": 1})
        
        if response.status_code == 200:
            return True, "Connected"
        else:
            return False, f"Connection error: {response.status_code}"
            
    except Exception as e:
        return False, f"Connection error: {str(e)[:50]}"

def save_chat_interaction(session_id, question, answer):
    """
    Save a chat interaction to Airtable
    
    Args:
        session_id: Unique identifier for the chat session
        question: The user's question
        answer: The assistant's answer
    
    Returns:
        The ID of the inserted record or None if failed
    """
    if not AIRTABLE_API_KEY or not AIRTABLE_BASE_ID or not AIRTABLE_TABLE_NAME:
        return None
        
    try:
        # Create record data
        record_data = {
            "fields": {
                "Session ID": session_id,
                "Question": question,
                "Answer": answer
                # Timestamp will be added automatically by Airtable
            }
        }
        
        # Insert into Airtable
        headers = {
            "Authorization": f"Bearer {AIRTABLE_API_KEY}",
            "Content-Type": "application/json"
        }
        
        url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE_NAME}"
        
        response = requests.post(url, headers=headers, json={"records": [record_data]})
        
        if response.status_code == 200:
            result = response.json()
            record_id = result["records"][0]["id"]
            return record_id
        else:
            return None
            
    except Exception as e:
        return None

def get_chat_history(session_id):
    """
    Retrieve chat history for a specific session
    
    Args:
        session_id: The session ID to retrieve history for
        
    Returns:
        List of chat messages formatted for the app or empty list if none found
    """
    if not AIRTABLE_API_KEY or not AIRTABLE_BASE_ID or not AIRTABLE_TABLE_NAME:
        print("Airtable client not initialized")
        return []
        
    try:
        # Retrieve chat history for the specified session
        
        headers = {
            "Authorization": f"Bearer {AIRTABLE_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Filter by session_id
        filter_formula = f"{{Session ID}} = '{session_id}'"
        url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE_NAME}"
        
        response = requests.get(
            url, 
            headers=headers, 
            params={
                "filterByFormula": filter_formula,
                # Use Timestamp as the column name
                "sort[0][field]": "Timestamp",
                "sort[0][direction]": "asc"
            }
        )
        
        if response.status_code != 200:
            return []
            
        result = response.json()
        records = result.get("records", [])
        
        # Convert to the format expected by the app
        formatted_history = []
        for record in records:
            fields = record.get("fields", {})
            
            # Add user question
            formatted_history.append({
                "role": "user",
                "content": fields.get("Question", "")
            })
            
            # Add assistant answer
            formatted_history.append({
                "role": "assistant",
                "content": fields.get("Answer", "")
            })
            
        return formatted_history
    except Exception as e:
        return []

def delete_chat_history(session_id):
    """
    Delete all chat interactions for a specific session
    
    Args:
        session_id: The session ID to delete history for
        
    Returns:
        True if successful, False otherwise
    """
    if not AIRTABLE_API_KEY or not AIRTABLE_BASE_ID or not AIRTABLE_TABLE_NAME:
        return False
        
    try:
        # Delete all records for the specified session
        
        # First, get all records for this session
        headers = {
            "Authorization": f"Bearer {AIRTABLE_API_KEY}",
            "Content-Type": "application/json"
        }
        
        filter_formula = f"{{Session ID}} = '{session_id}'"
        url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE_NAME}"
        
        response = requests.get(
            url, 
            headers=headers, 
            params={
                "filterByFormula": filter_formula,
                "fields": ["id"]  # Only get the IDs
            }
        )
        
        if response.status_code != 200:
            return False
            
        result = response.json()
        records = result.get("records", [])
        
        if not records:
            return True
            
        # Delete records in batches (Airtable has a limit of 10 records per delete request)
        record_ids = [record["id"] for record in records]
        batch_size = 10
        
        for i in range(0, len(record_ids), batch_size):
            batch = record_ids[i:i+batch_size]
            delete_url = f"{url}?records[]=" + "&records[]=".join(batch)
            
            delete_response = requests.delete(delete_url, headers=headers)
            
            if delete_response.status_code != 200:
                return False
                
        return True
    except Exception as e:
        return False

def get_unique_sessions():
    """
    Get a list of all unique session IDs with their first message
    
    Returns:
        List of dictionaries with session_id and first_message
    """
    if not AIRTABLE_API_KEY or not AIRTABLE_BASE_ID or not AIRTABLE_TABLE_NAME:
        print("Airtable client not initialized")
        return []
        
    try:
        # Get all unique chat sessions
        
        headers = {
            "Authorization": f"Bearer {AIRTABLE_API_KEY}",
            "Content-Type": "application/json"
        }
        
        url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE_NAME}"
        
        # Get all records
        response = requests.get(
            url, 
            headers=headers,
            params={
                # Use Timestamp as the column name
                "sort[0][field]": "Timestamp",
                "sort[0][direction]": "desc"
            }
        )
        
        if response.status_code != 200:
            print(f"Error retrieving from Airtable: {response.status_code} - {response.text}")
            return []
            
        result = response.json()
        records = result.get("records", [])
        
        # Group by session ID and get the first message
        sessions = {}
        for record in records:
            fields = record.get("fields", {})
            session_id = fields.get("Session ID")
            
            if session_id and session_id not in sessions:
                # Get timestamp from fields instead of record metadata
                created_time = fields.get("Timestamp", "")
                
                # Format the timestamp
                try:
                    dt = datetime.fromisoformat(created_time.replace("Z", "+00:00"))
                    formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
                except:
                    formatted_time = created_time
                
                sessions[session_id] = {
                    "session_id": session_id,
                    "first_message": fields.get("Question", "")[:50] + "...",
                    "created_at": formatted_time
                }
        
        return list(sessions.values())
    except Exception as e:
        return []
