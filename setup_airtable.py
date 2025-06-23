import os
import requests
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get Airtable credentials
AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY")
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_TABLE_NAME = os.getenv("AIRTABLE_TABLE_NAME")

# Verify Airtable credentials are present

if not AIRTABLE_API_KEY or not AIRTABLE_BASE_ID or not AIRTABLE_TABLE_NAME:
    print("Error: Missing Airtable credentials in environment variables")
    print("Please make sure AIRTABLE_API_KEY, AIRTABLE_BASE_ID, and AIRTABLE_TABLE_NAME are set in your .env file")
    exit(1)

# Set up headers for Airtable API requests
headers = {
    "Authorization": f"Bearer {AIRTABLE_API_KEY}",
    "Content-Type": "application/json"
}

# Check if the table exists
try:
    # First, list all tables in the base to verify access
    meta_url = f"https://api.airtable.com/v0/meta/bases/{AIRTABLE_BASE_ID}/tables"
    meta_response = requests.get(meta_url, headers=headers)
    
    # Access the specified table
    url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE_NAME}"
    response = requests.get(url, headers=headers, params={"maxRecords": 1})
    
    if response.status_code == 200:
        print("Successfully connected to Airtable!")
        print("Table exists and is accessible")
    elif response.status_code == 404:
        print("Table not found. Please create a table with the following fields:")
        print("  - Session ID (single line text)")
        print("  - Question (long text)")
        print("  - Answer (long text)")
        print("  - Timestamp (created time or formula)")
        exit(1)
    else:
        print(f"Error connecting to Airtable: {response.status_code} - {response.text}")
        exit(1)
        
    # Check if the table has the required fields
    
    # Get table metadata
    meta_url = f"https://api.airtable.com/v0/meta/bases/{AIRTABLE_BASE_ID}/tables"
    meta_response = requests.get(meta_url, headers=headers)
    
    if meta_response.status_code == 200:
        tables = meta_response.json().get("tables", [])
        target_table = None
        
        # Find our table
        for table in tables:
            if table.get("name") == AIRTABLE_TABLE_NAME or table.get("id") == AIRTABLE_TABLE_NAME:
                target_table = table
                break
                
        if target_table:
            fields = target_table.get("fields", [])
            field_names = [field.get("name") for field in fields]
            
            required_fields = ["Session ID", "Question", "Answer"]
            missing_fields = [field for field in required_fields if field not in field_names]
            
            if missing_fields:
                print(f"Missing required fields: {', '.join(missing_fields)}")
                print("Please add these fields to your Airtable table")
            else:
                print("All required fields are present")
        else:
            print("Could not find table metadata")
    else:
        print(f"Could not check table schema: {meta_response.status_code}")
        print("This might be due to permissions. Make sure your API key has schema access.")
        print("Continuing anyway, but you should verify the table has the required fields.")
    
    # Insert a test record to verify write permissions
    
    test_data = {
        "records": [
            {
                "fields": {
                    "Session ID": "test-session-id",
                    "Question": "This is a test question",
                    "Answer": "This is a test answer"
                }
            }
        ]
    }
    
    create_response = requests.post(url, headers=headers, json=test_data)
    
    if create_response.status_code == 200:
        record_id = create_response.json()["records"][0]["id"]
        print(f"Test record created successfully with ID: {record_id}")
        
        # Delete the test record
        delete_url = f"{url}/{record_id}"
        delete_response = requests.delete(delete_url, headers=headers)
        
        if delete_response.status_code == 200:
            print("Test record deleted successfully")
        else:
            print(f"Could not delete test record: {delete_response.status_code}")
    else:
        print(f"Could not create test record: {create_response.status_code}")
        print("Please check your table permissions and schema")
        exit(1)
        
except Exception as e:
    print(f"Error: {e}")
    exit(1)
    
print("Airtable setup complete! You can now run the main application.")
