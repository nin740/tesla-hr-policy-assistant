# main.py

import os
import base64
import uuid
import json
from datetime import datetime
from io import BytesIO
from PIL import Image
import requests
from dotenv import load_dotenv
import streamlit as st
from langchain_community.vectorstores import Qdrant
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate

# Common FAQ questions
FAQ_QUESTIONS = [
    "What is Tesla's policy on remote work?",
    "How many vacation days do employees get?",
    "What are the health insurance benefits?",
    "What is the parental leave policy?",
    "How does performance review work at Tesla?"
]
from airtable_integration import initialize_db, save_chat_message, get_session_history, delete_session_history, get_all_sessions

# Tesla branding constants
BRAND_PRIMARY_COLOR = "#e82127"  # Tesla Red
BRAND_SECONDARY_COLOR = "#000000"  # Tesla Black
BRAND_BACKGROUND_COLOR = "#121212"  # Dark background
BRAND_TEXT_COLOR = "#f5f5f5"  # Light text for dark background
BRAND_ACCENT_COLOR = "#333333"  # Subtle accent for panels
BRAND_LOGO_PATH = "tesla_logo.jpg"  # Logo path
BRAND_FONT = "'Gotham SSm', 'Helvetica Neue', Helvetica, Arial, sans-serif"

# Load environment variables
load_dotenv()

# Initialize session state for chat history, settings, and session ID
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'show_sources' not in st.session_state:
    st.session_state.show_sources = True

if "example_query" not in st.session_state:
    st.session_state.example_query = None

if "loaded_from_airtable" not in st.session_state:
    st.session_state.loaded_from_airtable = False

if "airtable_enabled" not in st.session_state:
    st.session_state.airtable_enabled = False  # Disable Airtable by default
    
if "local_chat_sessions" not in st.session_state:
    st.session_state.local_chat_sessions = {}  # Store sessions locally

# Initialize Airtable connection
airtable_connected, airtable_status = initialize_db()

# Flag to track if we're using Airtable or local storage
using_airtable = airtable_connected
st.session_state.airtable_enabled = using_airtable

if not using_airtable:
    # Falling back to local chat history storage
    pass
else:
    # Using Airtable for chat history storage
    pass

# Function to save chat history locally
def save_chat_history_locally():
    if st.session_state.session_id and st.session_state.chat_history:
        # Store the current session in the local_chat_sessions dictionary
        # Include timestamp for sorting
        current_time = datetime.now().isoformat()
        st.session_state.local_chat_sessions[st.session_state.session_id] = {
            "messages": st.session_state.chat_history.copy(),
            "timestamp": current_time,
            "first_message": st.session_state.chat_history[0]["content"] if st.session_state.chat_history else "New conversation"
        }

# Function to load chat history from local storage
def load_chat_history_from_local(session_id):
    if session_id in st.session_state.local_chat_sessions:
        st.session_state.chat_history = st.session_state.local_chat_sessions[session_id]["messages"].copy()
        return True
    return False

# Function to load chat history from Airtable
def load_chat_history_from_airtable():
    # Always attempt to load history when explicitly called, regardless of loaded_from_airtable flag
    if st.session_state.session_id and st.session_state.airtable_enabled:
        try:
            # Attempting to load chat history
            messages = get_session_history(st.session_state.session_id)
            if messages:
                st.session_state.chat_history = messages
                st.session_state.loaded_from_airtable = True
                # Successfully loaded messages from Airtable
                return True
            else:
                # No messages found for this session in Airtable
                pass
        except Exception as e:
            # Error loading chat history from Airtable
            pass
            st.session_state.airtable_enabled = False
    return False

# Load chat history from Airtable if available
load_chat_history_from_airtable()

# Function to initialize Azure OpenAI embeddings
def get_embeddings():
    try:
        # Check for required environment variables
        required_vars = [
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_API_VERSION",
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT"
        ]
        
        missing_vars = [var for var in required_vars if not os.environ.get(var)]
        
        if missing_vars:
            st.error(f"Missing required environment variables: {', '.join(missing_vars)}")
            return None
            
        return AzureOpenAIEmbeddings(
            azure_deployment=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"],
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        )
    except Exception as e:
        st.error(f"Error initializing embeddings: {str(e)}")
        return None

# Function to initialize Qdrant client with local connection
def get_qdrant_store():
    from qdrant_client import QdrantClient
    import os
    
    embedding = get_embeddings()
    
    try:
        # Check if we're in a cloud environment (Streamlit Cloud)
        is_cloud = os.environ.get('IS_STREAMLIT_CLOUD') == 'true'
        
        if is_cloud:
            # For cloud deployment, use in-memory Qdrant
            client = QdrantClient(":memory:")
            
            # Create the collection if it doesn't exist
            from qdrant_client.models import VectorParams, Distance
            try:
                client.get_collection(collection_name="hr-policies")
            except Exception:
                # Collection doesn't exist, create it
                client.create_collection(
                    collection_name="hr-policies",
                    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
                )
                # Note: In a real deployment, you would need to upload your vectors here
                # This is just a placeholder to prevent errors
        else:
            # For local development, use local file-based storage
            client = QdrantClient(path="./local_qdrant_db")
        
        return Qdrant(
            client=client,
            collection_name="hr-policies",
            embeddings=embedding
        )
    except Exception as e:
        st.error(f"Error connecting to vector database: {str(e)}")
        # Return a dummy store that won't crash the app but won't return results
        return None

# Function to initialize Azure OpenAI Chat model
def get_chat_model():
    try:
        # Check for required environment variables
        required_vars = [
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_API_VERSION",
            "AZURE_OPENAI_CHAT_DEPLOYMENT"
        ]
        
        missing_vars = [var for var in required_vars if not os.environ.get(var)]
        
        if missing_vars:
            st.error(f"Missing required environment variables: {', '.join(missing_vars)}")
            return None
            
        return AzureChatOpenAI(
            azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            temperature=0.3,
            max_tokens=500
        )
    except Exception as e:
        st.error(f"Error initializing chat model: {str(e)}")
        return None

# System prompt for the chat model
SYSTEM_PROMPT = """
You are an HR assistant that helps employees understand Tesla's policies and benefits.

Follow these guidelines for your answers:

## CONTENT GUIDELINES:
1. Be accurate, based on the provided HR policy documentation.
2. Include specific data like dollar amounts, plan names, and coverage tiers when available.
3. If you don't know the answer, just say that you don't know, don't try to make up an answer.
4. If the user's current message refers to a previous topic, use the last 2 Q&A pairs to infer the full context.
   For example, if they previously asked about remote work and now ask "What about interns?", interpret this as asking about
   remote work policies for interns.

## FORMATTING GUIDELINES (VERY IMPORTANT):
1. Keep answers concise and user-friendly - limit to 2-4 short paragraphs maximum (under 8 sentences total).
2. Start with a clear, direct answer (yes/no/summary), followed by key details like eligibility or duration.
3. Use bullet points for comparing options or listing multiple items.
4. Avoid essay-like structures or long-winded legal text unless explicitly requested.
5. End with a brief reference suggestion like: "Refer to the Benefits Guide for more details" or "Contact HR for specific eligibility questions."

Example format for a response about parental leave:

"Tesla offers paid parental leave to eligible full-time employees. The duration is X weeks for primary caregivers and Y weeks for secondary caregivers.

Both biological and adoptive parents are covered, including same-sex partners. Eligibility begins after [time period] of employment.

For complete details, check the Benefits Guide or speak with HR."
"""

# Custom prompt template that includes previous conversation context
template = """
You are an HR assistant that helps employees understand company policies.

Use the following pieces of context to answer the question at the end.
Follow these guidelines for your answers:
1. Be SPECIFIC and DETAILED - include concrete data like dollar amounts, plan names, coverage tiers, etc.
2. NEVER use vague phrases like "may vary" or "refer to the document" - provide the actual information.
3. Use bullet points or markdown formatting when comparing different options (e.g., PPO Base vs PPO Plus).
4. Organize information in a clear, structured way using headings and lists where appropriate.
5. If multiple sources provide different pieces of information, synthesize them into a comprehensive answer.
6. If you don't know the answer, just say that you don't know, don't try to make up an answer.
7. IMPORTANT: For follow-up questions, always maintain the context from previous messages. If the current question is ambiguous (e.g., "What about interns?"), interpret it as a continuation of the previous topic.

Context: {context}

Conversation History:
{conversation_history}

Current Question: {question}

Clarified Question: {clarified_question}

Answer:"""

PROMPT = PromptTemplate(
    template=template,
    input_variables=["context", "conversation_history", "question", "clarified_question"]
)

# Function to get answer with source documents
def get_answer(query):
    # Initialize vector store and chat model
    qdrant_store = get_qdrant_store()
    chat_model = get_chat_model()
    
    # Check if vector store or chat model is available
    if qdrant_store is None:
        return "Unable to process your request at this time. The vector database is not available. Please try again later or contact support.", []
        
    if chat_model is None:
        return "Unable to process your request at this time. The AI model is not available. Please try again later or contact support.", []
    
    # Create retriever with search parameters
    retriever = qdrant_store.as_retriever(
        search_kwargs={
            "k": 5,  # Retrieve top 5 most relevant chunks
            "score_threshold": 0.5  # Only include chunks with relevance score above threshold
        }
    )
    
    # Prepare messages list for chat completion format
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    
    # Add the last 2 Q&A pairs (4 messages) from chat history if available
    if st.session_state.chat_history:
        # Extract just the last 2 Q&A pairs (up to 4 messages)
        # We want to get complete pairs, so we need to be careful about slicing
        history = st.session_state.chat_history.copy()
        
        # Find the last 2 user messages (which start the Q&A pairs)
        user_indices = [i for i, msg in enumerate(history) if msg["role"] == "user"]
        
        # If we have at least 2 user messages, get the last 2 Q&A pairs
        if len(user_indices) >= 2:
            # Start from the second-to-last user message
            start_idx = user_indices[-2]
            context_messages = history[start_idx:]
        else:
            # If we have fewer than 2 user messages, just use what we have
            context_messages = history
        
        # Add these messages to our messages list
        for msg in context_messages:
            messages.append({"role": msg["role"], "content": msg["content"]})
    
    # Add the current query as the final user message
    messages.append({"role": "user", "content": query})
    
    # Retrieve relevant documents based on the query
    docs = retriever.get_relevant_documents(query)
    
    # Format the retrieved documents as context
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Add the context to the system message
    messages[0]["content"] += f"\n\nUse the following information to answer the user's question:\n{context}"
    
    # Generate response using the chat model with the messages format
    response = chat_model.invoke(messages)
    
    # Extract the answer from the response
    answer = response.content
    
    return {
        "answer": answer,
        "source_documents": docs
    }

# Function to load and display the Tesla logo
def display_tesla_logo():
    try:
        # Try to load the Tesla logo from the file
        try:
            # Load the official Tesla logo with transparent background
            logo = Image.open(BRAND_LOGO_PATH)
            return logo
        except FileNotFoundError:
            # If file not found, create a simple Tesla logo using text
            # Create a blank image with black background
            img = Image.new('RGB', (200, 100), color=(0, 0, 0))
            return img
    except Exception as e:
        # Error loading Tesla logo
        # Return a simple blank image as fallback
        return Image.new('RGB', (200, 100), color=(0, 0, 0))

# Helper function to convert image to base64 for inline display
def image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Function to apply Tesla styling to the app
def apply_tesla_styling():
    # Load CSS from external file for Tesla branding
    try:
        # Try relative path first
        css_path = "static/styles.css"
        if not os.path.exists(css_path):
            # Try absolute path based on script location
            script_dir = os.path.dirname(os.path.abspath(__file__))
            css_path = os.path.join(script_dir, "static/styles.css")
        
        with open(css_path, "r") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    except Exception as e:
        # Fallback to inline CSS if file can't be loaded
        st.markdown(f"""
        <style>
            /* Main elements - dark theme */
            .stApp {{background-color: {BRAND_BACKGROUND_COLOR};}}
            .main {{background-color: {BRAND_BACKGROUND_COLOR};}}
            h1, h2, h3 {{color: {BRAND_TEXT_COLOR}; font-family: {BRAND_FONT};}}
            p, li, div, span, label {{color: {BRAND_TEXT_COLOR} !important; font-family: {BRAND_FONT};}}
            
            /* Buttons */
            .stButton>button {{background-color: {BRAND_PRIMARY_COLOR}; color: white;}}
            
            /* Sidebar */
            [data-testid=stSidebar] {{background-color: {BRAND_SECONDARY_COLOR};}}
        </style>
        """, unsafe_allow_html=True)

# Function to clean up boilerplate headers from chunk text
def clean_chunk_text(text):
    # Remove common boilerplate headers
    boilerplate_patterns = [
        "Your Health Your Finances Your Eligibility",
        "Your Health Your Finances",
        "Your Eligibility",
        "TESLA, INC. CONFIDENTIAL INFORMATION",
        "TESLA EMPLOYEE HANDBOOK",
        "Your Health Your Family Your Perks"
    ]
    
    cleaned_text = text
    for pattern in boilerplate_patterns:
        cleaned_text = cleaned_text.replace(pattern, "")
    
    # Remove extra whitespace and newlines
    cleaned_text = "\n".join([line.strip() for line in cleaned_text.split("\n") if line.strip()])
    
    return cleaned_text

# Function to format and display source documents
def display_sources(source_docs):
    if not source_docs:
        return
    
    # Professional source document display
    with st.expander("Source References", expanded=False):
        st.markdown(f"<p style='font-size: 12px; color: #999999;'>The following information sources were used to generate this response:</p>", unsafe_allow_html=True)
        
        for i, doc in enumerate(source_docs):
            # Clean the text
            cleaned_content = clean_chunk_text(doc.page_content)
            
            # Create collapsible section for each source with professional styling
            with st.expander(f"Employee Handbook - Page {doc.metadata.get('page', 'Unknown')}"):
                st.markdown(f"<div style='font-size: 14px; line-height: 1.5;'>{cleaned_content}</div>", unsafe_allow_html=True)
            

# Streamlit UI
st.set_page_config(
    page_title="Tesla HR Policy Assistant",
    page_icon="T",  # T for Tesla
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply Tesla styling
apply_tesla_styling()

# Main panel - minimal, professional design
col1, col2, col3 = st.columns([1, 10, 1])
with col2:
    st.markdown("<h1 style='text-align: center; margin-top: 20px; margin-bottom: 10px; font-weight: 300;'>Tesla HR Policy Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; margin-bottom: 30px; color: #999999;'>Access information about Tesla employee benefits and policies</p>", unsafe_allow_html=True)
    st.divider()

# Define function to get unique sessions (either from Airtable or local storage)
def get_unique_sessions():
    # If Airtable is enabled and connected, use it
    if st.session_state.airtable_enabled:
        try:
            # Get unique session IDs with their latest timestamp
            sessions = get_all_sessions()
            
            # Convert to the format expected by the app
            formatted_sessions = []
            for session in sessions:
                # Format timestamp for display
                formatted_date = session["created_at"]
                if isinstance(formatted_date, str):
                    try:
                        # Try to parse and reformat the date if it's a string
                        dt = datetime.fromisoformat(formatted_date.replace("Z", "+00:00"))
                        formatted_date = dt.strftime('%Y-%m-%d %H:%M')
                    except:
                        # Keep as is if parsing fails
                        pass
                
                # Truncate first message if too long
                first_message = session["first_message"]
                if len(first_message) > 30:
                    first_message = first_message[:27] + "..."
                
                formatted_sessions.append({
                    "id": session["session_id"],
                    "label": f"{formatted_date}: {first_message}",
                    "timestamp": session["created_at"]
                })
            
            return formatted_sessions
        except Exception as e:
            print(f"Error getting unique sessions from Airtable: {e}")
            st.session_state.airtable_enabled = False
            # Fall back to local sessions
        
    # Use local storage if Airtable is not available
    sessions = []
    for session_id, session_data in st.session_state.local_chat_sessions.items():
        # Skip current session
        if session_id == st.session_state.session_id:
            continue
            
        # Format timestamp for display
        try:
            timestamp = datetime.fromisoformat(session_data['timestamp'])
            formatted_date = timestamp.strftime('%Y-%m-%d %H:%M')
        except:
            formatted_date = "Unknown date"
            
        # Get first message
        first_message = session_data.get('first_message', "New conversation")
        if len(first_message) > 30:
            first_message = first_message[:27] + "..."
            
        # Add session to list
        sessions.append({
            "id": session_id,
            "label": f"{formatted_date}: {first_message}",
            "timestamp": session_data['timestamp']
        })
    
    return sorted(sessions, key=lambda x: x.get('timestamp', ''), reverse=True)

# Sidebar with Tesla branding - centered with CSS
st.sidebar.markdown("""
<div style="display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; margin-top: 0px; margin-bottom: 0px;">
    <h1 style="margin-top: 0px; margin-bottom: 5px;">Tesla HR Policy Assistant</h1>
</div>
""", unsafe_allow_html=True)

# Center the Tesla logo with CSS
st.sidebar.markdown("""
<div style="display: flex; justify-content: center; margin-bottom: 0px;">
    <img src="https://upload.wikimedia.org/wikipedia/commons/e/e8/Tesla_logo.png" width="120">
</div>
""", unsafe_allow_html=True)

# Add spacing after the logo
st.sidebar.markdown("""
<div style="margin-bottom: 50px;"></div>
""", unsafe_allow_html=True)

# Add chat history section to sidebar (collapsible and collapsed by default)
with st.sidebar.expander("Chat History", expanded=False):
    # Get available chat sessions
    st.session_state.available_sessions = get_unique_sessions()
    
    # Refresh button for chat history
    if st.button("Refresh History", use_container_width=True, key="refresh_history_btn"):
        st.session_state.available_sessions = get_unique_sessions()
        st.rerun()
    
    # Display available sessions
    if st.session_state.available_sessions:
        # Display sessions without timestamps in the label
        for session in st.session_state.available_sessions:
            # Skip current session
            if session["id"] == st.session_state.session_id:
                continue
                
            # Extract just the first message without timestamp
            first_message = session.get("first_message", "")
            if not first_message and "label" in session:
                # Extract message from label if first_message is not available
                parts = session["label"].split(": ", 1)
                if len(parts) > 1:
                    first_message = parts[1]
                else:
                    first_message = "Chat session"
            
            # Create a button for each session with just the message
            if st.button(first_message, key=f"session_{session['id']}", use_container_width=True):
                # Load the selected session
                st.session_state.session_id = session["id"]
                st.session_state.chat_history = []
                st.session_state.loaded_from_airtable = False  # Reset this flag to force reload
                
                # Try to load from Airtable
                load_chat_history_from_airtable()
                
                st.rerun()
    else:
        st.write("No previous conversations found.")
        
    # Always show a button to start a new conversation
    if st.button("Start New Conversation", use_container_width=True, type="primary", key="new_chat_btn"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.chat_history = []
        st.session_state.loaded_from_airtable = False
        st.rerun()

# Separator
st.sidebar.markdown("---")

# FAQ section in sidebar
st.sidebar.markdown("### Quick FAQs")

# Function to immediately answer FAQ questions
def answer_faq_question(question):
    # Predefined answers for FAQ questions
    faq_answers = {
        "What is Tesla's policy on remote work?": "Tesla's remote work policy generally requires employees to be in the office for a minimum of 40 hours per week. Exceptions may be granted for exceptional contributors who cannot reasonably commute to an office, subject to executive approval. The company believes that in-person collaboration is essential for innovation and company culture.",
        
        "How many vacation days do employees get?": "Tesla provides full-time employees with a competitive Paid Time Off (PTO) package that typically includes:\n- 10-15 days of vacation annually, increasing with tenure\n- 7-9 paid holidays per year\n- Sick leave as required by local regulations\n- Parental leave benefits\nSpecific entitlements may vary based on location, position, and tenure with the company.",
        
        "What are the health insurance benefits?": "Tesla offers comprehensive health insurance benefits to eligible employees, including:\n- Medical, dental, and vision coverage\n- Health Savings Account (HSA) or Flexible Spending Account (FSA) options\n- Mental health resources and Employee Assistance Program\n- Wellness programs and incentives\nCoverage options and employee contributions vary by location and employment status.",
        
        "What is the parental leave policy?": "Tesla's parental leave policy provides:\n- Birth mothers: Up to 10 weeks of paid pregnancy disability leave plus additional parental bonding leave\n- All parents: Typically 6-8 weeks of paid parental bonding leave\n- Additional unpaid leave options in accordance with local regulations\nThe policy aims to support employees during this important life transition while complying with local laws.",
        
        "How does performance review work at Tesla?": "Tesla's performance review process typically includes:\n- Regular 1:1 meetings with managers\n- Semi-annual or annual formal reviews\n- Goal-setting aligned with company objectives\n- Peer feedback components\n- Performance ratings that influence compensation adjustments and advancement opportunities\nThe company emphasizes high standards and rewards exceptional performance."
    }
    
    # Display user question
    with st.chat_message("user"):
        st.write(question)
    
    # Add to chat history
    st.session_state.chat_history.append({"role": "user", "content": question})
    
    # Save to Airtable if enabled, otherwise save locally
    if st.session_state.airtable_enabled:
        try:
            save_chat_message(st.session_state.session_id, "user", question)
        except Exception as e:
            print(f"Error saving user message to Airtable: {e}")
            st.session_state.airtable_enabled = False
            save_chat_history_locally()
    else:
        save_chat_history_locally()
    
    # Get the answer from predefined answers or generate one
    if question in faq_answers:
        answer = faq_answers[question]
        sources = []
    else:
        # For non-predefined questions, use the regular answer generation
        with st.spinner():
            result = get_answer(question)
            answer = result["answer"]
            sources = result["source_documents"]
    
    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(answer)
        if st.session_state.show_sources and sources:
            display_sources(sources)
    
    # Add to chat history
    st.session_state.chat_history.append({
        "role": "assistant", 
        "content": answer,
        "sources": sources if sources else None
    })
    
    # Save to Airtable if enabled, otherwise save locally
    if st.session_state.airtable_enabled:
        try:
            save_chat_message(st.session_state.session_id, "assistant", answer, sources)
        except Exception as e:
            print(f"Error saving assistant message to Airtable: {e}")
            st.session_state.airtable_enabled = False
            save_chat_history_locally()
    else:
        save_chat_history_locally()
    
    # Force a rerun to update the UI
    st.rerun()

# Add FAQ buttons in the sidebar
for i, question in enumerate(FAQ_QUESTIONS):
    if st.sidebar.button(
        question,
        key=f"sidebar_faq_{i}",
        use_container_width=True,
        type="secondary"
    ):
        answer_faq_question(question)

# Separator
st.sidebar.markdown("---")

# Settings section
st.sidebar.markdown("### Settings")

# Toggle for showing sources
st.session_state.show_sources = st.sidebar.toggle("Show document sources", value=st.session_state.show_sources)

# Separator
st.sidebar.markdown("---")

# Main chat interface - sleek, professional layout
col1, col2, col3 = st.columns([1, 10, 1])
with col2:
    # Create a container structure to ensure chat input stays at bottom
    main_container = st.container()
    # Add padding at the top
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    
    # Create a precisely positioned chat input that aligns with the main content area
    st.markdown("""
    <style>
    /* Define exact measurements for positioning */
    :root {
        --sidebar-width: 300px; /* Exact sidebar width in pixels */
        --main-content-width: 800px;
    }
    
    /* Define the main content width and center position */
    .main .block-container {
        max-width: var(--main-content-width) !important;
        margin-left: auto !important;
        margin-right: auto !important;
        padding-bottom: 100px !important;
    }
    
    /* Position the chat input container with precise offset and centering */
    .element-container:has(.stChatInput) {
        position: fixed !important;
        bottom: 20px !important;
        width: 100% !important;
        z-index: 999999 !important;
        pointer-events: none !important;
        /* Use 50% positioning with precise offset to align with title */
        left: 50% !important;
        transform: translateX(calc(-50% + 90px)) !important; /* Nudge to the right by increasing offset to 90px */
        max-width: 800px !important;
    }
    
    /* Style the chat input itself */
    .stChatInput {
        max-width: var(--main-content-width) !important;
        width: 100% !important;
        pointer-events: auto !important;
        margin: 0 !important;
    }
    
    /* Style the textarea to look like ChatGPT's input */
    .stChatInput textarea, .stChatInput > div {
        background-color: #1e1e1e !important;
        border-radius: 20px !important;
        border: none !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15) !important;
        padding: 12px 16px !important;
    }
    
    /* Style the send button */
    .stChatInput button {
        border-radius: 20px !important;
        background-color: #1e1e1e !important;
        margin-left: 8px !important;
    }
    
    /* Hide the footer */
    footer {display: none !important;}
    
    /* Ensure the chat messages don't get hidden */
    .stChatMessage {margin-bottom: 20px !important;}
    
    /* Add responsive adjustments */
    @media (max-width: 992px) {
        .element-container:has(.stChatInput) {
            left: 0 !important; /* Reset position when sidebar collapses */
            padding: 0 5% !important; /* Add padding instead */
        }
    }
    
    @media (max-width: 768px) {
        .stChatInput {
            width: 100% !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Add an empty div with height to push content up
    st.markdown('<div style="height: 100px;"></div>', unsafe_allow_html=True)
    
    # Empty space for clean landing page
    if not st.session_state.chat_history:
        # Just add some vertical spacing
        st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
    
    # Chat container with custom styling - scrollable area
    chat_container = st.container()
    with chat_container:
        # Display chat history
        if st.session_state.chat_history:
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    with st.chat_message("user"):
                        st.write(message["content"])
                else:
                    with st.chat_message("assistant"):
                        # Use markdown for the answer to preserve formatting
                        st.markdown(message["content"])
                        if "sources" in message and st.session_state.show_sources:
                            display_sources(message["sources"])
    
    # Function to set example query (keeping this for sidebar FAQ functionality)
    def set_example_query(question):
        st.session_state.example_query = question
        st.rerun()
    
    # Add spacer to push chat input to bottom
    st.markdown("<div style='flex-grow: 1; min-height: 30px;'></div>", unsafe_allow_html=True)
    
    # Chat input at the bottom - styled for professional appearance
    query = st.chat_input("Ask a question about Tesla HR policies...")
    
    # Check if we have an example query from the sidebar
    if 'example_query' in st.session_state and st.session_state.example_query:
        query = st.session_state.example_query
        # Clear it so it doesn't get processed again
        st.session_state.example_query = None
    
    if query:
        # Display user message
        with st.chat_message("user"):
            st.write(query)
        
        # Add to chat history
        st.session_state.chat_history.append({"role": "user", "content": query})
        
        # Save to Airtable if enabled, otherwise save locally
        if st.session_state.airtable_enabled:
            try:
                save_chat_message(st.session_state.session_id, "user", query)
            except Exception as e:
                # Error saving message to Airtable
                # If we encounter an error, disable Airtable for the session
                st.session_state.airtable_enabled = False
                # Fall back to local storage
                save_chat_history_locally()
        else:
            # Save to local storage
            save_chat_history_locally()
        
        # Get answer with sources - professional loading indicator
        with st.spinner():
            try:
                result = get_answer(query)
                answer = result["answer"]
                sources = result["source_documents"]
                
                # Display assistant response
                with st.chat_message("assistant"):
                    # Use markdown to preserve formatting like bullet points and headings
                    st.markdown(answer)
                    if st.session_state.show_sources:
                        display_sources(sources)
                
                # Add to chat history
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": answer,
                    "sources": sources
                })
                
                # Save to Airtable if enabled, otherwise save locally
                if st.session_state.airtable_enabled:
                    try:
                        save_chat_message(st.session_state.session_id, "assistant", answer, sources)
                    except Exception as e:
                        # Error saving message to Airtable
                        # If we encounter an error, disable Airtable for the session
                        st.session_state.airtable_enabled = False
                        # Fall back to local storage
                        save_chat_history_locally()
                else:
                    # Save to local storage
                    save_chat_history_locally()
                
            except Exception as e:
                # Professional error message
                error_message = "Unable to process your request at this time. Please try again or contact HR support for assistance."
                with st.chat_message("assistant"):
                    st.markdown(f"<div style='color: {BRAND_PRIMARY_COLOR};'>{error_message}</div>", unsafe_allow_html=True)
                
                # Add error message to chat history
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": error_message
                })
                
                # Save to Airtable if enabled, otherwise save locally
                if st.session_state.airtable_enabled:
                    try:
                        save_chat_message(st.session_state.session_id, "assistant", error_message)
                    except Exception as e:
                        # Error saving message to Airtable
                        # If we encounter an error, disable Airtable for the session
                        st.session_state.airtable_enabled = False
                        # Fall back to local storage
                        save_chat_history_locally()
                else:
                    # Save to local storage
                    save_chat_history_locally()