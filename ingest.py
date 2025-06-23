import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import Qdrant

load_dotenv()

# Load PDF document and split into manageable chunks
loader = PyPDFLoader("data/Tesla_Employee_Handbook.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(docs)

# Initialize Azure OpenAI embedding model
embedding = AzureOpenAIEmbeddings(
    deployment=os.getenv("AZURE_EMBEDDING_DEPLOYMENT"),
    api_key=os.getenv("AZURE_EMBEDDING_KEY"),
    azure_endpoint=os.getenv("AZURE_EMBEDDING_ENDPOINT"),
    api_version=os.getenv("AZURE_EMBEDDING_API_VERSION")
)

# Create Qdrant vector store with document embeddings for semantic search
qdrant = Qdrant.from_documents(
    documents=chunks,
    embedding=embedding,  # from_documents() method uses 'embedding' (singular)
    path="./local_qdrant_db",  # Local storage path
    collection_name="hr-policies",
    force_recreate=True  # Recreate collection if it exists
)

# HR policy chunks embedded and stored in Qdrant