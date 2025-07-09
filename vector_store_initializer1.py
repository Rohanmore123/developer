import os
import logging
import requests
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(override=True)

# Azure OpenAI configuration for embeddings
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_STT_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY")
AZURE_OPENAI_API_VERSION = "2024-12-01-preview"
EMBEDDING_DEPLOYMENT = "text-embedding-3-small"

if not AZURE_OPENAI_API_KEY:
    logger.warning("❌ AZURE_OPENAI_API_KEY not set")
else:
    logger.info(f"🔑 Using Azure OpenAI API key: {AZURE_OPENAI_API_KEY[:8]}...{AZURE_OPENAI_API_KEY[-5:]}")
    logger.info(f"🔗 Azure OpenAI endpoint: {AZURE_OPENAI_ENDPOINT}")
    logger.info(f"📦 Embedding deployment: {EMBEDDING_DEPLOYMENT}")

# Azure SAS root URL
BLOB_SAS_URL = os.getenv("BLOB_SAS_URL") 
# BLOB_SAS_TOKEN= os.getenv("BLOB_SAS_TOKEN")  # e.g., sp=raw&st=2025-06-28T14:28:25Z&se=2025-06-28T22:28:25Z&sv=2024-11-04&sr=c&sig=T0dgo8OZJnSISDiVeoUyfLkloZirB4F5A4bWwk82HzQ%3D
BLOB_SAS_TOKEN = "sp=racw&st=2025-06-30T20:45:27Z&se=2026-01-06T04:45:27Z&sv=2024-11-04&sr=c&sig=lMLeFZ7wUL0YNXOAKL8iIXk%2Blj0hz4ad0D266OflGXk%3D"
if not BLOB_SAS_URL or not BLOB_SAS_TOKEN:
    logger.error("❌ Missing BLOB_SAS_URL in .env")
    raise RuntimeError("BLOB_SAS_URL is required")

# Index folders under the container (must match blob structure)
AZURE_INDEX_PATHS = {
    "dietician": "faiss_index/Deiticien_faiss_index",
    "general": "faiss_index/General_faiss_index",
    "psychologist": "faiss_index/General_faiss_index"
    
}

# Global vector stores
general_vector_store = None
psychologist_vector_store = None
dietician_vector_store = None

# Global retrievers
general_retriever = None
psychologist_retriever = None
dietician_retriever = None

def download_file_from_azure(blob_path: str, dest_path: str) -> bool:
    """Download a single file using the SAS URL."""
    try:
        # full_url = f"{BLOB_SAS_URL.rstrip('?')}/{blob_path}"
        full_url = f"{BLOB_SAS_URL}/{blob_path}?{BLOB_SAS_TOKEN}"  # Ensure token is properly formatted
        logger.info(f"⬇️ Downloading {full_url}")
        r = requests.get(full_url)
        r.raise_for_status()
        with open(dest_path, 'wb') as f:
            f.write(r.content)
        return True
    except Exception as e:
        logger.error(f"❌ Failed to download {blob_path}: {e}")
        return False

def load_faiss_index(index_name: str):
    """Download and load FAISS index for a given persona."""
    if index_name not in AZURE_INDEX_PATHS:
        logger.error(f"Invalid index name: {index_name}")
        return None

    folder_blob = AZURE_INDEX_PATHS[index_name]
    local_dir = os.path.join("azure_cache", index_name)
    os.makedirs(local_dir, exist_ok=True)

    faiss_path = os.path.join(local_dir, "index.faiss")
    pkl_path = os.path.join(local_dir, "index.pkl")

    # Download both files
    if not download_file_from_azure(f"{folder_blob}/index.faiss", faiss_path):
        return None
    if not download_file_from_azure(f"{folder_blob}/index.pkl", pkl_path):
        return None

    try:
        # Use Azure OpenAI embeddings for FAISS vector store loading
        embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_deployment=EMBEDDING_DEPLOYMENT
        )

        vector_store = FAISS.load_local(
            folder_path=local_dir,
            embeddings=embeddings,
            index_name="index",
            allow_dangerous_deserialization=True
        )
        logger.info(f"✅ Loaded FAISS index: {index_name}")
        return vector_store
    except Exception as e:
        logger.error(f"❌ Failed to load vector store '{index_name}': {e}")
        return None

def initialize_vector_stores():
    """Initialize all vector stores at application startup."""
    global general_vector_store, psychologist_vector_store, dietician_vector_store
    global general_retriever, psychologist_retriever, dietician_retriever

    logger.info("🚀 Initializing vector stores from Azure Blob...")
    
    dietician_vector_store = load_faiss_index("dietician")
    if not dietician_vector_store:
        return False
        
    general_vector_store = load_faiss_index("general")
    if not general_vector_store:
        return False

    psychologist_vector_store = general_vector_store
    logger.info("Psychologist vector store shares General index.")

    
    
    dietician_retriever = dietician_vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    general_retriever = general_vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    psychologist_retriever = psychologist_vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    logger.info("✅ All vector stores initialized successfully.")
    return True

# Expose for import
__all__ = [
    "initialize_vector_stores",
    "dietician_vector_store",
    "general_vector_store",
    "psychologist_vector_store",
    
    "general_retriever",
    "psychologist_retriever",
    "dietician_retriever",
]
# if __name__ == "__main__":
#     if initialize_vector_stores():
#         logger.info("Vector stores initialized successfully.")
#     else:
#         logger.error("Failed to initialize vector stores.")

