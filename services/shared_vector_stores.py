"""
Shared Vector Stores Module

This module provides centralized access to vector stores across the application.
It ensures that each vector store is loaded only once, even when used by multiple services.
"""

import os
import logging
from typing import Dict, Optional, Any
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()
logger = logging.getLogger(__name__)

# Constants
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Directory paths
GENERAL_INDEX_DIR = "resources/general_index"
PSYCHOLOGIST_INDEX_DIR = "resources/psychologist_index"
DIETICIAN_INDEX_DIR = "resources/dietician_index"
PDF_INDEX_DIR = "resources/pdf_index"
DOCTOR_INDEX_DIR = "resources/doctor_index"

# S3 Configuration
USE_S3 = os.getenv("USE_S3", "true").lower() == "true"
S3_BUCKET = os.getenv("PDF_BUCKET_NAME", "prasha-health-pdf")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# Initialize S3 client if enabled
s3_client = None
if USE_S3:
    try:
        # Log S3 configuration (without exposing full credentials)
        logger.info(f"S3 Configuration:")
        logger.info(f"  Bucket: {S3_BUCKET}")
        logger.info(f"  Region: {AWS_REGION}")
        logger.info(f"  Access Key: {AWS_ACCESS_KEY[:4]}...{AWS_ACCESS_KEY[-4:] if AWS_ACCESS_KEY and len(AWS_ACCESS_KEY) > 8 else '****'}")
        logger.info(f"  Secret Key: {'*' * 8}{AWS_SECRET_KEY[-4:] if AWS_SECRET_KEY and len(AWS_SECRET_KEY) > 8 else '****'}")

        s3_client = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY,
            region_name=AWS_REGION
        )
        logger.info("✅ AWS S3 client initialized successfully")
    except (ImportError, NoCredentialsError, ClientError) as e:
        logger.error(f"❌ Error initializing S3 client: {str(e)}")
        USE_S3 = False

# Vector store cache (initialized once)
_vector_stores = {
    "general": None,
    "psychologist": None,
    "dietician": None,
    "pdf": None,
    "doctor": None
}

def initialize_vector_store(index_dir: str, s3_prefix: Optional[str] = None, force_reload: bool = False):
    """
    Initialize a FAISS vector store from S3 only, with no local fallback.

    Args:
        index_dir: Local directory for the index (kept for API compatibility but not used)
        s3_prefix: S3 prefix for the index
        force_reload: Whether to force reload even if already in cache

    Returns:
        Initialized vector store or None if S3 loading fails
    """
    # Generate a cache key based on the S3 prefix
    cache_key = f"s3:{s3_prefix}"

    # Check if we already have this vector store in the cache
    if not force_reload:
        for key, store in _vector_stores.items():
            if store is not None and getattr(store, "_cache_key", None) == cache_key:
                logger.info(f"Using cached vector store for S3 prefix: {s3_prefix}")
                return store

    # Verify S3 is enabled and configured
    if not USE_S3 or not s3_client:
        logger.error("S3 is not enabled or configured. Cannot load vector store.")
        return None

    # Verify S3 prefix is provided
    if not s3_prefix:
        logger.error("S3 prefix is required but not provided. Cannot load vector store.")
        return None

    try:
        # Create temporary directory for downloading
        temp_dir = f"temp_{uuid.uuid4().hex}"
        os.makedirs(temp_dir, exist_ok=True)
        local_index_path = os.path.join(temp_dir, "index.faiss")
        local_metadata_path = os.path.join(temp_dir, "index.pkl")

        try:
            # Download files from S3
            logger.info(f"Loading FAISS index from S3: {S3_BUCKET}/{s3_prefix}/index.faiss")
            s3_client.download_file(S3_BUCKET, f"{s3_prefix}/index.faiss", local_index_path)
            s3_client.download_file(S3_BUCKET, f"{s3_prefix}/index.pkl", local_metadata_path)

            # Load the vector store
            embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
            vector_store = FAISS.load_local(
                temp_dir,
                embeddings,
                allow_dangerous_deserialization=True
            )

            logger.info(f"✅ Successfully loaded FAISS index from S3 with prefix: {s3_prefix}")

            # Add cache key as an attribute
            setattr(vector_store, "_cache_key", cache_key)

            # Clean up temporary directory
            try:
                for file in os.listdir(temp_dir):
                    os.remove(os.path.join(temp_dir, file))
                os.rmdir(temp_dir)
            except Exception as cleanup_error:
                logger.warning(f"Error cleaning up temporary directory: {str(cleanup_error)}")

            return vector_store

        except Exception as e:
            logger.error(f"Error loading from S3: {str(e)}")

            # Clean up temporary directory if it exists
            try:
                if os.path.exists(temp_dir):
                    for file in os.listdir(temp_dir):
                        os.remove(os.path.join(temp_dir, file))
                    os.rmdir(temp_dir)
            except Exception as cleanup_error:
                logger.warning(f"Error cleaning up temporary directory: {str(cleanup_error)}")

            return None

    except Exception as e:
        logger.error(f"Error initializing vector store: {str(e)}")
        return None

# S3 paths for different indexes
S3_PATHS = {
    "general": "faiss_index/general_index",
    "psychologist": "faiss_index/psychologist_index",
    "dietician": "dietician_index",  # Special case for backward compatibility
    "pdf": "faiss_index/pdf_index",
    "doctor": "faiss_index/doctor_index"
}

# Getter functions for each vector store
def get_general_vector_store():
    """Get the general vector store, initializing it from S3 if necessary."""
    global _vector_stores
    if _vector_stores["general"] is None:
        logger.info("Loading general vector store from S3")
        # Use a dummy local path for API compatibility
        _vector_stores["general"] = initialize_vector_store("", S3_PATHS["general"])
        if _vector_stores["general"] is None:
            logger.error("Failed to load general vector store from S3")
    return _vector_stores["general"]

def get_psychologist_vector_store():
    """Get the psychologist vector store, initializing it from S3 if necessary."""
    global _vector_stores
    if _vector_stores["psychologist"] is None:
        logger.info("Loading psychologist vector store from S3")
        # Use a dummy local path for API compatibility
        _vector_stores["psychologist"] = initialize_vector_store("", S3_PATHS["psychologist"])
        if _vector_stores["psychologist"] is None:
            logger.error("Failed to load psychologist vector store from S3")
    return _vector_stores["psychologist"]

def get_dietician_vector_store():
    """Get the dietician vector store, initializing it from S3 if necessary."""
    global _vector_stores
    if _vector_stores["dietician"] is None:
        logger.info("Loading dietician vector store from S3")
        # Use a dummy local path for API compatibility
        _vector_stores["dietician"] = initialize_vector_store("", S3_PATHS["dietician"])
        if _vector_stores["dietician"] is None:
            logger.error("Failed to load dietician vector store from S3")
    return _vector_stores["dietician"]

def get_pdf_vector_store():
    """Get the PDF vector store, initializing it from S3 if necessary."""
    global _vector_stores
    if _vector_stores["pdf"] is None:
        logger.info("Loading PDF vector store from S3")
        # Use a dummy local path for API compatibility
        _vector_stores["pdf"] = initialize_vector_store("", S3_PATHS["pdf"])
        if _vector_stores["pdf"] is None:
            logger.error("Failed to load PDF vector store from S3")
    return _vector_stores["pdf"]

def get_doctor_vector_store():
    """Get the doctor vector store, initializing it from S3 if necessary."""
    global _vector_stores
    if _vector_stores["doctor"] is None:
        logger.info("Loading doctor vector store from S3")
        # Use a dummy local path for API compatibility
        _vector_stores["doctor"] = initialize_vector_store("", S3_PATHS["doctor"])
        if _vector_stores["doctor"] is None:
            logger.error("Failed to load doctor vector store from S3")
    return _vector_stores["doctor"]

# Initialize all vector stores at module load time (optional)
def initialize_all_vector_stores():
    """Initialize all vector stores."""
    logger.info("Initializing all vector stores...")
    get_general_vector_store()
    get_psychologist_vector_store()
    get_dietician_vector_store()
    get_pdf_vector_store()
    get_doctor_vector_store()
    logger.info("All vector stores initialized.")
