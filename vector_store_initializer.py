"""
Vector Store Initializer

This module initializes FAISS vector stores from S3 at application startup.
It provides global access to the loaded vector stores for other modules.
"""

import os
import logging
import shutil
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(override=True)

# OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    logger.info(f"Initializing OpenAI client with API key: {OPENAI_API_KEY[:10]}...{OPENAI_API_KEY[-5:]}")
else:
    logger.warning("WARNING: OpenAI API key not found in environment variables.")

# S3 Configuration
USE_S3 = os.getenv("USE_S3", "true").lower() == "true"  # Default to true
S3_BUCKET = os.getenv("PDF_BUCKET_NAME", "prasha-health-pdf")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# S3 paths for different FAISS indexes - confirmed with S3 bucket listing
S3_PATHS = {
    "general": "faiss_index/general_index",
    "psychologist": "faiss_index/general_index",  # Using general index for psychologist
    "dietician": "faiss_index/dietician_index"
    # Removed PDF path to avoid unnecessary downloads
}

# Global vector stores that will be initialized once and shared
general_vector_store = None
psychologist_vector_store = None
dietician_vector_store = None
# Removed pdf_vector_store to avoid unnecessary downloads

# Global retrievers that will be initialized once and shared
general_retriever = None
psychologist_retriever = None
dietician_retriever = None
# Removed pdf_retriever to avoid unnecessary downloads

# Initialize S3 client
s3_client = None

def initialize_s3_client():
    """Initialize the S3 client."""
    global s3_client

    if not USE_S3:
        logger.warning("S3 is disabled. Vector stores will not be loaded.")
        return False

    try:
        import boto3
        from botocore.exceptions import NoCredentialsError, ClientError

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

        # Test S3 connection
        try:
            response = s3_client.list_buckets()
            buckets = [bucket['Name'] for bucket in response['Buckets']]
            logger.info(f"✅ AWS S3 client initialized successfully. Available buckets: {buckets}")

            # Check if our target bucket exists
            if S3_BUCKET in buckets:
                logger.info(f"✅ Target bucket '{S3_BUCKET}' found")

                # Test listing objects in the bucket
                try:
                    response = s3_client.list_objects_v2(
                        Bucket=S3_BUCKET,
                        MaxKeys=5
                    )
                    if 'Contents' in response:
                        logger.info(f"✅ Successfully listed objects in bucket. Sample keys: {[obj['Key'] for obj in response['Contents'][:3]]}")
                    else:
                        logger.info(f"⚠️ Bucket '{S3_BUCKET}' is empty or you don't have list permissions")
                except Exception as list_error:
                    logger.error(f"❌ Error listing objects in bucket: {str(list_error)}")
                    return False
            else:
                logger.error(f"❌ Target bucket '{S3_BUCKET}' not found in available buckets: {buckets}")
                return False
        except Exception as test_error:
            logger.error(f"❌ Error testing S3 connection: {str(test_error)}")
            return False

        return True
    except (ImportError, NoCredentialsError, ClientError) as e:
        logger.error(f"❌ Error initializing S3 client: {str(e)}")
        return False

def load_faiss_from_s3(index_name: str):
    """
    Load a FAISS index directly from S3.

    Args:
        index_name: Name of the index in the S3_PATHS dictionary

    Returns:
        Initialized vector store or None if S3 loading fails
    """
    if index_name not in S3_PATHS:
        logger.error(f"Invalid index name: {index_name}")
        return None

    s3_prefix = S3_PATHS[index_name]
    logger.info(f"Loading FAISS index for {index_name} from S3 prefix: {s3_prefix}")

    # Verify S3 is enabled and configured
    if not s3_client:
        logger.error("S3 client is not initialized. Cannot load vector store.")
        return None

    try:
        # Create a temporary directory within the project folder
        project_temp_dir = os.path.join(os.getcwd(), "temp_faiss_index")
        os.makedirs(project_temp_dir, exist_ok=True)

        # Create a specific directory for this index
        temp_dir = os.path.join(project_temp_dir, index_name)
        if os.path.exists(temp_dir):
            # Clean up any existing directory to avoid conflicts
            shutil.rmtree(temp_dir, ignore_errors=True)
        os.makedirs(temp_dir, exist_ok=True)

        logger.info(f"Created project-specific temporary directory for FAISS index: {temp_dir}")

        try:
            # List objects in the S3 bucket with the given prefix to find the index files
            logger.info(f"Listing objects in S3 bucket with prefix: {s3_prefix}")
            response = s3_client.list_objects_v2(
                Bucket=S3_BUCKET,
                Prefix=s3_prefix
            )

            if 'Contents' not in response or not response['Contents']:
                logger.error(f"No objects found in S3 with prefix {s3_prefix}")
                return None

            # Log the found objects
            keys = [obj['Key'] for obj in response['Contents']]
            logger.info(f"Found {len(keys)} objects in S3 with prefix {s3_prefix}")

            # Look for index.faiss and index.pkl files
            faiss_key = None
            pkl_key = None

            for key in keys:
                if key.endswith('index.faiss'):
                    faiss_key = key
                elif key.endswith('index.pkl'):
                    pkl_key = key

            if faiss_key and pkl_key:
                logger.info(f"Found index files: {faiss_key} and {pkl_key}")
            else:
                logger.error(f"Could not find index.faiss and index.pkl in S3 prefix {s3_prefix}")
                return None

            # Download the index files to the temporary directory
            faiss_path = os.path.join(temp_dir, "index.faiss")
            pkl_path = os.path.join(temp_dir, "index.pkl")

            logger.info(f"Downloading {faiss_key} to {faiss_path}")
            s3_client.download_file(S3_BUCKET, faiss_key, faiss_path)

            logger.info(f"Downloading {pkl_key} to {pkl_path}")
            s3_client.download_file(S3_BUCKET, pkl_key, pkl_path)

            logger.info(f"Successfully downloaded index files from S3")

            # Check if the pickle file is compressed (gzip)
            import gzip

            # Function to check if a file is gzip compressed
            def is_gzip_compressed(file_path):
                with open(file_path, 'rb') as f:
                    return f.read(2) == b'\x1f\x8b'  # gzip magic number

            # Check if the pickle file is compressed
            if is_gzip_compressed(pkl_path):
                logger.info(f"Detected gzip-compressed pickle file, decompressing...")
                # Read the compressed file
                with open(pkl_path, 'rb') as f:
                    compressed_data = f.read()

                # Decompress the data
                try:
                    decompressed_data = gzip.decompress(compressed_data)

                    # Write the decompressed data back to the file
                    with open(pkl_path, 'wb') as f:
                        f.write(decompressed_data)

                    logger.info(f"Successfully decompressed pickle file")
                except Exception as decompress_error:
                    logger.error(f"Error decompressing pickle file: {str(decompress_error)}")
                    return None

            # Initialize the embeddings
            embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

            # Load the vector store from the temporary directory
            logger.info(f"Loading FAISS index from temporary directory")
            vector_store = FAISS.load_local(
                folder_path=temp_dir,
                embeddings=embeddings,
                index_name="index",
                allow_dangerous_deserialization=True
            )

            logger.info(f"✅ Successfully loaded FAISS index from S3: {s3_prefix}")

            return vector_store
        finally:
            # Clean up the temporary directory
            if os.path.exists(temp_dir):
                logger.info(f"Cleaning up temporary directory: {temp_dir}")
                shutil.rmtree(temp_dir, ignore_errors=True)

    except Exception as e:
        logger.error(f"Error initializing vector store: {str(e)}")
        return None

def initialize_vector_stores():
    """Initialize all vector stores at application startup."""
    global general_vector_store, psychologist_vector_store, dietician_vector_store
    global general_retriever, psychologist_retriever, dietician_retriever

    logger.info("Initializing vector stores...")

    # Initialize S3 client
    if not initialize_s3_client():
        logger.error("Failed to initialize S3 client. Vector stores will not be loaded.")
        return False

    # Load the general index first
    general_vector_store = load_faiss_from_s3("general")
    if general_vector_store is None:
        logger.error("Failed to load general vector store from S3")
        return False

    # Use general index for psychologist as specified
    psychologist_vector_store = general_vector_store
    logger.info("Using general index for psychologist vector store")

    # Load dietician index
    dietician_vector_store = load_faiss_from_s3("dietician")
    if dietician_vector_store is None:
        logger.error("Failed to load dietician vector store from S3")
        return False

    # Create retrievers
    general_retriever = general_vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    psychologist_retriever = psychologist_vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    dietician_retriever = dietician_vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    logger.info("✅ Successfully initialized all vector stores")
    return True
