"""
PDF Processing Service for Specialized Indexes

This module extends the PDF processing service to support specialized indexes
for different medical specialties (General OPD, Physician, Dietician, Psychiatric).
"""

import os
import uuid
import logging
from datetime import datetime
from typing import Dict, Optional, Any, Tuple

import PyPDF2
from pdf2image import convert_from_path
import pytesseract
from fastapi import APIRouter, File, Form, UploadFile, HTTPException, BackgroundTasks
from pydantic import BaseModel
from langchain_text_splitters import RecursiveCharacterTextSplitter
# We still need OpenAIEmbeddings for adding new documents
from langchain_openai import OpenAIEmbeddings
# We don't need to import FAISS directly since we're using pre-loaded vector stores
# from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv
import boto3
from botocore.exceptions import NoCredentialsError

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Disable OpenAI and httpx logging to prevent excessive HTTP request logs
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


# Router
specialized_pdf_router = APIRouter(tags=["PDF Processing - Specialized"])

# Environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Get S3 configuration from environment variables
USE_S3 = os.getenv("USE_S3", "true").lower() == "true"
S3_BUCKET = os.getenv("PDF_BUCKET_NAME", "prasha-health-pdf")  # Use PDF_BUCKET_NAME from .env

# Get AWS credentials from environment variables
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")  # Use AWS_ACCESS_KEY_ID from .env
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")  # Use AWS_SECRET_ACCESS_KEY from .env
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# Log environment variables (without exposing full credentials)
logger.info("Environment variables:")
logger.info(f"  USE_S3: {USE_S3}")
logger.info(f"  S3_BUCKET: {S3_BUCKET}")
logger.info(f"  AWS_REGION: {AWS_REGION}")
logger.info(f"  AWS_ACCESS_KEY set: {'Yes' if AWS_ACCESS_KEY else 'No'}")
logger.info(f"  AWS_SECRET_KEY set: {'Yes' if AWS_SECRET_KEY else 'No'}")

# Directory for temporary files
TEMP_DIR = "temp"

# Ensure temp directory exists for temporary files
os.makedirs(TEMP_DIR, exist_ok=True)

# S3 paths for different indexes - match with vector_store_initializer.py
S3_PATHS = {
    "general_index": "faiss_index/general_index",
    "psychologist_index": "faiss_index/general_index",  # Using general index for psychologist
    "dietician_index": "faiss_index/dietician_index"  # Fixed path to match vector_store_initializer
}

# Initialize S3 client if enabled
s3_client = None
if USE_S3:
    try:
        # Check if AWS credentials are available
        if not AWS_ACCESS_KEY or not AWS_SECRET_KEY:
            logger.error("AWS credentials are missing. Please set AWS_ACCESS_KEY and AWS_SECRET_KEY environment variables.")
            logger.error("Attempting to use boto3's default credential provider chain...")

            # Try to initialize S3 client without explicit credentials
            # This will use the default credential provider chain (environment, ~/.aws/credentials, IAM role)
            try:
                s3_client = boto3.client('s3', region_name=AWS_REGION)
                logger.info("S3 client initialized using default credential provider chain")
            except Exception as default_cred_error:
                logger.error(f"Failed to initialize S3 client with default credentials: {str(default_cred_error)}")
                # Continue with S3 disabled
                USE_S3 = False
        else:
            # Log S3 configuration (without exposing full credentials)
            logger.info(f"S3 Configuration:")
            logger.info(f"  Bucket: {S3_BUCKET}")
            logger.info(f"  Region: {AWS_REGION}")
            logger.info(f"  Access Key: {AWS_ACCESS_KEY[:4]}...{AWS_ACCESS_KEY[-4:] if AWS_ACCESS_KEY and len(AWS_ACCESS_KEY) > 8 else '****'}")
            logger.info(f"  Secret Key: {'*' * 8}{AWS_SECRET_KEY[-4:] if AWS_SECRET_KEY and len(AWS_SECRET_KEY) > 8 else '****'}")

            # Initialize the S3 client with explicit credentials
            s3_client = boto3.client(
                's3',
                aws_access_key_id=AWS_ACCESS_KEY,
                aws_secret_access_key=AWS_SECRET_KEY,
                region_name=AWS_REGION
            )
            logger.info("S3 client initialized with explicit credentials")

        # If S3 client was successfully initialized, test the connection
        if s3_client:
            # Test S3 connection by listing buckets
            try:
                response = s3_client.list_buckets()
                buckets = [bucket['Name'] for bucket in response['Buckets']]
                logger.info(f"S3 connection successful. Available buckets: {buckets}")

                # Check if our target bucket exists
                if S3_BUCKET in buckets:
                    logger.info(f"Target bucket '{S3_BUCKET}' found")

                    # Test listing objects in the bucket
                    try:
                        response = s3_client.list_objects_v2(
                            Bucket=S3_BUCKET,
                            MaxKeys=5
                        )
                        if 'Contents' in response:
                            logger.info(f"Successfully listed objects in bucket. Sample keys: {[obj['Key'] for obj in response['Contents'][:3]]}")
                        else:
                            logger.info(f"Bucket '{S3_BUCKET}' is empty or you don't have list permissions")
                    except Exception as list_error:
                        logger.error(f"Error listing objects in bucket: {str(list_error)}")
                else:
                    logger.error(f"Target bucket '{S3_BUCKET}' not found in available buckets")
                    logger.error(f"Available buckets: {buckets}")
                    logger.error("Please check your AWS account and make sure the bucket exists")
            except Exception as test_error:
                logger.error(f"Error testing S3 connection: {str(test_error)}")
                import traceback
                logger.error(traceback.format_exc())
                # If we can't list buckets, S3 is not working
                s3_client = None
                USE_S3 = False
    except Exception as e:
        logger.error(f"Error initializing S3 client: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        # If there's any error, disable S3
        USE_S3 = False
        s3_client = None

# Final check if S3 is enabled
if USE_S3 and s3_client:
    logger.info("✅ S3 integration is enabled and working")
else:
    logger.warning("⚠️ S3 integration is disabled or not working")
    USE_S3 = False
    s3_client = None

# Use pre-loaded vector stores from the initializer
import vector_store_initializer

# Map our index names to the initializer's vector stores
# Use general vector store for all PDF processing to avoid unnecessary downloads
vector_stores = {
    "general_index": vector_store_initializer.general_vector_store,
    "psychologist_index": vector_store_initializer.psychologist_vector_store,
    "dietician_index": vector_store_initializer.dietician_vector_store,
    # Add pdf_index mapping to general_vector_store to avoid loading separate PDF vector store
    "pdf_index": vector_store_initializer.general_vector_store
}

logger.info("Using pre-loaded vector stores from initializer")

# Models
class SpecializedPDFProcessingResponse(BaseModel):
    """Response model for specialized PDF processing."""
    message: str
    pdf_id: str
    index_type: str
    num_pages: int
    extracted_text_sample: str
    processing_time: float
    s3_url: Optional[str] = None
    status: str = "processing"  # "processing", "completed", "failed"

class PDFProcessingStatus(BaseModel):
    """Status of a specific PDF processing job."""
    pdf_id: str
    index_type: str
    status: str  # "processing", "completed", "failed"
    num_pages: int
    extracted_text_sample: str
    processing_time: float
    s3_url: Optional[str] = None
    error: Optional[str] = None
    completed_at: Optional[str] = None

class IndexStatus(BaseModel):
    """Status of a specific index."""
    exists: bool
    document_count: Optional[int] = None
    last_updated: Optional[str] = None

class IndexStatusResponse(BaseModel):
    """Response model for index status."""
    general_index: IndexStatus
    psychologist_index: IndexStatus
    dietician_index: IndexStatus
    # Keep these for backward compatibility with existing clients
    physician_index: IndexStatus = None
    psychiatric_index: IndexStatus = None
    timestamp: str

# Dictionary to store processing status of PDFs
pdf_processing_status = {}

# Helper Functions
def save_to_s3(file_content: bytes, key: str) -> Optional[str]:
    """
    Save content to S3 bucket.

    Args:
        file_content: The content to save
        key: The S3 key (path) to save to

    Returns:
        S3 URL if successful, None otherwise
    """
    if not USE_S3 or not s3_client:
        logger.warning("S3 integration is disabled or not configured")
        return None

    try:
        # Check if we have the necessary environment variables
        if not AWS_ACCESS_KEY or not AWS_SECRET_KEY:
            logger.error("AWS credentials not set in environment variables")
            logger.error("Please set AWS_ACCESS_KEY and AWS_SECRET_KEY in your .env file")
            return None

        # Try to save the file to S3
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=key,
            Body=file_content
        )
        s3_url = f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{key}"
        logger.info(f"Successfully saved to S3: {s3_url}")
        return s3_url
    except NoCredentialsError:
        logger.error("AWS credentials not available or invalid")
        logger.error("Please check your AWS_ACCESS_KEY and AWS_SECRET_KEY in your .env file")
        return None
    except Exception as e:
        logger.error(f"Error saving to S3: {str(e)}")

        # Provide more specific guidance based on the error
        if "AccessDenied" in str(e):
            logger.error("Access Denied error. Please check your IAM permissions.")
            logger.error("Required permissions: s3:PutObject, s3:GetObject, s3:ListBucket, s3:DeleteObject")
            logger.error(f"Target bucket: {S3_BUCKET}")
            logger.error(f"Target key: {key}")
        elif "NoSuchBucket" in str(e):
            logger.error(f"Bucket '{S3_BUCKET}' does not exist in region '{AWS_REGION}'")
            logger.error("Please create the bucket or check the bucket name and region")

        import traceback
        logger.error(traceback.format_exc())
        return None

def save_index_to_s3(index_dir: str, s3_prefix: str) -> bool:
    """
    Save a FAISS index to S3 with optimized performance.

    Args:
        index_dir: Local directory containing the index
        s3_prefix: S3 prefix (directory) to save to

    Returns:
        True if successful, False otherwise
    """
    if not USE_S3 or not s3_client:
        logger.warning("S3 integration is disabled or not configured")
        return False

    try:
        # Skip permission check to save time - we'll catch permission errors during upload

        # Get list of files to upload
        files_to_upload = []
        for filename in os.listdir(index_dir):
            file_path = os.path.join(index_dir, filename)
            if os.path.isfile(file_path):
                files_to_upload.append((filename, file_path))

        logger.info(f"Found {len(files_to_upload)} files to upload to S3")

        # Check if we need to upload all files or can do incremental update
        try:
            # List existing files in S3
            response = s3_client.list_objects_v2(
                Bucket=S3_BUCKET,
                Prefix=f"{s3_prefix}/"
            )

            existing_files = {}
            if 'Contents' in response:
                for obj in response.get('Contents', []):
                    key = obj['Key']
                    # Extract just the filename from the key
                    filename = os.path.basename(key)
                    existing_files[filename] = obj

            logger.info(f"Found {len(existing_files)} existing files in S3")

            # Determine which files need to be uploaded (new or changed)
            files_to_upload_filtered = []
            for filename, file_path in files_to_upload:
                # If file doesn't exist in S3 or has different size, upload it
                if filename not in existing_files:
                    logger.info(f"File {filename} is new, will upload")
                    files_to_upload_filtered.append((filename, file_path))
                else:
                    # Compare file sizes as a quick check
                    local_size = os.path.getsize(file_path)
                    s3_size = existing_files[filename]['Size']
                    if local_size != s3_size:
                        logger.info(f"File {filename} has changed (size: {local_size} vs {s3_size}), will upload")
                        files_to_upload_filtered.append((filename, file_path))
                    else:
                        logger.info(f"File {filename} is unchanged, skipping upload")

            # Update our list to only include files that need uploading
            files_to_upload = files_to_upload_filtered
            logger.info(f"After filtering, {len(files_to_upload)} files need to be uploaded")

        except Exception as e:
            logger.warning(f"Error checking existing files in S3, will upload all files: {str(e)}")
            # Continue with uploading all files

        # If no files need uploading, we're done
        if not files_to_upload:
            logger.info(f"No files need to be uploaded to S3, skipping upload")
            return True

        # Use ThreadPoolExecutor for parallel uploads
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import gzip
        import io

        def upload_file(file_info):
            filename, file_path = file_info
            try:
                # Compress the file before uploading
                with open(file_path, 'rb') as f:
                    file_data = f.read()

                # Don't compress pickle files to ensure compatibility with vector_store_initializer
                if filename.endswith('.json'):
                    # Only compress JSON files, not pickle files
                    compressed_data = io.BytesIO()
                    with gzip.GzipFile(fileobj=compressed_data, mode='wb') as f:
                        f.write(file_data)
                    compressed_data.seek(0)

                    # Upload with compression headers
                    s3_client.put_object(
                        Bucket=S3_BUCKET,
                        Key=f"{s3_prefix}/{filename}",
                        Body=compressed_data,
                        ContentEncoding='gzip'
                    )
                    logger.info(f"Uploaded compressed {filename} to S3:{S3_BUCKET}/{s3_prefix}/{filename}")
                else:
                    # Upload pickle and binary files without compression
                    s3_client.put_object(
                        Bucket=S3_BUCKET,
                        Key=f"{s3_prefix}/{filename}",
                        Body=file_data
                    )
                    logger.info(f"Uploaded {filename} to S3:{S3_BUCKET}/{s3_prefix}/{filename}")

                return True
            except Exception as e:
                logger.error(f"Error uploading {filename} to S3: {str(e)}")
                return False

        # Configure S3 client for better performance
        s3_config = s3_client._client_config
        if hasattr(s3_config, 'max_pool_connections'):
            # Increase connection pool size for parallel uploads
            s3_config.max_pool_connections = 20

        # Use parallel uploads with ThreadPoolExecutor
        max_workers = min(10, len(files_to_upload))  # Don't use more workers than files
        logger.info(f"Starting parallel upload with {max_workers} workers")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all upload tasks
            future_to_file = {executor.submit(upload_file, file_info): file_info[0] for file_info in files_to_upload}

            # Process results as they complete
            success_count = 0
            for future in as_completed(future_to_file):
                filename = future_to_file[future]
                try:
                    if future.result():
                        success_count += 1
                except Exception as e:
                    logger.error(f"Exception uploading {filename}: {str(e)}")

        # Check if all uploads were successful
        if success_count == len(files_to_upload):
            logger.info(f"Successfully saved all {len(files_to_upload)} files to S3: {s3_prefix}")
            return True
        else:
            logger.warning(f"Partially saved index to S3: {success_count}/{len(files_to_upload)} files uploaded")
            return success_count > 0  # Return True if at least some files were uploaded

    except Exception as e:
        logger.error(f"Error saving index to S3: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def extract_text_from_pdf(file_path: str) -> Tuple[str, int]:
    """
    Extract text from a PDF file, using OCR if necessary.

    Args:
        file_path: Path to the PDF file

    Returns:
        Tuple of (extracted text, number of pages)
    """
    try:
        # Try direct text extraction first
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            num_pages = len(reader.pages)

            # Extract text from each page
            text = ""
            for page_num in range(num_pages):
                page = reader.pages[page_num]
                page_text = page.extract_text()

                # If page has little text, try OCR
                if not page_text or len(page_text.strip()) < 50:
                    logger.info(f"Page {page_num+1} has little text, trying OCR")
                    page_text = ocr_pdf_page(file_path, page_num)

                text += f"\n\n--- Page {page_num+1} ---\n\n{page_text}"

            return text, num_pages
    except Exception as e:
        logger.error(f"Error extracting text with PyPDF2: {str(e)}")
        # Fall back to full OCR
        return ocr_pdf_document(file_path)

def ocr_pdf_page(file_path: str, page_num: int) -> str:
    """
    Perform OCR on a specific page of a PDF.

    Args:
        file_path: Path to the PDF file
        page_num: Page number to OCR (0-based)

    Returns:
        Extracted text from the page
    """
    try:
        # Convert PDF page to image
        images = convert_from_path(file_path, first_page=page_num+1, last_page=page_num+1)

        if not images:
            return ""

        # Perform OCR on the image
        text = pytesseract.image_to_string(images[0])
        return text
    except Exception as e:
        logger.error(f"Error performing OCR on page {page_num}: {str(e)}")
        return ""

def ocr_pdf_document(file_path: str) -> Tuple[str, int]:
    """
    Perform OCR on an entire PDF document.

    Args:
        file_path: Path to the PDF file

    Returns:
        Tuple of (extracted text, number of pages)
    """
    try:
        # Convert all PDF pages to images
        images = convert_from_path(file_path)
        num_pages = len(images)

        # Perform OCR on each image
        text = ""
        for i, image in enumerate(images):
            page_text = pytesseract.image_to_string(image)
            text += f"\n\n--- Page {i+1} ---\n\n{page_text}"

        return text, num_pages
    except Exception as e:
        logger.error(f"Error performing OCR on document: {str(e)}")
        return f"Error extracting text: {str(e)}", 0

def clean_text(text: str) -> str:
    """
    Clean extracted text.

    Args:
        text: Text to clean

    Returns:
        Cleaned text
    """
    # Replace multiple newlines with a single newline
    import re
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Remove any non-printable characters
    text = ''.join(c for c in text if c.isprintable() or c in ['\n', '\t'])

    return text

def initialize_vector_store(_, s3_prefix: str = None):
    """
    Get the appropriate vector store based on the S3 prefix.

    Args:
        _: Unused parameter (kept for backward compatibility)
        s3_prefix: S3 prefix for the index

    Returns:
        Initialized vector store or None if failed
    """
    try:
        logger.info(f"Getting vector store for prefix: {s3_prefix}")

        # Map S3 prefixes to vector store keys
        prefix_to_index = {
            "faiss_index/general_index": "general_index",
            "faiss_index/psychologist_index": "psychologist_index",
            "faiss_index/dietician_index": "dietician_index"
        }

        # Get the index type for this prefix
        index_type = None
        for prefix, idx in prefix_to_index.items():
            if s3_prefix and s3_prefix in prefix:
                index_type = idx
                break

        if not index_type:
            logger.warning(f"Unknown S3 prefix: {s3_prefix}, using general_index as fallback")
            index_type = "general_index"

        # Get the vector store from our map
        vector_store = vector_stores.get(index_type)
        if vector_store:
            logger.info(f"Using pre-loaded vector store for {index_type}")
            return vector_store
        else:
            logger.warning(f"No pre-loaded vector store found for {index_type}")
            return None

    except Exception as e:
        logger.error(f"Error getting vector store: {str(e)}")
        return None

def add_to_vector_store(text: str, metadata: Dict[str, Any], index_type: str, chunk_size: int = 400, chunk_overlap: int = 50) -> bool:
    """
    Add text to the appropriate vector store with optimized performance.

    Args:
        text: Text to add
        metadata: Metadata for the text
        index_type: Type of index (general_index, physician_index, etc.)
        chunk_size: Size of text chunks in tokens
        chunk_overlap: Overlap between chunks in tokens

    Returns:
        True if successful, False otherwise
    """
    try:
        # Validate index type
        if index_type not in S3_PATHS:
            logger.error(f"Invalid index type: {index_type}")
            return False

        # Use the S3 paths defined at the top of the file
        s3_prefix = S3_PATHS.get(index_type, f"faiss_index/{index_type}")
        logger.info(f"Using S3 prefix: {s3_prefix} for index type: {index_type}")

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_text(text)
        metadatas = [metadata.copy() for _ in chunks]

        logger.info(f"Split text into {len(chunks)} chunks for vector processing")

        # Get the pre-loaded vector store from our map
        vector_store = vector_stores[index_type]

        # If vector store exists, add to it
        if vector_store is not None:
            # Initialize OpenAI embeddings for adding new documents
            try:
                # Use the smaller, faster model for better performance
                embeddings = OpenAIEmbeddings(
                    openai_api_key=OPENAI_API_KEY,
                    model="text-embedding-3-small",  # Faster, cheaper model
                    dimensions=1536  # Explicitly set dimensions for consistency
                )
            except Exception as e:
                logger.error(f"Error initializing OpenAI embeddings: {str(e)}")
                raise Exception(f"Failed to initialize embeddings: {str(e)}")

            # Process chunks in batches to improve performance

            # Define batch size for embedding API calls - smaller for better performance
            batch_size = 50  # Reduced from 100 for better performance and timeout handling

            # Function to process a batch of chunks
            def process_chunk_batch(batch_chunks, batch_metadatas):
                try:
                    # Reduce batch size if it's too large
                    if len(batch_chunks) > 100:
                        logger.info(f"Large batch detected ({len(batch_chunks)} chunks), reducing to smaller batches")
                        # Process in smaller sub-batches
                        sub_batch_size = 50
                        success = True
                        for j in range(0, len(batch_chunks), sub_batch_size):
                            sub_batch_chunks = batch_chunks[j:j+sub_batch_size]
                            sub_batch_metadatas = batch_metadatas[j:j+sub_batch_size]
                            logger.info(f"Processing sub-batch {j//sub_batch_size + 1}/{(len(batch_chunks) + sub_batch_size - 1)//sub_batch_size} ({len(sub_batch_chunks)} chunks)")
                            if not process_chunk_batch(sub_batch_chunks, sub_batch_metadatas):
                                success = False
                        return success

                    # Get embeddings for the batch in one API call
                    # Comment out excessive logging that generates too many log entries
                    # logger.info(f"Getting embeddings for batch of {len(batch_chunks)} chunks")
                    batch_embeddings = embeddings.embed_documents(batch_chunks)

                    if len(batch_embeddings) != len(batch_chunks):
                        logger.error(f"Embedding count mismatch: got {len(batch_embeddings)} embeddings for {len(batch_chunks)} chunks")
                        return False

                    # Comment out excessive logging that generates too many log entries
                    # logger.info(f"Successfully generated {len(batch_embeddings)} embeddings")

                    # Add to vector store with pre-computed embeddings
                    success_count = 0
                    for i, (chunk, chunk_metadata, embedding) in enumerate(zip(batch_chunks, batch_metadatas, batch_embeddings)):
                        try:
                            # Add individual document with its embedding
                            vector_store.add_texts(
                                texts=[chunk],
                                metadatas=[chunk_metadata],
                                embeddings=[embedding]  # Use pre-computed embedding
                            )
                            success_count += 1
                        except Exception as e:
                            logger.warning(f"Error adding chunk {i} to vector store: {str(e)}")

                    # Comment out excessive logging that generates too many log entries
                    # logger.info(f"Added {success_count}/{len(batch_chunks)} chunks to vector store")
                    return success_count > 0  # Return True if at least some chunks were added
                except Exception as e:
                    logger.error(f"Error processing batch: {str(e)}")
                    import traceback
                    logger.error(traceback.format_exc())
                    return False

            # Process chunks in batches
            total_batches = (len(chunks) + batch_size - 1) // batch_size
            logger.info(f"Processing {len(chunks)} chunks in {total_batches} batches")

            success = True
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i+batch_size]
                batch_metadatas = metadatas[i:i+batch_size]

                current_batch = i//batch_size + 1

                # Log progress for large documents only
                if total_batches > 10:  # Only for very large documents
                    logger.info(f"Processing batch {current_batch}/{total_batches}")

                batch_success = process_chunk_batch(batch_chunks, batch_metadatas)
                if not batch_success:
                    success = False

            if success:
                logger.info(f"Successfully added all {len(chunks)} chunks to vector store")
            else:
                logger.warning(f"Some chunks failed to be added to the vector store")

            # Save vector store to S3 in background
            if USE_S3 and s3_client:
                import threading

                def save_and_upload():
                    try:
                        # Create temporary directory for saving
                        temp_dir = os.path.join(TEMP_DIR, f"temp_save_{uuid.uuid4().hex}")
                        os.makedirs(temp_dir, exist_ok=True)

                        # Save locally first
                        vector_store.save_local(temp_dir)

                        # Upload to S3
                        s3_success = save_index_to_s3(temp_dir, s3_prefix)
                        if s3_success:
                            logger.info(f"Vector store saved to S3: {s3_prefix}")
                        else:
                            logger.warning(f"Failed to save vector store to S3")

                        # Clean up temporary directory
                        try:
                            for file in os.listdir(temp_dir):
                                os.remove(os.path.join(temp_dir, file))
                            os.rmdir(temp_dir)
                        except Exception:
                            pass  # Ignore cleanup errors

                    except Exception as save_error:
                        logger.error(f"Error in background S3 save: {str(save_error)}")

                # Start background thread for S3 upload
                save_thread = threading.Thread(target=save_and_upload)
                save_thread.daemon = True
                save_thread.start()

            # Return success since we've added the texts to the vector store
            return True
        else:
            # This should not happen since vector stores are pre-loaded
            logger.error(f"Vector store for {index_type} is not initialized")
            return False

    except Exception as e:
        logger.error(f"Error adding to vector store: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def process_specialized_pdf_background(
    file_path: str,
    pdf_id: str,
    index_type: str,
    title: str,
    description: str,
    chunk_size: int,
    chunk_overlap: int,
    file_content: Optional[bytes] = None,
    storage_location: str = "s3"
):
    """
    Process a PDF file in the background for a specialized index.
    Simplified version without timeouts - runs until completion.
    """
    start_time = datetime.now()
    s3_url = None

    # Initialize status
    pdf_processing_status[pdf_id] = {
        "pdf_id": pdf_id,
        "index_type": index_type,
        "status": "processing",
        "num_pages": 0,
        "extracted_text_sample": "Processing...",
        "processing_time": 0.0,
        "s3_url": None,
        "created_at": datetime.now().isoformat()
    }

    # Handle S3 upload in background if requested
    if USE_S3 and s3_client and storage_location == "s3" and file_content:
        try:
            s3_prefix = S3_PATHS.get(index_type, f"faiss_index/{index_type}")
            s3_key = f"{s3_prefix}/pdfs/{pdf_id}.pdf"
            s3_url = save_to_s3(file_content, s3_key)
            if s3_url:
                pdf_processing_status[pdf_id]["s3_url"] = s3_url
        except Exception as e:
            logger.warning(f"S3 upload error for PDF {pdf_id}: {str(e)}")

    try:
        logger.info(f"Processing PDF {pdf_id} for {index_type}")

        # Check if file exists
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            pdf_processing_status[pdf_id].update({
                "status": "failed",
                "error": f"File not found: {file_path}",
                "completed_at": datetime.now().isoformat()
            })
            return

        # Extract text from PDF
        try:
            text, num_pages = extract_text_from_pdf(file_path)
            logger.info(f"Extracted {len(text)} characters from {num_pages} pages")
        except Exception as e:
            error_msg = f"Text extraction error for PDF {pdf_id}: {str(e)}"
            logger.error(error_msg)
            pdf_processing_status[pdf_id].update({
                "status": "failed",
                "error": error_msg,
                "completed_at": datetime.now().isoformat()
            })
            return

        # Update status with number of pages
        pdf_processing_status[pdf_id]["num_pages"] = num_pages

        # Get a sample of the extracted text
        text_sample = text[:500] + "..." if len(text) > 500 else text
        pdf_processing_status[pdf_id]["extracted_text_sample"] = text_sample

        # Clean the text
        cleaned_text = clean_text(text)

        # Create metadata
        metadata = {
            "pdf_id": pdf_id,
            "title": title,
            "description": description,
            "num_pages": num_pages,
            "processed_at": datetime.now().isoformat(),
            "source": "pdf_upload",
            "index_type": index_type,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap
        }

        if s3_url:
            metadata["s3_url"] = s3_url

        # Check OpenAI API key
        if not OPENAI_API_KEY:
            error_msg = "OpenAI API key is not set. Cannot create embeddings."
            logger.error(error_msg)
            pdf_processing_status[pdf_id].update({
                "status": "failed",
                "error": error_msg,
                "completed_at": datetime.now().isoformat()
            })
            return

        # Add to vector store - runs until completion
        logger.info(f"Adding PDF {pdf_id} to {index_type} vector store")
        try:
            success = add_to_vector_store(
                cleaned_text,
                metadata,
                index_type,
                chunk_size,
                chunk_overlap
            )
        except Exception as e:
            error_msg = f"Vector processing error for PDF {pdf_id}: {str(e)}"
            logger.error(error_msg)
            pdf_processing_status[pdf_id].update({
                "status": "failed",
                "error": error_msg,
                "completed_at": datetime.now().isoformat()
            })
            return

        # Calculate processing time and update status
        processing_time = (datetime.now() - start_time).total_seconds()

        pdf_processing_status[pdf_id].update({
            "status": "completed" if success else "failed",
            "processing_time": processing_time,
            "completed_at": datetime.now().isoformat(),
            "error": None if success else "Failed to add to vector store"
        })

        if success:
            logger.info(f"PDF {pdf_id} processed successfully in {processing_time:.2f} seconds")
        else:
            logger.error(f"PDF {pdf_id} processing failed after {processing_time:.2f} seconds")

        # Clean up temporary file
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        logger.error(f"Error processing PDF {pdf_id}: {str(e)}")
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(error_traceback)

        # Update status to failed
        pdf_processing_status[pdf_id].update({
            "status": "failed",
            "error": str(e),
            "completed_at": datetime.now().isoformat()
        })

# API Endpoints
@specialized_pdf_router.get("/test-s3")
async def test_s3_connection():
    """
    Test S3 connection and permissions.

    This endpoint tests:
    1. S3 connection
    2. Bucket existence
    3. List permissions
    4. Write permissions
    5. Read permissions
    6. Delete permissions
    """
    # Check environment variables first
    env_info = {
        "AWS_ACCESS_KEY": "Set" if AWS_ACCESS_KEY else "Not set",
        "AWS_SECRET_KEY": "Set" if AWS_SECRET_KEY else "Not set",
        "AWS_REGION": AWS_REGION,
        "S3_BUCKET": S3_BUCKET,
        "USE_S3": USE_S3
    }

    if not USE_S3 or not s3_client:
        return {
            "status": "error",
            "message": "S3 is not enabled or client not initialized",
            "environment": env_info,
            "help": "Please set AWS_ACCESS_KEY and AWS_SECRET_KEY environment variables. You can do this by creating a .env file in the project root with these variables."
        }

    results = {
        "status": "testing",
        "bucket": S3_BUCKET,
        "region": AWS_REGION,
        "tests": {}
    }

    # Test 1: List buckets
    try:
        response = s3_client.list_buckets()
        buckets = [bucket['Name'] for bucket in response['Buckets']]
        results["tests"]["list_buckets"] = {
            "status": "success",
            "buckets": buckets
        }

        # Check if our bucket exists
        if S3_BUCKET in buckets:
            results["tests"]["bucket_exists"] = {
                "status": "success",
                "message": f"Bucket '{S3_BUCKET}' exists"
            }
        else:
            results["tests"]["bucket_exists"] = {
                "status": "error",
                "message": f"Bucket '{S3_BUCKET}' not found in available buckets"
            }
    except Exception as e:
        results["tests"]["list_buckets"] = {
            "status": "error",
            "message": str(e)
        }

    # Test 2: List objects in bucket
    try:
        response = s3_client.list_objects_v2(
            Bucket=S3_BUCKET,
            MaxKeys=5
        )
        if 'Contents' in response:
            results["tests"]["list_objects"] = {
                "status": "success",
                "sample_keys": [obj['Key'] for obj in response['Contents'][:3]]
            }
        else:
            results["tests"]["list_objects"] = {
                "status": "success",
                "message": "Bucket is empty or no objects found"
            }
    except Exception as e:
        results["tests"]["list_objects"] = {
            "status": "error",
            "message": str(e)
        }

    # Test 3: Write a test file
    test_key = f"test/s3_test_{uuid.uuid4()}.txt"
    try:
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=test_key,
            Body=b"This is a test file to verify S3 write permissions."
        )
        results["tests"]["write_object"] = {
            "status": "success",
            "key": test_key
        }

        # Test 4: Read the test file
        try:
            response = s3_client.get_object(
                Bucket=S3_BUCKET,
                Key=test_key
            )
            content = response['Body'].read().decode('utf-8')
            results["tests"]["read_object"] = {
                "status": "success",
                "content": content[:50] + "..." if len(content) > 50 else content
            }
        except Exception as e:
            results["tests"]["read_object"] = {
                "status": "error",
                "message": str(e)
            }

        # Test 5: Delete the test file
        try:
            s3_client.delete_object(
                Bucket=S3_BUCKET,
                Key=test_key
            )
            results["tests"]["delete_object"] = {
                "status": "success",
                "message": f"Successfully deleted {test_key}"
            }
        except Exception as e:
            results["tests"]["delete_object"] = {
                "status": "error",
                "message": str(e)
            }
    except Exception as e:
        results["tests"]["write_object"] = {
            "status": "error",
            "message": str(e)
        }

    # Overall status
    if all(test["status"] == "success" for test in results["tests"].values()):
        results["status"] = "success"
        results["message"] = "All S3 tests passed successfully"
    else:
        results["status"] = "error"
        results["message"] = "Some S3 tests failed. Check individual test results for details."

    return results

@specialized_pdf_router.post("/upload/specialized", response_model=SpecializedPDFProcessingResponse)
async def upload_specialized_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: str = Form(None),
    description: str = Form(None),
    index_type: str = Form(...),
    chunk_size: int = Form(400),
    chunk_overlap: int = Form(50),
    storage_location: str = Form("s3")
):
    """
    Upload and process a PDF file for a specialized index.

    Args:
        file: The PDF file to upload
        title: Optional title for the PDF
        description: Optional description for the PDF
        index_type: Type of index to add to (general_index, physician_index, etc.)
        chunk_size: Size of text chunks in tokens
        chunk_overlap: Overlap between chunks in tokens
        storage_location: Where to store the file (s3 or local)

    Returns:
        Processing status and information
    """
    # Validate index type
    valid_index_types = ["general_index", "psychologist_index", "dietician_index"]
    if index_type not in valid_index_types:
        raise HTTPException(status_code=400, detail=f"Invalid index type. Must be one of: {', '.join(valid_index_types)}")

    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    # Validate chunk size and overlap
    if chunk_size < 100 or chunk_size > 1000:
        raise HTTPException(status_code=400, detail="Chunk size must be between 100 and 1000 tokens")

    if chunk_overlap < 0 or chunk_overlap > 200:
        raise HTTPException(status_code=400, detail="Chunk overlap must be between 0 and 200 tokens")

    # Generate a unique ID for this PDF
    pdf_id = str(uuid.uuid4())
    logger.info(f"Starting PDF upload: {file.filename} -> {index_type} (ID: {pdf_id})")

    try:
        # Read the file content
        file_content = await file.read()

        # Validate file size (limit to 50MB)
        max_file_size = 50 * 1024 * 1024  # 50MB
        if len(file_content) > max_file_size:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size is {max_file_size // (1024*1024)}MB"
            )

        # Initialize processing status
        pdf_processing_status[pdf_id] = {
            "pdf_id": pdf_id,
            "index_type": index_type,
            "status": "processing",
            "num_pages": 0,
            "extracted_text_sample": "Processing...",
            "processing_time": 0.0,
            "s3_url": None,
            "created_at": datetime.now().isoformat()
        }

        # Save to temporary file
        temp_file_path = os.path.join(TEMP_DIR, f"{pdf_id}.pdf")
        with open(temp_file_path, "wb") as buffer:
            buffer.write(file_content)

        # Start background processing
        background_tasks.add_task(
            process_specialized_pdf_background,
            temp_file_path,
            pdf_id,
            index_type,
            title or file.filename,
            description,
            chunk_size,
            chunk_overlap,
            file_content,
            storage_location
        )

        # Return immediate response
        return SpecializedPDFProcessingResponse(
            message=f"PDF upload received and processing started for {index_type}. Check status at /pdf/processing/{pdf_id}",
            pdf_id=pdf_id,
            index_type=index_type,
            num_pages=0,
            extracted_text_sample="Processing...",
            processing_time=0.0,
            s3_url=None,
            status="processing"
        )
    except Exception as e:
        logger.error(f"Error processing specialized PDF upload: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

        # Provide more specific error messages based on the exception
        error_message = str(e)
        if "AccessDenied" in error_message:
            error_message = "Access denied to S3 bucket. Please check your AWS credentials and permissions."
        elif "NoSuchBucket" in error_message:
            error_message = f"S3 bucket '{S3_BUCKET}' does not exist. Please check your bucket configuration."
        elif "ConnectionError" in error_message:
            error_message = "Connection error. Please check your internet connection and try again."

        raise HTTPException(status_code=500, detail=f"Error processing PDF: {error_message}")

@specialized_pdf_router.get("/processing/{pdf_id}", response_model=PDFProcessingStatus)
async def get_pdf_processing_status(pdf_id: str):
    """
    Get the processing status of a specific PDF.

    Args:
        pdf_id: The ID of the PDF to check

    Returns:
        Processing status information
    """
    if pdf_id not in pdf_processing_status:
        raise HTTPException(status_code=404, detail=f"PDF with ID {pdf_id} not found")

    return pdf_processing_status[pdf_id]

@specialized_pdf_router.get("/processing/{pdf_id}/quick")
async def get_pdf_quick_status(pdf_id: str):
    """
    Get a quick status check for a PDF (just status and progress).

    Args:
        pdf_id: The ID of the PDF to check

    Returns:
        Quick status information
    """
    if pdf_id not in pdf_processing_status:
        raise HTTPException(status_code=404, detail=f"PDF with ID {pdf_id} not found")

    status_info = pdf_processing_status[pdf_id]
    return {
        "pdf_id": pdf_id,
        "status": status_info["status"],
        "progress": "completed" if status_info["status"] == "completed" else "processing",
        "num_pages": status_info.get("num_pages", 0),
        "processing_time": status_info.get("processing_time", 0.0),
        "error": status_info.get("error")
    }

@specialized_pdf_router.get("/index/status", response_model=IndexStatusResponse)
async def get_index_status():
    """
    Get the status of all specialized indexes.

    Returns:
        Status information for each index
    """
    try:
        # Initialize response with default values
        response = {
            "general_index": {"exists": False},
            "psychologist_index": {"exists": False},
            "dietician_index": {"exists": False},
            # Add backward compatibility fields
            "physician_index": {"exists": False},
            "psychiatric_index": {"exists": False},
            "timestamp": datetime.now().isoformat()
        }

        # Check each index in S3
        for index_type in S3_PATHS.keys():
            # Try to initialize the vector store from S3
            try:
                # Initialize the vector store if not already done
                if vector_stores[index_type] is None:
                    # Get the S3 prefix from the S3_PATHS dictionary
                    s3_prefix = S3_PATHS.get(index_type)
                    vector_stores[index_type] = initialize_vector_store("", s3_prefix)

                # If we successfully loaded the vector store, mark it as existing
                if vector_stores[index_type] is not None:
                    response[index_type]["exists"] = True

                    # Get document count if possible
                    try:
                        # This is a bit of a hack since FAISS doesn't expose document count directly
                        # We're using the index's docstore which contains all documents
                        doc_count = len(vector_stores[index_type].docstore._dict)
                        response[index_type]["document_count"] = doc_count

                        # Use current time as last updated since we can't get it from S3 easily
                        response[index_type]["last_updated"] = datetime.now().isoformat()
                    except Exception as e:
                        logger.error(f"Error getting document count for {index_type}: {str(e)}")
            except Exception as e:
                logger.error(f"Error checking index {index_type} in S3: {str(e)}")

        # For backward compatibility, copy psychologist data to physician and psychiatric fields
        if response["psychologist_index"]["exists"]:
            response["physician_index"] = response["psychologist_index"].copy()
            response["psychiatric_index"] = response["psychologist_index"].copy()

        return response
    except Exception as e:
        logger.error(f"Error getting index status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting index status: {str(e)}")

# Add a status endpoint to check the service status
@specialized_pdf_router.get("/status")
async def get_service_status():
    """
    Get the status of the PDF processing service.

    Returns:
        Status information for the service
    """
    try:
        # Check vector store status
        vector_store_status = "Loaded"
        for index_type in vector_stores:
            if not vector_stores[index_type]:
                vector_store_status = "Partially loaded"
                break

        # Check OpenAI API key
        openai_api_key_status = "Valid" if OPENAI_API_KEY else "Not set"

        # Check S3 status
        s3_status = {
            "enabled": USE_S3 and s3_client is not None,
            "bucket": S3_BUCKET,
            "region": AWS_REGION,
            "file_count": 0,
            "files": []
        }

        # If S3 is enabled, check for files
        if s3_status["enabled"]:
            try:
                # List objects in the bucket
                response = s3_client.list_objects_v2(
                    Bucket=S3_BUCKET,
                    MaxKeys=10
                )

                if 'Contents' in response:
                    s3_status["file_count"] = response.get('KeyCount', 0)
                    s3_status["files"] = [obj['Key'] for obj in response.get('Contents', [])[:10]]
            except Exception as e:
                logger.error(f"Error listing S3 objects: {str(e)}")

        # Check local directories
        faiss_directory = {
            "exists": os.path.exists("faiss_index"),
            "file_count": 0,
            "files": []
        }

        if faiss_directory["exists"]:
            try:
                faiss_files = []
                for root, _, files in os.walk("faiss_index"):
                    for file in files:
                        faiss_files.append(os.path.join(root, file))
                faiss_directory["file_count"] = len(faiss_files)
                faiss_directory["files"] = faiss_files[:10]  # Limit to 10 files
            except Exception as e:
                logger.error(f"Error listing FAISS files: {str(e)}")

        temp_directory = {
            "exists": os.path.exists(TEMP_DIR),
            "file_count": 0,
            "files": []
        }

        if temp_directory["exists"]:
            try:
                temp_files = [f for f in os.listdir(TEMP_DIR) if os.path.isfile(os.path.join(TEMP_DIR, f))]
                temp_directory["file_count"] = len(temp_files)
                temp_directory["files"] = temp_files[:10]  # Limit to 10 files
            except Exception as e:
                logger.error(f"Error listing temp files: {str(e)}")

        # Return the status
        return {
            "vector_store_status": vector_store_status,
            "openai_api_key_status": openai_api_key_status,
            "s3_status": s3_status,
            "faiss_directory": faiss_directory,
            "temp_directory": temp_directory,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting service status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting service status: {str(e)}")

# Vector stores are already initialized by the vector_store_initializer module
logger.info("Vector stores already initialized by vector_store_initializer module")

# Log the status of each vector store
for index_type in vector_stores:
    if vector_stores[index_type]:
        try:
            # Get document count if possible
            doc_count = len(vector_stores[index_type].index_to_docstore_id)
            logger.info(f"✅ {index_type} vector store is loaded with {doc_count} documents")
        except Exception as e:
            logger.info(f"✅ {index_type} vector store is loaded")
    else:
        logger.warning(f"⚠️ {index_type} vector store is not loaded")
