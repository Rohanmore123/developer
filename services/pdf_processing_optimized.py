"""
PDF Processing Service - Optimized Version

This module provides an optimized version of the PDF processing service
that connects with S3 bucket FAISS indexes for general and dietician specialties.
It uses HNSW indexing for better performance and is designed for AWS cloud deployment.
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
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv
import boto3
from botocore.exceptions import ClientError  # Used for handling S3 errors

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import OpenAIEmbeddings with fallback
try:
    # Try to import from langchain_openai (new version)
    from langchain_openai import OpenAIEmbeddings
    logger.info("Using OpenAIEmbeddings from langchain_openai")
except ImportError:
    # Fall back to old import path
    from langchain.embeddings.openai import OpenAIEmbeddings
    logger.warning("Using deprecated OpenAIEmbeddings from langchain.embeddings.openai")

# Router
optimized_pdf_router = APIRouter(tags=["PDF Processing - Optimized"])

# Environment variables - force reload to ensure we get the latest value
load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
logger.info(f"Using OpenAI API key: {OPENAI_API_KEY[:10]}...{OPENAI_API_KEY[-5:] if OPENAI_API_KEY else 'None'}")

# S3 configuration
USE_S3 = True  # Always use S3 for this optimized version
S3_BUCKET = os.getenv("PDF_BUCKET_NAME", "prasha-health-pdf")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# Log environment variables (without exposing full credentials)
logger.info("Environment variables:")
logger.info(f"  S3_BUCKET: {S3_BUCKET}")
logger.info(f"  AWS_REGION: {AWS_REGION}")
logger.info(f"  AWS_ACCESS_KEY set: {'Yes' if AWS_ACCESS_KEY else 'No'}")
logger.info(f"  AWS_SECRET_KEY set: {'Yes' if AWS_SECRET_KEY else 'No'}")

# Temporary directory for PDF files during processing (files are deleted after processing)
TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

# S3 paths for different indexes - only include general and dietician
# Adjusted to match the existing S3 bucket structure
S3_PATHS = {
    "general_index": "faiss_index/general_index",
    "dietician_index": "faiss_index/dietician_index"  # Updated to use consistent path structure
}

# Initialize S3 client
s3_client = None
try:
    if AWS_ACCESS_KEY and AWS_SECRET_KEY:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY,
            region_name=AWS_REGION
        )
        logger.info("S3 client initialized with explicit credentials")
    else:
        s3_client = boto3.client('s3', region_name=AWS_REGION)
        logger.info("S3 client initialized using default credential provider chain")

    # Test S3 connection
    response = s3_client.list_buckets()
    buckets = [bucket['Name'] for bucket in response['Buckets']]
    logger.info(f"S3 connection successful. Available buckets: {buckets}")

    if S3_BUCKET in buckets:
        logger.info(f"Target bucket '{S3_BUCKET}' found")
    else:
        logger.error(f"Target bucket '{S3_BUCKET}' not found in available buckets")
        raise Exception(f"Bucket {S3_BUCKET} not found")

except Exception as e:
    logger.error(f"Error initializing S3 client: {str(e)}")
    raise Exception(f"Failed to initialize S3 client: {str(e)}")

# Use pre-loaded vector stores from the initializer
import vector_store_initializer

# Map our index names to the initializer's vector stores
vector_stores = {
    "general_index": vector_store_initializer.general_vector_store,
    "dietician_index": vector_store_initializer.dietician_vector_store
}

logger.info("Using pre-loaded vector stores from initializer")

# Models
class OptimizedPDFProcessingResponse(BaseModel):
    """Response model for optimized PDF processing."""
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
    s3_path: Optional[str] = None
    document_count: Optional[int] = None
    pdf_count: Optional[int] = None
    last_updated: Optional[str] = None
    faiss_size: Optional[int] = None
    pkl_size: Optional[int] = None
    pdf_samples: Optional[list] = None
    directory_exists: Optional[bool] = None
    object_count: Optional[int] = None
    error: Optional[str] = None

class IndexStatusResponse(BaseModel):
    """Response model for index status."""
    general_index: IndexStatus
    dietician_index: IndexStatus
    timestamp: str

# Dictionary to store processing status of PDFs
pdf_processing_status = {}

# Helper Functions
def save_to_s3(file_content: bytes, key: str) -> Optional[str]:
    """Save content to S3 bucket."""
    try:
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=key,
            Body=file_content
        )
        s3_url = f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{key}"
        logger.info(f"Successfully saved to S3: {s3_url}")
        return s3_url
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        error_message = e.response.get('Error', {}).get('Message', str(e))
        logger.error(f"S3 ClientError: {error_code} - {error_message}")
        raise
    except Exception as e:
        logger.error(f"Error saving to S3: {str(e)}")
        raise

def extract_text_from_pdf(file_path: str) -> Tuple[str, int]:
    """Extract text from a PDF file, using OCR if necessary."""
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            num_pages = len(reader.pages)

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
    """Perform OCR on a specific page of a PDF."""
    try:
        images = convert_from_path(file_path, first_page=page_num+1, last_page=page_num+1)
        if not images:
            return ""
        text = pytesseract.image_to_string(images[0])
        return text
    except Exception as e:
        logger.error(f"Error performing OCR on page {page_num}: {str(e)}")
        return ""

def ocr_pdf_document(file_path: str) -> Tuple[str, int]:
    """Perform OCR on an entire PDF document."""
    try:
        images = convert_from_path(file_path)
        num_pages = len(images)

        text = ""
        for i, image in enumerate(images):
            page_text = pytesseract.image_to_string(image)
            text += f"\n\n--- Page {i+1} ---\n\n{page_text}"

        return text, num_pages
    except Exception as e:
        logger.error(f"Error performing OCR on document: {str(e)}")
        return f"Error extracting text: {str(e)}", 0

def clean_text(text: str) -> str:
    """Clean extracted text."""
    import re
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = ''.join(c for c in text if c.isprintable() or c in ['\n', '\t'])
    return text

def initialize_vector_store(s3_prefix: str):
    """Get the appropriate vector store based on the S3 prefix."""
    try:
        logger.info(f"Getting vector store for prefix: {s3_prefix}")

        # Map S3 prefixes to vector store keys
        prefix_to_key = {
            "faiss_index/general_index": "general_index",
            "faiss_index/dietician_index": "dietician_index"
        }

        # Get the key for this prefix
        key = prefix_to_key.get(s3_prefix)
        if not key:
            logger.warning(f"Unknown S3 prefix: {s3_prefix}, using general_index as fallback")
            key = "general_index"

        # Get the vector store from our map
        vector_store = vector_stores.get(key)
        if vector_store:
            logger.info(f"Using pre-loaded vector store for {key}")
            return vector_store
        else:
            logger.warning(f"No pre-loaded vector store found for {key}")
            return None

    except Exception as e:
        logger.error(f"Error getting vector store: {str(e)}")
        return None

def add_to_vector_store(text: str, metadata: Dict[str, Any], index_type: str, chunk_size: int = 500, chunk_overlap: int = 50) -> bool:
    """
    Add text to the appropriate vector store with HNSW indexing.

    Args:
        text: Text to add
        metadata: Metadata for the text
        index_type: Type of index (general_index or dietician_index)
        chunk_size: Size of text chunks in tokens (default: 500)
        chunk_overlap: Overlap between chunks in tokens (default: 50)

    Returns:
        True if successful, False otherwise
    """
    try:
        # Validate index type
        if index_type not in S3_PATHS:
            logger.error(f"Invalid index type: {index_type}")
            return False

        # Use the S3 paths defined at the top of the file
        s3_prefix = S3_PATHS.get(index_type)

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_text(text)
        logger.info(f"Split text into {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")

        # Create metadata for each chunk
        metadatas = [metadata.copy() for _ in chunks]
        logger.info(f"Created metadata for {len(metadatas)} chunks")

        # Initialize OpenAI embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

        # Get or initialize the vector store
        vector_store = vector_stores[index_type]
        if vector_store is None:
            vector_store = initialize_vector_store(s3_prefix)

        # If vector store still doesn't exist, create a new one with HNSW indexing
        if vector_store is None:
            logger.info(f"Creating new FAISS HNSW vector store for {index_type} with {len(chunks)} chunks")

            # Use HNSW index type for better performance
            vector_store = FAISS.from_texts(
                chunks,
                embeddings,
                metadatas=metadatas,
                # HNSW index parameters for optimal performance
                index_kwargs={"space": "cosine", "M": 16, "ef_construction": 200, "ef_search": 100}
            )
            logger.info(f"✅ Successfully created FAISS HNSW index for {index_type}")
            vector_stores[index_type] = vector_store
        else:
            # Add to existing vector store
            logger.info(f"Adding to existing vector store for {index_type}")
            vector_store.add_texts(chunks, metadatas=metadatas)
            logger.info(f"✅ Successfully added {len(chunks)} chunks to existing index")
            vector_stores[index_type] = vector_store

        # Save directly to S3
        try:
            logger.info(f"Saving vector store to S3: {s3_prefix}")

            # Serialize the vector store to bytes
            # This creates two files: index.faiss and index.pkl
            # Check which method signature to use (different versions of langchain have different signatures)
            if hasattr(vector_store, 'serialize_to_bytes'):
                # New method
                logger.info("Using vector_store.serialize_to_bytes() method")
                faiss_bytes, pkl_bytes = vector_store.serialize_to_bytes()
            else:
                # Old method
                logger.info("Using FAISS.serialize_to_bytes() class method")
                faiss_bytes, pkl_bytes = FAISS.serialize_to_bytes(vector_store)
            logger.info(f"Successfully serialized vector store to bytes: {len(faiss_bytes)} bytes for FAISS, {len(pkl_bytes)} bytes for PKL")

            # Upload directly to S3
            s3_client.put_object(
                Bucket=S3_BUCKET,
                Key=f"{s3_prefix}/index.faiss",
                Body=faiss_bytes
            )
            logger.info(f"Successfully uploaded index.faiss to S3")

            s3_client.put_object(
                Bucket=S3_BUCKET,
                Key=f"{s3_prefix}/index.pkl",
                Body=pkl_bytes
            )
            logger.info(f"Successfully uploaded index.pkl to S3")

            logger.info(f"✅ Successfully saved vector store to S3: {s3_prefix}")
            return True
        except Exception as save_error:
            logger.error(f"Error saving vector store to S3: {str(save_error)}")
            return False
    except Exception as e:
        logger.error(f"Error adding to vector store: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

async def process_pdf_background(
    file_path: str,
    pdf_id: str,
    index_type: str,
    title: str,
    description: str,
    chunk_size: int,
    chunk_overlap: int,
    s3_url: str
):
    """
    Process a PDF file in the background for the specified index.

    Args:
        file_path: Path to the temporary PDF file
        pdf_id: Unique ID for the PDF
        index_type: Type of index to add to
        title: Title for the PDF
        description: Description for the PDF
        chunk_size: Size of text chunks in tokens
        chunk_overlap: Overlap between chunks in tokens
        s3_url: S3 URL where the PDF is stored
    """
    start_time = datetime.now()

    # Initialize status in the global dictionary
    pdf_processing_status[pdf_id] = {
        "pdf_id": pdf_id,
        "index_type": index_type,
        "status": "processing",
        "num_pages": 0,
        "extracted_text_sample": "Processing...",
        "processing_time": 0.0,
        "s3_url": s3_url
    }

    try:
        logger.info(f"Processing PDF {pdf_id} for {index_type}")

        # Check if temporary file exists
        if not os.path.exists(file_path):
            logger.error(f"Temporary file not found: {file_path}")
            pdf_processing_status[pdf_id].update({
                "status": "failed",
                "error": f"Temporary file not found: {file_path}",
                "completed_at": datetime.now().isoformat()
            })
            return

        logger.info(f"Temporary file exists, size: {os.path.getsize(file_path)} bytes")

        # Extract text from PDF
        logger.info(f"Extracting text from PDF {pdf_id}")
        text, num_pages = extract_text_from_pdf(file_path)
        logger.info(f"Extracted {len(text)} characters from {num_pages} pages")

        # Update status with number of pages
        pdf_processing_status[pdf_id]["num_pages"] = num_pages

        # Get a sample of the extracted text
        text_sample = text[:500] + "..." if len(text) > 500 else text
        pdf_processing_status[pdf_id]["extracted_text_sample"] = text_sample

        # Clean the text
        logger.info(f"Cleaning text for PDF {pdf_id}")
        cleaned_text = clean_text(text)
        logger.info(f"Cleaned text length: {len(cleaned_text)} characters")

        # Create metadata
        metadata = {
            "pdf_id": pdf_id,
            "title": title,
            "description": description,
            "num_pages": num_pages,
            "processed_at": datetime.now().isoformat(),
            "source": "pdf_upload_optimized",
            "index_type": index_type,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "s3_url": s3_url
        }

        logger.info(f"Created metadata for PDF {pdf_id}")

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

        # Add to vector store with HNSW indexing
        logger.info(f"Adding PDF {pdf_id} to {index_type} with HNSW indexing")
        success = add_to_vector_store(
            cleaned_text,
            metadata,
            index_type,
            chunk_size,
            chunk_overlap
        )
        logger.info(f"Vector store update result: {success}")

        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"PDF {pdf_id} processed in {processing_time:.2f} seconds")

        # Update status to completed
        pdf_processing_status[pdf_id].update({
            "status": "completed" if success else "failed",
            "processing_time": processing_time,
            "completed_at": datetime.now().isoformat(),
            "error": None if success else "Failed to add to vector store"
        })
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
    finally:
        # Always clean up temporary file
        if os.path.exists(file_path):
            logger.info(f"Removing temporary file {file_path}")
            try:
                os.remove(file_path)
            except Exception as e:
                logger.error(f"Error removing temporary file {file_path}: {str(e)}")

# API Endpoints
@optimized_pdf_router.post("/upload", response_model=OptimizedPDFProcessingResponse)
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: str = Form(None),
    description: str = Form(None),
    index_type: str = Form(...),
    chunk_size: int = Form(500),
    chunk_overlap: int = Form(50)
):
    """
    Upload and process a PDF file for the specified index.

    This endpoint uploads a PDF file to S3, extracts text, and adds it to the specified FAISS index.
    The index is created with HNSW indexing for better performance and stored in S3.

    Args:
        file: The PDF file to upload
        title: Optional title for the PDF
        description: Optional description for the PDF
        index_type: Type of index to add to (general_index or dietician_index)
        chunk_size: Size of text chunks in tokens (default: 500)
        chunk_overlap: Overlap between chunks in tokens (default: 50)

    Returns:
        Processing status and information including the PDF ID for status tracking
    """
    # Log the request
    logger.info(f"Received PDF upload request: index_type={index_type}")
    logger.info(f"File: {file.filename}, chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")

    # Validate index type
    valid_index_types = list(S3_PATHS.keys())
    if index_type not in valid_index_types:
        logger.warning(f"Invalid index type: {index_type}")
        raise HTTPException(status_code=400, detail=f"Invalid index type. Must be one of: {', '.join(valid_index_types)}")

    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        logger.warning(f"Invalid file type: {file.filename}")
        raise HTTPException(status_code=400, detail="File must be a PDF")

    # Validate chunk size and overlap
    if chunk_size < 100 or chunk_size > 1000:
        logger.warning(f"Invalid chunk size: {chunk_size}")
        raise HTTPException(status_code=400, detail="Chunk size must be between 100 and 1000 tokens")

    if chunk_overlap < 0 or chunk_overlap > 200:
        logger.warning(f"Invalid chunk overlap: {chunk_overlap}")
        raise HTTPException(status_code=400, detail="Chunk overlap must be between 0 and 200 tokens")

    # Generate a unique ID for this PDF
    pdf_id = str(uuid.uuid4())
    logger.info(f"Generated PDF ID: {pdf_id}")

    try:
        # Read the file content
        file_content = await file.read()
        logger.info(f"Read file content: {len(file_content)} bytes")

        # Get the S3 prefix from the S3_PATHS dictionary
        s3_prefix = S3_PATHS.get(index_type)

        # Save PDF to S3 in the pdfs folder within the index directory
        # This matches the existing S3 bucket structure where PDFs are stored in a pdfs folder
        s3_key = f"{s3_prefix}/pdfs/{pdf_id}.pdf"
        logger.info(f"Using S3 key: {s3_key}")

        try:
            s3_url = save_to_s3(file_content, s3_key)
            logger.info(f"PDF saved to S3: {s3_url}")
        except Exception as s3_error:
            logger.error(f"Failed to save PDF to S3: {str(s3_error)}")
            raise HTTPException(status_code=500, detail="Failed to save PDF to S3")

        # Create temp directory if it doesn't exist
        if not os.path.exists(TEMP_DIR):
            logger.info(f"Creating temporary directory: {TEMP_DIR}")
            os.makedirs(TEMP_DIR, exist_ok=True)

        # Save to temporary file for processing
        temp_file_path = os.path.join(TEMP_DIR, f"{pdf_id}.pdf")
        with open(temp_file_path, "wb") as buffer:
            buffer.write(file_content)
        logger.info(f"Saved temporary file for processing: {temp_file_path}")

        # Process the PDF in the background
        logger.info(f"Starting background processing task for {pdf_id}")
        background_tasks.add_task(
            process_pdf_background,
            temp_file_path,
            pdf_id,
            index_type,
            title or file.filename,
            description,
            chunk_size,
            chunk_overlap,
            s3_url
        )

        # Return immediate response
        logger.info(f"Returning response for PDF {pdf_id}")
        return OptimizedPDFProcessingResponse(
            message=f"PDF upload received and processing started for {index_type}",
            pdf_id=pdf_id,
            index_type=index_type,
            num_pages=0,  # Will be updated during background processing
            extracted_text_sample="Processing...",
            processing_time=0.0,
            s3_url=s3_url
        )
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error processing PDF upload: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@optimized_pdf_router.get("/processing/{pdf_id}", response_model=PDFProcessingStatus)
async def get_pdf_processing_status(pdf_id: str):
    """
    Get the status of a PDF processing job.

    Args:
        pdf_id: The ID of the PDF processing job

    Returns:
        Status information for the processing job
    """
    if pdf_id not in pdf_processing_status:
        raise HTTPException(status_code=404, detail=f"PDF processing job with ID {pdf_id} not found")

    return pdf_processing_status[pdf_id]

@optimized_pdf_router.get("/index/status", response_model=IndexStatusResponse)
async def get_index_status():
    """
    Get the status of all indexes.

    Returns:
        Status information for all indexes including S3 paths and document counts
    """
    try:
        # Initialize response with default values
        response = {
            "general_index": {"exists": False},
            "dietician_index": {"exists": False},
            "timestamp": datetime.now().isoformat()
        }

        # Check each index
        for index_type, s3_prefix in S3_PATHS.items():
            try:
                # Add S3 path information to response
                response[index_type]["s3_path"] = s3_prefix

                # Try to head the index files
                try:
                    # Check if index.faiss exists
                    faiss_response = s3_client.head_object(Bucket=S3_BUCKET, Key=f"{s3_prefix}/index.faiss")
                    pkl_response = s3_client.head_object(Bucket=S3_BUCKET, Key=f"{s3_prefix}/index.pkl")

                    # If we get here, the index exists
                    response[index_type]["exists"] = True

                    # Add last modified date from S3 metadata
                    if 'LastModified' in faiss_response:
                        response[index_type]["last_updated"] = faiss_response['LastModified'].isoformat()
                    else:
                        response[index_type]["last_updated"] = datetime.now().isoformat()

                    # Add file sizes
                    if 'ContentLength' in faiss_response:
                        response[index_type]["faiss_size"] = faiss_response['ContentLength']
                    if 'ContentLength' in pkl_response:
                        response[index_type]["pkl_size"] = pkl_response['ContentLength']

                    # Try to get document count from vector store
                    if vector_stores[index_type] is not None:
                        try:
                            doc_count = len(vector_stores[index_type].docstore._dict)
                            response[index_type]["document_count"] = doc_count
                        except Exception as count_error:
                            logger.warning(f"Could not get document count for {index_type}: {str(count_error)}")

                    # Check for PDFs folder
                    try:
                        pdf_list_response = s3_client.list_objects_v2(
                            Bucket=S3_BUCKET,
                            Prefix=f"{s3_prefix}/pdfs/"
                        )

                        if 'Contents' in pdf_list_response:
                            pdf_count = len(pdf_list_response['Contents'])
                            response[index_type]["pdf_count"] = pdf_count

                            # Add sample PDF names (up to 5)
                            if pdf_count > 0:
                                pdf_samples = []
                                for i, obj in enumerate(pdf_list_response['Contents']):
                                    if i >= 5:  # Limit to 5 samples
                                        break
                                    pdf_samples.append(os.path.basename(obj['Key']))
                                response[index_type]["pdf_samples"] = pdf_samples
                    except Exception as pdf_error:
                        logger.warning(f"Error checking PDFs for {index_type}: {str(pdf_error)}")

                except Exception as head_error:
                    logger.info(f"Index files not found for {index_type}: {str(head_error)}")
                    response[index_type]["exists"] = False

                    # Check if the directory exists even if files don't
                    try:
                        list_response = s3_client.list_objects_v2(
                            Bucket=S3_BUCKET,
                            Prefix=f"{s3_prefix}/"
                        )

                        if 'Contents' in list_response and len(list_response['Contents']) > 0:
                            response[index_type]["directory_exists"] = True
                            response[index_type]["object_count"] = len(list_response['Contents'])
                        else:
                            response[index_type]["directory_exists"] = False
                    except Exception as list_error:
                        logger.warning(f"Error listing objects for {index_type}: {str(list_error)}")

            except Exception as e:
                logger.error(f"Error checking index status for {index_type}: {str(e)}")
                response[index_type] = {"exists": False, "error": str(e), "s3_path": s3_prefix}

        return response
    except Exception as e:
        logger.error(f"Error getting index status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting index status: {str(e)}")

# Initialize vector stores on startup
for index_type, s3_prefix in S3_PATHS.items():
    logger.info(f"Initializing {index_type} vector store with S3 prefix: {s3_prefix}")
    vector_stores[index_type] = initialize_vector_store(s3_prefix)
    if vector_stores[index_type] is None:
        logger.warning(f"⚠️ No existing {index_type} vector store found in S3")
