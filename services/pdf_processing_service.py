import os
import uuid
import logging
import tempfile
import shutil
from typing import List, Dict, Any, Optional, Tuple, BinaryIO
from datetime import datetime
from io import BytesIO
import re
import json
import boto3
from botocore.exceptions import NoCredentialsError

# PDF processing
import PyPDF2
import pytesseract
from pdf2image import convert_from_path, convert_from_bytes
from PIL import Image

# Text processing
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

# Vector store
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from services.shared_vector_stores import get_pdf_vector_store

# FastAPI
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
pdf_router = APIRouter()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PDF_FAISS_INDEX_DIR = "resources/pdf_faiss_index"
TEMP_DIR = "temp_pdf_files"
STATIC_JWT_TOKEN = os.getenv("STATIC_JWT_TOKEN")

# AWS S3 Configuration
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
PDF_BUCKET_NAME = os.getenv("PDF_BUCKET_NAME", "prasha-health-pdf")
USE_S3 = os.getenv("USE_S3", "True").lower() == "true"

# Initialize S3 client if credentials are available
s3_client = None
if USE_S3 and AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION
        )
        logger.info("✅ AWS S3 client initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize AWS S3 client: {str(e)}")
else:
    logger.warning("⚠️ AWS S3 integration is disabled or credentials are missing")

# Ensure temp directory exists
os.makedirs(TEMP_DIR, exist_ok=True)

# Models
class PDFProcessingResponse(BaseModel):
    """Response model for PDF processing."""
    message: str
    pdf_id: str
    num_pages: int
    extracted_text_sample: str
    processing_time: float
    s3_url: Optional[str] = None

class PDFSearchResponse(BaseModel):
    """Response model for PDF search."""
    results: List[Dict[str, Any]]
    query: str
    total_results: int

# S3 Helper Functions
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
        s3_client.put_object(
            Bucket=PDF_BUCKET_NAME,
            Key=key,
            Body=file_content
        )
        s3_url = f"https://{PDF_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{key}"
        logger.info(f"Successfully saved to S3: {s3_url}")
        return s3_url
    except NoCredentialsError:
        logger.error("AWS credentials not available")
        return None
    except Exception as e:
        logger.error(f"Error saving to S3: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def get_from_s3(key: str) -> Optional[bytes]:
    """
    Get content from S3 bucket.

    Args:
        key: The S3 key (path) to get

    Returns:
        Content if successful, None otherwise
    """
    if not USE_S3 or not s3_client:
        logger.warning("S3 integration is disabled or not configured")
        return None

    try:
        response = s3_client.get_object(
            Bucket=PDF_BUCKET_NAME,
            Key=key
        )
        return response['Body'].read()
    except Exception as e:
        logger.error(f"Error getting from S3: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def delete_from_s3(key: str) -> bool:
    """
    Delete content from S3 bucket.

    Args:
        key: The S3 key (path) to delete

    Returns:
        True if successful, False otherwise
    """
    if not USE_S3 or not s3_client:
        logger.warning("S3 integration is disabled or not configured")
        return False

    try:
        s3_client.delete_object(
            Bucket=PDF_BUCKET_NAME,
            Key=key
        )
        logger.info(f"Successfully deleted from S3: {key}")
        return True
    except Exception as e:
        logger.error(f"Error deleting from S3: {str(e)}")
        return False

# FAISS S3 Helper Functions
def save_faiss_to_s3(vector_store):
    """
    Save FAISS index to S3.

    Args:
        vector_store: The FAISS vector store to save

    Returns:
        True if successful, False otherwise
    """
    if not USE_S3 or not s3_client:
        logger.warning("S3 integration is disabled or not configured")
        return False

    try:
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save the vector store locally first
            local_path = os.path.join(temp_dir, "faiss_index")
            vector_store.save_local(local_path)

            # Upload each file to S3
            for filename in os.listdir(local_path):
                file_path = os.path.join(local_path, filename)
                with open(file_path, 'rb') as f:
                    s3_client.put_object(
                        Bucket=PDF_BUCKET_NAME,
                        Key=f"faiss_index/{filename}",
                        Body=f
                    )

            logger.info(f"Successfully saved FAISS index to S3 bucket: {PDF_BUCKET_NAME}/faiss_index/")
            return True
    except Exception as e:
        logger.error(f"Error saving FAISS index to S3: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def load_faiss_from_s3(embeddings):
    """
    Load FAISS index from S3.

    Args:
        embeddings: The embeddings to use for the vector store

    Returns:
        FAISS vector store if successful, None otherwise
    """
    if not USE_S3 or not s3_client:
        logger.warning("S3 integration is disabled or not configured")
        return None

    try:
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create the faiss_index directory
            local_path = os.path.join(temp_dir, "faiss_index")
            os.makedirs(local_path, exist_ok=True)

            # List all files in the S3 faiss_index directory
            response = s3_client.list_objects_v2(
                Bucket=PDF_BUCKET_NAME,
                Prefix="faiss_index/"
            )

            if 'Contents' not in response:
                logger.warning("No FAISS index found in S3")
                return None

            # Download each file
            for obj in response['Contents']:
                key = obj['Key']
                filename = os.path.basename(key)
                if filename:  # Skip the directory itself
                    file_path = os.path.join(local_path, filename)
                    s3_client.download_file(
                        Bucket=PDF_BUCKET_NAME,
                        Key=key,
                        Filename=file_path
                    )

            # Load the vector store
            vector_store = FAISS.load_local(
                local_path,
                embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info("✅ Loaded FAISS index from S3")
            return vector_store
    except Exception as e:
        logger.error(f"Error loading FAISS index from S3: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

# Initialize FAISS vector store
def initialize_pdf_vector_store():
    """Initialize or load the FAISS vector store for PDF content."""
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

        # Try to load from S3 first if enabled
        if USE_S3 and s3_client:
            logger.info("Attempting to load FAISS index from S3...")
            vector_store = load_faiss_from_s3(embeddings)
            if vector_store:
                return vector_store
            logger.info("No FAISS index found in S3, checking local storage...")

        # Check if the FAISS index already exists locally
        if os.path.exists(PDF_FAISS_INDEX_DIR):
            vector_store = FAISS.load_local(
                PDF_FAISS_INDEX_DIR,
                embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info("✅ Loaded existing FAISS index from local storage")

            # If S3 is enabled, save to S3 as well
            if USE_S3 and s3_client:
                logger.info("Saving local FAISS index to S3...")
                save_faiss_to_s3(vector_store)

            return vector_store

        # Create a new empty vector store
        vector_store = FAISS.from_texts(["PDF processing service initialized."], embeddings)

        # Save the vector store locally
        os.makedirs(PDF_FAISS_INDEX_DIR, exist_ok=True)
        vector_store.save_local(PDF_FAISS_INDEX_DIR)
        logger.info("✅ Created new FAISS index in local storage")

        # If S3 is enabled, save to S3 as well
        if USE_S3 and s3_client:
            logger.info("Saving new FAISS index to S3...")
            save_faiss_to_s3(vector_store)

        return vector_store
    except Exception as e:
        logger.error(f"Error initializing PDF vector store: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

# Get the vector store from shared module
logger.info("Getting PDF vector store from shared module...")
vector_store = get_pdf_vector_store()

# Helper functions
def extract_text_from_pdf(file_path: str) -> Tuple[str, int]:
    """
    Extract text from a PDF file using PyPDF2.
    Falls back to OCR if needed.

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

                # If page has no text, try OCR
                if not page_text or len(page_text.strip()) < 50:  # Arbitrary threshold
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
        return "Error extracting text from PDF", 0

def clean_text(text: str) -> str:
    """
    Clean and preprocess the extracted text.

    Args:
        text: Raw extracted text

    Returns:
        Cleaned text
    """
    try:
        # Basic cleaning
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
        text = re.sub(r'[^\w\s.,;:!?()-]', '', text)  # Remove special characters

        # More advanced cleaning could be added here

        return text
    except Exception as e:
        logger.error(f"Error cleaning text: {str(e)}")
        return text

def add_to_vector_store(text: str, metadata: Dict[str, Any]) -> bool:
    """
    Add the extracted text to the FAISS vector store.

    Args:
        text: Cleaned text to add
        metadata: Metadata about the PDF

    Returns:
        Success status
    """
    try:
        logger.info(f"Starting to add text to vector store for PDF {metadata.get('pdf_id')}")

        # Split text into chunks
        logger.info("Splitting text into chunks")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_text(text)

        logger.info(f"Created {len(chunks)} chunks from text")

        if not chunks:
            logger.warning("No chunks created from text")
            return False

        # Create metadata for each chunk
        metadatas = [metadata.copy() for _ in chunks]
        logger.info(f"Created metadata for {len(metadatas)} chunks")

        # Add to vector store
        logger.info("Initializing OpenAI embeddings with API key")
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

        # If vector store doesn't exist yet, create it
        global vector_store
        if vector_store is None:
            logger.info("Vector store is None, creating new one")
            vector_store = FAISS.from_texts(chunks, embeddings, metadatas=metadatas)
            logger.info(f"Creating directory: {PDF_FAISS_INDEX_DIR}")
            os.makedirs(PDF_FAISS_INDEX_DIR, exist_ok=True)
            logger.info("Saving vector store locally")
            vector_store.save_local(PDF_FAISS_INDEX_DIR)

            # If S3 is enabled, save to S3 as well
            if USE_S3 and s3_client:
                logger.info("Saving new vector store to S3...")
                save_faiss_to_s3(vector_store)
        else:
            # Add to existing vector store
            logger.info("Adding to existing vector store")
            vector_store.add_texts(chunks, metadatas=metadatas)
            logger.info("Saving updated vector store locally")
            vector_store.save_local(PDF_FAISS_INDEX_DIR)

            # If S3 is enabled, save to S3 as well
            if USE_S3 and s3_client:
                logger.info("Saving updated vector store to S3...")
                save_faiss_to_s3(vector_store)

        logger.info(f"Successfully added {len(chunks)} chunks to vector store")
        return True
    except Exception as e:
        logger.error(f"Error adding to vector store: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

# API Endpoints
@pdf_router.post("/upload", response_model=PDFProcessingResponse)
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: str = Form(None),
    description: str = Form(None)
):
    """
    Upload and process a PDF file.

    Args:
        file: The PDF file to upload
        title: Optional title for the PDF
        description: Optional description for the PDF

    Returns:
        Processing status and information
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    # Generate a unique ID for this PDF
    pdf_id = str(uuid.uuid4())
    s3_url = None

    try:
        # Read the file content
        file_content = await file.read()

        # If S3 is enabled, save to S3
        if USE_S3 and s3_client:
            logger.info(f"Saving PDF {pdf_id} to S3...")
            s3_key = f"pdfs/{pdf_id}.pdf"
            s3_url = save_to_s3(file_content, s3_key)

            if s3_url:
                logger.info(f"PDF saved to S3: {s3_url}")

                # Process the PDF in the background using S3
                background_tasks.add_task(
                    process_pdf_background_s3,
                    s3_key,
                    pdf_id,
                    title or file.filename,
                    description
                )
            else:
                logger.warning("Failed to save PDF to S3, falling back to local storage")
                # Fall back to local storage
                s3_url = None

        # If S3 is not enabled or failed, use local storage
        if not s3_url:
            # Create a temporary file
            temp_file_path = os.path.join(TEMP_DIR, f"{pdf_id}.pdf")

            # Save the uploaded file locally
            with open(temp_file_path, "wb") as buffer:
                buffer.write(file_content)

            # Process the PDF in the background using local storage
            background_tasks.add_task(
                process_pdf_background,
                temp_file_path,
                pdf_id,
                title or file.filename,
                description
            )

        # Return immediate response
        return PDFProcessingResponse(
            message="PDF upload received and processing started",
            pdf_id=pdf_id,
            num_pages=0,  # Will be updated during background processing
            extracted_text_sample="Processing...",
            processing_time=0.0,
            s3_url=s3_url
        )
    except Exception as e:
        logger.error(f"Error processing PDF upload: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

async def process_pdf_background_s3(s3_key: str, pdf_id: str, title: str, description: str):
    """
    Process a PDF file from S3 in the background.

    Args:
        s3_key: S3 key (path) to the PDF file
        pdf_id: Unique ID for the PDF
        title: Title for the PDF
        description: Description for the PDF
    """
    start_time = datetime.now()
    temp_file_path = None

    try:
        logger.info(f"Starting background processing of PDF {pdf_id} from S3 key {s3_key}")

        # Get the PDF from S3
        pdf_content = get_from_s3(s3_key)
        if not pdf_content:
            logger.error(f"Failed to get PDF from S3: {s3_key}")
            return

        logger.info(f"Downloaded PDF from S3, size: {len(pdf_content)} bytes")

        # Create a temporary file
        temp_file_path = os.path.join(TEMP_DIR, f"{pdf_id}_temp.pdf")
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(pdf_content)

        logger.info(f"Saved PDF to temporary file: {temp_file_path}")

        # Extract text from PDF
        logger.info(f"Extracting text from PDF {pdf_id}")
        text, num_pages = extract_text_from_pdf(temp_file_path)
        logger.info(f"Extracted {len(text)} characters from {num_pages} pages")

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
            "source": "pdf_upload",
            "s3_key": s3_key
        }
        logger.info(f"Created metadata for PDF {pdf_id}: {metadata}")

        # Check OpenAI API key
        if not OPENAI_API_KEY:
            logger.error("OpenAI API key is not set. Cannot create embeddings.")
            return

        # Add to vector store
        logger.info(f"Adding PDF {pdf_id} to vector store")
        success = add_to_vector_store(cleaned_text, metadata)
        logger.info(f"Vector store update result: {success}")

        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()

        logger.info(f"PDF {pdf_id} processed successfully in {processing_time:.2f} seconds")

        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            logger.info(f"Removing temporary file {temp_file_path}")
            os.remove(temp_file_path)

    except Exception as e:
        logger.error(f"Error in background PDF processing from S3: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            logger.info(f"Removing temporary file {temp_file_path} after error")
            os.remove(temp_file_path)

async def process_pdf_background(file_path: str, pdf_id: str, title: str, description: str):
    """
    Process a PDF file in the background.

    Args:
        file_path: Path to the PDF file
        pdf_id: Unique ID for the PDF
        title: Title for the PDF
        description: Description for the PDF
    """
    start_time = datetime.now()

    try:
        logger.info(f"Starting background processing of PDF {pdf_id} at {file_path}")

        # Check if file exists
        if not os.path.exists(file_path):
            logger.error(f"PDF file not found at {file_path}")
            return

        logger.info(f"File exists, size: {os.path.getsize(file_path)} bytes")

        # Extract text from PDF
        logger.info(f"Extracting text from PDF {pdf_id}")
        text, num_pages = extract_text_from_pdf(file_path)
        logger.info(f"Extracted {len(text)} characters from {num_pages} pages")

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
            "source": "pdf_upload"
        }
        logger.info(f"Created metadata for PDF {pdf_id}: {metadata}")

        # Check OpenAI API key
        if not OPENAI_API_KEY:
            logger.error("OpenAI API key is not set. Cannot create embeddings.")
            return

        # Add to vector store
        logger.info(f"Adding PDF {pdf_id} to vector store")
        success = add_to_vector_store(cleaned_text, metadata)
        logger.info(f"Vector store update result: {success}")

        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()

        logger.info(f"PDF {pdf_id} processed successfully in {processing_time:.2f} seconds")

        # Clean up temporary file
        if os.path.exists(file_path):
            logger.info(f"Removing temporary file {file_path}")
            os.remove(file_path)

    except Exception as e:
        logger.error(f"Error in background PDF processing: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

        # Clean up temporary file
        if os.path.exists(file_path):
            logger.info(f"Removing temporary file {file_path} after error")
            os.remove(file_path)

@pdf_router.get("/status")
async def get_pdf_service_status():
    """
    Get the status of the PDF processing service.
    """
    try:
        # Check if vector store is initialized
        vector_store_status = "Initialized" if vector_store else "Not initialized"

        # Check if temp directory exists
        temp_dir_exists = os.path.exists(TEMP_DIR)

        # Check if FAISS index directory exists
        faiss_dir_exists = os.path.exists(PDF_FAISS_INDEX_DIR)

        # Count files in temp directory
        temp_files = []
        if temp_dir_exists:
            temp_files = os.listdir(TEMP_DIR)

        # Count files in FAISS index directory
        faiss_files = []
        if faiss_dir_exists:
            faiss_files = os.listdir(PDF_FAISS_INDEX_DIR)

        # Check OpenAI API key
        openai_key_status = "Set" if OPENAI_API_KEY else "Not set"

        # Check S3 status
        s3_status = "Enabled" if USE_S3 and s3_client else "Disabled"

        # List S3 files if enabled
        s3_files = []
        if USE_S3 and s3_client:
            try:
                # List PDFs
                pdf_response = s3_client.list_objects_v2(
                    Bucket=PDF_BUCKET_NAME,
                    Prefix="pdfs/"
                )
                if 'Contents' in pdf_response:
                    for obj in pdf_response['Contents']:
                        s3_files.append(obj['Key'])

                # List FAISS index files
                faiss_response = s3_client.list_objects_v2(
                    Bucket=PDF_BUCKET_NAME,
                    Prefix="faiss_index/"
                )
                if 'Contents' in faiss_response:
                    for obj in faiss_response['Contents']:
                        s3_files.append(obj['Key'])
            except Exception as e:
                logger.error(f"Error listing S3 files: {str(e)}")

        return {
            "vector_store_status": vector_store_status,
            "temp_directory": {
                "exists": temp_dir_exists,
                "path": os.path.abspath(TEMP_DIR),
                "files": temp_files,
                "file_count": len(temp_files)
            },
            "faiss_directory": {
                "exists": faiss_dir_exists,
                "path": os.path.abspath(PDF_FAISS_INDEX_DIR),
                "files": faiss_files,
                "file_count": len(faiss_files)
            },
            "s3_status": {
                "enabled": s3_status == "Enabled",
                "bucket": PDF_BUCKET_NAME if USE_S3 else None,
                "region": AWS_REGION if USE_S3 else None,
                "files": s3_files,
                "file_count": len(s3_files)
            },
            "openai_api_key_status": openai_key_status,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting PDF service status: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error getting PDF service status: {str(e)}")

@pdf_router.get("/search", response_model=PDFSearchResponse)
async def search_pdfs(query: str, limit: int = 5):
    """
    Search for content in processed PDFs.

    Args:
        query: Search query
        limit: Maximum number of results to return

    Returns:
        Search results
    """
    if not vector_store:
        raise HTTPException(status_code=500, detail="Vector store not initialized")

    try:
        # Create retriever
        retriever = vector_store.as_retriever(search_kwargs={"k": limit})

        # Perform search
        docs = retriever.get_relevant_documents(query)

        # Format results
        results = []
        for doc in docs:
            results.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })

        return PDFSearchResponse(
            results=results,
            query=query,
            total_results=len(results)
        )
    except Exception as e:
        logger.error(f"Error searching PDFs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error searching PDFs: {str(e)}")
