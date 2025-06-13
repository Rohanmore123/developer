import os
import uuid
import json
import base64
import asyncio
import tempfile
import concurrent.futures
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
import logging
import boto3
from botocore.exceptions import NoCredentialsError
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import desc
from database.database import SessionLocal
from database.database import get_db
from model.model_correct import (
    ChatMessage, Patient, Doctor, Appointment, OnboardingQuestion, DiaryEntry,
    EmotionAnalysis, MedicalHistory, Prescription, DoctorAvailability
)
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from services.shared_vector_stores import get_doctor_vector_store, get_pdf_vector_store

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
ai_audio_smart_chat_doc_router = APIRouter()

# OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
try:
    OPENAI_CLIENT = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
    if not OPENAI_CLIENT:
        logger.warning("OpenAI client could not be initialized. Check your API key.")
except ImportError:
    logger.warning("OpenAI package not installed. Some features may not work.")
    OPENAI_CLIENT = None

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
        import traceback
        logger.error(traceback.format_exc())
else:
    logger.warning("⚠️ AWS S3 integration is disabled or credentials are missing")

# FAISS index directories
DOCTOR_FAISS_INDEX_DIR = "resources/doctor_faiss_index"
PDF_FAISS_INDEX_DIR = "resources/pdf_faiss_index"

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

def load_faiss_from_s3(embeddings, prefix="faiss_index/"):
    """
    Load FAISS index from S3.

    Args:
        embeddings: The embeddings to use for the vector store
        prefix: The S3 prefix (directory) to load from

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
                Prefix=prefix
            )

            if 'Contents' not in response:
                logger.warning(f"No FAISS index found in S3 with prefix: {prefix}")
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
            logger.info(f"✅ Loaded FAISS index from S3 with prefix: {prefix}")
            return vector_store
    except Exception as e:
        logger.error(f"Error loading FAISS index from S3: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

# System prompt for the doctor assistant
SYSTEM_PROMPT = """
You are an AI assistant for doctors, helping them access patient information and providing summaries.
Your responses should be professional, concise, and focused on providing relevant medical information.
Always maintain patient confidentiality and provide factual information based on the data provided.
When referencing PDF documents, cite the source and provide specific information from the document.
"""

def initialize_doctor_faiss_index():
    """Initialize the FAISS vector store with medical information for doctors."""
    try:
        # Check if the FAISS index already exists
        if os.path.exists(DOCTOR_FAISS_INDEX_DIR):
            embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
            vector_store = FAISS.load_local(
                DOCTOR_FAISS_INDEX_DIR,
                embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info("✅ Loaded existing FAISS index for doctor service")
            return vector_store

        # If index doesn't exist, create a new one with medical information
        # Read the medical information text
        try:
            with open("resources/medical_info.txt", "r", encoding="utf-8") as f:
                medical_text = f.read()
        except FileNotFoundError:
            # Create a basic medical info file if it doesn't exist
            medical_text = """
            This is placeholder medical information. Replace with actual medical guidelines,
            treatment protocols, and other relevant information for doctors.
            """
            os.makedirs("resources", exist_ok=True)
            with open("resources/medical_info.txt", "w", encoding="utf-8") as f:
                f.write(medical_text)

        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_text(medical_text)

        # Create embeddings and vector store
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vector_store = FAISS.from_texts(chunks, embeddings)

        # Save the vector store
        os.makedirs(DOCTOR_FAISS_INDEX_DIR, exist_ok=True)
        vector_store.save_local(DOCTOR_FAISS_INDEX_DIR)
        logger.info(f"✅ Created and saved new FAISS index with {len(chunks)} chunks")

        return vector_store
    except Exception as e:
        logger.error(f"Error initializing FAISS vector store: {str(e)}")
        return None

def initialize_pdf_faiss_index():
    """Initialize the FAISS vector store with PDF content from S3."""
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

        # Try to load from S3 first if enabled
        if USE_S3 and s3_client:
            logger.info("Attempting to load FAISS index from S3...")
            vector_store = load_faiss_from_s3(embeddings)
            if vector_store:
                # Save a local copy for faster access next time
                os.makedirs(PDF_FAISS_INDEX_DIR, exist_ok=True)
                vector_store.save_local(PDF_FAISS_INDEX_DIR)
                logger.info("✅ Loaded FAISS index from S3 and saved locally")
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

            # If S3 is enabled but we loaded from local, save to S3 as well to ensure sync
            if USE_S3 and s3_client:
                logger.info("Syncing local FAISS index to S3...")
                save_faiss_to_s3(vector_store)

            return vector_store

        # Create a minimal vector store if we can't load from S3 or local
        logger.warning("Creating a minimal PDF vector store as fallback")
        vector_store = FAISS.from_texts(
            ["This is a placeholder for PDF content. Please upload PDFs to populate this index."],
            embeddings
        )

        # Save the minimal vector store locally
        os.makedirs(PDF_FAISS_INDEX_DIR, exist_ok=True)
        vector_store.save_local(PDF_FAISS_INDEX_DIR)
        logger.info("✅ Created new minimal FAISS index in local storage")

        # If S3 is enabled, save to S3 as well
        if USE_S3 and s3_client:
            logger.info("Saving new minimal FAISS index to S3...")
            save_faiss_to_s3(vector_store)

        return vector_store
    except Exception as e:
        logger.error(f"Error initializing PDF FAISS vector store: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

# Get vector stores from shared module
logger.info("Getting vector stores from shared module...")
doctor_vector_store = get_doctor_vector_store()
pdf_vector_store = get_pdf_vector_store()

# Create retrievers
doctor_retriever = doctor_vector_store.as_retriever(search_kwargs={"k": 5}) if doctor_vector_store else None
pdf_retriever = pdf_vector_store.as_retriever(search_kwargs={"k": 5}) if pdf_vector_store else None

def fetch_doctor_info(doctor_id: str, db: Session) -> Dict[str, Any]:
    """Fetch basic information about the doctor."""
    try:
        # Use a more selective query to avoid schema issues
        try:
            # First try with the standard column names
            doctor_query = db.query(
                Doctor.doctor_id,
                Doctor.title,
                Doctor.first_name,
                Doctor.middle_name,
                Doctor.last_name,
                Doctor.specialization,
                Doctor.email,
                Doctor.phone,
                Doctor.language,
                Doctor.address,
                Doctor.treatment
            ).filter(Doctor.doctor_id == doctor_id)
        except Exception as e:
            logger.error(f"Error creating doctor query: {str(e)}")
            # Fallback to minimal query if there are schema issues
            doctor_query = db.query(
                Doctor.doctor_id,
                Doctor.first_name,
                Doctor.last_name
            ).filter(Doctor.doctor_id == doctor_id)

        result = doctor_query.first()

        if not result:
            return {
                "name": "Unknown Doctor",
                "specialization": "Unknown",
                "email": "Unknown",
                "phone": "Unknown",
                "religion": "Unknown",
                "language": "Unknown",
                "address": "Unknown"
            }

        # Format name with title if available
        name = f"{result.title} " if result.title else ""
        name += f"{result.first_name or ''}"
        name += f" {result.middle_name}" if result.middle_name else ""
        name += f" {result.last_name or ''}"
        name = name.strip()
        if not name:
            name = "Doctor (No Name)"

        # Create the doctor info dictionary with safe attribute access
        doctor_info = {
            "name": name,
            "specialization": result.specialization or "Not specified",
            "email": result.email or "Not specified",
            "phone": result.phone or "Not specified",
            "religion": "Not available in database",  # Skip the problematic column
            "language": result.language or "Not specified",
            "address": result.address or "Not specified",
            "consultation_fee": "Not available in database",  # Skip the problematic column
            "treatment": result.treatment or "Not specified"
        }

        # Note: We've removed the fallback approach for religion to avoid unreachable code warnings

        return doctor_info
    except Exception as e:
        logger.error(f"Error fetching doctor info: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "name": "Unknown Doctor",
            "specialization": "Unknown",
            "email": "Unknown",
            "phone": "Unknown",
            "religion": "Unknown",
            "language": "Unknown",
            "address": "Unknown"
        }

def fetch_doctor_appointments(doctor_id: str, db: Session, days_ahead: int = 7) -> List[Dict[str, Any]]:
    """Fetch upcoming appointments for a doctor."""
    try:
        # Calculate date range
        start_date = datetime.now().date()
        end_date = start_date + timedelta(days=days_ahead)

        # Use a more selective query to avoid schema issues
        appointments = db.query(
            Appointment.appointment_id,
            Appointment.patient_id,
            Appointment.doctor_id,
            Appointment.appointment_date,
            Appointment.appointment_time,
            Appointment.visit_reason,
            Appointment.status,
            Appointment.notes
        ).filter(
            Appointment.doctor_id == doctor_id,
            Appointment.appointment_date >= start_date,
            Appointment.appointment_date <= end_date,
            Appointment.status == "Scheduled"
        ).order_by(Appointment.appointment_date, Appointment.appointment_time).all()

        # Format appointments
        result = []
        for appointment in appointments:
            try:
                # Get patient info
                patient = db.query(Patient).filter(Patient.patient_id == appointment.patient_id).first()

                # Format patient name with proper handling of None values
                if patient:
                    patient_name = f"{patient.first_name or ''} {patient.last_name or ''}".strip()
                    if not patient_name:
                        patient_name = "Patient (No Name)"
                else:
                    patient_name = "Unknown Patient"

                # Format appointment time with proper error handling
                try:
                    appointment_time = appointment.appointment_time.strftime("%H:%M") if appointment.appointment_time else "No time specified"
                except Exception as time_error:
                    logger.error(f"Error formatting appointment time: {str(time_error)}")
                    appointment_time = "Error formatting time"

                # Format appointment date with proper error handling
                try:
                    appointment_date = appointment.appointment_date.strftime("%Y-%m-%d") if appointment.appointment_date else "No date specified"
                except Exception as date_error:
                    logger.error(f"Error formatting appointment date: {str(date_error)}")
                    appointment_date = "Error formatting date"

                # Create appointment data with safe attribute access
                appointment_data = {
                    "appointment_id": str(appointment.appointment_id),
                    "patient_id": str(appointment.patient_id),
                    "patient_name": patient_name,
                    "date": appointment_date,
                    "time": appointment_time,
                    "reason": appointment.visit_reason or "No reason specified",
                    "notes": appointment.notes or "No notes"
                }

                # Safely add consultation_type if it exists
                try:
                    appointment_data["type"] = getattr(appointment, "consultation_type", None) or "No type specified"
                except Exception:
                    appointment_data["type"] = "No type specified"

                result.append(appointment_data)
            except Exception as appointment_error:
                logger.error(f"Error processing appointment: {str(appointment_error)}")
                # Continue to the next appointment

        return result
    except Exception as e:
        logger.error(f"Error fetching doctor appointments: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return []

def fetch_patient_info(patient_id: str, db: Session) -> Dict[str, Any]:
    """Fetch basic information about a patient."""
    try:
        # Use a more selective query to avoid schema issues
        patient_query = db.query(
            Patient.patient_id,
            Patient.first_name,
            Patient.last_name,
            Patient.dob,
            Patient.gender,
            Patient.health_score,
            Patient.under_medications,
            Patient.treatment,
            Patient.language,
            Patient.address
        ).filter(Patient.patient_id == patient_id)

        patient = patient_query.first()

        if not patient:
            return {
                "name": "Unknown Patient",
                "age": 0,
                "gender": "Unknown",
                "health_score": 0,
                "under_medications": False,
                "treatment": "Unknown",
                "religion": "Unknown",
                "language": "Unknown",
                "address": "Unknown"
            }

        # Calculate age from DOB with error handling
        try:
            today = datetime.now()
            dob = patient.dob
            if dob:
                age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
            else:
                age = "Unknown"
        except Exception as age_error:
            logger.error(f"Error calculating age: {str(age_error)}")
            age = "Error calculating age"

        # Format name with proper handling of None values
        first_name = patient.first_name or ""
        last_name = patient.last_name or ""
        name = f"{first_name} {last_name}".strip()
        if not name:
            name = "Patient (No Name)"

        # Handle treatment which might be a JSONB field
        treatment_info = "Unknown"
        if patient.treatment:
            try:
                if isinstance(patient.treatment, dict):
                    treatment_info = json.dumps(patient.treatment)
                elif isinstance(patient.treatment, str):
                    treatment_info = patient.treatment
                else:
                    treatment_info = str(patient.treatment)
            except Exception as treatment_error:
                logger.error(f"Error formatting treatment info: {str(treatment_error)}")
                treatment_info = "Error formatting treatment info"

        return {
            "name": name,
            "age": age,
            "gender": patient.gender or "Not specified",
            "health_score": patient.health_score or 0,
            "under_medications": patient.under_medications or False,
            "treatment": treatment_info,
            "religion": "Not available in database",  # Skip the problematic column
            "language": patient.language or "Not specified",
            "address": patient.address or "Not specified"
        }
    except Exception as e:
        logger.error(f"Error fetching patient info: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "name": "Unknown Patient",
            "age": 0,
            "gender": "Unknown",
            "health_score": 0,
            "under_medications": False,
            "treatment": "Unknown",
            "religion": "Unknown",
            "language": "Unknown",
            "address": "Unknown"
        }

def fetch_medical_history(patient_id: str, db: Session) -> List[Dict[str, Any]]:
    """Fetch medical history for a patient."""
    try:
        history_records = db.query(MedicalHistory).filter(
            MedicalHistory.patient_id == patient_id
        ).order_by(desc(MedicalHistory.diagnosed_date)).all()

        result = []
        for record in history_records:
            # Get doctor info
            doctor = db.query(Doctor).filter(Doctor.doctor_id == record.doctor_id).first()
            doctor_name = f"Dr. {doctor.first_name} {doctor.last_name}" if doctor else "Unknown Doctor"

            result.append({
                "diagnosis": record.diagnosis,
                "treatment": record.treatment,
                "diagnosed_date": record.diagnosed_date.strftime("%Y-%m-%d") if record.diagnosed_date else "Unknown",
                "doctor": doctor_name,
                "notes": record.additional_notes
            })

        return result
    except Exception as e:
        logger.error(f"Error fetching medical history: {str(e)}")
        return []

def fetch_prescriptions(patient_id: str, db: Session) -> List[Dict[str, Any]]:
    """Fetch prescriptions for a patient."""
    try:
        prescriptions = db.query(Prescription).filter(
            Prescription.patient_id == patient_id
        ).order_by(desc(Prescription.start_date)).all()

        result = []
        for prescription in prescriptions:
            # Get doctor info
            doctor = db.query(Doctor).filter(Doctor.doctor_id == prescription.doctor_id).first()
            doctor_name = f"Dr. {doctor.first_name} {doctor.last_name}" if doctor else "Unknown Doctor"

            result.append({
                "medication": prescription.medication_name,
                "dosage": prescription.dosage,
                "instructions": prescription.instructions,
                "start_date": prescription.start_date.strftime("%Y-%m-%d") if prescription.start_date else "Unknown",
                "end_date": prescription.end_date.strftime("%Y-%m-%d") if prescription.end_date else "Ongoing",
                "status": prescription.status,
                "doctor": doctor_name
            })

        return result
    except Exception as e:
        logger.error(f"Error fetching prescriptions: {str(e)}")
        return []

def fetch_emotion_analysis(patient_id: str, db: Session, days_back: int = 30) -> Dict[str, Any]:
    """Fetch emotion analysis summary for a patient."""
    try:
        # Calculate the date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        # Query emotion analysis records
        emotions = db.query(EmotionAnalysis).filter(
            EmotionAnalysis.patient_id == patient_id,
            EmotionAnalysis.analyzed_at >= start_date,
            EmotionAnalysis.analyzed_at <= end_date
        ).all()

        if not emotions:
            return {
                "dominant_emotion": "No data",
                "emotion_distribution": {},
                "days_analyzed": days_back,
                "total_messages_analyzed": 0
            }

        # Count emotions
        emotion_counts = {}
        for emotion in emotions:
            category = emotion.emotion_category
            emotion_counts[category] = emotion_counts.get(category, 0) + 1

        # Find dominant emotion
        dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0] if emotion_counts else "No data"

        # Calculate percentages
        total = sum(emotion_counts.values())
        emotion_distribution = {emotion: (count / total) * 100 for emotion, count in emotion_counts.items()}

        return {
            "dominant_emotion": dominant_emotion,
            "emotion_distribution": emotion_distribution,
            "days_analyzed": days_back,
            "total_messages_analyzed": total
        }
    except Exception as e:
        logger.error(f"Error fetching emotion analysis: {str(e)}")
        return {
            "dominant_emotion": "Error",
            "emotion_distribution": {},
            "days_analyzed": days_back,
            "total_messages_analyzed": 0
        }

def fetch_diary_entries(patient_id: str, db: Session, limit: int = 5) -> List[Dict[str, Any]]:
    """Fetch recent diary entries for a patient."""
    try:
        entries = db.query(DiaryEntry).filter(
            DiaryEntry.patient_id == patient_id
        ).order_by(desc(DiaryEntry.created_at)).limit(limit).all()

        result = []
        for entry in entries:
            result.append({
                "date": entry.created_at.strftime("%Y-%m-%d %H:%M") if entry.created_at else "Unknown",
                "notes": entry.notes
            })

        return result
    except Exception as e:
        logger.error(f"Error fetching diary entries: {str(e)}")
        return []

def fetch_upcoming_appointments(patient_id: str, db: Session) -> List[Dict[str, Any]]:
    """Fetch upcoming appointments for a patient."""
    try:
        # Get current date
        current_date = datetime.now().date()

        # Query appointments
        appointments = db.query(Appointment).filter(
            Appointment.patient_id == patient_id,
            Appointment.appointment_date >= current_date,
            Appointment.status == "Scheduled"
        ).order_by(Appointment.appointment_date, Appointment.appointment_time).all()

        result = []
        for appointment in appointments:
            # Get doctor info
            doctor = db.query(Doctor).filter(Doctor.doctor_id == appointment.doctor_id).first()
            doctor_name = f"Dr. {doctor.first_name} {doctor.last_name}" if doctor else "Unknown Doctor"

            result.append({
                "date": appointment.appointment_date.strftime("%Y-%m-%d"),
                "time": appointment.appointment_time.strftime("%H:%M"),
                "doctor": doctor_name,
                "reason": appointment.visit_reason,
                "type": appointment.consultation_type
            })

        return result
    except Exception as e:
        logger.error(f"Error fetching upcoming appointments: {str(e)}")
        return []

# Define constants for chat participants
AI_DOCTOR_ASSISTANT_ID = "ai-doctor-assistant"

def fetch_chat_history(doctor_id: str, db: Session) -> List[Dict[str, Any]]:
    """Fetch chat history for a doctor with the AI assistant.

    Args:
        doctor_id: The doctor's ID
        db: Database session

    Returns:
        List of chat messages
    """
    try:
        # Query messages between the doctor and the AI assistant
        messages = db.query(ChatMessage).filter(
            ((ChatMessage.sender_id == doctor_id) & (ChatMessage.receiver_id == AI_DOCTOR_ASSISTANT_ID)) |
            ((ChatMessage.sender_id == AI_DOCTOR_ASSISTANT_ID) & (ChatMessage.receiver_id == doctor_id))
        ).order_by(ChatMessage.timestamp.desc()).limit(10).all()

        chat_history = []
        for msg in messages:
            role = "doctor" if msg.sender_id == doctor_id else "assistant"
            chat_history.append({
                "role": role,
                "message": msg.message_text,
                "timestamp": msg.timestamp.strftime("%Y-%m-%d %H:%M:%S") if msg.timestamp else "Unknown"
            })

        # Reverse to get chronological order
        chat_history.reverse()
        logger.info(f"Retrieved {len(chat_history)} chat messages for doctor {doctor_id}")
        return chat_history
    except Exception as e:
        logger.error(f"Error fetching chat history: {str(e)}")
        return []

def extract_patient_id(query: str) -> Optional[str]:
    """
    Extract patient ID from a query string.

    This is a simple implementation that looks for patterns like "patient id: XXX" or "patient: XXX".
    In a production environment, you might want to use a more sophisticated approach like NER.
    """
    if not OPENAI_CLIENT:
        # Simple regex-based approach as fallback
        import re
        patterns = [
            r"patient\s+id[:\s]+([a-f0-9\-]{36})",
            r"patient[:\s]+([a-f0-9\-]{36})"
        ]

        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1)

        return None

    # Use OpenAI to extract the patient ID
    try:
        prompt = f"""
        Extract the patient ID from the following query. Patient IDs are UUID strings (36 characters with hyphens).
        If no patient ID is found, respond with "None".

        Query: {query}

        Patient ID:
        """

        response = OPENAI_CLIENT.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts patient IDs from text."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=0.1
        )

        extracted_id = response.choices[0].message.content.strip()

        # Check if the extracted ID looks like a UUID
        import re
        if re.match(r"^[a-f0-9\-]{36}$", extracted_id, re.IGNORECASE):
            return extracted_id

        return None
    except Exception as e:
        logger.error(f"Error extracting patient ID: {str(e)}")
        return None

def generate_patient_summary(patient_id: str, db: Session = None) -> str:
    """Generate a comprehensive summary of a patient for the doctor."""
    if not OPENAI_CLIENT:
        return "OpenAI client not initialized. Cannot generate patient summary."

    try:
        # Initialize data containers
        patient_info = {}
        medical_history = []
        prescriptions = []
        emotion_analysis = {}
        upcoming_appointments = []
        diary_entries = []

        # Fetch patient info with a new session
        try:
            new_db = SessionLocal()
            try:
                patient_info = fetch_patient_info(patient_id, new_db)
            except Exception as e:
                logger.error(f"Error fetching patient info: {str(e)}")
                patient_info = {"name": "Unknown Patient", "error": "Failed to fetch patient information"}
            finally:
                new_db.close()
        except Exception as db_error:
            logger.error(f"Error creating new database session for patient info: {str(db_error)}")

        # Fetch medical history with a new session
        try:
            new_db = SessionLocal()
            try:
                medical_history = fetch_medical_history(patient_id, new_db)
            except Exception as e:
                logger.error(f"Error fetching medical history: {str(e)}")
            finally:
                new_db.close()
        except Exception as db_error:
            logger.error(f"Error creating new database session for medical history: {str(db_error)}")

        # Fetch prescriptions with a new session
        try:
            new_db = SessionLocal()
            try:
                prescriptions = fetch_prescriptions(patient_id, new_db)
            except Exception as e:
                logger.error(f"Error fetching prescriptions: {str(e)}")
            finally:
                new_db.close()
        except Exception as db_error:
            logger.error(f"Error creating new database session for prescriptions: {str(db_error)}")

        # Fetch emotion analysis with a new session
        try:
            new_db = SessionLocal()
            try:
                emotion_analysis = fetch_emotion_analysis(patient_id, new_db)
            except Exception as e:
                logger.error(f"Error fetching emotion analysis: {str(e)}")
                emotion_analysis = {"dominant_emotion": "Unknown", "emotion_distribution": {}}
            finally:
                new_db.close()
        except Exception as db_error:
            logger.error(f"Error creating new database session for emotion analysis: {str(db_error)}")

        # Fetch upcoming appointments with a new session
        try:
            new_db = SessionLocal()
            try:
                upcoming_appointments = fetch_upcoming_appointments(patient_id, new_db)
            except Exception as e:
                logger.error(f"Error fetching upcoming appointments: {str(e)}")
            finally:
                new_db.close()
        except Exception as db_error:
            logger.error(f"Error creating new database session for upcoming appointments: {str(db_error)}")

        # Fetch diary entries with a new session
        try:
            new_db = SessionLocal()
            try:
                diary_entries = fetch_diary_entries(patient_id, new_db)
            except Exception as e:
                logger.error(f"Error fetching diary entries: {str(e)}")
            finally:
                new_db.close()
        except Exception as db_error:
            logger.error(f"Error creating new database session for diary entries: {str(db_error)}")

        # Create a comprehensive context for OpenAI
        context = f"""
        ## Patient Information
        Name: {patient_info['name']}
        Age: {patient_info['age']}
        Gender: {patient_info['gender']}
        Health Score: {patient_info['health_score']}
        Under Medications: {'Yes' if patient_info['under_medications'] else 'No'}
        Treatment: {patient_info['treatment']}

        ## Medical History
        {json.dumps(medical_history, indent=2) if medical_history else "No medical history available."}

        ## Current Prescriptions
        {json.dumps(prescriptions, indent=2) if prescriptions else "No active prescriptions."}

        ## Emotional State (Last 30 Days)
        Dominant Emotion: {emotion_analysis['dominant_emotion']}
        Emotion Distribution: {json.dumps(emotion_analysis['emotion_distribution'], indent=2)}
        Messages Analyzed: {emotion_analysis['total_messages_analyzed']}

        ## Upcoming Appointments
        {json.dumps(upcoming_appointments, indent=2) if upcoming_appointments else "No upcoming appointments."}

        ## Recent Diary Entries
        {json.dumps(diary_entries, indent=2) if diary_entries else "No recent diary entries."}
        """

        # Generate the summary using OpenAI
        prompt = f"""
        Create a concise medical summary for a doctor about the following patient.
        Focus on the most important medical information, current treatment, emotional state, and upcoming appointments.
        Keep the summary professional, factual, and to the point.

        {context}
        """

        response = OPENAI_CLIENT.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.3
        )

        summary = response.choices[0].message.content.strip()
        return summary

    except Exception as e:
        logger.error(f"Error generating patient summary: {str(e)}")
        return f"Error generating patient summary: {str(e)}"

def retrieve_relevant_information(query: str) -> List[str]:
    """Retrieve relevant information from both doctor and PDF vector stores."""
    results = []

    # Retrieve from doctor knowledge base
    if doctor_retriever:
        try:
            docs = doctor_retriever.get_relevant_documents(query)
            results.extend([f"[Medical Knowledge] {doc.page_content}" for doc in docs])
        except Exception as e:
            logger.error(f"Error retrieving from doctor knowledge base: {str(e)}")

    # Retrieve from PDF knowledge base
    if pdf_retriever:
        try:
            docs = pdf_retriever.get_relevant_documents(query)
            results.extend([f"[PDF Document] {doc.page_content}" for doc in docs])
        except Exception as e:
            logger.error(f"Error retrieving from PDF knowledge base: {str(e)}")

    return results

def process_doctor_query(query: str, doctor_id: str, db: Session = None, chat_history=None, patient_id=None, relevant_info=None):
    """Process a query from a doctor and generate a response."""
    if not OPENAI_CLIENT:
        return "OpenAI client not initialized. Cannot process query."

    try:
        # Initialize context
        context_parts = []

        # Add doctor information - use a new session to avoid transaction issues
        try:
            new_db = SessionLocal()
            try:
                doctor_info = fetch_doctor_info(doctor_id, new_db)
                context_parts.append(f"## Doctor Information\nName: {doctor_info['name']}\nSpecialization: {doctor_info['specialization']}")
            except Exception as doctor_error:
                logger.error(f"Error fetching doctor info: {str(doctor_error)}")
                context_parts.append("## Doctor Information\nUnable to fetch doctor information due to a database error.")
            finally:
                new_db.close()
        except Exception as db_error:
            logger.error(f"Error creating new database session for doctor info: {str(db_error)}")
            context_parts.append("## Doctor Information\nUnable to fetch doctor information due to a database error.")

        # If no patient ID is provided, try to extract it from the query
        if not patient_id:
            patient_id = extract_patient_id(query)

        # If patient ID is found, generate a patient summary - use a new session
        if patient_id:
            try:
                new_db = SessionLocal()
                try:
                    patient_summary = generate_patient_summary(patient_id, new_db)
                    context_parts.append(f"## Patient Summary\n{patient_summary}")
                except Exception as patient_error:
                    logger.error(f"Error generating patient summary: {str(patient_error)}")
                    context_parts.append("## Patient Summary\nUnable to fetch patient information due to a database error.")
                finally:
                    new_db.close()
            except Exception as db_error:
                logger.error(f"Error creating new database session for patient summary: {str(db_error)}")
                context_parts.append("## Patient Summary\nUnable to fetch patient information due to a database error.")

        # Fetch upcoming appointments for the doctor - use a new session
        try:
            new_db = SessionLocal()
            try:
                appointments = fetch_doctor_appointments(doctor_id, new_db)
                if appointments:
                    context_parts.append("## Upcoming Appointments\n" + json.dumps(appointments, indent=2))
            except Exception as appointment_error:
                logger.error(f"Error fetching doctor appointments: {str(appointment_error)}")
                context_parts.append("## Upcoming Appointments\nUnable to fetch appointments due to a database error.")
            finally:
                new_db.close()
        except Exception as db_error:
            logger.error(f"Error creating new database session for appointments: {str(db_error)}")
            context_parts.append("## Upcoming Appointments\nUnable to fetch appointments due to a database error.")

        # Use pre-retrieved information if provided, otherwise retrieve it now
        if relevant_info is None:
            relevant_info = retrieve_relevant_information(query)

        if relevant_info:
            context_parts.append("## Relevant Information")
            for i, info in enumerate(relevant_info, 1):
                context_parts.append(f"{i}. {info}")

        # Add chat history for context if available
        if chat_history and len(chat_history) > 0:
            context_parts.append("## Recent Conversation")
            # Include last 5 messages for context
            for msg in chat_history[-5:]:
                role = msg.get("role", "unknown")
                message = msg.get("message", "")
                context_parts.append(f"{role}: {message}")

        # Combine all context parts
        context = "\n\n".join(context_parts)

        # Generate response using OpenAI
        prompt = f"""
        Doctor's Query: {query}

        {context}

        Please provide a helpful, professional response to the doctor's query.
        If the query is about a specific patient and patient information is available, include relevant details.
        If the query is about appointments, provide appointment information.
        If the query is a general medical question, provide relevant medical information.
        """

        response = OPENAI_CLIENT.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.3
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        logger.error(f"Error processing doctor query: {str(e)}")
        return f"I apologize, but I encountered an error while processing your query: {str(e)}"

def generate_audio_from_text(text: str, voice: str = "alloy"):
    """Generate audio from text using OpenAI's text-to-speech API."""
    if not OPENAI_CLIENT:
        logger.warning("OpenAI client not initialized. Cannot generate audio.")
        return None

    # Validate voice parameter
    valid_voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
    if voice not in valid_voices:
        logger.warning(f"Invalid voice: {voice}. Using default voice 'alloy'.")
        voice = "alloy"

    try:
        logger.info(f"Generating audio with voice: {voice}")
        response = OPENAI_CLIENT.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text
        )

        # Get audio data as bytes
        audio_data = response.content

        # Convert to base64 for sending over WebSocket
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')

        return audio_base64
    except Exception as e:
        logger.error(f"Error generating audio: {str(e)}")
        return None

@ai_audio_smart_chat_doc_router.websocket("/audio-smart-chat-doc/{doctor_id}")
async def audio_smart_chat_for_doctors(websocket: WebSocket, doctor_id: str, db: Session = Depends(get_db)):
    """
    WebSocket endpoint for smart chat with doctors with audio support.

    Args:
        websocket: WebSocket connection
        doctor_id: The doctor's ID
        db: Database session
    """
    logger.info(f"=== NEW DOCTOR WEBSOCKET CONNECTION REQUEST ===")
    logger.info(f"Doctor ID: {doctor_id}")
    logger.info(f"Query parameters: {websocket.query_params}")
    logger.info(f"Headers: {websocket.headers}")

    # Accept the connection first
    await websocket.accept()
    logger.info(f"WebSocket connection accepted for doctor: {doctor_id}")

    # Get JWT token from query parameters
    token = None
    if "token" in websocket.query_params:
        token = websocket.query_params["token"]
        logger.info(f"Found token in query parameters")

    # Check if we have a token
    if not token:
        logger.info(f"No token provided for doctor: {doctor_id}")
        await websocket.send_text(json.dumps({
            "response": "Error: Authentication required. Please provide a valid JWT token as a query parameter (token=...).",
            "error": "authentication_required"
        }))
        await websocket.close()
        return

    # Get static JWT token from environment
    STATIC_JWT_TOKEN = os.getenv("STATIC_JWT_TOKEN")

    # Validate token (simple check for static token)
    if token != STATIC_JWT_TOKEN:
        logger.info(f"Invalid token provided for doctor: {doctor_id}")
        await websocket.send_text(json.dumps({
            "response": "Error: Invalid authentication token.",
            "error": "invalid_token"
        }))
        await websocket.close()
        return

    logger.info(f"Token validated for doctor: {doctor_id}")

    # Load chat history from database
    chat_history = fetch_chat_history(doctor_id, db)
    logger.info(f"Loaded {len(chat_history)} messages from chat history for doctor {doctor_id}")

    # Send welcome message
    welcome_message = f"Welcome, Doctor. I'm your AI assistant. I can help you with patient information, appointments, and medical questions. How can I assist you today?"

    # Create a unique response ID
    response_id = str(uuid.uuid4())

    # Save welcome message to database
    try:
        # Create a new session to avoid transaction issues
        new_db = SessionLocal()
        try:
            new_db.add(ChatMessage(
                chat_message_id=str(uuid.uuid4()),
                sender_id=AI_DOCTOR_ASSISTANT_ID,
                receiver_id=doctor_id,
                message_text=welcome_message,
                timestamp=datetime.now()
            ))
            new_db.commit()
            logger.info(f"✅ Saved welcome message to database for doctor {doctor_id}")
        except Exception as inner_error:
            new_db.rollback()  # Roll back the transaction on error
            logger.error(f"❌ Error saving welcome message to database: {str(inner_error)}")
        finally:
            new_db.close()
    except Exception as db_error:
        logger.error(f"❌ Error creating new database session: {str(db_error)}")

    # Add welcome message to chat history if it's not already in the database
    if not chat_history or chat_history[-1].get("message") != welcome_message:
        chat_history.append({
            "role": "assistant",
            "message": welcome_message,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    # Send welcome message to client
    await websocket.send_text(json.dumps({
        "response": welcome_message,
        "response_id": response_id
    }))

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            data = json.loads(data)
            logger.info(f"Received message from doctor {doctor_id}: {data}")

            # Extract user input from text or audio
            user_input = ""

            # Handle audio input
            if "audio" in data:
                try:
                    if not OPENAI_CLIENT:
                        raise Exception("OpenAI client not initialized")

                    audio_data = base64.b64decode(data["audio"])
                    with open("temp_doc.mp3", "wb") as f:
                        f.write(audio_data)

                    with open("temp_doc.mp3", "rb") as f:
                        transcript = OPENAI_CLIENT.audio.transcriptions.create(
                            model="whisper-1",
                            file=f,
                            language="en"
                        )

                    user_input = transcript.text.strip()

                    # Send the transcription back to the client
                    await websocket.send_text(json.dumps({
                        "transcription": user_input
                    }))

                    logger.info(f"Transcribed audio: {user_input}")

                    # Wait a moment to ensure the transcription is displayed before processing
                    await asyncio.sleep(0.5)
                except Exception as e:
                    logger.error(f"Error processing audio: {str(e)}")
                    # Send error to client
                    await websocket.send_text(json.dumps({
                        "error": "Audio processing failed",
                        "response": f"Error processing audio: {str(e)}"
                    }))
                    continue
            else:
                user_input = data.get("text", "").strip()

            if not user_input:
                await websocket.send_text(json.dumps({
                    "error": "No input provided",
                    "response": "Please provide either text or audio input."
                }))
                continue

            logger.info(f"Processing input from doctor {doctor_id}: {user_input}")

            # Process context and generate response in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                # Task 1: Extract patient ID if mentioned
                def extract_patient():
                    return extract_patient_id(user_input)

                # Task 2: Retrieve relevant information
                def retrieve_info():
                    return retrieve_relevant_information(user_input)

                # Execute both tasks in parallel
                patient_id_future = executor.submit(extract_patient)
                relevant_info_future = executor.submit(retrieve_info)

                # Wait for both tasks to complete
                patient_id = patient_id_future.result()
                # Store relevant info for use in the process_doctor_query function
                relevant_info = relevant_info_future.result()

            # Save user message to database with transaction handling
            try:
                # Create a new session to avoid transaction issues
                new_db = SessionLocal()
                try:
                    new_db.add(ChatMessage(
                        chat_message_id=str(uuid.uuid4()),
                        sender_id=doctor_id,
                        receiver_id=AI_DOCTOR_ASSISTANT_ID,
                        message_text=user_input,
                        timestamp=datetime.now()
                    ))
                    new_db.commit()
                    logger.info(f"✅ Saved user message to database for doctor {doctor_id}")
                except Exception as inner_error:
                    new_db.rollback()  # Roll back the transaction on error
                    logger.error(f"❌ Error saving user message to database: {str(inner_error)}")
                finally:
                    new_db.close()
            except Exception as db_error:
                logger.error(f"❌ Error creating new database session: {str(db_error)}")
                # Continue anyway - we can still try to generate a response

            # Add user message to chat history
            chat_history.append({"role": "doctor", "message": user_input, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})

            # Process the doctor query with all the context
            response = process_doctor_query(
                query=user_input,
                doctor_id=doctor_id,
                db=db,
                chat_history=chat_history,
                patient_id=patient_id,
                relevant_info=relevant_info
            )

            # Get the voice preference from the data
            voice = data.get("voice", "alloy")
            logger.info(f"Using voice: {voice} for response")

            # Generate audio from the response text
            audio_base64 = generate_audio_from_text(response, voice)

            # Save AI response to database with transaction handling
            try:
                # Create a new session to avoid transaction issues
                new_db = SessionLocal()
                try:
                    # Extract keywords from the response for future search
                    keywords = []
                    if OPENAI_CLIENT:
                        try:
                            keyword_prompt = f"Extract 3-5 key medical terms or concepts from this text as a comma-separated list: {response}"
                            keyword_response = OPENAI_CLIENT.chat.completions.create(
                                model="gpt-3.5-turbo",
                                messages=[
                                    {"role": "system", "content": "You extract key medical terms from text."},
                                    {"role": "user", "content": keyword_prompt}
                                ],
                                max_tokens=50,
                                temperature=0.1
                            )
                            keywords = keyword_response.choices[0].message.content.strip().split(',')
                            keywords = [k.strip() for k in keywords]
                        except Exception as kw_error:
                            logger.error(f"Error extracting keywords: {str(kw_error)}")

                    new_db.add(ChatMessage(
                        chat_message_id=str(uuid.uuid4()),
                        sender_id=AI_DOCTOR_ASSISTANT_ID,
                        receiver_id=doctor_id,
                        message_text=response,
                        extracted_keywords=json.dumps(keywords) if keywords else None,
                        timestamp=datetime.now()
                    ))
                    new_db.commit()
                    logger.info(f"✅ Saved AI response to database for doctor {doctor_id}")
                except Exception as inner_error:
                    new_db.rollback()  # Roll back the transaction on error
                    logger.error(f"❌ Error saving AI response to database: {str(inner_error)}")
                finally:
                    new_db.close()
            except Exception as db_error:
                logger.error(f"❌ Error creating new database session: {str(db_error)}")
                # Continue anyway - we can still send the response to the client

            # Add AI response to chat history
            chat_history.append({"role": "assistant", "message": response, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})

            # Keep chat history limited to last 10 exchanges (20 messages)
            if len(chat_history) > 20:
                chat_history = chat_history[-20:]

            # Create a unique response ID to help client detect duplicates
            response_id = str(uuid.uuid4())

            # Send response to client with audio
            response_data = {
                "response": response,
                "response_id": response_id
            }

            # Add audio if available
            if audio_base64:
                response_data["audio"] = audio_base64

            # Add patient ID if found
            if patient_id:
                response_data["patient_id"] = patient_id

            # Send the response
            await websocket.send_text(json.dumps(response_data))

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for doctor: {doctor_id}")
    except Exception as e:
        logger.error(f"Error in WebSocket connection: {str(e)}")
        try:
            await websocket.send_text(json.dumps({
                "error": "Server error",
                "response": f"An unexpected error occurred: {str(e)}"
            }))
        except:
            pass  # If we can't send the error, just log it
