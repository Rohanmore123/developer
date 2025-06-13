import os
import uuid
import json
import base64
import asyncio
import tempfile
import logging
import concurrent.futures
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dotenv import load_dotenv

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from sqlalchemy.orm import Session
from sqlalchemy import desc, text

from database.database import SessionLocal, get_db
from model.model_correct import (
    ChatMessage, Patient, Doctor, Appointment, OnboardingQuestion, DiaryEntry,
    EmotionAnalysis, MedicalHistory, Prescription
)

from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create router
chat_final_router = APIRouter()

# Initialize OpenAI client
try:
    # Force reload the API key from environment
    load_dotenv(override=True)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    if OPENAI_API_KEY:
        logger.info(f"Initializing OpenAI client with API key: {OPENAI_API_KEY[:10]}...{OPENAI_API_KEY[-5:]}")
        OPENAI_CLIENT = OpenAI(api_key=OPENAI_API_KEY)
    else:
        logger.warning("WARNING: OpenAI API key not found in environment variables.")
        OPENAI_CLIENT = None

    if not OPENAI_CLIENT:
        logger.warning("WARNING: OpenAI client could not be initialized. Check your API key.")
except ImportError:
    logger.warning("WARNING: OpenAI package not installed. Some features may not work.")
    OPENAI_CLIENT = None

# AI Doctor ID (fixed ID for the AI doctor)
AI_DOCTOR_ID = "00000000-0000-0000-0000-000000000000"

# S3 paths for different FAISS indexes - confirmed with S3 bucket listing
S3_PATHS = {
    "general": "faiss_index/general_index",
    "psychologist": "faiss_index/general_index",  # Using general index for psychologist
    "dietician": "faiss_index/dietician_index"
    # Removed PDF path to avoid unnecessary downloads
}

# Dictionary to cache vector stores
_vector_stores = {
    "general": None,
    "psychologist": None,
    "dietician": None
    # Removed PDF vector store to avoid unnecessary downloads
}

# S3 Configuration
USE_S3 = os.getenv("USE_S3", "true").lower() == "true"  # Default to true
S3_BUCKET = os.getenv("PDF_BUCKET_NAME", "prasha-health-pdf")  # Use the correct bucket name
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")  # Use the correct environment variable name
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")  # Use the correct environment variable name
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# Initialize S3 client if enabled
s3_client = None
if USE_S3:
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
            else:
                logger.error(f"❌ Target bucket '{S3_BUCKET}' not found in available buckets: {buckets}")
        except Exception as test_error:
            logger.error(f"❌ Error testing S3 connection: {str(test_error)}")
            USE_S3 = False
    except (ImportError, NoCredentialsError, ClientError) as e:
        logger.error(f"❌ Error initializing S3 client: {str(e)}")
        USE_S3 = False

# Persona system prompts
GENERAL_OPD_PROMPT = """
You are an experienced General mental health OPD (Outpatient Department) doctor Named Tova with 15 years of experience.
you are working for a PrashaSync, which is an AI-driven platform that revolutionizes how people access and receive mental healthcare through personalized therapeutic experiences.
Your role is to conduct a thorough assessment of the patient, understand their symptoms,
medical history, and current concerns. Be warm, empathetic, and professional.

IMPORTANT: You have been provided with:
1. The patient's medical records and history from the database
2. The patient's onboarding health information and questionnaire responses
3. Relevant medical information retrieved from our knowledge base

Review this information carefully to understand the patient's background and health concerns.

Your goals are to:
1. Have a meaningful conversation with the patient to understand their health concerns
2. Gather comprehensive information about their symptoms, duration, severity, and medical history
3. Ask clarifying questions to better understand their condition
4. Provide general guidance, reassurance, and initial recommendations
5. Do not refer specialist immediately after starting the new conversation first have some dialogue exchange for better understanding about patients mental helath and symptoms
5. Determine if a specialist referral is needed based on the conversation


IMPORTANT GUIDELINES FOR SPECIALIST REFERRAL and strictly follow them:
- DO NOT suggest a specialist immediately. Take time to understand the patient's concerns first.
- Only suggest a specialist when:
  a) The patient explicitly asks to speak with a specialist, OR
  b) You've had at least 6-7 exchanges and have gathered sufficient information to determine that specialized care is needed
- When suggesting a specialist, choose the most appropriate one:
  - Suggest "psychologist" for mental health, emotional, psychological, or behavioral issues
  - Suggest "dietician" for nutrition, weight management, diet, or food-related concerns
- Create response in plain text dont use numbers or bullets or any other formatting
- keep responses short and  dont create longer responses, keep it like a real conversation

The system will automatically transfer the patient to the suggested specialist when you set the "suggested_specialist" field.

❗Reply ONLY in this JSON structure:
{
  "response": "Your empathetic and helpful response to the patient",
  "extracted_keywords": ["keyword1", "keyword2"], #based on user input keywords related to mood or mental health
  "suggested_specialist": "psychologist|dietician|none"
}

Only output the JSON (no explanation, no markdown).

The "suggested_specialist" field should be "none" until you've had sufficient conversation with the patient and determined that specialized care is needed or the patient has requested it.
"""

PSYCHOLOGIST_PROMPT = """
You are a psychologist named 'Ori' with 15 years of experience,
you are working for a PrashaSync, which is an AI-driven platform that revolutionizes how people access and receive mental healthcare through personalized therapeutic experiences.
You are having a warm, casual, and supportive conversation with a patient. Your primary goal is to connect empathetically and help them feel heard and understood,
just like you would in a relaxed chat in your office.

IMPORTANT: You have been provided with:
1. The patient's medical records and history from the database
2. The patient's onboarding health information and questionnaire responses
3. The patient's previous conversation with the General OPD doctor
4. Relevant medical information retrieved from our knowledge base

Review this information carefully to understand the patient's background, concerns, and why they were
referred to you. Pay special attention to any mental health history, medications, previous diagnoses,
and the concerns they expressed to the General OPD doctor.

Focus on mental health concerns, emotional well-being, and psychological symptoms. Provide thoughtful,
evidence-based guidance while maintaining a compassionate approach.

REFERRAL BACK TO GENERAL OPD:
If at any point you determine that:
- The patient's concerns are primarily non-psychiatric in nature
- They would be better served by the General OPD doctor
- They ask about issues outside your specialty
- They explicitly request to speak with the General OPD doctor
# TIP's for perfect response
- keep responses short and to the point dont create longer responses, keep like a real conversation
- If it is about therapy or treatment then suggest user a treatment plan go step by step, try to understand the concern and guide or suggest therapy to them
- or when you expect your response to be long, break the answer into a clear step-by-step plan. After presenting each step, ask the user: "Can I continue with the plan?" Wait for their confirmation before sharing the next step. This approach helps the user understand and follow the plan more easily without having to remember everything at once. Keep your responses simple and plain, and do not use any formatting
- If you think that response will be longer then create multiple response such that user can remember and follow it easily also ask then can i continue with the plan.
- never say I can't provide you therapy and look for a therapist, keep confidence in you , you are a great psychologist-
- Do not hallucinate look for chat history, or context related to chat and try to answer in a confident manner.

You can refer them back by including "refer_to_general": true in your response.

❗Reply ONLY in this JSON structure:
{
  "response": "Your empathetic and helpful response to the patient",
  "extracted_keywords": ["keyword1", "keyword2"], #based on user input keywords related to mood or mental health
  "refer_to_general": false,
  "refer_to_specialist": null
}

Only output the JSON (no explanation, no markdown).

Set "refer_to_general" to true only when you believe the patient should be transferred back to the General OPD doctor and transfer only if you think it needs to be transfered otherwise avoid transferring.
Set "refer_to_specialist" to "dietician" when the patient asks about nutrition, diet, food, eating habits, or weight management that would be better handled by a dietician, mostly if user asks about diet plan or nutrition, you can handle simple questions related to diet which does not require dieticien.
"""

# DIETICIAN_PROMPT = """
# You are a dietician with 15 years of experience specializing in nutritional counseling and dietary management.
# You provide expert advice on nutrition, diet plans, and healthy eating habits.

# IMPORTANT: You have been provided with:
# 1. The patient's medical records and history from the database
# 2. The patient's previous conversation with the General OPD doctor
# 3. Relevant medical information retrieved from our knowledge base

# Review this information carefully to understand the patient's background, concerns, and why they were
# referred to you. Pay special attention to their medical history, current medications, previous diagnoses,
# test results, and the physical symptoms they described to the General OPD doctor.

# Focus on physical symptoms, medical conditions, and evidence-based treatment options. Provide thorough
# assessments and clear explanations while maintaining a professional and compassionate approach.

# REFERRAL BACK TO GENERAL OPD:
# If at any point you determine that:
# - The patient's concerns are primarily not related to physical health conditions
# - They would be better served by the General OPD doctor for coordination of care
# - They ask about issues outside your specialty
# - They explicitly request to speak with the General OPD doctor
# - Create response in plain text dont use numbers or bullets or any other formatting, Use formatting only when user ask about treatment plan or step or something similar
# - Create response in plain text dont use numbers or bullets or any other formatting and dont create longer responses such that user feel too much to remember and have to waitfor longer time.

# You can refer them back by including "refer_to_general": true in your response.

# Always respond in a JSON format with the following structure:
# {
#   "response": "Your professional and helpful response to the patient",
#   "extracted_keywords": ["keyword1", "keyword2"],
#   "refer_to_general": false
# }

# Set "refer_to_general" to true only when you believe the patient should be transferred back to the General OPD doctor.
# """

DIETICIAN_PROMPT = """
You are a dietician named Maya with 15 years of experience specializing in nutritional counseling and dietary management.
you are working for a PrashaSync, which is an AI-driven platform that revolutionizes how people access and receive mental healthcare through personalized therapeutic experiences.
You provide expert advice on nutrition, diet plans, and healthy eating habits.

IMPORTANT: You have been provided with:
1. The patient's medical records and history from the database
2. The patient's onboarding health information and questionnaire responses
3. The patient's previous conversation with the General OPD doctor
4. Relevant nutritional and medical information retrieved from our knowledge base

Review this information carefully to understand the patient's background, concerns, and why they were
referred to you. Pay special attention to their weight history, dietary habits, food allergies or intolerances,
medical conditions that affect nutrition (like diabetes or hypertension), and any nutritional concerns
they expressed to the General OPD doctor.

Focus on personalized nutritional needs, evidence-based dietary recommendations, and practical meal planning
advice. Consider the patient's lifestyle, preferences, and medical conditions when providing guidance.
Maintain a supportive and encouraging approach.

REFERRAL BACK TO GENERAL OPD:
If at any point you determine that:
- The patient's concerns are primarily not related to nutrition or diet
- They would be better served by the General OPD doctor
- They have medical concerns that should be addressed first
- They ask about issues outside your specialty
- They explicitly request to speak with the General OPD doctor

Tips:-
(keep responses short and to the point dont create longer responses, keep like a real conversation
When the user asks for a diet plan, generate a concise response. If the response is likely to be long, divide it into smaller parts.
Begin by sharing only the first portion of the plan. At the end, ask the user: "Would you like me to continue?" Only proceed if the user agrees.
Keep all responses plain, without any formatting such as bold text, bullet points, or italics.)

You can refer them back by including "refer_to_general": true in your response.

❗Reply ONLY in this JSON structure:
{
  "response": "Your professional and helpful response to the patient",
  "extracted_keywords": ["keyword1", "keyword2"], #based on user input keywords related to nutrition or diet
  "refer_to_general": false,
  "refer_to_specialist": null
}

Only output the JSON (no explanation, no markdown).

Set "refer_to_general" to true only when you believe the patient should be transferred back to the General OPD doctor and transfer only if you think it needs to be transfered otherwise avoid transferring.
Set "refer_to_specialist" to "psychologist" when the patient asks about mental health, anxiety, depression, stress, or psychological concerns that would be better handled by a psychologist.
"""

# Function to load FAISS index directly from S3
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

    # Check if we already have this vector store in the cache
    if _vector_stores[index_name] is not None:
        logger.info(f"Using cached vector store for {index_name}")
        return _vector_stores[index_name]

    # Verify S3 is enabled and configured
    if not USE_S3 or not s3_client:
        logger.error("S3 is not enabled or configured. Cannot load vector store.")
        return None

    try:
        # Use project-specific temporary files that are properly cleaned up
        import os
        import shutil

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
            logger.info(f"Found keys: {keys}")

            # Look for index.faiss and index.pkl files
            faiss_key = None
            pkl_key = None

            # Try different possible paths based on the S3 structure
            possible_paths = [
                # Direct path
                (f"{s3_prefix}/index.faiss", f"{s3_prefix}/index.pkl"),
                # Nested in index directory
                (f"{s3_prefix}/index/index.faiss", f"{s3_prefix}/index/index.pkl")
            ]

            # Check if any of the possible paths exist in the listing
            for faiss_path, pkl_path in possible_paths:
                if faiss_path in keys and pkl_path in keys:
                    faiss_key = faiss_path
                    pkl_key = pkl_path
                    logger.info(f"Found matching index files: {faiss_key} and {pkl_key}")
                    break

            # If not found with exact paths, try with endswith
            if not faiss_key or not pkl_key:
                for key in keys:
                    if key.endswith('index.faiss'):
                        faiss_key = key
                    elif key.endswith('index.pkl'):
                        pkl_key = key

                if faiss_key and pkl_key:
                    logger.info(f"Found index files by suffix: {faiss_key} and {pkl_key}")

            # If we still haven't found the files, return None
            if not faiss_key or not pkl_key:
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

            # Initialize the embeddings
            embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

            # Load the vector store from the temporary directory
            # When loading an existing index, we don't need to specify index_kwargs
            # The index parameters are already stored in the index file
            logger.info(f"Loading FAISS index from temporary directory")
            vector_store = FAISS.load_local(
                folder_path=temp_dir,
                embeddings=embeddings,
                index_name="index",
                allow_dangerous_deserialization=True
            )

            logger.info(f"✅ Successfully loaded FAISS index from S3: {s3_prefix}")

            # Cache the vector store
            _vector_stores[index_name] = vector_store

            return vector_store
        finally:
            # Clean up the temporary directory
            if 'temp_dir' in locals() and os.path.exists(temp_dir):
                logger.info(f"Cleaning up temporary directory: {temp_dir}")
                shutil.rmtree(temp_dir, ignore_errors=True)

    except Exception as e:
        logger.error(f"Error initializing vector store: {str(e)}")
        return None

# No fallback mechanism - we require the actual FAISS index from S3

# Use pre-loaded vector stores from the initializer
import vector_store_initializer

# Get vector stores from the initializer
logger.info("Using pre-loaded vector stores from initializer...")
general_vector_store = vector_store_initializer.general_vector_store
psychologist_vector_store = vector_store_initializer.psychologist_vector_store
dietician_vector_store = vector_store_initializer.dietician_vector_store
# Removed pdf_vector_store to avoid unnecessary downloads

# Get retrievers from the initializer
general_retriever = vector_store_initializer.general_retriever
psychologist_retriever = vector_store_initializer.psychologist_retriever
dietician_retriever = vector_store_initializer.dietician_retriever
# Removed pdf_retriever to avoid unnecessary downloads

logger.info("Successfully accessed all pre-loaded vector stores")

# Create a thread pool executor for parallel operations
# Using max_workers=None will use the default value (min(32, os.cpu_count() + 4))
thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=None)

# Register shutdown handler to clean up resources
import atexit

@atexit.register
def cleanup_resources():
    """Clean up resources when the application shuts down."""
    logger.info("Shutting down thread pool...")
    thread_pool.shutdown(wait=True)
    logger.info("Thread pool shutdown complete.")

# Function to retrieve information from vector store
def retrieve_information(query, retriever):
    """Retrieve relevant information from a vector store."""
    try:
        docs = retriever.get_relevant_documents(query)
        return [doc.page_content for doc in docs]
    except Exception as e:
        logger.error(f"Error retrieving information: {str(e)}")
        return []

# Function to get patient information
def get_patient_info(patient_id: str, db: Session) -> Dict:
    """Get patient information from the database."""
    try:
        patient = db.query(Patient).filter(Patient.patient_id == patient_id).first()
        if not patient:
            return {"error": "Patient not found"}

        # Get medical history
        medical_history = db.query(MedicalHistory).filter(
            MedicalHistory.patient_id == patient_id
        ).order_by(desc(MedicalHistory.created_at)).all()

        # Get prescriptions
        prescriptions = db.query(Prescription).filter(
            Prescription.patient_id == patient_id
        ).order_by(desc(Prescription.created_at)).all()

        # Get onboarding questions and answers
        onboarding_questions = db.query(OnboardingQuestion).filter(
            OnboardingQuestion.patient_id == patient_id
        ).order_by(OnboardingQuestion.timestamp).all()

        # Get appointments using raw SQL to avoid ORM column mismatch issues
        appointments_query = """
            SELECT
                appointment_id,
                patient_id,
                doctor_id,
                appointment_date,
                appointment_time,
                visit_reason,
                status,
                notes,
                created_at,
                updated_at
            FROM appointments
            WHERE patient_id = :patient_id
            ORDER BY appointment_date DESC
        """
        appointments = db.execute(
            text(appointments_query),
            {"patient_id": patient_id}
        ).fetchall()

        # Calculate age from date of birth
        age = None
        if patient.dob:
            today = datetime.now()
            age = today.year - patient.dob.year - ((today.month, today.day) < (patient.dob.month, patient.dob.day))

        # Format patient info
        patient_info = {
            "patient_id": str(patient.patient_id),  # Convert UUID to string
            "name": f"{patient.first_name} {patient.last_name}",
            "gender": patient.gender,
            "dob": patient.dob.strftime("%Y-%m-%d") if patient.dob else None,
            "age": age,  # Add calculated age
            "medical_history": [
                {
                    "condition": mh.diagnosis,  # Using diagnosis instead of condition
                    "notes": mh.additional_notes or mh.treatment or "",  # Using additional_notes or treatment as fallback
                    "date": mh.diagnosed_date.strftime("%Y-%m-%d") if mh.diagnosed_date else (
                        mh.created_at.strftime("%Y-%m-%d") if mh.created_at else None
                    )
                } for mh in medical_history
            ],
            "prescriptions": [
                {
                    "medication": p.medication_name,  # Using medication_name instead of medication
                    "dosage": p.dosage,
                    "instructions": p.instructions,
                    "date": p.start_date.strftime("%Y-%m-%d") if p.start_date else (
                        p.created_at.strftime("%Y-%m-%d") if p.created_at else None
                    )
                } for p in prescriptions
            ],
            "appointments": [
                {
                    "date": a[3].strftime("%Y-%m-%d") if a[3] else None,  # appointment_date
                    "time": a[4].strftime("%H:%M") if a[4] else None,     # appointment_time
                    "reason": a[5],                                       # visit_reason
                    "status": a[6]                                        # status
                } for a in appointments
            ],
            "onboarding_questions": [
                {
                    "question": q.question,
                    "answer": q.answer,
                    "category": q.category,
                    "date": q.timestamp.strftime("%Y-%m-%d") if q.timestamp else None
                } for q in onboarding_questions
            ]
        }

        # Add additional patient fields if they exist
        if hasattr(patient, 'country'):
            patient_info['country'] = patient.country
        if hasattr(patient, 'timezone'):
            patient_info['timezone'] = patient.timezone
        if hasattr(patient, 'preferences'):
            patient_info['preferences'] = patient.preferences
        if hasattr(patient, 'interests'):
            patient_info['interests'] = patient.interests
        if hasattr(patient, 'treatment'):
            patient_info['treatment'] = patient.treatment
        if hasattr(patient, 'isOnboarded'):
            patient_info['isOnboarded'] = patient.isOnboarded

        return patient_info
    except Exception as e:
        logger.error(f"Error getting patient info: {str(e)}")
        return {"error": f"Error retrieving patient information: {str(e)}"}

# Function to get chat history
def get_chat_history(patient_id: str, limit: int = 20, db: Session = None) -> List[Dict]:
    """Get recent chat history for a patient."""
    try:
        if not db:
            db = SessionLocal()

        messages = db.query(ChatMessage).filter(
            (ChatMessage.sender_id == patient_id) | (ChatMessage.receiver_id == patient_id)
        ).order_by(desc(ChatMessage.createdAt)).limit(limit).all()

        # Reverse to get chronological order
        messages.reverse()

        # Format messages
        chat_history = []
        for msg in messages:
            role = "user" if msg.sender_id == patient_id else "assistant"
            chat_history.append({
                "role": role,
                "content": msg.message_text
            })

        return chat_history
    except Exception as e:
        logger.error(f"Error getting chat history: {str(e)}")
        return []
    finally:
        if db and db != SessionLocal():
            db.close()

# Function to generate AI response with streaming
async def generate_ai_response_streaming(
    user_input: str,
    patient_info: Dict,
    chat_history: List[Dict],
    system_prompt: str,
    retriever,
    websocket: WebSocket,
    current_persona: str = "general"
) -> Dict:
    """Generate AI response using OpenAI API with streaming support."""
    try:
        if not OPENAI_CLIENT:
            return {
                "response": "I'm sorry, but the AI service is currently unavailable. Please try again later.",
                "extracted_keywords": ["error", "unavailable"],
                "suggested_specialist": "none"
            }

        # Prepare context (same as before)
        retrieval_query = user_input

        # Extract medical info in parallel
        def extract_medical_conditions():
            if patient_info.get("medical_history"):
                medical_conditions = [item.get("condition", "") for item in patient_info.get("medical_history", [])]
                return [c for c in medical_conditions if c]
            return []

        def extract_medications():
            if patient_info.get("prescriptions"):
                medications = [item.get("medication", "") for item in patient_info.get("prescriptions", [])]
                return [m for m in medications if m]
            return []

        def extract_onboarding_answers():
            if patient_info.get("onboarding_questions"):
                health_categories = ["health", "medical", "symptoms", "lifestyle", "diet", "exercise", "mental_health"]
                answers = []
                for item in patient_info.get("onboarding_questions", []):
                    answer = item.get("answer", "")
                    category = item.get("category", "").lower()
                    if answer and (not category or category in health_categories):
                        answers.append(answer)
                return answers
            return []

        # Get medical context
        medical_conditions = extract_medical_conditions()
        medications = extract_medications()
        onboarding_answers = extract_onboarding_answers()

        # Enhance retrieval query
        if medical_conditions:
            retrieval_query += " " + " ".join(medical_conditions)
        if medications:
            retrieval_query += " " + " ".join(medications)
        if onboarding_answers:
            retrieval_query += " " + " ".join(onboarding_answers[:5])

        # Retrieve information
        relevant_info = retrieve_information(retrieval_query, retriever)

        # Format patient context
        patient_context = f"""
PATIENT PROFILE:
Name: {patient_info.get('name', 'Unknown')}
Gender: {patient_info.get('gender', 'Unknown')}
Date of Birth: {patient_info.get('dob', 'Unknown')}
Age: {patient_info.get('age', 'Unknown')} years
Country: {patient_info.get('country', 'Unknown')}
Timezone: {patient_info.get('timezone', 'Unknown')}

ONBOARDING HEALTH INFORMATION:
{json.dumps(patient_info.get('onboarding_questions', []), indent=2)}

MEDICAL HISTORY:
{json.dumps(patient_info.get('medical_history', []), indent=2)}

CURRENT MEDICATIONS:
{json.dumps(patient_info.get('prescriptions', []), indent=2)}

RECENT APPOINTMENTS:
{json.dumps(patient_info.get('appointments', []), indent=2)}

ADDITIONAL INFORMATION:
Preferences: {json.dumps(patient_info.get('preferences', {}), indent=2)}
Interests: {json.dumps(patient_info.get('interests', {}), indent=2)}
Treatment: {json.dumps(patient_info.get('treatment', {}), indent=2)}
"""

        # Format relevant information
        if relevant_info:
            context = "RELEVANT MEDICAL INFORMATION FROM KNOWLEDGE BASE:\n\n"
            for i, info in enumerate(relevant_info, 1):
                context += f"--- Information {i} ---\n{info}\n\n"
        else:
            context = "No specific medical information found in knowledge base."

        # Prepare messages for OpenAI
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": f"PATIENT DATABASE INFORMATION:\n{patient_context}"},
            {"role": "system", "content": f"KNOWLEDGE BASE INFORMATION:\n{context}"}
        ]

        # Add chat history (limited to last 10 messages)
        for msg in chat_history[-10:]:
            messages.append(msg)

        # Add current user input
        messages.append({"role": "user", "content": user_input})

        # 🚀 STREAMING IMPLEMENTATION
        logger.info("🌊 Starting streaming response generation...")

        # Determine voice for audio streaming
        voice = "nova"  # Default
        if system_prompt == PSYCHOLOGIST_PROMPT:
            voice = "onyx"
        elif system_prompt == DIETICIAN_PROMPT:
            voice = "shimmer"

        # Initialize streaming audio manager
        audio_manager = StreamingAudioManager(voice)

        # Create streaming request
        stream = OPENAI_CLIENT.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=messages,
            temperature=0.1,
            max_tokens=550,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            response_format={"type": "json_object"},
            stream=True
        )

        # Stream processing variables
        complete_response = ""
        response_text = ""
        first_chunk_sent = False
        response_started = False

        # Process streaming chunks
        streaming_complete = False

        for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                complete_response += content

                logger.info(f"🔍 Chunk: '{content}'")

                # 🚀 DUAL APPROACH: Stream response text + collect complete JSON
                if not response_started:
                    # Look for the start of response field
                    if '"response":' in complete_response:
                        response_started = True
                        logger.info("🌊 Response field detected, starting streaming...")

                        # Find where the actual response text starts
                        start_patterns = ['"response": "', '"response":"']
                        for pattern in start_patterns:
                            if pattern in complete_response:
                                start_idx = complete_response.find(pattern) + len(pattern)
                                initial_text = complete_response[start_idx:]

                                # Send initial text if we have any
                                if initial_text and not initial_text.startswith('"'):
                                    # Clean up any partial content
                                    clean_initial = initial_text.split('",')[0]  # Stop at first quote-comma
                                    if clean_initial and len(clean_initial) > 0:
                                        logger.info(f"🌊 Streaming initial text: '{clean_initial}'")
                                        await websocket.send_text(json.dumps({
                                            "type": "streaming_text",
                                            "content": clean_initial,
                                            "is_complete": False
                                        }))
                                        await audio_manager.add_text_chunk(clean_initial, websocket)
                                        response_text = clean_initial
                                break

                elif response_started and not streaming_complete:
                    # We're in streaming mode - check if this chunk ends the response
                    if '",' in content or '"}' in content:
                        # This chunk contains the end of response
                        end_markers = ['",', '"}']
                        final_chunk = content

                        for marker in end_markers:
                            if marker in content:
                                final_chunk = content.split(marker)[0]
                                break

                        if final_chunk.strip():
                            logger.info(f"🌊 Final streaming chunk: '{final_chunk}'")
                            await websocket.send_text(json.dumps({
                                "type": "streaming_text",
                                "content": final_chunk,
                                "is_complete": True
                            }))
                            await audio_manager.add_text_chunk(final_chunk, websocket)
                            response_text += final_chunk

                        logger.info("🌊 Response streaming complete! Continuing to collect metadata...")
                        streaming_complete = True
                        # DON'T break - continue collecting the rest of the JSON
                    else:
                        # Continue streaming this chunk - allow ALL content including spaces
                        if content and content != '\n':  # Only filter newlines, allow everything else including spaces and quotes
                            logger.info(f"🌊 Streaming chunk: '{content}'")

                            # Create the message to send
                            streaming_message = {
                                "type": "streaming_text",
                                "content": content,
                                "is_complete": False
                            }

                            # Log the exact JSON being sent
                            logger.info(f"🔍 DEBUG: Sending streaming JSON: {json.dumps(streaming_message)}")

                            await websocket.send_text(json.dumps(streaming_message))
                            await audio_manager.add_text_chunk(content, websocket)
                            response_text += content

                # Continue collecting chunks even after streaming is complete
                # This ensures we get the full JSON with metadata

        # 🚀 PROCESS COMPLETE RESPONSE
        logger.info("🌊 Streaming complete, processing full response...")

        # Clean up the complete response
        ai_response = complete_response.strip('```json').strip('```').strip()
        ai_response = ai_response.replace('{\n','{').replace('\n}','}').replace(",\n",",").replace('\n','###')
        ai_response = ai_response.strip().replace('###', '')

        try:
            response_json = json.loads(ai_response)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {str(e)}")
            logger.error(f"GPT said: {repr(ai_response)}")
            # Provide a fallback response
            response_json = {
                "response": response_text if response_text else "I apologize, but I encountered an error processing your request. Could you please rephrase?",
                "extracted_keywords": ["error"],
                "suggested_specialist": "none" if system_prompt == GENERAL_OPD_PROMPT else None
            }

        # Ensure required fields are present
        if "response" not in response_json:
            response_json["response"] = response_text if response_text else "I apologize, but I couldn't generate a proper response. Please try again."

        if "extracted_keywords" not in response_json:
            response_json["extracted_keywords"] = []

        if "suggested_specialist" not in response_json and system_prompt == GENERAL_OPD_PROMPT:
            response_json["suggested_specialist"] = "none"

        # Handle refer_to_general and refer_to_specialist fields for specialist prompts
        if system_prompt != GENERAL_OPD_PROMPT:
            if "refer_to_general" not in response_json:
                response_json["refer_to_general"] = False
            if "refer_to_specialist" not in response_json:
                response_json["refer_to_specialist"] = None

        # 🎵 Finalize streaming audio and send completion signal
        await audio_manager.finalize_audio(websocket, response_json["response"])

        # 🌊 Send simple completion signal to finish streaming message
        await websocket.send_text(json.dumps({
            "type": "streaming_complete",
            "extracted_keywords": response_json.get("extracted_keywords", []),
            "current_persona": current_persona
        }))

        logger.info(f"🌊 Streaming response complete. Total length: {len(response_json['response'])}")
        return response_json

    except Exception as e:
        logger.error(f"Error generating streaming AI response: {str(e)}")
        # Send error to client
        await websocket.send_text(json.dumps({
            "type": "streaming_error",
            "error": str(e)
        }))
        return {
            "response": "I apologize, but I encountered an error while processing your request. Please try again.",
            "extracted_keywords": ["error"],
            "suggested_specialist": "none" if system_prompt == GENERAL_OPD_PROMPT else None
        }

# Function to generate AI response (non-streaming fallback)
def generate_ai_response(
    user_input: str,
    patient_info: Dict,
    chat_history: List[Dict],
    system_prompt: str,
    retriever
) -> Dict:
    """Generate AI response using OpenAI API."""
    try:
        if not OPENAI_CLIENT:
            return {
                "response": "I'm sorry, but the AI service is currently unavailable. Please try again later.",
                "extracted_keywords": ["error", "unavailable"],
                "suggested_specialist": "none"
            }

        # Retrieve relevant information based on both user input and patient history
        # Combine user input with key medical terms from patient history for better retrieval
        retrieval_query = user_input

        # Start parallel tasks for extracting medical conditions, medications, and onboarding answers
        # This runs the extraction in parallel with other operations
        def extract_medical_conditions():
            if patient_info.get("medical_history"):
                medical_conditions = [item.get("condition", "") for item in patient_info.get("medical_history", [])]
                return [c for c in medical_conditions if c]
            return []

        def extract_medications():
            if patient_info.get("prescriptions"):
                medications = [item.get("medication", "") for item in patient_info.get("prescriptions", [])]
                return [m for m in medications if m]
            return []

        def extract_onboarding_answers():
            if patient_info.get("onboarding_questions"):
                # Extract answers from onboarding questions, focusing on health-related categories
                health_categories = ["health", "medical", "symptoms", "lifestyle", "diet", "exercise", "mental_health"]
                answers = []

                for item in patient_info.get("onboarding_questions", []):
                    answer = item.get("answer", "")
                    category = item.get("category", "").lower()

                    # Include all answers, but prioritize health-related ones
                    if answer and (not category or category in health_categories):
                        answers.append(answer)

                return answers
            return []

        # Submit tasks to thread pool
        medical_conditions_future = thread_pool.submit(extract_medical_conditions)
        medications_future = thread_pool.submit(extract_medications)
        onboarding_answers_future = thread_pool.submit(extract_onboarding_answers)

        # Get results from futures
        medical_conditions = medical_conditions_future.result()
        medications = medications_future.result()
        onboarding_answers = onboarding_answers_future.result()

        # Enhance retrieval query with medical conditions, medications, and onboarding answers
        if medical_conditions:
            retrieval_query += " " + " ".join(medical_conditions)
        if medications:
            retrieval_query += " " + " ".join(medications)
        if onboarding_answers:
            # Add the most relevant onboarding answers to the query
            # Limit to first 5 answers to avoid making the query too long
            retrieval_query += " " + " ".join(onboarding_answers[:5])

        # Retrieve information using the enhanced query
        # This is still sequential as it depends on the enhanced query
        relevant_info = retrieve_information(retrieval_query, retriever)

        # Format patient info for context with clear sections
        patient_context = f"""
PATIENT PROFILE:
Name: {patient_info.get('name', 'Unknown')}
Gender: {patient_info.get('gender', 'Unknown')}
Date of Birth: {patient_info.get('dob', 'Unknown')}
Age: {patient_info.get('age', 'Unknown')} years
Country: {patient_info.get('country', 'Unknown')}
Timezone: {patient_info.get('timezone', 'Unknown')}

ONBOARDING HEALTH INFORMATION:
{json.dumps(patient_info.get('onboarding_questions', []), indent=2)}

MEDICAL HISTORY:
{json.dumps(patient_info.get('medical_history', []), indent=2)}

CURRENT MEDICATIONS:
{json.dumps(patient_info.get('prescriptions', []), indent=2)}

RECENT APPOINTMENTS:
{json.dumps(patient_info.get('appointments', []), indent=2)}

ADDITIONAL INFORMATION:
Preferences: {json.dumps(patient_info.get('preferences', {}), indent=2)}
Interests: {json.dumps(patient_info.get('interests', {}), indent=2)}
Treatment: {json.dumps(patient_info.get('treatment', {}), indent=2)}
"""

        # Format relevant information with clear section headers
        if relevant_info:
            context = "RELEVANT MEDICAL INFORMATION FROM KNOWLEDGE BASE:\n\n"
            for i, info in enumerate(relevant_info, 1):
                context += f"--- Information {i} ---\n{info}\n\n"
        else:
            context = "No specific medical information found in knowledge base."

        # Prepare messages for OpenAI with enhanced context
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": f"PATIENT DATABASE INFORMATION:\n{patient_context}"},
            {"role": "system", "content": f"KNOWLEDGE BASE INFORMATION:\n{context}"}
        ]

        # Add chat history (limited to last 10 messages to save tokens)
        for msg in chat_history[-10:]:
            messages.append(msg)

        # Add current user input
        messages.append({"role": "user", "content": user_input})

        # Log messages being sent to LLM
        messages_to_log = []
        for msg in messages:
            # Create a copy of the message to avoid modifying the original
            msg_copy = msg.copy()
            # Truncate content if it's too long
            if len(msg_copy.get('content', '')) > 500:
                msg_copy['content'] = msg_copy['content'][:500] + "... [truncated]"
            messages_to_log.append(msg_copy)

        logger.info(f"Messages sent to LLM: {json.dumps(messages_to_log, indent=2)}")

        # print("Messages sent to LLM::::", messages)

        # Generate response using the same model and configuration as smart_agent.py
        response = OPENAI_CLIENT.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=messages,
            temperature=0.1,
            max_tokens=550,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            response_format={"type": "json_object"}
        )

        # Parse response with improved error handling similar to smart_agent.py
        response_text = response.choices[0].message.content.strip()

        # Clean up the response text similar to smart_agent.py
        ai_response = response_text.strip('```json').strip('```').strip().replace('{\n','{').replace('\n}','}').replace(",\n",",").replace('\n','###')
        ai_response = ai_response.strip().replace('###', '')

        logger.info(f"Generated response: {ai_response}")

        try:
            response_json = json.loads(ai_response)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {str(e)}")
            logger.error(f"GPT said: {repr(ai_response)}")
            # Provide a fallback response
            response_json = {
                "response": "I apologize, but I encountered an error processing your request. Could you please rephrase?",
                "extracted_keywords": ["error"],
                "suggested_specialist": "none" if system_prompt == GENERAL_OPD_PROMPT else None
            }

        # Log response from LLM
        logger.info(f"Response from LLM: {response_text}")

        # Ensure required fields are present
        if "response" not in response_json:
            response_json["response"] = "I apologize, but I couldn't generate a proper response. Please try again."

        if "extracted_keywords" not in response_json:
            response_json["extracted_keywords"] = []

        if "suggested_specialist" not in response_json and system_prompt == GENERAL_OPD_PROMPT:
            response_json["suggested_specialist"] = "none"

         # Handle refer_to_general and refer_to_specialist fields for specialist prompts
        if system_prompt != GENERAL_OPD_PROMPT:
            if "refer_to_general" not in response_json:
                response_json["refer_to_general"] = False
            if "refer_to_specialist" not in response_json:
                response_json["refer_to_specialist"] = None

        return response_json
    except Exception as e:
        logger.error(f"Error generating AI response: {str(e)}")
        return {
            "response": "I apologize, but I encountered an error while processing your request. Please try again.",
            "extracted_keywords": ["error"],
            "suggested_specialist": "none" if system_prompt == GENERAL_OPD_PROMPT else None
        }

# Function to generate audio from text
def generate_audio(text: str, voice: str = "nova") -> Optional[str]:
    """Generate audio from text using OpenAI API."""
    try:
        if not OPENAI_CLIENT:
            return None

        # Generate audio
        response = OPENAI_CLIENT.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text
        )

        # Get audio data
        audio_data = response.content

        # Convert to base64
        base64_audio = base64.b64encode(audio_data).decode("utf-8")

        return base64_audio
    except Exception as e:
        logger.error(f"Error generating audio: {str(e)}")
        return None

# Streaming audio generation manager
class StreamingAudioManager:
    def __init__(self, voice: str = "nova"):
        self.voice = voice
        self.accumulated_text = ""
        self.audio_queue = []
        self.is_generating = False

    async def add_text_chunk(self, text_chunk: str, websocket: WebSocket):
        """Add text chunk and generate audio when we have enough text."""
        self.accumulated_text += text_chunk

        # Generate audio when we have a complete sentence or enough text
        if (("." in self.accumulated_text or "!" in self.accumulated_text or "?" in self.accumulated_text)
            and len(self.accumulated_text) > 30) or len(self.accumulated_text) > 100:

            if not self.is_generating:
                self.is_generating = True
                # Generate audio for accumulated text
                audio_future = thread_pool.submit(generate_audio, self.accumulated_text, self.voice)

                try:
                    audio_data = audio_future.result(timeout=10)  # 10 second timeout
                    if audio_data:
                        # Send audio chunk to client
                        await websocket.send_text(json.dumps({
                            "type": "streaming_audio",
                            "audio": audio_data,
                            "text": self.accumulated_text
                        }))
                        logger.info(f"🎵 Generated streaming audio for: '{self.accumulated_text[:30]}...'")

                        # Reset for next chunk
                        self.accumulated_text = ""
                except Exception as e:
                    logger.error(f"Error generating streaming audio: {str(e)}")
                finally:
                    self.is_generating = False

    async def finalize_audio(self, websocket: WebSocket, complete_text: str):
        """Generate final audio for any remaining text."""
        if self.accumulated_text.strip():
            try:
                audio_data = generate_audio(self.accumulated_text, self.voice)
                if audio_data:
                    await websocket.send_text(json.dumps({
                        "type": "streaming_audio_final",
                        "audio": audio_data,
                        "text": self.accumulated_text
                    }))
                    logger.info(f"🎵 Generated final streaming audio chunk")
            except Exception as e:
                logger.error(f"Error generating final streaming audio: {str(e)}")

        # Also generate complete audio as fallback
        try:
            complete_audio = generate_audio(complete_text, self.voice)
            if complete_audio:
                await websocket.send_text(json.dumps({
                    "type": "complete_audio",
                    "audio": complete_audio,
                    "text": complete_text
                }))
                logger.info(f"🎵 Generated complete audio as fallback")
        except Exception as e:
            logger.error(f"Error generating complete audio: {str(e)}")

# 🔧 Function to detect transfer requests early
def is_transfer_request(text: str) -> tuple[bool, str]:
    """
    Detect if user is requesting a transfer to a specific specialist.
    Returns (is_transfer, target_specialist)
    """
    text_lower = text.lower()

    # Direct specialist requests
    if any(phrase in text_lower for phrase in [
        "transfer me to", "switch to", "change to", "connect me to",
        "i want to talk to", "can i speak to", "i need to see"
    ]):
        if any(word in text_lower for word in ["psychologist", "psychology", "mental health", "therapist"]):
            return True, "psychologist"
        elif any(word in text_lower for word in ["dietician", "nutritionist", "diet", "nutrition"]):
            return True, "dietician"
        elif any(word in text_lower for word in ["general", "general doctor", "gp", "primary care"]):
            return True, "general"

    # Direct specialist mentions
    if any(phrase in text_lower for phrase in [
        "psychologist", "mental health specialist", "therapist"
    ]) and any(word in text_lower for word in ["please", "want", "need", "transfer", "switch"]):
        return True, "psychologist"

    if any(phrase in text_lower for phrase in [
        "dietician", "nutritionist", "nutrition specialist"
    ]) and any(word in text_lower for word in ["please", "want", "need", "transfer", "switch"]):
        return True, "dietician"

    if any(phrase in text_lower for phrase in [
        "general doctor", "primary care", "gp"
    ]) and any(word in text_lower for word in ["please", "want", "need", "transfer", "switch"]):
        return True, "general"

    return False, None

# 🔧 Function to handle specialist transfers with predefined messages
async def handle_specialist_transfer(
    websocket: WebSocket,
    user_input: str,
    current_persona: str,
    target_specialist: str,
    patient_info: dict,
    chat_history: list
) -> tuple[str, dict]:
    """
    Handle specialist transfer with predefined messages only.
    Returns (new_persona, response_data)
    """

    # Predefined transition messages
    transition_messages = {
        ("general", "psychologist"): "I'll connect you with our psychologist Doctor Ori who specializes in mental health. Transferring you now.",
        ("general", "dietician"): "I'll connect you with our dietician Doctor Maya who specializes in nutrition. Transferring you now.",
        ("psychologist", "general"): "I believe your concerns would be better addressed by our general doctor Tova. I'm transferring you back to them now.",
        ("dietician", "general"): "I believe your concerns would be better addressed by our general doctor Tova. I'm transferring you back to them now.",
        ("psychologist", "dietician"): "For your nutrition concerns, our dietician Doctor Maya would be more appropriate. I'm transferring you to them now.",
        ("dietician", "psychologist"): "For your mental health concerns, our psychologist Doctor Ori would be better suited. I'm transferring you to them now."
    }

    # Welcome messages for new specialists
    welcome_messages = {
        "psychologist": "You are now speaking with psychologist Doctor Ori. How can I help you today?",
        "dietician": "Hello! I'm Doctor Maya, your dietician. I'm here to help with your nutrition and dietary needs. How can I assist you today?",
        "general": "Hello, I'm Dr. Tova, a general practitioner with 15 years of experience. How can I assist you today?"
    }

    # Voice mapping
    voices = {
        "general": "nova",
        "psychologist": "onyx",
        "dietician": "shimmer"
    }

    # Get transition message
    transition_key = (current_persona, target_specialist)
    transition_msg = transition_messages.get(transition_key, f"Transferring you to {target_specialist} now.")

    # Send transition message with current specialist's voice
    current_voice = voices[current_persona]
    transition_audio = generate_audio(transition_msg, current_voice)

    await websocket.send_text(json.dumps({
        "response": transition_msg,
        "extracted_keywords": ["transition", target_specialist],
        "audio": transition_audio,
        "response_id": str(uuid.uuid4()),
        "current_persona": current_persona
    }))

    logger.info(f"🔄 Sent transition message from {current_persona} to {target_specialist}")

    # Send welcome message from new specialist
    welcome_msg = welcome_messages[target_specialist]
    new_voice = voices[target_specialist]
    welcome_audio = generate_audio(welcome_msg, new_voice)

    await websocket.send_text(json.dumps({
        "response": welcome_msg,
        "extracted_keywords": ["greeting", "introduction"],
        "audio": welcome_audio,
        "response_id": str(uuid.uuid4()),
        "current_persona": target_specialist
    }))

    logger.info(f"🎯 Sent welcome message from {target_specialist}")

    # Return new persona and dummy response data
    return target_specialist, {
        "response": welcome_msg,
        "extracted_keywords": ["greeting", "introduction"],
        "suggested_specialist": "none"
    }

# WebSocket endpoint for chat
@chat_final_router.websocket("/chat-final/{patient_id}")
async def chat_websocket(websocket: WebSocket, patient_id: str, db: Session = Depends(get_db)):
    await websocket.accept()

    # Initialize connection state
    current_persona = "general"  # Start with General OPD
    authenticated = False

    # Pre-loaded data containers (will be populated after authentication)
    patient_info = None
    chat_history = []  # In-memory chat history

    # Performance tracking
    import time
    connection_start = time.time()

    try:
        # Wait for authentication
        while not authenticated:
            try:
                # Receive authentication message
                auth_message = await websocket.receive_text()
                auth_data = json.loads(auth_message)

                # Check for token
                if "token" not in auth_data:
                    await websocket.send_text(json.dumps({
                        "error": "Authentication token required"
                    }))
                    continue

                # Validate token (simplified for now)
                token = auth_data["token"]
                if not token or token == "invalid":
                    await websocket.send_text(json.dumps({
                        "error": "Invalid authentication token"
                    }))
                    continue

                # Authentication successful
                authenticated = True

                # 🚀 OPTIMIZATION: Pre-load ALL patient data at connection time
                logger.info(f"🔄 Pre-loading all patient data for {patient_id}...")
                data_load_start = time.time()

                # Load patient information
                patient_info = get_patient_info(patient_id, db)
                if "error" in patient_info:
                    await websocket.send_text(json.dumps({
                        "error": patient_info["error"]
                    }))
                    return

                # Load initial chat history (last 10 messages) into memory
                chat_history = get_chat_history(patient_id, limit=10, db=db)

                data_load_time = time.time() - data_load_start
                logger.info(f"✅ Pre-loaded patient data in {data_load_time:.2f}s - Patient: {patient_info['name']}, History: {len(chat_history)} messages")

                # Send welcome message
                welcome_message = {
                    "response": f"Hello {patient_info['name']}! I'm Dr. Tova, a general practitioner. How can I help you today?",
                    "extracted_keywords": ["greeting", "introduction"],
                    "suggested_specialist": "none"
                }

                # Generate audio in parallel if voice is specified
                voice = auth_data.get("voice", "nova")
                audio_future = thread_pool.submit(generate_audio, welcome_message["response"], voice)

                # Get audio data from future
                audio_data = audio_future.result()

                # Send response
                await websocket.send_text(json.dumps({
                    "response": welcome_message["response"],
                    "extracted_keywords": welcome_message["extracted_keywords"],
                    "suggested_specialist": welcome_message["suggested_specialist"],
                    "audio": audio_data,
                    "response_id": str(uuid.uuid4())
                }))

            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "error": "Invalid JSON format"
                }))
            except Exception as e:
                logger.error(f"Authentication error: {str(e)}")
                await websocket.send_text(json.dumps({
                    "error": f"Authentication error: {str(e)}"
                }))

        # Main chat loop
        while True:
            # Receive message
            message = await websocket.receive_text()
            data = json.loads(message)

            # Handle text input
            if "text" in data:
                # 🚀 OPTIMIZATION: Start timing for this request
                request_start = time.time()
                user_input = data["text"]
                voice = data.get("voice", "nova")

                # Set voice based on current persona
                if current_persona == "psychologist":
                    voice = "onyx"  # Male voice for psychologist Doctor Ori
                elif current_persona == "dietician":
                    voice = "shimmer"  # Female voice for dietician Doctor Maya
                else:  # general
                    voice = "nova"  # Female voice for general doctor Tova

                logger.info(f"🎯 Processing text input: '{user_input[:50]}...' (Persona: {current_persona})")

                # 🔧 EARLY DETECTION: Check for transfer requests BEFORE AI processing
                is_transfer, target_specialist = is_transfer_request(user_input)

                if is_transfer and target_specialist:
                    logger.info(f"🔄 Early transfer detection: {current_persona} → {target_specialist}")

                    # Add user message to chat history
                    chat_history.append({"role": "user", "content": user_input})

                    # Save user message to database in parallel
                    def save_user_message_to_database():
                        try:
                            with SessionLocal() as thread_db:
                                now = datetime.now()
                                thread_db.add(ChatMessage(
                                    chat_message_id=str(uuid.uuid4()),
                                    sender_id=patient_id,
                                    receiver_id=AI_DOCTOR_ID,
                                    message_text=user_input,
                                    createdAt=now,
                                ))
                                thread_db.commit()
                                logger.info("✅ Saved user message to database")
                                return True
                        except Exception as e:
                            logger.error(f"❌ Error saving user message: {str(e)}")
                            return False

                    # Submit database task to thread pool (non-blocking)
                    db_future = thread_pool.submit(save_user_message_to_database)

                    # Handle transfer with predefined messages only
                    current_persona, response_data = await handle_specialist_transfer(
                        websocket, user_input, current_persona, target_specialist, patient_info, chat_history
                    )

                    # Add transition messages to chat history
                    chat_history.append({"role": "assistant", "content": response_data["response"]})

                    # Calculate timing
                    total_time = time.time() - request_start
                    logger.info(f"🔄 Transfer completed in {total_time:.2f}s")

                    # Skip AI processing and continue to next message
                    continue
                # 🚀 NORMAL AI PROCESSING: Add to in-memory chat history immediately
                chat_history.append({"role": "user", "content": user_input})

                # 🚀 OPTIMIZATION: Start database save in parallel (non-blocking)
                def save_user_message_to_database():
                    try:
                        with SessionLocal() as thread_db:
                            now = datetime.now()
                            thread_db.add(ChatMessage(
                                chat_message_id=str(uuid.uuid4()),
                                sender_id=patient_id,
                                receiver_id=AI_DOCTOR_ID,
                                message_text=user_input,
                                createdAt=now,
                            ))
                            thread_db.commit()
                            logger.info("✅ Saved user message to database")
                            return True
                    except Exception as e:
                        logger.error(f"❌ Error saving user message: {str(e)}")
                        return False

                # Submit database task to thread pool (non-blocking)
                db_future = thread_pool.submit(save_user_message_to_database)

                # 🚀 OPTIMIZATION: Select persona and retriever (using pre-loaded data)
                system_prompt = GENERAL_OPD_PROMPT
                retriever = general_retriever

                if current_persona == "psychologist":
                    system_prompt = PSYCHOLOGIST_PROMPT
                    retriever = psychologist_retriever
                elif current_persona == "dietician":
                    system_prompt = DIETICIAN_PROMPT
                    retriever = dietician_retriever

                # 🚀 STREAMING: Generate AI response with streaming support
                ai_start = time.time()
                response_data = await generate_ai_response_streaming(
                    user_input,
                    patient_info,  # Using pre-loaded patient info
                    chat_history,  # Using in-memory chat history
                    system_prompt,
                    retriever,
                    websocket,  # Pass websocket for streaming
                    current_persona  # Pass current persona
                )
                ai_time = time.time() - ai_start
                logger.info(f"🌊 Streaming AI response completed in {ai_time:.2f}s")

                # Check if specialist wants to refer to another specialist directly
                if current_persona != "general" and response_data.get("refer_to_specialist"):
                    target_specialist = response_data.get("refer_to_specialist")
                    if target_specialist in ["psychologist", "dietician"] and target_specialist != current_persona:
                        # Direct specialist-to-specialist transfer
                        current_specialist_voice = "onyx" if current_persona == "psychologist" else "shimmer"

                        # Generate specialist-specific transition message
                        if current_persona == "psychologist" and target_specialist == "dietician":
                            transition_msg = f"For your nutrition concerns, our dietician Doctor Maya would be more appropriate. I'm transferring you to them now."
                        elif current_persona == "dietician" and target_specialist == "psychologist":
                            transition_msg = f"For your mental health concerns, our psychologist Doctor Ori would be better suited. I'm transferring you to them now."

                        # Update current persona
                        current_persona = target_specialist

                        # Update voice based on new persona
                        if current_persona == "psychologist":
                            voice = "onyx"  # Male voice for psychologist Doctor Ori
                        elif current_persona == "dietician":
                            voice = "shimmer"  # Female voice for dietician Doctor Maya

                        # Send transition message with current specialist's voice
                        audio_future = thread_pool.submit(generate_audio, transition_msg, current_specialist_voice)
                        audio_data = audio_future.result()

                        await websocket.send_text(json.dumps({
                            "response": transition_msg,
                            "extracted_keywords": ["transition", target_specialist],
                            "audio": audio_data,
                            "response_id": str(uuid.uuid4())
                        }))

                        # Update system prompt and retriever
                        if current_persona == "psychologist":
                            system_prompt = PSYCHOLOGIST_PROMPT
                            retriever = psychologist_retriever
                        elif current_persona == "dietician":
                            system_prompt = DIETICIAN_PROMPT
                            retriever = dietician_retriever

                        # Generate new response from target specialist with STREAMING
                        response_data = await generate_ai_response_streaming(
                            f"Hello, I'm a patient who was just transferred to you from another specialist, First introduce yourself. My original question was: {user_input}",
                            patient_info,
                            chat_history,
                            system_prompt,
                            retriever,
                            websocket,
                            current_persona
                        )

                        # No need to send additional response - streaming already handled it

                # Check if specialist wants to refer back to General OPD
                elif current_persona != "general" and response_data.get("refer_to_general", False):
                    # Determine the specialist voice before changing the persona
                    specialist_voice = "onyx" if current_persona == "psychologist" else "shimmer"

                    # Update current persona to general
                    current_persona = "general"

                    # Start audio generation in parallel with the specialist's voice
                    transition_msg = f"It sounds like our general doctor Tova would be best suited to help you. I'm transferring you back to them now."
                    audio_future = thread_pool.submit(generate_audio, transition_msg, specialist_voice)

                    # Get audio data from future
                    audio_data = audio_future.result()

                    await websocket.send_text(json.dumps({
                        "response": transition_msg,
                        "extracted_keywords": ["transition", "general"],
                        "audio": audio_data,
                        "response_id": str(uuid.uuid4())
                    }))

                    # Update system prompt and retriever
                    system_prompt = GENERAL_OPD_PROMPT
                    retriever = general_retriever

                    # Generate new response from General OPD with STREAMING
                    response_data = await generate_ai_response_streaming(
                        f"Hello, I'm a patient who was just transferred back to you from a specialist, First Introduce yourself. My original question was: {user_input}",
                        patient_info,
                        chat_history,
                        system_prompt,
                        retriever,
                        websocket,
                        current_persona
                    )

                    # No need to send additional response - streaming already handled it

                # Check if specialist suggestion is made
                if current_persona == "general" and response_data.get("suggested_specialist") not in ["none", None]:
                    suggested = response_data.get("suggested_specialist")
                    if suggested in ["psychologist", "dietician"]:
                        # Automatically transfer to the suggested specialist
                        specialist_names = {
                            "psychologist": "psychologist Doctor Ori",
                            "dietician": "dietician Doctor Maya"
                        }

                        # Save original general response (optional)
                        initial_general_response = response_data["response"]

                        # Prepare transition message
                        transition_msg = (
                            f"Based on our conversation, I believe our {specialist_names[suggested]} would be better suited to help you. "
                            f"I'm transferring you to them now."
                        )

                        # Send transition message with general doctor’s voice (nova)
                        transition_audio = thread_pool.submit(generate_audio, transition_msg, "nova").result()
                        await websocket.send_text(json.dumps({
                            "response": transition_msg,
                            "extracted_keywords": ["transition", suggested],
                            "audio": transition_audio,
                            "response_id": str(uuid.uuid4())
                        }))

                        # Update persona and voice
                        current_persona = suggested
                        voice = "onyx" if current_persona == "psychologist" else "shimmer"

                        # Update system prompt and retriever
                        if current_persona == "psychologist":
                            system_prompt = PSYCHOLOGIST_PROMPT
                            retriever = psychologist_retriever
                        else:
                            system_prompt = DIETICIAN_PROMPT
                            retriever = dietician_retriever

                        # Generate specialist response
                        specialist_intro = (
                            f"Hello, I'm a patient who was just transferred to you. First introduce yourself. "
                            f"My original question was: {user_input}"
                        )
                        response_data = await generate_ai_response_streaming(
                            specialist_intro,
                            patient_info,
                            chat_history,
                            system_prompt,
                            retriever,
                            websocket,
                            current_persona
                        )

                        # No need to send additional response - streaming already handled it


                # # Check if specialist suggestion is made
                # if current_persona == "general" and response_data.get("suggested_specialist") not in ["none", None]:
                #     suggested = response_data.get("suggested_specialist")
                #     if suggested in ["psychologist", "dietician"]:
                #         # Automatically transfer to the suggested specialist
                #         specialist_names = {
                #             "psychologist": "psychologist Doctor Ori",
                #             "dietician": "dietician Doctor Maya"
                #         }

                #         # Update current persona
                #         current_persona = suggested

                #         # Update voice based on new persona
                #         if current_persona == "psychologist":
                #             voice = "onyx"  # Male voice for psychologist Doctor Ori
                #         elif current_persona == "dietician":
                #             voice = "shimmer"  # Female voice for dietician Doctor Maya

                #         # Add transition message to the response
                #         response_data["response"] += f"\n\nI'm transferring you to our {specialist_names[suggested]} for more specialized care."

                #         # Generate transition message and start audio generation in parallel
                #         transition_msg = f"I'm transferring you to our {specialist_names[suggested]} now."
                #         # Use nova (female voice) for the transition message since it's still the general doctor speaking
                #         audio_future = thread_pool.submit(generate_audio, transition_msg, "nova")

                #         # Get audio data from future
                #         audio_data = audio_future.result()

                #         await websocket.send_text(json.dumps({
                #             "response": transition_msg,
                #             "extracted_keywords": ["transition", suggested],
                #             "audio": audio_data,
                #             "response_id": str(uuid.uuid4())
                #         }))

                #         # Update system prompt and retriever for next response
                #         if current_persona == "psychologist":
                #             system_prompt = PSYCHOLOGIST_PROMPT
                #             retriever = psychologist_retriever
                #         elif current_persona == "dietician":
                #             system_prompt = DIETICIAN_PROMPT
                #             retriever = dietician_retriever

                #         # Generate new response from specialist
                #         response_data = generate_ai_response(
                #             f"Hello, I'm a patient who was just transferred to you, First Introduce yourself. My original question was: {user_input}",
                #             patient_info,
                #             chat_history,
                #             system_prompt,
                #             retriever
                #         )

                #         # Set the appropriate voice for the specialist
                #         if current_persona == "psychologist":
                #             voice = "onyx"  # Male voice for psychologist Doctor Ori
                #         elif current_persona == "dietician":
                #             voice = "shimmer"  # Female voice for dietician Doctor Maya

                #         # Send the specialist response to frontend
                #         audio_future = thread_pool.submit(generate_audio, response_data["response"], voice)
                #         audio_data = audio_future.result()

                #         await websocket.send_text(json.dumps({
                #             "response": response_data["response"],
                #             "extracted_keywords": response_data["extracted_keywords"],
                #             "audio": audio_data,
                #             "response_id": str(uuid.uuid4()),
                #             "current_persona": current_persona
                #         }))

                # # 🔄 Handle general → specialist transition based on user confirmation
                # if current_persona == "general" and any(keyword in user_input.lower() for keyword in ["yes", "connect", "specialist", "switch"]):
                #     # Check last few assistant messages for suggested specialist
                #     for msg in reversed(chat_history[-5:]):
                #         if msg["role"] == "assistant" and "suggested_specialist" in msg:
                #             suggested = msg["suggested_specialist"]

                #             if suggested in ["psychologist", "dietician"]:
                #                 # ⬇ Update persona and voice
                #                 current_persona = suggested
                #                 voice = "onyx" if current_persona == "psychologist" else "shimmer"

                #                 # 🗣️ Create transition message
                #                 transition_msg = (
                #                     f"Based on our conversation, I believe our {current_persona} would be best suited to address your concerns. "
                #                     f"I'm transferring you to them now. All your information and our conversation history will be shared with them for continuity of care."
                #                 )

                #                 # 📤 Send transition message
                #                 await websocket.send_text(json.dumps({
                #                     "response": transition_msg,
                #                     "extracted_keywords": ["transition", current_persona],
                #                     "audio": generate_audio(transition_msg, voice),
                #                     "response_id": str(uuid.uuid4())
                #                 }))

                #                 # 🧠 Update system prompt and retriever
                #                 if current_persona == "psychologist":
                #                     system_prompt = PSYCHOLOGIST_PROMPT
                #                     retriever = psychologist_retriever
                #                 else:
                #                     system_prompt = DIETICIAN_PROMPT
                #                     retriever = dietician_retriever

                #                 # 🤖 Generate specialist's introduction/response
                #                 response_data = generate_ai_response(
                #                     f"Hello, I'm a patient who was just transferred to you. First introduce yourself. My original question was: {user_input}",
                #                     patient_info,
                #                     chat_history,
                #                     system_prompt,
                #                     retriever
                #                 )

                #                 # 🔊 Generate audio in background
                #                 audio_future = thread_pool.submit(generate_audio, response_data["response"], voice)
                #                 audio_data = audio_future.result()

                #                 # 📤 Send specialist response
                #                 await websocket.send_text(json.dumps({
                #                     "response": response_data["response"],
                #                     "extracted_keywords": response_data["extracted_keywords"],
                #                     "audio": audio_data,
                #                     "response_id": str(uuid.uuid4()),
                #                     "current_persona": current_persona
                #                 }))

               


                # Handle specialist transition request
                if current_persona == "general" and any(keyword in user_input.lower() for keyword in ["yes", "connect", "specialist", "switch"]):
                    # Look for specialist in previous response
                    for msg in reversed(chat_history[-5:]):
                        if msg["role"] == "assistant" and "suggested_specialist" in msg:
                            suggested = msg.get("suggested_specialist")
                            if suggested in ["psychologist", "dietician"]:
                                current_persona = suggested

                                # Update voice based on new persona
                                if current_persona == "psychologist":
                                    voice = "onyx"  # Male voice for psychologist Doctor Ori
                                elif current_persona == "dietician":
                                    voice = "shimmer"  # Female voice for dietician Doctor Maya

                                # Generate transition message with more context
                                specialist_names = {
                                    "psychologist": "psychologist",
                                    "dietician": "dietician"
                                }
                                transition_msg = f"Based on our conversation, I believe our {specialist_names[suggested]} would be best suited to address your concerns. I'm transferring you to them now. All your information and our conversation history will be shared with them for continuity of care."
                                await websocket.send_text(json.dumps({
                                    "response": transition_msg,
                                    "extracted_keywords": ["transition", suggested],
                                    "audio": generate_audio(transition_msg, voice),
                                    "response_id": str(uuid.uuid4())
                                }))

                                # Update system prompt and retriever for next response
                                if current_persona == "psychologist":
                                    system_prompt = PSYCHOLOGIST_PROMPT
                                    retriever = psychologist_retriever
                                elif current_persona == "dietician":
                                    system_prompt = DIETICIAN_PROMPT
                                    retriever = dietician_retriever

                                # Generate new response from specialist with STREAMING
                                response_data = await generate_ai_response_streaming(
                                    f"Hello, I'm a patient who was just transferred to you, First Introduce Yourself. My original question was: {user_input}",
                                    patient_info,
                                    chat_history,
                                    system_prompt,
                                    retriever,
                                    websocket,
                                    current_persona
                                )

                                # No need to send additional response - streaming already handled it

                # 🚀 STREAMING COMPLETE: Add AI response to chat history and save to database
                response_id = str(uuid.uuid4())

                # Add AI response to in-memory chat history immediately
                chat_history.append({"role": "assistant", "content": response_data["response"]})

                def save_ai_response_to_database():
                    try:
                        with SessionLocal() as thread_db:
                            now = datetime.now()
                            thread_db.add(ChatMessage(
                                chat_message_id=response_id,
                                sender_id=AI_DOCTOR_ID,
                                receiver_id=patient_id,
                                message_text=response_data["response"],
                                extracted_keywords=",".join(response_data["extracted_keywords"]),
                                createdAt=now,
                            ))
                            thread_db.commit()
                            logger.info("✅ Saved AI response to database")
                            return True
                    except Exception as e:
                        logger.error(f"❌ Error saving AI response: {str(e)}")
                        return False

                # 🚀 OPTIMIZATION: Save to database in background
                db_future = thread_pool.submit(save_ai_response_to_database)

                # Calculate total request time
                total_time = time.time() - request_start
                logger.info(f"🏁 Total streaming request processed in {total_time:.2f}s (AI: {ai_time:.2f}s)")

                # ✅ STREAMING ONLY - No traditional response needed (streaming already displays complete message)

            # Handle audio input
            elif "audio" in data:
                # 🚀 OPTIMIZATION: Start timing for audio processing
                audio_request_start = time.time()
                audio_base64 = data["audio"]
                voice = data.get("voice", "nova")

                try:
                    logger.info(f"🎤 Processing audio input...")

                    # Decode audio
                    transcription_start = time.time()
                    audio_bytes = base64.b64decode(audio_base64)

                    # Create temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_file:
                        temp_file.write(audio_bytes)
                        temp_file_path = temp_file.name

                    # Transcribe audio
                    with open(temp_file_path, "rb") as audio_file:
                        transcription = OPENAI_CLIENT.audio.transcriptions.create(
                            model="whisper-1",
                            file=audio_file
                        )

                    # Clean up temporary file
                    os.unlink(temp_file_path)

                    # Get transcribed text
                    user_input = transcription.text
                    transcription_time = time.time() - transcription_start
                    logger.info(f"🎯 Audio transcribed in {transcription_time:.2f}s: '{user_input[:50]}...'")

                    # Send transcription back to client
                    await websocket.send_text(json.dumps({
                        "transcription": user_input
                    }))

                    # Save user message to database in parallel
                    def save_user_message_to_database():
                        try:
                            # Create a new session for this thread
                            with SessionLocal() as thread_db:
                                # Set timestamp, createdAt,
                                now = datetime.now()
                                thread_db.add(ChatMessage(
                                    chat_message_id=str(uuid.uuid4()),
                                    sender_id=patient_id,
                                    receiver_id=AI_DOCTOR_ID,
                                    message_text=user_input,
                                    # timestamp=now,
                                    createdAt=now,  # Required non-null field in the database
                                      # Required non-null field in the database
                                ))
                                thread_db.commit()
                                logger.info("✅ Saved user message to database")
                                return True
                        except Exception as e:
                            logger.error(f"❌ Error saving user message: {str(e)}")
                            return False

                    # Submit database task to thread pool
                    db_future = thread_pool.submit(save_user_message_to_database)

                    # 🔧 EARLY DETECTION: Check for transfer requests BEFORE AI processing (same as text input)
                    is_transfer, target_specialist = is_transfer_request(user_input)

                    if is_transfer and target_specialist:
                        logger.info(f"🔄 Early transfer detection (audio): {current_persona} → {target_specialist}")

                        # Add user message to chat history
                        chat_history.append({"role": "user", "content": user_input})

                        # Handle transfer with predefined messages only
                        current_persona, response_data = await handle_specialist_transfer(
                            websocket, user_input, current_persona, target_specialist, patient_info, chat_history
                        )

                        # Add transition messages to chat history
                        chat_history.append({"role": "assistant", "content": response_data["response"]})

                        # Calculate timing
                        total_audio_time = time.time() - audio_request_start
                        logger.info(f"🔄 Audio transfer completed in {total_audio_time:.2f}s")

                        # Skip AI processing and continue to next message
                        continue

                    # 🚀 NORMAL AI PROCESSING: Add to in-memory chat history immediately
                    chat_history.append({"role": "user", "content": user_input})

                    # Select appropriate persona and retriever (using pre-loaded data)
                    system_prompt = GENERAL_OPD_PROMPT
                    retriever = general_retriever
                    voice = "nova"  # Default voice for general doctor

                    if current_persona == "psychologist":
                        system_prompt = PSYCHOLOGIST_PROMPT
                        retriever = psychologist_retriever
                        voice = "onyx"  # Male voice for psychologist Doctor Ori
                    elif current_persona == "dietician":
                        system_prompt = DIETICIAN_PROMPT
                        retriever = dietician_retriever
                        voice = "shimmer"  # Female voice for dietician Doctor Maya

                    # 🚀 STREAMING: Generate AI response with streaming support
                    ai_start = time.time()
                    response_data = await generate_ai_response_streaming(
                        user_input,
                        patient_info,  # Using pre-loaded patient info
                        chat_history,  # Using in-memory chat history
                        system_prompt,
                        retriever,
                        websocket,  # Pass websocket for streaming
                        current_persona  # Pass current persona
                    )
                    ai_time = time.time() - ai_start
                    logger.info(f"🌊 Streaming AI response completed in {ai_time:.2f}s")

                    # Check if specialist wants to refer to another specialist directly
                    if current_persona != "general" and response_data.get("refer_to_specialist"):
                        target_specialist = response_data.get("refer_to_specialist")
                        if target_specialist in ["psychologist", "dietician"] and target_specialist != current_persona:
                            # Direct specialist-to-specialist transfer
                            current_specialist_voice = "onyx" if current_persona == "psychologist" else "shimmer"

                            # Generate specialist-specific transition message
                            if current_persona == "psychologist" and target_specialist == "dietician":
                                transition_msg = f"For your nutrition concerns, our dietician Doctor Maya would be more appropriate. I'm transferring you directly to them now."
                            elif current_persona == "dietician" and target_specialist == "psychologist":
                                transition_msg = f"For your mental health concerns, our psychologist Doctor Ori would be better suited. I'm transferring you directly to them now."

                            # Update current persona
                            current_persona = target_specialist

                            # Update voice based on new persona
                            if current_persona == "psychologist":
                                voice = "onyx"  # Male voice for psychologist Doctor Ori
                            elif current_persona == "dietician":
                                voice = "shimmer"  # Female voice for dietician Doctor Maya

                            # Send transition message with current specialist's voice
                            await websocket.send_text(json.dumps({
                                "response": transition_msg,
                                "extracted_keywords": ["transition", target_specialist],
                                "audio": generate_audio(transition_msg, current_specialist_voice),
                                "response_id": str(uuid.uuid4())
                            }))

                            # Update system prompt and retriever
                            if current_persona == "psychologist":
                                system_prompt = PSYCHOLOGIST_PROMPT
                                retriever = psychologist_retriever
                            elif current_persona == "dietician":
                                system_prompt = DIETICIAN_PROMPT
                                retriever = dietician_retriever

                            # Generate new response from target specialist with STREAMING
                            response_data = await generate_ai_response_streaming(
                                f"Hello, I'm a patient who was just transferred to you from another specialist, First Introduce Yourself. My original question was: {user_input}",
                                patient_info,
                                chat_history,
                                system_prompt,
                                retriever,
                                websocket,
                                current_persona
                            )

                            # No need to send additional response - streaming already handled it


                            # # Send the specialist response to frontend
                            # audio_future = thread_pool.submit(generate_audio, response_data["response"], voice)
                            # audio_data = audio_future.result()

                            # await websocket.send_text(json.dumps({
                            #     "response": response_data["response"],
                            #     "extracted_keywords": response_data["extracted_keywords"],
                            #     "audio": audio_data,
                            #     "response_id": str(uuid.uuid4()),
                            #     "current_persona": current_persona
                            # }))


                    # Check if specialist wants to refer back to General OPD
                    elif current_persona != "general" and response_data.get("refer_to_general", False):
                        # Determine the specialist voice before changing the persona
                        specialist_voice = "onyx" if current_persona == "psychologist" else "shimmer"

                        # Update current persona to general
                        current_persona = "general"

                        # Generate transition message
                        transition_msg = f"I believe your concerns would be better addressed by our general doctor Tova. I'm transferring you back to them now."
                        await websocket.send_text(json.dumps({
                            "response": transition_msg,
                            "extracted_keywords": ["transition", "general"],
                            "audio": generate_audio(transition_msg, specialist_voice),  # Use the specialist's voice
                            "response_id": str(uuid.uuid4())
                        }))

                        # Update system prompt and retriever
                        system_prompt = GENERAL_OPD_PROMPT
                        retriever = general_retriever

                        # Generate new response from General OPD with STREAMING
                        response_data = await generate_ai_response_streaming(
                            f"Hello, I'm a patient who was just transferred back to you from a specialist, First introduce yourself. My original question was: {user_input}",
                            patient_info,
                            chat_history,
                            system_prompt,
                            retriever,
                            websocket,
                            current_persona
                        )

                        # No need to send additional response - streaming already handled it

                    # Check if specialist suggestion is made
                    if current_persona == "general" and response_data.get("suggested_specialist") not in ["none", None]:
                        suggested = response_data.get("suggested_specialist")
                        if suggested in ["psychologist", "dietician"]:
                            # Automatically transfer to the suggested specialist
                            specialist_names = {
                                "psychologist": "psychologist Doctor Ori",
                                "dietician": "dietician Doctor Maya"
                            }

                            # Update current persona
                            current_persona = suggested

                            # Add transition message to the response
                            response_data["response"] += f"\n\nI'm automatically transferring you to our {specialist_names[suggested]} for more specialized care."

                            # Generate transition message
                            transition_msg = f"I'm transferring you to our {suggested} specialist now."
                            await websocket.send_text(json.dumps({
                                "response": transition_msg,
                                "extracted_keywords": ["transition", suggested],
                                "audio": generate_audio(transition_msg, voice),
                                "response_id": str(uuid.uuid4())
                            }))

                            # Update system prompt and retriever for next response
                            if current_persona == "psychologist":
                                system_prompt = PSYCHOLOGIST_PROMPT
                                retriever = psychologist_retriever
                            elif current_persona == "dietician":
                                system_prompt = DIETICIAN_PROMPT
                                retriever = dietician_retriever

                            # Generate new response from specialist with STREAMING
                            response_data = await generate_ai_response_streaming(
                                f"Hello, I'm a patient who was just transferred to you, First Introduce yourself. My original question was: {user_input}",
                                patient_info,
                                chat_history,
                                system_prompt,
                                retriever,
                                websocket,
                                current_persona
                            )

                            # No need to send additional response - streaming already handled it

                            # Set the appropriate voice for the specialist
                            if current_persona == "psychologist":
                                voice = "onyx"  # Male voice for psychologist Doctor Ori
                            elif current_persona == "dietician":
                                voice = "shimmer"  # Female voice for dietician Doctor Maya

                    # Handle specialist transition request
                    if current_persona == "general" and any(keyword in user_input.lower() for keyword in ["yes", "connect", "specialist", "switch"]):
                        # Look for specialist in previous response
                        for msg in reversed(chat_history[-5:]):
                            if msg["role"] == "assistant" and "suggested_specialist" in msg:
                                suggested = msg.get("suggested_specialist")
                                if suggested in ["psychologist", "dietician"]:
                                    current_persona = suggested

                                    # Generate transition message with more context
                                    specialist_names = {
                                        "psychologist": "psychologist Doctor Ori",
                                        "dietician": "dietician Doctor Maya"
                                    }
                                    transition_msg = f"Based on our conversation, I believe our {specialist_names[suggested]} would be best suited to address your concerns. I'm transferring you to them now. All your information and our conversation history will be shared with them for continuity of care."
                                    await websocket.send_text(json.dumps({
                                        "response": transition_msg,
                                        "extracted_keywords": ["transition", suggested],
                                        "audio": generate_audio(transition_msg, "nova"),  # Use nova (female voice) for the transition message
                                        "response_id": str(uuid.uuid4())
                                    }))

                                    # Update system prompt and retriever for next response
                                    if current_persona == "psychologist":
                                        system_prompt = PSYCHOLOGIST_PROMPT
                                        retriever = psychologist_retriever
                                    elif current_persona == "dietician":
                                        system_prompt = DIETICIAN_PROMPT
                                        retriever = dietician_retriever

                                    # Generate new response from specialist with STREAMING
                                    response_data = await generate_ai_response_streaming(
                                        f"Hello, I'm a patient who was just transferred to you, First Introduce Yourself. My original question was: {user_input}",
                                        patient_info,
                                        chat_history,
                                        system_prompt,
                                        retriever,
                                        websocket,
                                        current_persona
                                    )

                                    # No need to send additional response - streaming already handled it



                    # 🚀 STREAMING COMPLETE: Add AI response to chat history and save to database
                    chat_history.append({"role": "assistant", "content": response_data["response"]})

                    # 🚀 OPTIMIZATION: Save to database in background
                    response_id = str(uuid.uuid4())

                    def save_ai_response_to_database():
                        try:
                            with SessionLocal() as thread_db:
                                now = datetime.now()
                                thread_db.add(ChatMessage(
                                    chat_message_id=response_id,
                                    sender_id=AI_DOCTOR_ID,
                                    receiver_id=patient_id,
                                    message_text=response_data["response"],
                                    extracted_keywords=",".join(response_data["extracted_keywords"]),
                                    createdAt=now,
                                ))
                                thread_db.commit()
                                logger.info("✅ Saved AI response to database")
                                return True
                        except Exception as e:
                            logger.error(f"❌ Error saving AI response: {str(e)}")
                            return False

                    # Save to database in background
                    db_future = thread_pool.submit(save_ai_response_to_database)

                    # Calculate total audio request time
                    total_audio_time = time.time() - audio_request_start
                    logger.info(f"🏁 Total audio streaming request processed in {total_audio_time:.2f}s (Transcription: {transcription_time:.2f}s, AI: {ai_time:.2f}s)")

                    # ✅ STREAMING ONLY - No traditional response needed (streaming already displays complete message)

                except Exception as e:
                    logger.error(f"Error processing audio: {str(e)}")
                    await websocket.send_text(json.dumps({
                        "error": f"Error processing audio: {str(e)}"
                    }))

            # Handle direct specialist selection
            elif "select_specialist" in data:
                specialist = data["select_specialist"]

                # Always use nova voice for the transition message (general doctor's voice)
                voice = "nova"  # Female voice for the transition message

                # We'll set the specialist-specific voice after the transition

                if specialist in ["general", "psychologist", "dietician"]:
                    current_persona = specialist

                    # Generate transition message
                    specialist_names = {
                        "general": "general doctor Tova",
                        "psychologist": "psychologist Doctor Ori",
                        "dietician": "dietician Doctor Maya"
                    }

                    transition_msg = f"You are now speaking with {specialist_names[specialist]}. How can I help you today?"
                    if specialist == "general":
                        voice = "nova"  # Male voice for psychologist Doctor Ori
                    elif specialist == "psychologist":
                        voice = "onyx"  # Male voice for psychologist Doctor Ori
                    elif specialist == "dietician":
                        voice = "shimmer"  # Female voice for dietician Doctor Maya

                    # Generate audio
                    audio_data = generate_audio(transition_msg, voice)

                    # Save transition message to database
                    response_id = str(uuid.uuid4())
                    try:
                        # Set timestamp, createdAt,
                        now = datetime.now()
                        db.add(ChatMessage(
                            chat_message_id=response_id,
                            sender_id=AI_DOCTOR_ID,
                            receiver_id=patient_id,
                            message_text=transition_msg,
                            extracted_keywords="transition," + specialist,
                            # timestamp=now,
                            createdAt=now,  # Required non-null field in the database

                        ))
                        db.commit()
                        logger.info("✅ Saved transition message to database")
                    except Exception as e:
                        db.rollback()
                        logger.error(f"❌ Error saving transition message: {str(e)}")

                    # Update chat history
                    chat_history.append({"role": "assistant", "content": transition_msg})

                    # Send response
                    await websocket.send_text(json.dumps({
                        "response": transition_msg,
                        "extracted_keywords": ["transition", specialist],
                        "audio": audio_data,
                        "response_id": response_id,
                        "current_persona": current_persona
                    }))
                else:
                    await websocket.send_text(json.dumps({
                        "error": f"Invalid specialist: {specialist}"
                    }))

            else:
                await websocket.send_text(json.dumps({
                    "error": "Invalid message format. Expected 'text', 'audio', or 'select_specialist'."
                }))

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for patient {patient_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        try:
            await websocket.send_text(json.dumps({
                "error": f"An error occurred: {str(e)}"
            }))
        except:
            pass


