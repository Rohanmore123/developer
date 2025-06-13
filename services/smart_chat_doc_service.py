import os
import uuid
import json
import sys
import base64
import asyncio
import concurrent.futures
import threading
from io import BytesIO
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
import numpy as np
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Header
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import desc
from database.database import SessionLocal
from model.model_correct import (
    Patient, Doctor, Appointment, DiaryEntry,
    EmotionAnalysis, MedicalHistory, Prescription, ChatMessage
)
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
smart_chat_doc_router = APIRouter()

# AI Doctor ID for chat messages
AI_DOCTOR_ID = "00000000-0000-0000-0000-000000000000"

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

# FAISS index directory
DOCTOR_FAISS_INDEX_DIR = "resources/doctor_faiss_index"

# System prompt for the doctor assistant
SYSTEM_PROMPT = """
You are an AI assistant for Mental doctors, helping them access patient information or providing them with the necessary information about the queries and providing summaries.
Your responses must be:
1. Be professional and factual you can take help of imformation provided to you
2. Keep responses concise and to the point and add necessary context, if necessary then you can increase the length of the response.
3. In plain text format (no markdown, no bullet points, no numbered lists)
4. Focus only on the most relevant medical information
5. Avoid unnecessary explanations or verbose language
6. Use medical terminology appropriate for a doctor
7. You can ask for clarifications if the query is ambiguous

Always maintain patient confidentiality and provide only essential information based on the data provided.
Avoid lengthy explanations and focus on delivering key points in a compact format.
"""

def initialize_faiss_vector_store():
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
        # Read the medical information text (you'll need to create this file)
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

# Initialize the vector store
vector_store = initialize_faiss_vector_store()
retriever = vector_store.as_retriever(search_kwargs={"k": 5}) if vector_store else None

def fetch_doctor_appointments(doctor_id: str, db: Session, days_ahead: int = 7):
    """
    Fetch upcoming appointments for a doctor.

    Args:
        doctor_id: The doctor's ID
        db: Database session
        days_ahead: Number of days to look ahead for appointments

    Returns:
        List of upcoming appointments
    """
    try:
        today = datetime.now().date()
        end_date = today + timedelta(days=days_ahead)

        # Use raw SQL to avoid ORM issues with missing columns
        from sqlalchemy.sql import text

        query = text(f"""
            SELECT
                appointment_id,
                patient_id,
                doctor_id,
                appointment_date,
                appointment_time,
                visit_reason,
                status,
                notes
            FROM
                appointments
            WHERE
                doctor_id = :doctor_id
                AND appointment_date >= :start_date
                AND appointment_date <= :end_date
            ORDER BY
                appointment_date, appointment_time
        """)

        result_proxy = db.execute(
            query,
            {
                "doctor_id": doctor_id,
                "start_date": today,
                "end_date": end_date
            }
        )

        appointments = result_proxy.fetchall()

        result = []
        for appt in appointments:
            try:
                # Get patient info
                patient = db.query(Patient).filter(Patient.patient_id == appt.patient_id).first()
                patient_name = f"{patient.first_name} {patient.last_name}" if patient else "Unknown Patient"

                # Format date and time
                appt_date = appt.appointment_date
                appt_time = appt.appointment_time

                if isinstance(appt_date, str):
                    date_str = appt_date
                else:
                    date_str = appt_date.strftime("%Y-%m-%d")

                if isinstance(appt_time, str):
                    time_str = appt_time
                else:
                    time_str = appt_time.strftime("%H:%M")

                result.append({
                    "appointment_id": str(appt.appointment_id),
                    "patient_id": str(appt.patient_id),
                    "patient_name": patient_name,
                    "date": date_str,
                    "time": time_str,
                    "reason": appt.visit_reason if appt.visit_reason else "General checkup",
                    "status": appt.status if appt.status else "Scheduled"
                })
            except Exception as inner_e:
                logger.error(f"Error processing appointment: {str(inner_e)}")
                # Continue with next appointment
                continue

        return result
    except Exception as e:
        logger.error(f"Error fetching doctor appointments: {str(e)}")
        # Return empty list but don't break the transaction
        return []

def fetch_patient_info(patient_id: str, db: Session):
    """
    Fetch comprehensive patient information.

    Args:
        patient_id: The patient's ID
        db: Database session

    Returns:
        Dictionary with patient information
    """
    try:
        patient = db.query(Patient).filter(Patient.patient_id == patient_id).first()
        if not patient:
            return None

        # Handle treatment field which might be JSONB in the database
        treatment_info = patient.treatment
        if isinstance(treatment_info, dict):
            treatment_str = json.dumps(treatment_info)
        else:
            treatment_str = str(treatment_info)

        return {
            "patient_id": str(patient.patient_id),
            "name": f"{patient.first_name} {patient.last_name}",
            "age": patient.age,
            "gender": patient.gender,
            "dob": patient.dob.strftime("%Y-%m-%d") if patient.dob else None,
            "language": patient.language,
            "address": patient.address,
            "phone": patient.phone,
            "treatment": treatment_str,
            "health_score": patient.health_score,
            "under_medications": patient.under_medications
        }
    except Exception as e:
        logger.error(f"Error fetching patient info: {str(e)}")
        return None

def fetch_medical_history(patient_id: str, db: Session):
    """
    Fetch medical history for a patient.

    Args:
        patient_id: The patient's ID
        db: Database session

    Returns:
        List of medical history records
    """
    try:
        history = db.query(MedicalHistory).filter(MedicalHistory.patient_id == patient_id).all()
        return [
            {
                "diagnosis": item.diagnosis,
                "treatment": item.treatment,
                "date": item.diagnosed_date.strftime("%Y-%m-%d") if item.diagnosed_date else None,
                "notes": item.additional_notes
            }
            for item in history
        ] if history else []
    except Exception as e:
        logger.error(f"Error fetching medical history: {str(e)}")
        return []

def fetch_prescriptions(patient_id: str, db: Session):
    """
    Fetch prescriptions for a patient.

    Args:
        patient_id: The patient's ID
        db: Database session

    Returns:
        List of prescriptions
    """
    try:
        prescriptions = db.query(Prescription).filter(Prescription.patient_id == patient_id).all()
        return [
            {
                "medication": item.medication_name,
                "dosage": item.dosage,
                "instructions": item.instructions,
                "start_date": item.start_date.strftime("%Y-%m-%d") if item.start_date else None,
                "end_date": item.end_date.strftime("%Y-%m-%d") if item.end_date else None,
                "status": item.status
            }
            for item in prescriptions
        ] if prescriptions else []
    except Exception as e:
        logger.error(f"Error fetching prescriptions: {str(e)}")
        return []

def fetch_emotion_analysis(patient_id: str, db: Session, days_back: int = 30):
    """
    Fetch emotion analysis for a patient.

    Args:
        patient_id: The patient's ID
        db: Database session
        days_back: Number of days to look back

    Returns:
        Dictionary with emotion analysis summary
    """
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
        dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0]

        # Calculate average sentiment (if available)
        avg_sentiment = 0
        if hasattr(emotions[0], 'confidence_score'):
            total_sentiment = sum(e.confidence_score for e in emotions if e.confidence_score is not None)
            avg_sentiment = round(total_sentiment / len(emotions), 2)

        return {
            "dominant_emotion": dominant_emotion,
            "emotion_distribution": emotion_counts,
            "average_sentiment": avg_sentiment,
            "days_analyzed": days_back,
            "total_messages_analyzed": len(emotions)
        }
    except Exception as e:
        logger.error(f"Error fetching emotion analysis: {str(e)}")
        return {
            "dominant_emotion": "Error",
            "emotion_distribution": {},
            "days_analyzed": days_back,
            "total_messages_analyzed": 0,
            "error": str(e)
        }

def fetch_diary_entries(patient_id: str, db: Session, limit: int = 5):
    """
    Fetch recent diary entries for a patient.

    Args:
        patient_id: The patient's ID
        db: Database session
        limit: Maximum number of entries to return

    Returns:
        List of diary entries
    """
    try:
        entries = db.query(DiaryEntry).filter(
            DiaryEntry.patient_id == patient_id
        ).order_by(desc(DiaryEntry.created_at)).limit(limit).all()

        return [
            {
                "event_id": entry.event_id,
                "notes": entry.notes,
                "created_at": entry.created_at.strftime("%Y-%m-%d %H:%M:%S") if entry.created_at else None
            }
            for entry in entries
        ] if entries else []
    except Exception as e:
        logger.error(f"Error fetching diary entries: {str(e)}")
        return []

def fetch_upcoming_appointments(patient_id: str, db: Session):
    """
    Fetch upcoming appointments for a patient.

    Args:
        patient_id: The patient's ID
        db: Database session

    Returns:
        List of upcoming appointments
    """
    try:
        today = datetime.now().date()

        appointments = db.query(Appointment).filter(
            Appointment.patient_id == patient_id,
            Appointment.appointment_date >= today
        ).order_by(Appointment.appointment_date, Appointment.appointment_time).all()

        result = []
        for appt in appointments:
            doctor = db.query(Doctor).filter(Doctor.doctor_id == appt.doctor_id).first()
            doctor_name = f"{doctor.first_name} {doctor.last_name}" if doctor else "Unknown Doctor"

            result.append({
                "appointment_id": str(appt.appointment_id),
                "doctor_name": doctor_name,
                "date": appt.appointment_date.strftime("%Y-%m-%d"),
                "time": appt.appointment_time.strftime("%H:%M"),
                "reason": appt.visit_reason,
                "status": appt.status
            })

        return result
    except Exception as e:
        logger.error(f"Error fetching upcoming appointments: {str(e)}")
        return []

def fetch_chat_history(doctor_id: str, db: Session, limit: int = 20):
    """
    Fetch chat history between doctor and AI assistant.

    Args:
        doctor_id: The doctor's ID
        db: Database session
        limit: Maximum number of messages to return

    Returns:
        List of chat messages between doctor and AI
    """
    try:
        # Fetch messages between doctor and AI_DOCTOR_ID only
        messages = db.query(ChatMessage).filter(
            ((ChatMessage.sender_id == doctor_id) & (ChatMessage.receiver_id == AI_DOCTOR_ID)) |
            ((ChatMessage.sender_id == AI_DOCTOR_ID) & (ChatMessage.receiver_id == doctor_id))
        ).order_by(desc(ChatMessage.createdAt)).limit(limit).all()

        # Reverse to get chronological order
        messages.reverse()

        # Format messages for chat history
        chat_history = []
        for msg in messages:
            role = "doctor" if msg.sender_id == doctor_id else "assistant"
            chat_history.append({
                "role": role,
                "message": msg.message_text,
                "timestamp": msg.createdAt.strftime("%Y-%m-%d %H:%M:%S") if msg.createdAt else "Unknown"
            })

        logger.info(f"Retrieved {len(chat_history)} chat messages for doctor {doctor_id}")
        return chat_history
    except Exception as e:
        logger.error(f"Error fetching chat history for doctor {doctor_id}: {str(e)}")
        return []

def save_chat_message(sender_id: str, receiver_id: str, message_text: str, db: Session):
    """
    Save a chat message to the database.

    Args:
        sender_id: ID of the message sender
        receiver_id: ID of the message receiver
        message_text: The message content
        db: Database session

    Returns:
        True if successful, False otherwise
    """
    try:
        chat_message = ChatMessage(
            chat_message_id=str(uuid.uuid4()),
            sender_id=sender_id,
            receiver_id=receiver_id,
            message_text=message_text,
            extracted_keywords=None,
            createdAt=datetime.now(),
            # updatedAt=datetime.now()
        )

        db.add(chat_message)
        db.commit()
        logger.info(f"✅ Saved chat message from {sender_id} to {receiver_id}")
        return True
    except Exception as e:
        logger.error(f"❌ Error saving chat message: {str(e)}")
        db.rollback()
        return False

# Removed extract_patient_id function - patient ID will be provided as optional input

def generate_patient_summary(patient_id: str, db: Session):
    """
    Generate a comprehensive summary for a patient using OpenAI.

    Args:
        patient_id: The patient's ID
        db: Database session

    Returns:
        A summary of the patient's information
    """
    if not OPENAI_CLIENT:
        return "OpenAI client not initialized. Cannot generate patient summary."

    try:
        # Gather all relevant patient data
        patient_info = fetch_patient_info(patient_id, db)
        if not patient_info:
            return f"No patient found with ID: {patient_id}"

        medical_history = fetch_medical_history(patient_id, db)
        prescriptions = fetch_prescriptions(patient_id, db)
        emotion_analysis = fetch_emotion_analysis(patient_id, db)
        upcoming_appointments = fetch_upcoming_appointments(patient_id, db)
        diary_entries = fetch_diary_entries(patient_id, db)

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

        ## Emotional State
        Dominant Emotion: {emotion_analysis['dominant_emotion']}
        Emotion Distribution: {json.dumps(emotion_analysis['emotion_distribution'])}

        ## Upcoming Appointments
        {json.dumps(upcoming_appointments, indent=2) if upcoming_appointments else "No upcoming appointments."}

        ## Recent Diary Entries
        {json.dumps(diary_entries, indent=2) if diary_entries else "No recent diary entries."}
        """

        # Generate the summary using OpenAI
        prompt = f"""
        Create an extremely concise medical summary (5-7 lines max) for a doctor about this patient.
        Focus ONLY on critical medical information, current treatment, and emotional state.
        Use plain text format with no formatting, bullet points, or unnecessary details.
        Be direct and use medical terminology appropriate for a doctor.

        {context}
        """

        response = OPENAI_CLIENT.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            max_tokens=250,
            temperature=0.2
        )

        summary = response.choices[0].message.content.strip()
        return summary

    except Exception as e:
        logger.error(f"Error generating patient summary: {str(e)}")
        return f"Error generating patient summary: {str(e)}"

def retrieve_relevant_information(query: str):
    """
    Retrieve relevant information from the FAISS vector store.

    Args:
        query: The search query

    Returns:
        List of relevant documents
    """
    if not retriever:
        return []

    try:
        docs = retriever.get_relevant_documents(query)
        return [doc.page_content for doc in docs]
    except Exception as e:
        logger.error(f"Error retrieving information: {str(e)}")
        return []

def process_doctor_query(query: str, doctor_id: str, chat_history=None, patient_id=None):
    """
    Process a query from a doctor and generate a response.

    Args:
        query: The doctor's query
        doctor_id: The doctor's ID
        chat_history: List of previous chat messages for context
        patient_id: Optional patient ID to include patient information in context

    Returns:
        Response to the doctor's query
    """
    if not OPENAI_CLIENT:
        return "OpenAI client not initialized. Cannot process query."

    try:
        # Create a new database session to avoid transaction issues
        from database.database import SessionLocal
        new_db = SessionLocal()

        try:
            # Initialize context
            context = ""

            # Add chat history context if available
            if chat_history and len(chat_history) > 0:
                context += "\n## Recent Conversation History\n"
                # Get last 10 exchanges (20 messages) for context
                recent_history = chat_history[-20:] if len(chat_history) > 20 else chat_history
                logger.info(f"Including {len(recent_history)} chat history messages in context")
                for msg in recent_history:
                    role = "Doctor" if msg["role"] == "doctor" else "AI Assistant"
                    context += f"{role}: {msg['message']}\n"
                context += "\n"
            else:
                logger.info("No chat history available for context")

            # If patient ID is provided, generate a patient summary
            if patient_id:
                logger.info(f"Patient ID provided: {patient_id}")
                patient_summary = generate_patient_summary(patient_id, new_db)
                context += f"\n## Patient Summary\n{patient_summary}\n\n"
                logger.info(f"Patient summary added to context (length: {len(patient_summary)})")
            else:
                logger.info("No patient ID provided")

            # Retrieve relevant information from FAISS
            relevant_info = retrieve_relevant_information(query)
            if relevant_info:
                logger.info(f"Retrieved {len(relevant_info)} relevant documents from FAISS")
                context += "\n## Relevant Medical Information\n"
                for i, info in enumerate(relevant_info, 1):
                    context += f"{i}. {info}\n"
            else:
                logger.info("No relevant information retrieved from FAISS")

            # Fetch doctor's upcoming appointments
            appointments = fetch_doctor_appointments(doctor_id, new_db)
            if "appointment" in query.lower() and appointments:
                logger.info(f"Retrieved {len(appointments)} appointments for doctor")
                context += "\n## Your Upcoming Appointments\n"
                context += json.dumps(appointments, indent=2)
            else:
                logger.info("No appointments retrieved or query not about appointments")

            # Log the total context being sent to LLM
            logger.info(f"Total context length: {len(context)} characters")
            logger.info(f"Context preview: {context[:200]}..." if len(context) > 200 else f"Full context: {context}")

            # Generate response using OpenAI
            prompt = f"""
            Doctor's Query: {query}

            {context}

            Based on the conversation history and available context, provide a direct, concise response (5-7 lines max) to this doctor's query.
            Use plain text only - no markdown, no bullet points, no numbered lists.
            Consider the previous conversation when formulating your response to maintain continuity.
            If about a patient, include only the most critical clinical information.
            If about appointments, provide only essential details in a compact format.
            If a general medical question, give only key facts without elaboration.
            If this is a follow-up question, reference the previous discussion appropriately.
            """

            logger.info(f"Sending prompt to LLM (total length: {len(prompt)} characters)")

            response = OPENAI_CLIENT.chat.completions.create(
                model="gpt-4o-2024-11-20",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=250,
                temperature=0.2
            )

            logger.info(f"Received response from LLM: {response.choices[0].message.content[:100]}...")

            return response.choices[0].message.content.strip()
        finally:
            # Always close the new database session
            new_db.close()

    except Exception as e:
        logger.error(f"Error processing doctor query: {str(e)}")

        # Provide a more helpful error message based on the type of error
        if "UndefinedColumn" in str(e) or "InFailedSqlTransaction" in str(e):
            return "I'm having trouble accessing the database at the moment. This might be due to a schema mismatch. Please try a different query or contact technical support if this persists."

        # For general medical queries that don't require database access, try to provide a response
        if "appointment" not in query.lower() and not patient_id:
            try:
                # Attempt to answer general medical questions even if database fails
                prompt = f"""
                Doctor's Query: {query}

                Provide a very concise (5-7 lines max) response to this general medical question.
                Use plain text only with no formatting or bullet points.
                Include only essential medical facts without elaboration.
                Focus on clinical relevance for a medical professional.
                """

                response = OPENAI_CLIENT.chat.completions.create(
                    model="gpt-4o-2024-11-20",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=250,
                    temperature=0.2
                )

                return response.choices[0].message.content.strip()
            except Exception as inner_e:
                logger.error(f"Error in fallback response: {str(inner_e)}")

        return f"I encountered an error while processing your query. Please try again or rephrase your question."

@smart_chat_doc_router.websocket("/smart-chat-doc/{doctor_id}")
async def smart_chat_for_doctors(websocket: WebSocket, doctor_id: str):
    """
    WebSocket endpoint for smart chat with doctors.

    Args:
        websocket: WebSocket connection
        doctor_id: The doctor's ID
    """
    logger.info(f"=== NEW DOCTOR WEBSOCKET CONNECTION REQUEST ===")
    logger.info(f"Doctor ID: {doctor_id}")
    # logger.info(f"Query parameters: {websocket.query_params}")
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
    from os import getenv
    from dotenv import load_dotenv

    # Load environment variables
    load_dotenv()

    # Get static JWT token
    STATIC_JWT_TOKEN = getenv("STATIC_JWT_TOKEN")

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

    # Validate doctor exists
    try:
        # Create a new session to avoid transaction issues
        new_db = SessionLocal()

        try:
            doctor = new_db.query(Doctor).filter(Doctor.doctor_id == doctor_id).first()
            if not doctor:
                logger.info(f"Doctor not found in WebSocket: {doctor_id}")
                await websocket.send_text(json.dumps({
                    "response": "Error: Doctor not found.",
                    "error": "invalid_doctor_id"
                }))
                await websocket.close()
                new_db.close()
                return
            logger.info(f"Doctor validated in WebSocket: {doctor.first_name} {doctor.last_name}")
        finally:
            new_db.close()
    except Exception as e:
        logger.error(f"Error validating doctor: {str(e)}")
        await websocket.send_text(json.dumps({
            "response": "Error: Could not validate doctor.",
            "error": "validation_error"
        }))
        await websocket.close()
        return

    # Initialize chat history by loading from database
    chat_history = []
    try:
        # Load existing chat history from database
        db_session = SessionLocal()
        try:
            chat_history = fetch_chat_history(doctor_id, db_session, limit=20)
            logger.info(f"Loaded {len(chat_history)} existing chat messages for doctor {doctor_id}")
        finally:
            db_session.close()
    except Exception as e:
        logger.error(f"Error loading chat history: {str(e)}")
        chat_history = []

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            data = json.loads(data)

            # Get optional patient_id from the message
            patient_id = data.get("patient_id", None)
            if patient_id:
                logger.info(f"Patient ID received in message: {patient_id}")

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

                    # Send transcription back to client before processing
                    await websocket.send_text(json.dumps({
                        "transcription": user_input
                    }))

                    # Log the transcription
                    logger.info(f"Audio transcription for doctor {doctor_id}: {user_input}")
                except Exception as e:
                    logger.error(f"Error processing audio: {str(e)}")
                    await websocket.send_text(json.dumps({
                        "response": f"Error processing audio: {str(e)}",
                        "error": "audio_processing_error"
                    }))
                    continue
            else:
                user_input = data.get("text", "").strip()

            if not user_input:
                await websocket.send_text(json.dumps({
                    "response": "Please enter a message.",
                    "error": "empty_input"
                }))
                continue

            logger.info(f"Received message from doctor {doctor_id}: {user_input}")

            # Save doctor's message to database
            db_session = SessionLocal()
            try:
                save_chat_message(doctor_id, AI_DOCTOR_ID, user_input, db_session)
            except Exception as e:
                logger.error(f"Error saving doctor message: {str(e)}")
            finally:
                db_session.close()

            # Process the query with threading for better performance
            # Pass the chat history and optional patient_id for context
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(process_doctor_query, user_input, doctor_id, chat_history, patient_id)
                response = future.result()

            # Save AI response to database
            db_session = SessionLocal()
            try:
                save_chat_message(AI_DOCTOR_ID, doctor_id, response, db_session)
            except Exception as e:
                logger.error(f"Error saving AI response: {str(e)}")
            finally:
                db_session.close()

            # Update in-memory chat history
            chat_history.append({"role": "doctor", "message": user_input, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
            chat_history.append({"role": "assistant", "message": response, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})

            # Keep chat history limited to last 10 exchanges (20 messages)
            if len(chat_history) > 20:
                chat_history = chat_history[-20:]

            # Send response to client
            await websocket.send_text(json.dumps({
                "response": response
            }))

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for doctor: {doctor_id}")
    except Exception as e:
        logger.error(f"Error in doctor WebSocket: {str(e)}")
        try:
            await websocket.send_text(json.dumps({
                "response": "An error occurred. Please try again.",
                "error": str(e)
            }))
        except:
            pass
