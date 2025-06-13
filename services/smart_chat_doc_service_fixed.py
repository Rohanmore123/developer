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
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException, Header
from fastapi.responses import JSONResponse
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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
smart_chat_doc_router = APIRouter()

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
You are an AI assistant for doctors, helping them access patient information and providing summaries.
Your responses should be professional, concise, and focused on providing relevant medical information.
Always maintain patient confidentiality and provide factual information based on the data provided.
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

        appointments = db.query(Appointment).filter(
            Appointment.doctor_id == doctor_id,
            Appointment.appointment_date >= today,
            Appointment.appointment_date <= end_date
        ).order_by(Appointment.appointment_date, Appointment.appointment_time).all()

        result = []
        for appt in appointments:
            patient = db.query(Patient).filter(Patient.patient_id == appt.patient_id).first()
            patient_name = f"{patient.first_name} {patient.last_name}" if patient else "Unknown Patient"

            result.append({
                "appointment_id": str(appt.appointment_id),
                "patient_id": str(appt.patient_id),
                "patient_name": patient_name,
                "date": appt.appointment_date.strftime("%Y-%m-%d"),
                "time": appt.appointment_time.strftime("%H:%M"),
                "reason": appt.visit_reason,
                "status": appt.status
            })

        return result
    except Exception as e:
        logger.error(f"Error fetching doctor appointments: {str(e)}")
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

def extract_patient_id(text: str):
    """
    Extract patient ID from text if present.
    This is a simple implementation that looks for UUIDs.

    Args:
        text: The input text

    Returns:
        Extracted patient ID or None
    """
    import re
    # UUID pattern
    uuid_pattern = r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'
    matches = re.findall(uuid_pattern, text.lower())

    return matches[0] if matches else None

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
        Create a concise medical summary for a doctor about the following patient.
        Focus on the most important medical information, current treatment, emotional state, and upcoming appointments.
        Keep the summary professional, factual, and to the point.

        {context}
        """

        response = OPENAI_CLIENT.chat.completions.create(
            model="gpt-3.5-turbo",  # Using a more widely available model
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

def process_doctor_query(query: str, doctor_id: str, db: Session, chat_history=None, patient_id=None, relevant_info=None):
    """
    Process a query from a doctor and generate a response.

    Args:
        query: The doctor's query
        doctor_id: The doctor's ID
        db: Database session
        chat_history: Previous chat messages (optional)
        patient_id: Patient ID if detected in the query (optional)
        relevant_info: Pre-retrieved relevant information (optional)

    Returns:
        Response to the doctor's query
    """
    if not OPENAI_CLIENT:
        return "OpenAI client not initialized. Cannot process query."

    try:
        # If patient_id is not provided, try to extract it from the query
        if patient_id is None:
            patient_id = extract_patient_id(query)
            logger.info(f"Extracted patient ID from query: {patient_id}")

        # Initialize context
        context = ""

        # If patient ID is found, generate a patient summary
        if patient_id:
            patient_summary = generate_patient_summary(patient_id, db)
            context += f"\n## Patient Summary\n{patient_summary}\n\n"

        # Use pre-retrieved information if provided, otherwise retrieve it now
        if relevant_info is None:
            relevant_info = retrieve_relevant_information(query)
            logger.info(f"Retrieved {len(relevant_info)} relevant documents")

        if relevant_info:
            context += "\n## Relevant Medical Information\n"
            for i, info in enumerate(relevant_info, 1):
                context += f"{i}. {info}\n"

        # Fetch doctor's upcoming appointments
        appointments = fetch_doctor_appointments(doctor_id, db)
        if appointments:
            # Always include appointments if available, but highlight them if the query is about appointments
            if "appointment" in query.lower():
                context += "\n## Your Upcoming Appointments (You asked about these)\n"
            else:
                context += "\n## Your Upcoming Appointments\n"
            context += json.dumps(appointments, indent=2)

        # Prepare messages for OpenAI
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

        # Add chat history if available
        if chat_history and len(chat_history) > 0:
            # Add a summary of the conversation context
            context += "\n## Previous Conversation\n"

            # Add up to 5 most recent exchanges to the context for reference
            recent_history = chat_history[-10:] if len(chat_history) > 10 else chat_history
            for msg in recent_history:
                if msg["role"] == "doctor":
                    context += f"Doctor: {msg['message']}\n"
                    # Also add to messages for OpenAI
                    messages.append({"role": "user", "content": msg["message"]})
                else:
                    context += f"Assistant: {msg['message']}\n"
                    # Also add to messages for OpenAI
                    messages.append({"role": "assistant", "content": msg["message"]})

        # Add the current query with context
        current_prompt = f"""
        Doctor's Query: {query}

        {context}

        Please provide a helpful, professional response to the doctor's query.
        If the query is about a specific patient and patient information is available, include relevant details.
        If the query is about appointments, provide appointment information.
        If the query is a general medical question, provide relevant medical information.
        """

        # Add the current query to messages
        messages.append({"role": "user", "content": current_prompt})

        # Generate response using OpenAI
        response = OPENAI_CLIENT.chat.completions.create(
            model="gpt-3.5-turbo",  # Using a more widely available model
            messages=messages,
            max_tokens=500,
            temperature=0.3
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        logger.error(f"Error processing doctor query: {str(e)}")
        return f"Error processing your query: {str(e)}"

def create_conversation_summary(chat_history, current_question):
    """
    Create a summary of the conversation to improve context retrieval for follow-up questions.

    Args:
        chat_history: List of previous messages in the conversation
        current_question: The current question from the doctor

    Returns:
        A string containing a summary of the conversation that can be used for retrieval
    """
    if not chat_history:
        return current_question

    # If we have OpenAI client, use it to generate a better summary
    if OPENAI_CLIENT:
        try:
            # Format the chat history
            formatted_history = ""
            for i, msg in enumerate(chat_history[-6:]):  # Use last 6 messages at most
                role = "Doctor" if msg.get("role") == "doctor" else "AI"
                formatted_history += f"{role}: {msg.get('message', '')}\n"

            # Create the prompt for summarization
            prompt = f"""
            I need to summarize a conversation to improve context retrieval for a follow-up question.

            Previous conversation:
            {formatted_history}

            Current question: "{current_question}"

            Create a comprehensive search query that combines the main topics from the conversation with the current question.
            Focus on medical terms and concepts that would help retrieve relevant information.
            The query should be self-contained and understandable without the conversation history.

            Query:
            """

            # Generate the summary
            response = OPENAI_CLIENT.chat.completions.create(
                model="gpt-3.5-turbo",  # Using a smaller model for efficiency
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that creates search queries based on conversation context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.3
            )

            summary = response.choices[0].message.content.strip()
            return summary
        except Exception as e:
            logger.error(f"Error generating conversation summary: {str(e)}")
            # Fall back to simple concatenation

    # Simple fallback: combine the last message with the current question
    if len(chat_history) > 0:
        last_message = chat_history[-1].get("message", "")
        return f"{last_message} {current_question}"

    return current_question

@smart_chat_doc_router.websocket("/smart-chat-doc/{doctor_id}")
async def smart_chat_for_doctors(websocket: WebSocket, doctor_id: str, db: Session = Depends(get_db)):
    """
    WebSocket endpoint for smart chat with doctors.

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

    # Initialize chat history
    chat_history = []

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            data = json.loads(data)

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

            # Debug chat history
            logger.info(f"Current chat history ({len(chat_history)} messages):")
            for i, msg in enumerate(chat_history[-4:]):  # Show last 4 messages for brevity
                logger.info(f"  {i+1}. {msg.get('role')}: {msg.get('message')[:50]}...")

            # Process context and generate response with threading for better performance
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                # Task 1: Create conversation summary and retrieve relevant documents
                def process_context():
                    # Create a conversation summary for better context retrieval
                    conversation_summary = create_conversation_summary(chat_history, user_input)
                    logger.info(f"Created conversation summary: {conversation_summary}")

                    # Use the summary for retrieval if it's a follow-up question, otherwise use the original input
                    retrieval_query = conversation_summary if len(chat_history) > 0 else user_input
                    logger.info(f"Using retrieval query: {retrieval_query}")

                    # Get relevant documents
                    relevant_info = retrieve_relevant_information(retrieval_query)
                    logger.info(f"Retrieved {len(relevant_info)} relevant documents")

                    return relevant_info

                # Task 2: Check if the query contains a patient ID
                def check_patient_id():
                    patient_id = extract_patient_id(user_input)
                    if patient_id:
                        logger.info(f"Detected patient ID in query: {patient_id}")
                        return patient_id
                    return None

                # Execute both tasks in parallel
                relevant_info_future = executor.submit(process_context)
                patient_id_future = executor.submit(check_patient_id)

                # Get results
                relevant_info = relevant_info_future.result()
                patient_id = patient_id_future.result()

            # Now process the doctor query with all the context
            response = process_doctor_query(
                query=user_input,
                doctor_id=doctor_id,
                db=db,
                chat_history=chat_history,
                patient_id=patient_id,
                relevant_info=relevant_info
            )

            # Update chat history
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
