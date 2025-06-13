import os
import uuid
import json
import base64
import asyncio
import concurrent.futures
from datetime import datetime
from dotenv import load_dotenv
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Request
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import desc
from database.database import SessionLocal
from database.database import get_db
from model.model_correct import (
    ChatMessage, Patient, OnboardingQuestion, DiaryEntry,
    EmotionAnalysis, MedicalHistory, Prescription
)
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from services.smart_agent import SmartAgent
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
ai_talk_router = APIRouter()

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

# AI Therapist ID (fixed ID for the AI therapist)
AI_THERAPIST_ID = "00000000-0000-0000-0000-000000000001"

# HNSW index name
HNSW_INDEX_DIR = "resources/hnsw_index"

def initialize_hnsw_vector_store():
    """Initialize the FAISS HNSW vector store with mental health information."""
    try:
        # Check if the HNSW index already exists
        if os.path.exists(HNSW_INDEX_DIR):
            embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
            vector_store = FAISS.load_local(
                HNSW_INDEX_DIR,
                embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info("✅ Loaded existing FAISS HNSW index")
            return vector_store

        # Read the mental health information text
        with open("resources/mental_health_info.txt", "r", encoding="utf-8") as f:
            mental_health_text = f.read()

        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_text(mental_health_text)

        # Create embeddings and vector store
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vector_store = FAISS.from_texts(chunks, embeddings)

        # Save the vector store
        os.makedirs(HNSW_INDEX_DIR, exist_ok=True)
        vector_store.save_local(HNSW_INDEX_DIR)
        logger.info("✅ Created and saved new FAISS HNSW index")
        return vector_store
    except Exception as e:
        logger.error(f"❌ Error initializing HNSW vector store: {str(e)}")
        # Return a minimal vector store with a placeholder text if there's an error
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        return FAISS.from_texts(["Mental health information placeholder."], embeddings)

# Initialize the vector store
vector_store = initialize_hnsw_vector_store()

# Create a retriever
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5},
    threshold=0.5
)

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

def fetch_chat_history(patient_id: str, db: Session):
    """Fetch chat history for a patient with the AI therapist."""
    try:
        messages = db.query(ChatMessage).filter(
            ((ChatMessage.sender_id == patient_id) & (ChatMessage.receiver_id == AI_THERAPIST_ID)) |
            ((ChatMessage.sender_id == AI_THERAPIST_ID) & (ChatMessage.receiver_id == patient_id))
        ).order_by(ChatMessage.timestamp.desc()).limit(10).all()

        chat_history = []
        for msg in messages:
            role = "patient" if msg.sender_id == patient_id else "therapist"
            chat_history.append({
                "role": role,
                "message": msg.message_text,
                "timestamp": msg.timestamp.strftime("%Y-%m-%d %H:%M:%S") if msg.timestamp else "Unknown"
            })

        # Reverse to get chronological order
        chat_history.reverse()
        return chat_history
    except Exception as e:
        logger.error(f"Error fetching chat history: {str(e)}")
        return []

def generate_audio_from_text(text: str, voice: str = "nova"):
    """Generate audio from text using OpenAI's text-to-speech API.

    Args:
        text (str): The text to convert to speech
        voice (str): The voice to use. Options: alloy, echo, fable, onyx, nova, shimmer

    Returns:
        str: Base64 encoded audio data or None if generation failed
    """
    if not OPENAI_CLIENT:
        logger.error("OpenAI client not initialized. Cannot generate audio.")
        return None

    # Validate voice parameter
    valid_voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
    if voice not in valid_voices:
        logger.warning(f"Invalid voice: {voice}. Using default voice 'nova'.")
        voice = "nova"

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

def generate_therapist_response(patient_id: str, user_input: str, chat_history: list, db: Session):
    """Generate a response from the AI therapist."""
    try:
        # Get patient information
        patient = db.query(Patient).filter(Patient.patient_id == patient_id).first()
        patient_name = f"{patient.first_name} {patient.last_name}" if patient else "Unknown Patient"
        
        # Create a conversation summary for better context
        conversation_context = "\n".join([f"{msg['role']}: {msg['message']}" for msg in chat_history[-5:]])
        
        # Retrieve relevant mental health information
        relevant_info = retrieve_relevant_information(user_input)
        mental_health_context = "\n\n".join(relevant_info)
        
        # Create the system prompt
        system_prompt = f"""
        You are a compassionate psychiatric therapist having a conversation with {patient_name}. 
        Your responses should be warm, empathetic, and conversational - as if you're speaking to them face-to-face.
        
        IMPORTANT GUIDELINES:
        1. Keep your responses brief (2-3 sentences maximum) to maintain a natural conversation flow
        2. DO NOT diagnose or name any specific mental health disorders
        3. Focus on being supportive and understanding
        4. Use a conversational tone - speak as if you're in the room with them
        5. Respond directly to what they're saying without clinical language
        6. If they express distress, acknowledge their feelings and offer gentle support
        
        Your goal is to make the patient feel heard and understood in a natural conversation.
        """
        
        # Create the messages for the API call
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Add chat history
        for msg in chat_history[-5:]:
            role = "user" if msg["role"] == "patient" else "assistant"
            messages.append({"role": role, "content": msg["message"]})
        
        # Add the current user input
        messages.append({"role": "user", "content": user_input})
        
        # Generate the response
        response = OPENAI_CLIENT.chat.completions.create(
            model="gpt-4",
            messages=messages,
            max_tokens=150,
            temperature=0.7
        )
        
        message_text = response.choices[0].message.content.strip()
        
        # Extract keywords for subtitles/captions
        keyword_prompt = f"Extract 3-5 key emotional or therapeutic concepts from this text as a comma-separated list: {message_text}"
        keyword_response = OPENAI_CLIENT.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You extract key emotional and therapeutic concepts from text."},
                {"role": "user", "content": keyword_prompt}
            ],
            max_tokens=50,
            temperature=0.1
        )
        
        keywords = keyword_response.choices[0].message.content.strip().split(',')
        keywords = [k.strip() for k in keywords]
        
        return {
            "response": message_text,
            "extracted_keywords": keywords
        }
    except Exception as e:
        logger.error(f"Error generating therapist response: {str(e)}")
        return {
            "response": "I'm sorry, I'm having trouble processing your request right now. Could we try again?",
            "extracted_keywords": ["technical difficulty", "apology", "retry"]
        }

# WebSocket endpoint for AI Talk
@ai_talk_router.websocket("/ai-talk/{patient_id}")
async def ai_talk_websocket(websocket: WebSocket, patient_id: str, db: Session = Depends(get_db)):
    logger.info(f"\n\n=== NEW AI TALK WEBSOCKET CONNECTION REQUEST ===")
    logger.info(f"Patient ID: {patient_id}")
    logger.info(f"Query parameters: {websocket.query_params}")
    logger.info(f"Headers: {websocket.headers}")
    
    # Accept the connection first
    await websocket.accept()
    logger.info(f"WebSocket connection accepted for patient: {patient_id}")

    # Get JWT token from query parameters
    token = None
    if "token" in websocket.query_params:
        token = websocket.query_params["token"]
        logger.info(f"Found token in query parameters")

    # Check if we have a token
    if not token:
        logger.warning(f"No token provided for patient: {patient_id}")
        await websocket.send_text(json.dumps({
            "response": "Error: Authentication required. Please provide a valid JWT token as a query parameter (token=...).",
            "extracted_keywords": []
        }))
        await websocket.close()
        return

    # Initialize chat history
    chat_history = fetch_chat_history(patient_id, db)
    logger.info(f"Fetched chat history for patient: {patient_id}. {len(chat_history)} messages found.")

    # Send welcome message
    welcome_message = "Hello! I'm Dr. Nova, your virtual therapist. How are you feeling today?"
    welcome_audio = generate_audio_from_text(welcome_message, "nova")
    
    # Save welcome message to database
    try:
        db.add(ChatMessage(
            chat_message_id=str(uuid.uuid4()),
            sender_id=AI_THERAPIST_ID,
            receiver_id=patient_id,
            message_text=welcome_message,
            timestamp=datetime.now()
        ))
        db.commit()
        logger.info(f"✅ Saved welcome message to database for patient {patient_id}")
    except Exception as e:
        db.rollback()
        logger.error(f"❌ Error saving welcome message to database: {str(e)}")
    
    # Add welcome message to chat history
    chat_history.append({
        "role": "therapist",
        "message": welcome_message,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    
    # Send welcome message to client
    await websocket.send_text(json.dumps({
        "response": welcome_message,
        "audio": welcome_audio,
        "extracted_keywords": ["greeting", "introduction", "inquiry"],
        "response_id": str(uuid.uuid4())
    }))

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            data = json.loads(data)
            logger.info(f"Received message from patient {patient_id}: {data}")

            # Extract user input from text or audio
            user_input = ""

            # Handle audio input
            if "audio" in data:
                try:
                    if not OPENAI_CLIENT:
                        raise Exception("OpenAI client not initialized")

                    audio_data = base64.b64decode(data["audio"])
                    with open("temp.mp3", "wb") as f:
                        f.write(audio_data)

                    with open("temp.mp3", "rb") as f:
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
                    logger.error(f"❌ Error processing audio: {str(e)}")
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

            logger.info(f"Processing input from patient {patient_id}: {user_input}")

            # Save user message to database
            try:
                db.add(ChatMessage(
                    chat_message_id=str(uuid.uuid4()),
                    sender_id=patient_id,
                    receiver_id=AI_THERAPIST_ID,
                    message_text=user_input,
                    timestamp=datetime.now()
                ))
                db.commit()
                logger.info(f"✅ Saved user message to database for patient {patient_id}")
            except Exception as e:
                db.rollback()
                logger.error(f"❌ Error saving user message to database: {str(e)}")

            # Add user message to chat history
            chat_history.append({
                "role": "patient",
                "message": user_input,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

            # Generate response
            response_data = generate_therapist_response(patient_id, user_input, chat_history, db)
            message_text = response_data["response"]
            extracted_keywords = response_data["extracted_keywords"]
            
            logger.info(f"Generated response: {message_text}")
            logger.info(f"Extracted keywords: {extracted_keywords}")

            # Get the voice preference from the data
            voice = data.get("voice", "nova")
            logger.info(f"Using voice: {voice} for response")

            # Generate audio from the response text
            audio_base64 = generate_audio_from_text(message_text, voice)

            # Generate a unique response ID
            response_id = str(uuid.uuid4())

            # Save AI response to database
            try:
                db.add(ChatMessage(
                    chat_message_id=str(uuid.uuid4()),
                    sender_id=AI_THERAPIST_ID,
                    receiver_id=patient_id,
                    message_text=message_text,
                    extracted_keywords=json.dumps(extracted_keywords) if extracted_keywords else None,
                    timestamp=datetime.now()
                ))
                db.commit()
                logger.info(f"✅ Saved AI response to database for patient {patient_id}")
            except Exception as e:
                db.rollback()
                logger.error(f"❌ Error saving AI response to database: {str(e)}")

            # Add AI response to chat history
            chat_history.append({
                "role": "therapist",
                "message": message_text,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

            # Send response to client
            await websocket.send_text(json.dumps({
                "response": message_text,
                "audio": audio_base64,
                "extracted_keywords": extracted_keywords,
                "response_id": response_id
            }))

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for patient: {patient_id}")
    except Exception as e:
        logger.error(f"Error in WebSocket connection: {str(e)}")
        try:
            await websocket.close()
        except:
            pass

# Route to serve the AI Talk HTML page
@ai_talk_router.get("/ai-talk")
async def get_ai_talk_page():
    return FileResponse("static/ai_talk.html")
