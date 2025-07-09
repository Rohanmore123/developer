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
from asyncio import wait_for

from database.database import SessionLocal, get_db, get_db_safe
from model.model_correct import (
    Conversation, ChatMessage, Patient, Doctor, Appointment, OnboardingQuestion, DiaryEntry,
    EmotionAnalysis, MedicalHistory, Prescription
)
import vector_store_initializer
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create router
chat_final_router3 = APIRouter()

from openai import AzureOpenAI

azure_openai_api_key = os.getenv("azure_openai_api_key")
azure_openai_endpoint = os.getenv("azure_openai_endpoint")
model_name = "gpt-4o"
deployment = "gpt-4o"

api_version = "2024-12-01-preview"
if not azure_openai_api_key or not azure_openai_endpoint:
    logger.error("Azure OpenAI API key or endpoint not set in environment variables.")
    raise ValueError("Azure OpenAI API key and endpoint must be set in environment variables.")

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=azure_openai_endpoint,
    api_key=azure_openai_api_key,
)


# Initialize Azure OpenAI audio client
azure_openai_audio_api_key = os.getenv("azure_openai_audio_api_key")
azure_openai_audio_endpoint = os.getenv("azure_openai_audio_endpoint")
if not azure_openai_audio_api_key or not azure_openai_audio_endpoint:
    logger.error("Azure OpenAI audio API key or endpoint not set in environment variables.")
    raise ValueError("Azure OpenAI audio API key and endpoint must be set in environment variables.")
OPENAI_audio_CLIENT = AzureOpenAI(
    api_key=azure_openai_audio_api_key,
    azure_endpoint=azure_openai_audio_endpoint,
    api_version="2025-03-01-preview"
)

transcription_client = AzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_STT_API_KEY"],
    api_version="2025-03-01-preview",  # Use the latest API version that supports transcription
    azure_endpoint=os.environ["AZURE_OPENAI_STT_ENDPOINT"]  # e.g., "https://your-resource.openai.azure.com/"
)


# AI Doctor IDs for different personas
AI_DOCTOR_IDS = {
    "psychologist": "00000000-0000-0000-0000-000000000000",
    "dietician": "11111111-1111-1111-1111-111111111111"
}

# S3 paths for different FAISS indexes
S3_PATHS = {
    "psychologist": "faiss_index/general_index",  # Using general index for psychologist
    "dietician": "faiss_index/dietician_index"
}

# Dictionary to cache vector stores
_vector_stores = {
    "psychologist": None,
    "dietician": None
}

# Get vector stores from the initializer
logger.info("Using pre-loaded vector stores from initializer...")
psychologist_vector_store = vector_store_initializer.psychologist_vector_store
dietician_vector_store = vector_store_initializer.dietician_vector_store
# Removed pdf_vector_store to avoid unnecessary downloads

# Get retrievers from the initializer
psychologist_retriever = vector_store_initializer.psychologist_retriever
dietician_retriever = vector_store_initializer.dietician_retriever
# Removed pdf_retriever to avoid unnecessary downloads

logger.info("Successfully accessed all pre-loaded vector stores")

# Create a thread pool executor for parallel operations
# Using max_workers=None will use the default value (min(32, os.cpu_count() + 4))

# Thread pool for database operations
thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)

# Persona system prompts
PSYCHOLOGIST_PROMPT = """
You are a psychologist named 'Ori' with 15 years of experience,
you are working for a PrashaSync, which is an AI-driven platform that revolutionizes how people access and receive mental healthcare through personalized therapeutic experiences.
You are having a warm, casual, and supportive conversation with a patient. Your primary goal is to connect empathetically and help them feel heard and understood,
just like you would in a relaxed chat in your office.

IMPORTANT: You have been provided with:
1. The patient's medical records and history from the database
2. The patient's onboarding health information and questionnaire responses
3. Relevant medical information retrieved from our knowledge base

Your conversation style should be:
- Warm and empathetic, like talking to a trusted friend
- Use simple, everyday language (avoid medical jargon)
- Ask follow-up questions to understand their feelings better
- Validate their emotions and experiences
- Offer gentle insights and coping strategies when appropriate
- Be genuinely curious about their well-being

Remember: This is a supportive conversation, not a formal therapy session. Focus on making them feel comfortable and understood.

CRITICAL: You must respond in valid JSON format with these exact fields:
{
    "response": "Your empathetic response here",
    "extracted_keywords": ["keyword1", "keyword2", "keyword3"],
    "suggested_specialist": "none"
}

Only output the JSON (no explanation, no markdown).
"""

DIETICIAN_PROMPT = """
You are a dietician named Maya with 15 years of experience specializing in nutritional counseling and dietary management.
you are working for a PrashaSync, which is an AI-driven platform that revolutionizes how people access and receive mental healthcare through personalized therapeutic experiences.
You provide expert advice on nutrition, diet plans, and healthy eating habits.

IMPORTANT: You have been provided with:
1. The patient's medical records and history from the database
2. The patient's onboarding health information and questionnaire responses
3. Relevant medical information retrieved from our knowledge base

Your conversation style should be:
- Professional yet approachable and friendly
- Focus on practical, actionable nutrition advice
- Consider the patient's lifestyle, preferences, and medical conditions
- Provide evidence-based nutritional guidance
- Ask about eating habits, food preferences, and dietary restrictions
- Offer meal planning suggestions and healthy alternatives

CRITICAL: You must respond in valid JSON format with these exact fields:
{
    "response": "Your nutritional guidance response here",
    "extracted_keywords": ["keyword1", "keyword2", "keyword3"],
    "suggested_specialist": "none"
}

Only output the JSON (no explanation, no markdown).
"""

# S3 and FAISS utilities
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

#

# Helper functions
def get_patient_info(patient_id: str, db: Session) -> Dict:
    """Get comprehensive patient information."""
    try:
        # Get patient basic info
        patient = db.query(Patient).filter(Patient.patient_id == patient_id).first()
        if not patient:
            return {"error": "Patient not found"}

        # Calculate age
        age = None
        if patient.dob:
            from datetime import date
            today = date.today()
            age = today.year - patient.dob.year - ((today.month, today.day) < (patient.dob.month, patient.dob.day))

        # Get onboarding questions
        onboarding_questions = db.query(OnboardingQuestion).filter(
            OnboardingQuestion.patient_id == patient_id
        ).order_by(desc(OnboardingQuestion.timestamp)).limit(10).all()

        # Get medical history
        medical_history = db.query(MedicalHistory).filter(
            MedicalHistory.patient_id == patient_id
        ).order_by(desc(MedicalHistory.created_at)).limit(5).all()

        # Get recent diary entries
        diary_entries = db.query(DiaryEntry).filter(
            DiaryEntry.patient_id == patient_id
        ).order_by(desc(DiaryEntry.created_at)).limit(5).all()

        # Get recent emotion analysis
        emotion_analysis = db.query(EmotionAnalysis).filter(
            EmotionAnalysis.patient_id == patient_id
        ).order_by(desc(EmotionAnalysis.analyzed_at)).limit(3).all()

        # Construct patient name
        patient_name = f"{patient.first_name} {patient.last_name}".strip()

        return {
            "patient": {
                "id": str(patient.patient_id),  # Convert UUID to string
                "name": patient_name,
                "age": age,
                "gender": patient.gender,
                "phone": patient.phone,
                "email": patient.user.email if patient.user else None,
                "date_of_birth": patient.dob.isoformat() if patient.dob else None,
                "address": patient.address,
                "language": patient.language,
                "religion": patient.religion,
                "health_score": patient.health_score,
                "under_medications": patient.under_medications,
                "preferences": patient.preferences,
                "interests": patient.interests,
                "treatment": patient.treatment
            },
            "onboarding_questions": [
                {
                    "question": q.question,
                    "answer": q.answer,
                    "timestamp": q.timestamp.isoformat() if q.timestamp else None
                } for q in onboarding_questions
            ],
            "medical_history": [
                {
                    "condition": h.diagnosis,
                    "diagnosis_date": h.diagnosed_date.isoformat() if h.diagnosed_date else None,
                    "treatment": h.treatment,
                    "notes": h.additional_notes
                } for h in medical_history
            ],
            "diary_entries": [
                {
                    "content": d.notes,
                    "created_at": d.created_at.isoformat() if d.created_at else None
                } for d in diary_entries
            ],
            "emotion_analysis": [
                {
                    "dominant_emotion": e.emotion_category,
                    "confidence": float(e.confidence_score) if e.confidence_score else None,
                    "created_at": e.analyzed_at.isoformat() if e.analyzed_at else None
                } for e in emotion_analysis
            ]
        }

    except Exception as e:
        logger.error(f"Error getting patient info: {str(e)}")
        return {"error": f"Database error: {str(e)}"}

def get_chat_history_summary(chat_history: List[Dict], max_messages: int = 10) -> str:
    """Get a summary of recent chat history."""
    if not chat_history:
        return "No previous conversation history."

    recent_messages = chat_history[-max_messages:]
    summary_parts = []

    for msg in recent_messages:
        role = "Patient" if msg["role"] == "user" else "AI"
        content = msg["content"][:200] + "..." if len(msg["content"]) > 200 else msg["content"]
        summary_parts.append(f"{role}: {content}")

    return "\n".join(summary_parts)

# ============== CONVERSATION MANAGEMENT ==============

def get_user_id_from_patient_id(patient_id: str, db: Session) -> str:
    """Get user_id from patient_id."""
    try:
        from model.model_correct import Patient
        patient = db.query(Patient).filter(Patient.patient_id == patient_id).first()
        if patient:
            return str(patient.user_id)
        else:
            raise ValueError(f"Patient with ID {patient_id} not found")
    except Exception as e:
        logger.error(f"Error getting user_id for patient {patient_id}: {str(e)}")
        raise

def get_or_create_conversation(conversation_id: str, patient_id: str, persona: str, db: Session) -> Conversation:
    """Get existing conversation or create new one."""
    try:
        # Get user_id from patient_id
        user_id = get_user_id_from_patient_id(patient_id, db)

        if conversation_id:
            # Try to get existing conversation
            conversation = db.query(Conversation).filter(
                Conversation.conversation_id == conversation_id,
                Conversation.patient_id == patient_id
            ).first()

            if conversation:
                logger.info(f"Found existing conversation {conversation_id}")
                return conversation

        # Create new conversation (either conversation_id is None or not found)
        conversation = Conversation(
            user_id=user_id,
            patient_id=patient_id,
            title="New Conversation",
            createdat=datetime.now(),
            isarchived=False
        )
        db.add(conversation)
        db.flush()  # Get the auto-generated conversation_id
        db.commit()
        logger.info(f"Created new conversation {conversation.conversation_id} for patient {patient_id} (user {user_id})")
        return conversation

    except Exception as e:
        logger.error(f"Error managing conversation: {str(e)}")
        db.rollback()
        raise

def create_new_conversation(patient_id: str, db: Session) -> Conversation:
    """Create a new conversation and return it."""
    try:
        # Get user_id from patient_id
        user_id = get_user_id_from_patient_id(patient_id, db)

        conversation = Conversation(
            user_id=user_id,
            patient_id=patient_id,
            title="New Conversation",
            createdat=datetime.now(),
            isarchived=False
        )
        db.add(conversation)
        db.flush()  # Get the auto-generated conversation_id
        db.commit()
        logger.info(f"Created new conversation {conversation.conversation_id} for patient {patient_id} (user {user_id})")
        return conversation

    except Exception as e:
        logger.error(f"Error creating new conversation: {str(e)}")
        db.rollback()
        raise

def get_conversation_history(conversation_id: str, db: Session) -> List[ChatMessage]:
    """Get chat history for a conversation."""
    try:
        messages = db.query(ChatMessage).filter(
            ChatMessage.conversation_id == conversation_id
        ).order_by(ChatMessage.createdAt.asc()).all()

        logger.info(f"Retrieved {len(messages)} messages for conversation {conversation_id}")
        return messages

    except Exception as e:
        logger.error(f"Error retrieving conversation history: {str(e)}")
        return []

def generate_conversation_title(messages: List[ChatMessage], patient_id: str) -> str:
    """Generate a title for the conversation based on exactly 3 messages using GPT."""
    try:
        # Check if we have exactly 4 or 5 messages
        if len(messages) != 5 and len(messages) != 4:
            logger.info(f"Title generation skipped: {len(messages)} messages (need exactly 4 or 5)")
            return None

        # Build conversation history for GPT
        conversation_history = []
        for msg in messages:
            role = "user" if msg.sender_id == patient_id else "assistant"
            conversation_history.append({
                "role": role,
                "content": msg.message_text
            })

        # Create prompt for title generation
        title_prompt = """Based on the following conversation between a patient and a healthcare AI, generate a short, descriptive title (maximum 4-5 words) that captures the main topic or concern discussed.

The title should be:
- Concise and clear
- Professional and empathetic
- Focused on the main health topic extract based on chat
- Suitable for a medical chat interface

Conversation:
"""

        for msg in conversation_history:
            speaker = "Patient" if msg["role"] == "user" else "AI Doctor"
            title_prompt += f"{speaker}: {msg['content']}\n"

        title_prompt += "\nGenerate only the title, nothing else:"

        # Call GPT to generate title
        try:
            response = client.chat.completions.create(
                model=deployment,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates concise, professional titles for medical conversations.striclty output only title and nothing else"},
                    {"role": "user", "content": title_prompt}
                ],
                temperature=0.3,
                max_tokens=20,
                top_p=1.0
            )

            generated_title = response.choices[0].message.content.strip()

            # Clean up the title (remove quotes, extra punctuation)
            generated_title = generated_title.strip('"\'').strip()

            # Validate title length and content
            if len(generated_title) > 50:
                generated_title = generated_title[:47] + "..."

            if not generated_title or len(generated_title) < 3:
                # Fallback to keyword-based title
                return generate_fallback_title(messages, patient_id)

            logger.info(f"Generated GPT title: '{generated_title}'")
            return generated_title

        except Exception as gpt_error:
            logger.error(f"GPT title generation failed: {str(gpt_error)}")
            # Fallback to keyword-based title
            return generate_fallback_title(messages, patient_id)

    except Exception as e:
        logger.error(f"Error in title generation: {str(e)}")
        return "Health Conversation"

def generate_fallback_title(messages: List[ChatMessage], patient_id: str) -> str:
    """Generate a fallback title based on keywords when GPT fails."""
    try:
        # Get user messages for keyword analysis
        user_messages = [msg for msg in messages if msg.sender_id == patient_id]
        combined_text = " ".join([msg.message_text for msg in user_messages])
        text_lower = combined_text.lower()

        # Keyword-based title generation
        if any(word in text_lower for word in ['anxiety', 'anxious', 'worry', 'stress', 'nervous']):
            return "Anxiety and Stress Support"
        elif any(word in text_lower for word in ['depression', 'sad', 'down', 'mood', 'depressed']):
            return "Mood and Mental Health"
        elif any(word in text_lower for word in ['diet', 'food', 'eating', 'nutrition', 'meal']):
            return "Nutrition and Diet Planning"
        elif any(word in text_lower for word in ['sleep', 'insomnia', 'tired', 'rest']):
            return "Sleep and Rest Issues"
        elif any(word in text_lower for word in ['relationship', 'family', 'friend', 'partner']):
            return "Relationship Support"
        elif any(word in text_lower for word in ['pain', 'hurt', 'ache', 'medication']):
            return "Pain and Medication"
        elif any(word in text_lower for word in ['exercise', 'fitness', 'workout', 'activity']):
            return "Fitness and Exercise"
        else:
            return "Health and Wellness Chat"

    except Exception as e:
        logger.error(f"Error in fallback title generation: {str(e)}")
        return "Health Conversation"

def update_conversation_title(conversation_id: str, title: str, db: Session):
    """Update conversation title."""
    try:
        conversation = db.query(Conversation).filter(
            Conversation.conversation_id == conversation_id
        ).first()

        if conversation and (conversation.title == "New Conversation" or not conversation.title):
            conversation.title = title
            db.commit()
            logger.info(f"Updated conversation {conversation_id} title: {title}")

    except Exception as e:
        logger.error(f"Error updating conversation title: {str(e)}")
        db.rollback()

def save_message_to_database(patient_id: str, message: str, sender: str, persona: str, conversation_id: str, db: Session):
    """Save message to database."""
    try:
        ai_doctor_id = AI_DOCTOR_IDS.get(persona, AI_DOCTOR_IDS["psychologist"])

        # Determine sender and receiver IDs based on sender type
        if sender == "patient":
            sender_id = patient_id
            receiver_id = ai_doctor_id
        else:  # sender == "doctor"
            sender_id = ai_doctor_id
            receiver_id = patient_id

        chat_message = ChatMessage(
            chat_message_id=str(uuid.uuid4()),
            sender_id=sender_id,
            receiver_id=receiver_id,
            message_text=message,
            patient_id=patient_id,
            conversation_id=conversation_id,
            createdAt=datetime.now()
        )
        db.add(chat_message)
        db.commit()
        logger.info(f"Message saved to database for {persona}")
    except Exception as e:
        logger.error(f"Error saving message to database: {str(e)}")
        db.rollback()

# Audio processing utilities
import traceback

def clean_text(text):
    import re
    # Remove non-printable and problematic characters
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', text)
    # Remove directional markers, soft hyphens, etc.
    text = re.sub(r'[\u200B-\u200F\u202A-\u202E\u2060\uFEFF]', '', text)
    return text

class StreamingAudioManager:
    def __init__(self, voice: str = "nova"):
        self.voice = voice
        self.accumulated_text = ""
        self.audio_queue = []
        self.is_generating = False
        self.all_streamed_text = ""  # Track all text that was streamed
        self.streaming_complete = False  # Track if streaming is complete
        self.final_chunk_processed = asyncio.Event()
        self.streamed_audio_chunks = []

    # async def add_text_chunk(self, text_chunk: str, websocket: WebSocket):
    #     """Add text chunk and generate audio when we have enough text."""
    #     self.accumulated_text += text_chunk
    #     self.all_streamed_text += text_chunk
    #     # Generate audio more frequently for faster response - reduced thresholds
    #     if (("." in self.accumulated_text or "!" in self.accumulated_text or "?" in self.accumulated_text)
    #         and len(self.accumulated_text) > 20) or len(self.accumulated_text) > 50:


    #     # # 🎵 AGGRESSIVE STREAMING: Generate audio more frequently for immediate response
    #     # should_generate = False

    #     # # Check for natural breaks (sentence endings)
    #     # if ("." in self.accumulated_text or "!" in self.accumulated_text or "?" in self.accumulated_text):
    #     #     should_generate = True
    #     # # Check for comma breaks with reasonable length
    #     # elif ("," in self.accumulated_text and len(self.accumulated_text) > 8):
    #     #     should_generate = True
    #     # # Check for word boundaries with shorter thresholds
    #     # elif (" " in self.accumulated_text and len(self.accumulated_text) > 12):
    #     #     should_generate = True
    #     # # Force generation after moderate length to avoid long delays
    #     # elif len(self.accumulated_text) > 25:
    #     #     should_generate = True
    #     # # 🚀 IMMEDIATE START: Generate audio for very first chunks to avoid silence
    #     # elif len(self.audio_queue) == 0 and len(self.accumulated_text) > 6:
    #     #     should_generate = True

    #     # if should_generate:
    #         text_to_process = self.accumulated_text.strip()
    #         self.accumulated_text = ""  # Reset for next chunk

    #         if text_to_process and not self.is_generating:
    #             self.is_generating = True
    #             try:
    #                 # Generate audio for this chunk
    #                 audio_data = await self.generate_audio_chunk(text_to_process)
    #                 if audio_data:
    #                     self.audio_queue.append(audio_data)
    #                     self.streamed_audio_chunks.append(audio_data)

    #                     # Send audio chunk to client immediately
    #                     audio_b64 = base64.b64encode(audio_data).decode("utf-8")
    #                     await websocket.send_text(json.dumps({
    #                         "type": "streaming_audio",
    #                         "audio": audio_b64,
    #                         "text": text_to_process,
    #                         "sequence": len(self.audio_queue)
    #                     }))

    #                     logger.info(f"🎵 Generated streaming audio for: '{text_to_process[:30]}...' (length: {len(text_to_process)})")

    #             except Exception as e:
    #                 logger.error(f"Error generating streaming audio: {str(e)}")
    #             finally:
    #                 self.is_generating = False
    async def add_text_chunk(self, text_chunk: str, websocket: WebSocket):
        """Add text chunk and generate audio when we have enough text."""
        self.accumulated_text += text_chunk
        self.all_streamed_text += text_chunk
        
        # 🎵 FIXED: More aggressive streaming with multiple conditions
        should_generate = False
        
        # Check for natural sentence breaks
        if any(punct in self.accumulated_text for punct in [".","!", "?"]):
            should_generate = True
        # Check for clause breaks with reasonable length
        elif "," in self.accumulated_text and len(self.accumulated_text) > 20:
            should_generate = True
        # # Check for word boundaries with shorter thresholds
        # elif " " in self.accumulated_text and len(self.accumulated_text) > 20:
        #     should_generate = True
        # Force generation after moderate length to avoid long delays
        elif len(self.accumulated_text) > 40:
            should_generate = True
        # # 🚀 CRITICAL FIX: Generate audio for very first chunks to avoid silence
        # elif len(self.audio_queue) == 0 and len(self.accumulated_text) > 8:
        #     should_generate = True
        # 🚀 ADDITIONAL FIX: If we have a decent chunk and haven't generated audio recently
        # elif len(self.accumulated_text) > 25 and not self.is_generating:
        #     should_generate = True
        # # 🎵 EMERGENCY FIX: If we have ANY text and no audio has been generated yet
        # elif len(self.accumulated_text) > 5 and len(self.audio_queue) == 0:
        #     should_generate = True

        if should_generate:
            text_to_process = self.accumulated_text.strip()
            self.accumulated_text = ""  # Reset for next chunk

            if text_to_process and not self.is_generating:
                self.is_generating = True
                try:
                    # Generate audio for this chunk
                    audio_data = await self.generate_audio_chunk(text_to_process)
                    if audio_data:
                        self.audio_queue.append(audio_data)
                        self.streamed_audio_chunks.append(audio_data)

                        # Send audio chunk to client immediately
                        audio_b64 = base64.b64encode(audio_data).decode("utf-8")
                        await websocket.send_text(json.dumps({
                            "type": "streaming_audio",
                            "audio": audio_b64,
                            "text": text_to_process,
                            "sequence": len(self.audio_queue)
                        }))

                        logger.info(f"🎵 Generated streaming audio for: '{text_to_process[:30]}...' (length: {len(text_to_process)})")

                except Exception as e:
                    logger.error(f"Error generating streaming audio: {str(e)}")
                finally:
                    self.is_generating = False
    async def generate_audio_chunk(self, text: str) -> bytes:
        """Generate audio for a text chunk."""
        try:
            if not text.strip():
                return None

            # Clean the text
            clean_text_content = clean_text(text)
            if not clean_text_content.strip():
                return None

            # Generate audio using Azure OpenAI
            response = OPENAI_audio_CLIENT.audio.speech.create(
                model="tts",  # Azure deployment name
                voice=self.voice,
                input=clean_text_content,
                response_format="mp3"
            )

            return response.content

        except Exception as e:
            logger.error(f"Error generating audio chunk: {str(e)}")
            # Don't fail the entire response if audio generation fails
            return None

    async def finalize_audio(self, websocket: WebSocket, complete_text: str):
        """Generate complete audio for the full response."""
        try:
            if not complete_text.strip():
                logger.warning("No complete text provided for finalization")
                return

            # Clean the complete text
            clean_complete_text = clean_text(complete_text)
            if not clean_complete_text.strip():
                logger.warning("Complete text is empty after cleaning")
                return

            logger.info(f"🎵 Generating complete audio for {len(clean_complete_text)} characters")

            # Generate complete audio
            response = OPENAI_audio_CLIENT.audio.speech.create(
                model="tts",  # Azure deployment name
                voice=self.voice,
                input=clean_complete_text,
                response_format="mp3"
            )

            complete_audio_b64 = base64.b64encode(response.content).decode("utf-8")

            # Send complete audio to client
            await websocket.send_text(json.dumps({
                "type": "complete_audio",
                "audio": complete_audio_b64,
                "text": complete_text,
                "voice": self.voice,
                "streaming_chunks_count": len(self.audio_queue)
            }))

            logger.info(f"✅ Complete audio sent successfully")

        except Exception as e:
            logger.error(f"❌ Error in finalize_audio: {str(e)}")
            # Send empty audio response so the UI doesn't hang waiting
            await websocket.send_text(json.dumps({
                "type": "complete_audio",
                "audio": "",
                "text": complete_text,
                "error": f"Audio generation failed: {str(e)}"
            }))

# Function to generate AI response with streaming
async def generate_ai_response_streaming(
    user_input: str,
    patient_info: Dict,
    chat_history: List[Dict],
    system_prompt: str,
    retriever,
    websocket: WebSocket,
    current_persona: str = "psychologist"
) -> Dict:
    """Generate AI response using OpenAI API with streaming support."""
    try:
        # if not OPENAI_CLIENT:
        #     return {
        #         "response": "I'm sorry, but the AI service is currently unavailable. Please try again later.",
        #         "extracted_keywords": ["error", "unavailable"],
        #         "suggested_specialist": "none"
        #     }

        # 🚀 STREAMING IMPLEMENTATION
        logger.info("🌊 Starting streaming response generation...")

        # Determine voice for audio streaming
        voice = "onyx" if current_persona == "psychologist" else "shimmer"

        # Initialize streaming audio manager
        audio_manager = StreamingAudioManager(voice)

        # Get relevant context from vector store
        context_docs = []
        if retriever:
            try:
                context_docs = retriever.get_relevant_documents(user_input)
                logger.info(f"Retrieved {len(context_docs)} context documents")
            except Exception as e:
                logger.warning(f"Error retrieving context: {str(e)}")

        # Prepare context
        context = "\n".join([doc.page_content for doc in context_docs[:3]])
        chat_summary = get_chat_history_summary(chat_history)

        # Prepare messages for OpenAI
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": f"Patient Information: {json.dumps(patient_info, indent=2)}"},
            {"role": "system", "content": f"Recent Chat History: {chat_summary}"},
            {"role": "system", "content": f"Relevant Medical Context: {context}"},
            {"role": "user", "content": user_input}
        ]

        # 🌊 Start streaming response
        await websocket.send_text(json.dumps({
            "type": "streaming_start",
            "persona": current_persona,
            "voice": voice
        }))

        # Generate streaming response
        response_stream = client.chat.completions.create(
            model=deployment,
            messages=messages,
            temperature=0.1,
            max_tokens=850,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            response_format={"type": "json_object"},
            stream=True
        )

        # Collect streaming response
        full_response = ""
        response_text = ""
        response_started = False
        streaming_complete = False

        for chunk in response_stream:
            # Check if chunk has choices and content
            if chunk.choices and len(chunk.choices) > 0 and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content

                # 🚀 SIMPLE APPROACH: Look for response field start, then stream directly
                if not response_started:
                    # Look for the start of response field
                    if '"response":' in full_response:
                        response_started = True
                        logger.info("🌊 Response field detected, starting streaming...")
                        # Skip the '"response":"' part and start streaming from the actual content
                        continue

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
                                "text": final_chunk,
                                "persona": current_persona,
                                "is_complete": True
                            }))
                            await audio_manager.add_text_chunk(final_chunk, websocket)
                            response_text += final_chunk

                        logger.info("🌊 Response streaming complete! Continuing to collect metadata...")
                        streaming_complete = True
                        # DON'T break - continue collecting the rest of the JSON
                    else:
                        # Continue streaming this chunk - allow ALL content including spaces
                        if content and content != '\n':  # Only filter newlines, allow everything else
                            logger.info(f"🌊 Streaming chunk: '{content}'")

                            # Send text chunk to client
                            await websocket.send_text(json.dumps({
                                "type": "streaming_text", 
                                "text": content,
                                "persona": current_persona,
                                "is_complete": False
                            }))

                            # Add to audio manager for streaming audio generation
                            await audio_manager.add_text_chunk(content, websocket)
                            response_text += content
                            # await asyncio.sleep(0.15)  # Small delay to simulate real-time streaming

        # Process any remaining text in audio manager
        if audio_manager.accumulated_text.strip():
            remaining_text = audio_manager.accumulated_text.strip()
            audio_manager.accumulated_text = ""
            try:
                audio_data = await audio_manager.generate_audio_chunk(remaining_text)
                if audio_data:
                    audio_manager.audio_queue.append(audio_data)
                    audio_b64 = base64.b64encode(audio_data).decode("utf-8")
                    await websocket.send_text(json.dumps({
                        "type": "streaming_audio",
                        "audio": audio_b64,
                        "text_chunk": remaining_text,
                        "chunk_index": len(audio_manager.audio_queue) - 1
                    }))
            except Exception as e:
                logger.error(f"Error processing remaining audio: {str(e)}")

        # 🚀 PROCESS COMPLETE RESPONSE
        logger.info("🌊 Streaming complete, processing full response...")

    

        # Clean up and parse the full JSON
        ai_response = full_response.strip('```json').strip('```').strip()
        ai_response = ai_response.replace('{\n', '{').replace('\n}', '}').replace(",\n", ",").replace('\n', '###').strip().replace('###', '')

        try:
            response_json = json.loads(ai_response)
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON parsing failed: {str(e)}")
            logger.error(f"🧾 Raw AI Response: {repr(ai_response)}")

            # Fallback
            response_json = {
                "response": response_text if response_text else "I apologize, but I encountered an error processing your request. Could you please rephrase?",
                "extracted_keywords": ["error"],
                "suggested_specialist": "none"
            }

        # Defensive check: ensure 'response' exists and is not empty
        response_text_final = response_json.get("response", "").strip()
        if not response_text_final:
            logger.warning("⚠️ Empty 'response' field in response_json. Adding fallback.")
            response_text_final = "I'm sorry, I didn't quite catch that. Could you please rephrase?"
            response_json["response"] = response_text_final

        # Ensure required fields are present
        response_json.setdefault("extracted_keywords", [])
        response_json.setdefault("suggested_specialist", "none")

        # Ensure required fields are present
        if "response" not in response_json:
            response_json["response"] = response_text_final if response_text_final else "I apologize, but I couldn't generate a proper response. Please try again."

        if "extracted_keywords" not in response_json:
            response_json["extracted_keywords"] = []

        if "suggested_specialist" not in response_json:
            response_json["suggested_specialist"] = "none"

        # Generate complete audio for full response
        await audio_manager.finalize_audio(websocket, response_json["response"])

        # 🌊 Send enhanced completion signal with audio information
        await websocket.send_text(json.dumps({
            "type": "streaming_complete",
            "extracted_keywords": response_json.get("extracted_keywords", []),
            "current_persona": current_persona,
            "audio_chunks_generated": len(audio_manager.audio_queue),
            "complete_text_available": True,
            "streaming_text_length": len(audio_manager.all_streamed_text),
            "final_text_length": len(response_json["response"])
        }))

        logger.info(f"🌊 Streaming response completed for {current_persona}")
        return response_json

    except Exception as e:
        logger.error(f"Error in generate_ai_response_streaming: {str(e)}")
        return {
            "response": f"I apologize, but I encountered an error while processing your request. Please try again.",
            "extracted_keywords": ["error", "technical"],
            "suggested_specialist": "none"
        }

# WebSocket endpoint for chat
@chat_final_router3.websocket("/chat-final3/{patient_id}/{persona}")
async def chat_websocket(websocket: WebSocket, patient_id: str, persona: str, db: Session = Depends(get_db)):
    await websocket.accept()

    # Validate persona
    if persona not in ["psychologist", "dietician"]:
        await websocket.send_text(json.dumps({
            "error": "Invalid persona. Must be 'psychologist' or 'dietician'"
        }))
        await websocket.close()
        return

    # Initialize connection state
    current_persona = persona
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

                # 🚀 CONVERSATION: Get conversation_id from auth data (optional)
                conversation_id = auth_data.get("conversation_id")
                if conversation_id:
                    conversation_id = str(conversation_id)  # Ensure it's a string

                logger.info(f"🔄 Pre-loading patient data and conversation for {patient_id}...")
                data_load_start = time.time()

                # Load patient information
                patient_info = get_patient_info(patient_id, db)
                if "error" in patient_info:
                    await websocket.send_text(json.dumps({
                        "error": patient_info["error"]
                    }))
                    return

                # 🚀 CONVERSATION: Get or create conversation and load history
                try:
                    if conversation_id:
                        # Use existing conversation
                        conversation = get_or_create_conversation(conversation_id, patient_id, current_persona, db)
                        conversation_id = str(conversation.conversation_id)
                    else:
                        # Create new conversation
                        conversation = create_new_conversation(patient_id, db)
                        conversation_id = str(conversation.conversation_id)
                        logger.info(f"Created new conversation {conversation_id} for new session")

                    conversation_messages = get_conversation_history(conversation_id, db)

                    # Convert to chat history format
                    for msg in conversation_messages:
                        role = "user" if msg.sender_id == patient_id else "assistant"
                        chat_history.append({"role": role, "content": msg.message_text})

                    logger.info(f"Loaded {len(chat_history)} messages from conversation {conversation_id}")

                    # Send conversation info to frontend
                    history_data = []
                    if conversation_messages:
                        for msg in conversation_messages:
                            history_data.append({
                                "sender": "user" if msg.sender_id == patient_id else "assistant",
                                "message": msg.message_text,
                                "timestamp": msg.createdAt.isoformat(),
                                "persona": current_persona
                            })

                    await websocket.send_text(json.dumps({
                        "type": "conversation_history",
                        "messages": history_data,
                        "conversation_id": conversation_id,
                        "title": conversation.title,
                        "is_new_conversation": len(conversation_messages) == 0
                    }))

                except Exception as e:
                    logger.error(f"Error loading conversation: {str(e)}")
                    await websocket.send_text(json.dumps({
                        "error": f"Failed to load conversation: {str(e)}"
                    }))
                    return

                data_load_time = time.time() - data_load_start
                logger.info(f"🚀 Data loading completed in {data_load_time:.2f}s")

                # Send success response with patient info
                await websocket.send_text(json.dumps({
                    "type": "auth_success",
                    "patient_info": patient_info,
                    "current_persona": current_persona,
                    "chat_history_loaded": len(chat_history),
                    "connection_time": time.time() - connection_start
                }))

                # 🎯 Send initial welcome message from the selected specialist
                logger.info(f"🎯 Sending initial welcome message from {current_persona}")

                # Determine system prompt and retriever for the selected persona
                system_prompt = PSYCHOLOGIST_PROMPT if current_persona == "psychologist" else DIETICIAN_PROMPT
                retriever = psychologist_retriever if current_persona == "psychologist" else dietician_retriever

                # Generate welcome message based on chat history
                if len(chat_history) > 0:
                    welcome_prompt = f"Hello! You are now talking to a patient. Create a brief summary of our previous conversations and send a warm welcome message to continue our chat or start with new concerns."
                else:
                    welcome_prompt = f"Hello! You are now talking to a new patient. Please introduce yourself and send a warm welcome message to start our conversation."

                try:
                    # Generate initial welcome message with streaming
                    response_data = await wait_for(generate_ai_response_streaming(
                        welcome_prompt,
                        patient_info,
                        chat_history,
                        system_prompt,
                        retriever,
                        websocket,
                        current_persona
                    ), timeout=60)

                    # Add welcome message to chat history
                    chat_history.append({"role": "assistant", "content": response_data["response"]})

                    # Save welcome message to database
                    def save_welcome_message_to_database():
                        try:
                            with SessionLocal() as db_session:
                                save_message_to_database(patient_id, response_data["response"], "doctor", current_persona, conversation_id, db_session)
                        except Exception as e:
                            logger.error(f"Error saving welcome message: {str(e)}")

                    # Submit welcome message save to thread pool
                    thread_pool.submit(save_welcome_message_to_database)

                    logger.info(f"✅ Initial welcome message sent from {current_persona}")

                except Exception as e:
                    logger.error(f"Error generating welcome message: {str(e)}")
                    # Send a simple fallback welcome message
                    fallback_message = f"Hello! I'm Dr. {'Ori' if current_persona == 'psychologist' else 'Maya'}, your {current_persona}. How can I help you today?"
                    await websocket.send_text(json.dumps({
                        "type": "streaming_start",
                        "persona": current_persona
                    }))
                    await websocket.send_text(json.dumps({
                        "type": "streaming_text",
                        "text": fallback_message,
                        "persona": current_persona
                    }))
                    await websocket.send_text(json.dumps({
                        "type": "streaming_complete",
                        "extracted_keywords": ["greeting", "introduction"],
                        "current_persona": current_persona
                    }))

            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "error": "Invalid JSON format"
                }))
            except Exception as e:
                logger.error(f"Authentication error: {str(e)}")
                await websocket.send_text(json.dumps({
                    "error": f"Authentication failed: {str(e)}"
                }))

        # 🚀 MAIN CHAT LOOP: Handle messages after authentication
        logger.info(f"🎯 Starting main chat loop for {current_persona} with patient {patient_id}")

        while True:
            try:
                # Receive message
                message = await websocket.receive_text()
                data = json.loads(message)

                # Performance tracking
                request_start = time.time()

                # Handle different message types
                if data.get("type") == "ping":
                    # Handle client heartbeat ping
                    await websocket.send_text(json.dumps({"type": "pong"}))
                    continue
                elif data.get("type") == "pong":
                    # Handle client pong response
                    logger.debug("💓 Client pong received")
                    continue
                elif "text" in data:
                    user_input = data["text"].strip()
                    voice = data.get("voice", "nova")

                    if not user_input:
                        continue

                    logger.info(f"💬 Processing text message: {user_input[:100]}...")

                    # 🚀 NORMAL AI PROCESSING: Add to in-memory chat history immediately
                    chat_history.append({"role": "user", "content": user_input})

                    # Database save function (non-blocking)
                    def save_user_message_to_database():
                        try:
                            with SessionLocal() as db_session:
                                save_message_to_database(patient_id, user_input, "patient", current_persona, conversation_id, db_session)
                        except Exception as e:
                            logger.error(f"Error saving user message: {str(e)}")

                    # Submit database task to thread pool (non-blocking)
                    db_future = thread_pool.submit(save_user_message_to_database)

                    # 🚀 OPTIMIZATION: Select persona and retriever (using pre-loaded data)
                    system_prompt = PSYCHOLOGIST_PROMPT if current_persona == "psychologist" else DIETICIAN_PROMPT
                    retriever = psychologist_retriever if current_persona == "psychologist" else dietician_retriever

                    # 🚀 STREAMING: Generate AI response with streaming support
                    try:
                        response_data = await wait_for(generate_ai_response_streaming(
                            user_input,
                            patient_info,
                            chat_history,
                            system_prompt,
                            retriever,
                            websocket,
                            current_persona
                        ), timeout=90)

                        # Add AI response to chat history
                        chat_history.append({"role": "assistant", "content": response_data["response"]})

                        # Database save function for AI response (non-blocking)
                        def save_ai_message_to_database():
                            try:
                                with SessionLocal() as db_session:
                                    save_message_to_database(patient_id, response_data["response"], "doctor", current_persona, conversation_id, db_session)
                            except Exception as e:
                                logger.error(f"Error saving AI message: {str(e)}")

                        # Submit AI message save to thread pool
                        ai_db_future = thread_pool.submit(save_ai_message_to_database)

                        # Calculate timing
                        total_time = time.time() - request_start
                        logger.info(f"✅ {current_persona.title()} response completed in {total_time:.2f}s")

                        # 🚀 TITLE GENERATION: Check if we need to generate a title after 3 messages
                        try:
                            def check_and_generate_title():
                                try:
                                    with SessionLocal() as db_session:
                                        messages = get_conversation_history(conversation_id, db_session)
                                        if len(messages) == 4 or len(messages) == 5:  # After exactly 5 messages
                                            title = generate_conversation_title(messages, patient_id)
                                            if title:
                                                update_conversation_title(conversation_id, title, db_session)
                                                logger.info(f"Generated title for conversation {conversation_id}: {title}")
                                except Exception as e:
                                    logger.error(f"Error generating conversation title: {str(e)}")

                            # Submit title generation to thread pool (non-blocking)
                            thread_pool.submit(check_and_generate_title)
                        except Exception as e:
                            logger.error(f"Error submitting title generation task: {str(e)}")

                    except asyncio.TimeoutError:
                        logger.error("AI response generation timed out")
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "message": "Response generation timed out. Please try again."
                        }))
                    except Exception as e:
                        logger.error(f"Error generating AI response: {str(e)}")
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "message": "An error occurred while generating the response."
                        }))

                elif "audio" in data:
                    # Handle audio input
                    try:
                        audio_data = base64.b64decode(data["audio"])
                        voice = data.get("voice", "nova")

                        logger.info(f"🎤 Processing audio message ({len(audio_data)} bytes)")

                        # Save audio to temporary file
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_file:
                            temp_file.write(audio_data)
                            temp_file_path = temp_file.name

                        try:
                            # Transcribe audio using OpenAI Whisper
                            with open(temp_file_path, "rb") as audio_file:
                                transcript = transcription_client.audio.transcriptions.create(
                                    model="gpt-4o-transcribe",
                                    file=audio_file,
                                    response_format="text"
                                )

                            user_input = transcript.strip()
                            logger.info(f"🎤 Transcribed: {user_input}")

                            if not user_input:
                                await websocket.send_text(json.dumps({
                                    "type": "error",
                                    "message": "Could not transcribe audio. Please try again."
                                }))
                                continue

                            # Send transcription to client
                            await websocket.send_text(json.dumps({
                                "type": "transcription",
                                "text": user_input
                            }))

                            # Process the transcribed text same as text input
                            chat_history.append({"role": "user", "content": user_input})

                            # Database save function (non-blocking)
                            def save_user_message_to_database():
                                try:
                                    with SessionLocal() as db_session:
                                        save_message_to_database(patient_id, user_input, "patient", current_persona, conversation_id, db_session)
                                except Exception as e:
                                    logger.error(f"Error saving user message: {str(e)}")

                            # Submit database task to thread pool (non-blocking)
                            db_future = thread_pool.submit(save_user_message_to_database)

                            # Generate AI response with streaming
                            system_prompt = PSYCHOLOGIST_PROMPT if current_persona == "psychologist" else DIETICIAN_PROMPT
                            retriever = psychologist_retriever if current_persona == "psychologist" else dietician_retriever

                            try:
                                response_data = await wait_for(generate_ai_response_streaming(
                                    user_input,
                                    patient_info,
                                    chat_history,
                                    system_prompt,
                                    retriever,
                                    websocket,
                                    current_persona
                                ), timeout=90)

                                # Add AI response to chat history
                                chat_history.append({"role": "assistant", "content": response_data["response"]})

                                # Database save function for AI response (non-blocking)
                                def save_ai_message_to_database():
                                    try:
                                        with SessionLocal() as db_session:
                                            save_message_to_database(patient_id, response_data["response"], "doctor", current_persona, db_session)
                                    except Exception as e:
                                        logger.error(f"Error saving AI message: {str(e)}")

                                # Submit AI message save to thread pool
                                ai_db_future = thread_pool.submit(save_ai_message_to_database)

                                # Calculate timing
                                total_time = time.time() - request_start
                                logger.info(f"✅ {current_persona.title()} audio response completed in {total_time:.2f}s")

                                # 🚀 TITLE GENERATION: Check if we need to generate a title after 3 messages
                                try:
                                    def check_and_generate_title():
                                        try:
                                            with SessionLocal() as db_session:
                                                messages = get_conversation_history(conversation_id, db_session)
                                                if len(messages) == 4 or len(messages) == 5:  # After exactly 4 or 5 messages
                                                    title = generate_conversation_title(messages, patient_id)
                                                    if title:
                                                        update_conversation_title(conversation_id, title, db_session)
                                                        logger.info(f"Generated title for conversation {conversation_id}: {title}")
                                        except Exception as e:
                                            logger.error(f"Error generating conversation title: {str(e)}")

                                    # Submit title generation to thread pool (non-blocking)
                                    thread_pool.submit(check_and_generate_title)
                                except Exception as e:
                                    logger.error(f"Error submitting title generation task: {str(e)}")

                            except asyncio.TimeoutError:
                                logger.error("AI response generation timed out")
                                await websocket.send_text(json.dumps({
                                    "type": "error",
                                    "message": "Response generation timed out. Please try again."
                                }))
                            except Exception as e:
                                logger.error(f"Error generating AI response: {str(e)}")
                                await websocket.send_text(json.dumps({
                                    "type": "error",
                                    "message": "An error occurred while generating the response."
                                }))

                        finally:
                            # Clean up temporary file
                            try:
                                os.unlink(temp_file_path)
                            except:
                                pass

                    except Exception as e:
                        logger.error(f"Error processing audio: {str(e)}")
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "message": "Error processing audio input."
                        }))

                else:
                    # Log unknown message types for debugging, but don't treat as error
                    logger.debug(f"Unhandled message type: {data.get('type', 'unknown')}")

            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected for patient {patient_id}")
                break
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "error": "Invalid JSON format"
                }))
            except Exception as e:
                logger.error(f"Error in chat loop: {str(e)}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "An unexpected error occurred."
                }))

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected during authentication for patient {patient_id}")
    except Exception as e:
        logger.error(f"Unexpected error in WebSocket connection: {str(e)}")
    finally:
        logger.info(f"WebSocket connection closed for patient {patient_id} ({current_persona})")

# Health check endpoint
@chat_final_router3.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "chat_final3",
        "personas": ["psychologist", "dietician"],
        "openai_available": client is not None
    }
