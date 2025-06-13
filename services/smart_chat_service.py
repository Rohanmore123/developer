import os
import uuid
import json
import sys
import base64
import asyncio
import concurrent.futures
from io import BytesIO
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException, Header
# from auth.dependencies import get_current_active_user
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import desc
from database.database import SessionLocal
from database.database import get_db
from model.model_correct import (
    ChatMessage, Patient, OnboardingQuestion, DiaryEntry,
    EmotionAnalysis, MedicalHistory, Prescription
)
from openai import OpenAI
# # LangChain imports
# from langchain.agents import AgentExecutor, create_openai_tools_agent
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_openai import ChatOpenAI
# from langchain.tools import tool
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
# from langchain.chains import ConversationalRetrievalChain
# from langchain.memory import ConversationBufferMemory
from services.smart_agent import SmartAgent

# Load environment variables
load_dotenv()
smart_chat_router = APIRouter()

# OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
try:

    OPENAI_CLIENT = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
    if not OPENAI_CLIENT:
        print("WARNING: OpenAI client could not be initialized. Check your API key.")
except ImportError:
    print("WARNING: OpenAI package not installed. Some features may not work.")
    OPENAI_CLIENT = None

# AI Doctor ID (fixed ID for the AI doctor)
# AI_DOCTOR_ID = "a0e61dd9-2824-4d84-ae10-4cb9b0f4b321"
AI_DOCTOR_ID="00000000-0000-0000-0000-000000000000"



import faiss
import os


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
            print("✅ Loaded existing FAISS HNSW index")
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

        # Use the standard FAISS.from_texts method which handles all the complexity
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vector_store = FAISS.from_texts(chunks, embeddings)

        # If you want to use HNSW specifically, you can replace the index after creation
        # This is a more reliable approach than trying to build it from scratch
        if hasattr(vector_store, '_index'):
            # Get dimension from existing index
            vector_dim = vector_store._index.d

            # Create HNSW index
            hnsw_index = faiss.IndexHNSWFlat(vector_dim, 32)  # 32 is M, can be tuned
            hnsw_index.hnsw.efConstruction = 40  # Controls construction complexity

            # If there are vectors in the original index, copy them to the new index
            if vector_store._index.ntotal > 0:
                # This is a more complex operation that requires direct access to the index
                # For simplicity, we'll just recreate the HNSW index from scratch
                # Get the embeddings from the original documents
                embedded_vectors = embeddings.embed_documents(chunks)
                # Add them to the new index
                hnsw_index.add(np.array(embedded_vectors).astype("float32"))

            # Replace the index in the vector store
            vector_store._index = hnsw_index

        # Save to local
        vector_store.save_local(HNSW_INDEX_DIR)
        print("✅ Created and saved new HNSW FAISS index")
        return vector_store

    except Exception as e:
        print(f"❌ Error initializing HNSW vector store: {str(e)}")
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        return FAISS.from_texts(["Mental health information placeholder."], embeddings)


# # Initialize FAISS vector store
# def initialize_vector_store():
#     """Initialize the FAISS vector store with mental health information."""
#     try:
#         # Check if the FAISS index already exists
#         if os.path.exists("resources/faiss_index"):
#             # Load the existing index with allow_dangerous_deserialization=True
#             embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
#             vector_store = FAISS.load_local(
#                 "resources/faiss_index",
#                 embeddings,
#                 allow_dangerous_deserialization=True
#             )
#             print("✅ Loaded existing FAISS index")
#             return vector_store

#         # Create a new index
#         with open("resources/mental_health_info.txt", "r", encoding="utf-8") as f:
#             mental_health_text = f.read()

#         # Split the text into chunks
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=500,
#             chunk_overlap=100,
#             separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""]
#         )
#         chunks = text_splitter.split_text(mental_health_text)

#         # Create embeddings and store in FAISS
#         embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
#         vector_store = FAISS.from_texts(chunks, embeddings)

#         # Save the index
#         vector_store.save_local("resources/faiss_index")
#         print("✅ Created and saved new FAISS index")
#         return vector_store
#     except Exception as e:
#         print(f"❌ Error initializing vector store: {str(e)}")
#         # Return a minimal vector store with a placeholder text if there's an error
#         embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
#         return FAISS.from_texts(["Mental health information placeholder."], embeddings)

# Initialize the vector store
vector_store = initialize_hnsw_vector_store()

# Create a retriever
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5},
    threshold=0.5
)

# Cache for patient data to avoid redundant database queries
patient_data_cache = {}

def get_cached_patient_data(patient_id: str, db: Session, max_age_seconds: int = 300):
    """Get patient data from cache or fetch it if not available or expired."""
    current_time = datetime.now()

    # Check if data is in cache and not expired
    if patient_id in patient_data_cache:
        cache_entry = patient_data_cache[patient_id]
        age = (current_time - cache_entry["timestamp"]).total_seconds()

        if age < max_age_seconds:
            print(f"Using cached patient data for {patient_id} (age: {age:.2f} seconds)")
            return cache_entry["data"]

    # Fetch data in parallel
    print(f"Cache miss for {patient_id}, fetching data in parallel")
    data = fetch_patient_data_parallel(patient_id, db)

    # Update cache
    patient_data_cache[patient_id] = {
        "data": data,
        "timestamp": current_time
    }

    return data

# Helper function to return JSON responses
def return_json(data, code, message):
    return {'data': data, 'code': code, 'message': message}

# Function to create a conversation summary for better context retrieval
def create_conversation_summary(chat_history, current_question):
    """
    Create a summary of the conversation to improve context retrieval for follow-up questions.

    Args:
        chat_history: List of previous messages in the conversation
        current_question: The current question from the user

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
                role = "User" if msg.get("role") == "patient" else "AI"
                formatted_history += f"{role}: {msg.get('message', '')}\n"

            # Create the prompt for summarization
            prompt = f"""
            I need to summarize a conversation to improve context retrieval for a follow-up question.

            Previous conversation:
            {formatted_history}

            Current question: "{current_question}"

            Create a comprehensive search query that combines the main topics from the conversation with the current question.
            Focus on mental health related terms and concepts that would help retrieve relevant information.
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
            print(f"Error generating conversation summary: {str(e)}")
            # Fall back to simple concatenation

    # Simple fallback: combine the last message with the current question
    if len(chat_history) > 0:
        last_message = chat_history[-1].get("message", "")
        return f"{last_message} {current_question}"

    return current_question


# Create a LangChain agent
def create_agent(db: Session):
    """Create a LangChain agent with all available tools."""
    # try:
    #     # Try to import and use the LangChain agent
    #     from services.langchain_agent import create_langchain_agent
    #     print("Using LangChain agent")
    #     return create_langchain_agent(db, retriever)
    # except ImportError as e:
    #     print(f"LangChain imports not available, using original SmartAgent: {str(e)}")
    #     # Fall back to the original SmartAgent implementation if LangChain is not available

    return SmartAgent(db)

# Parallel data fetching functions
def fetch_patient_data_parallel(patient_id: str, db: Session):
    """Fetch all patient data in parallel using ThreadPoolExecutor."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        # Create tasks for each data fetch operation
        patient_info_future = executor.submit(fetch_patient_info, patient_id, db)
        diary_entries_future = executor.submit(fetch_diary_entries, patient_id, db)
        emotion_analysis_future = executor.submit(fetch_emotion_analysis, patient_id, db)
        medical_history_future = executor.submit(fetch_medical_history, patient_id, db)
        prescriptions_future = executor.submit(fetch_prescriptions, patient_id, db)
        chat_history_future = executor.submit(fetch_chat_history, patient_id, db)

        # Wait for all tasks to complete and get results
        patient_info = patient_info_future.result()
        diary_entries = diary_entries_future.result()
        emotion_analysis = emotion_analysis_future.result()
        medical_history = medical_history_future.result()
        prescriptions = prescriptions_future.result()
        chat_history = chat_history_future.result()

        return {
            "patient_info": patient_info,
            "diary_entries": diary_entries,
            "emotion_analysis": emotion_analysis,
            "medical_history": medical_history,
            "prescriptions": prescriptions,
            "chat_history": chat_history
        }

def fetch_patient_info(patient_id: str, db: Session):
    """Fetch patient information from the database."""
    try:
        patient = db.query(Patient).filter(Patient.patient_id == patient_id).first()
        if patient:
            # Handle treatment field which might be JSONB in the database
            treatment_info = patient.treatment
            if isinstance(treatment_info, dict):
                treatment_str = json.dumps(treatment_info)
            else:
                treatment_str = str(treatment_info) if treatment_info else ""

            return {
                "name": f"{patient.first_name} {patient.last_name}",
                "gender": patient.gender,
                "health_score": patient.health_score,
                "under_medications": patient.under_medications,
                "treatment": treatment_str,
                "region": patient.region if hasattr(patient, 'region') else None,
                "timezone": patient.timezone if hasattr(patient, 'timezone') else None,
                "isOnboarded": patient.isOnboarded if hasattr(patient, 'isOnboarded') else None
            }
        return None
    except Exception as e:
        print(f"Error fetching patient info: {str(e)}")
        return None

def fetch_diary_entries(patient_id: str, db: Session):
    """Fetch diary entries for a patient."""
    try:
        entries = db.query(DiaryEntry).filter(DiaryEntry.patient_id == patient_id).order_by(DiaryEntry.created_at.desc()).limit(5).all()
        return [{"date": entry.created_at.strftime("%Y-%m-%d %H:%M:%S") if entry.created_at else "Unknown",
                 "content": entry.notes} for entry in entries] if entries else []
    except Exception as e:
        print(f"Error fetching diary entries: {str(e)}")
        return []

def fetch_emotion_analysis(patient_id: str, db: Session):
    """Fetch emotion analysis for a patient."""
    try:
        emotions = db.query(EmotionAnalysis).filter(EmotionAnalysis.patient_id == patient_id).order_by(EmotionAnalysis.analyzed_at.desc()).limit(5).all()
        return [{"date": emotion.analyzed_at.strftime("%Y-%m-%d %H:%M:%S") if emotion.analyzed_at else "Unknown",
                 "emotion": emotion.emotion_category,
                 "confidence": float(emotion.confidence_score) if emotion.confidence_score else 0.0}
                for emotion in emotions] if emotions else []
    except Exception as e:
        print(f"Error fetching emotion analysis: {str(e)}")
        return []

def fetch_medical_history(patient_id: str, db: Session):
    """Fetch medical history for a patient."""
    try:
        history = db.query(MedicalHistory).filter(MedicalHistory.patient_id == patient_id).all()
        return [{"diagnosis": item.diagnosis, "treatment": item.treatment, "date": item.diagnosed_date.strftime("%Y-%m-%d")} for item in history] if history else []
    except Exception as e:
        print(f"Error fetching medical history: {str(e)}")
        return []

def fetch_prescriptions(patient_id: str, db: Session):
    """Fetch prescriptions for a patient."""
    try:
        # Use status field instead of active field
        prescriptions = db.query(Prescription).filter(Prescription.patient_id == patient_id, Prescription.status == "Active").all()
        return [{"medication": p.medication_name,
                 "dosage": p.dosage,
                 "instructions": p.instructions,
                 "start_date": p.start_date.strftime("%Y-%m-%d") if p.start_date else "Unknown",
                 "end_date": p.end_date.strftime("%Y-%m-%d") if p.end_date else "Ongoing"}
                for p in prescriptions] if prescriptions else []
    except Exception as e:
        print(f"Error fetching prescriptions: {str(e)}")
        return []

def fetch_chat_history(patient_id: str, db: Session):
    """Fetch chat history for a patient."""
    try:
        messages = db.query(ChatMessage).filter(
            ((ChatMessage.sender_id == patient_id) & (ChatMessage.receiver_id == AI_DOCTOR_ID)) |
            ((ChatMessage.sender_id == AI_DOCTOR_ID) & (ChatMessage.receiver_id == patient_id))
        ).order_by(ChatMessage.timestamp.desc()).limit(10).all()

        chat_history = []
        for msg in messages:
            role = "patient" if msg.sender_id == patient_id else "doctor"
            chat_history.append({
                "role": role,
                "message": msg.message_text,
                "timestamp": msg.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            })

        # Reverse to get chronological order
        chat_history.reverse()
        return chat_history
    except Exception as e:
        print(f"Error fetching chat history: {str(e)}")
        return []

#

# WebSocket endpoint for smart chat with AI
@smart_chat_router.websocket("/smart-chat/{patient_id}")
async def smart_chat_with_ai(websocket: WebSocket, patient_id: str, db: Session = Depends(get_db)):
    print(f"\n\n=== NEW WEBSOCKET CONNECTION REQUEST ===")
    print(f"Patient ID: {patient_id}")
    print(f"Query parameters: {websocket.query_params}")
    print(f"Headers: {websocket.headers}")
    print(f"=== END CONNECTION REQUEST ===\n\n")
    """
    WebSocket endpoint for smart chat with AI.

    Args:
        websocket (WebSocket): WebSocket connection.
        patient_id (str): Patient ID.
        db (Session): Database session.

        input (str): User input.
        text (str): Text input.
        {
            "text": "Hello, how are you?",
        }
        audio (str): Audio input.
        {
            "audio": "base64 encoded audio data",
        }

    Returns:
        dict: JSON response with AI response and extracted keywords.
    """
    # Accept the connection first
    await websocket.accept()
    print(f"WebSocket connection accepted for patient: {patient_id}")

    # Get JWT token from query parameters
    token = None
    if "token" in websocket.query_params:
        token = websocket.query_params["token"]
        print(f"Found token in query parameters")

    # Check if we have a token
    if not token:
        print(f"No token provided for patient: {patient_id}")
        await websocket.send_text(json.dumps({
            "response": "Error: Authentication required. Please provide a valid JWT token as a query parameter (token=...).",
            "extracted_keywords": []
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
        print(f"Invalid token provided for patient: {patient_id}")
        await websocket.send_text(json.dumps({
            "response": "Error: Invalid authentication token.",
            "extracted_keywords": []
        }))
        await websocket.close()
        return

    print(f"Token validated for patient: {patient_id}")

    # Validate patient ID
    try:
        # Create a new session to avoid transaction issues

        new_db = SessionLocal()

        try:
            patient = new_db.query(Patient).filter(Patient.patient_id == patient_id).first()
            if not patient:
                print(f"Patient not found in WebSocket: {patient_id}")
                await websocket.send_text(json.dumps({
                    "response": "Error: Patient not found.",
                    "extracted_keywords": []
                }))
                await websocket.close()
                new_db.close()
                return
            print(f"Patient validated in WebSocket: {patient.first_name} {patient.last_name}")
        finally:
            new_db.close()
    except Exception as e:
        print(f"Error validating patient: {str(e)}")
        await websocket.send_text(json.dumps({
            "response": "Error: Could not validate patient.",
            "extracted_keywords": []
        }))
        await websocket.close()
        return

    # Create the agent
    agent_executor = create_agent(db)

    # Initialize default values for message_text and extracted_keywords
    # message_text = "Welcome to the smart chat service."
    # extracted_keywords = []

    # Initialize chat history
    chat_history = []

    # Fetch initial patient data using cache
    print(f"Fetching patient data for patient: {patient_id}")
    start_time = datetime.now()
    patient_data = get_cached_patient_data(patient_id, db)
    end_time = datetime.now()
    print(f"Patient data fetching completed in {(end_time - start_time).total_seconds()} seconds")
    try:
        while True:
            data = await websocket.receive_json()

            # Get the user input
            user_input = ""
            if "text" in data:
                user_input = data["text"]
                print(f"Received text input: {user_input}")

                # Process the user input first
                if not user_input:
                    await websocket.send_text(json.dumps({
                        "response": "Please enter a message.",
                        "extracted_keywords": []
                    }))
                    continue

                # Continue with the rest of the processing

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

                    print(f"Transcribed audio: {user_input}")
                except Exception as e:
                    print(f"❌ Error processing audio: {str(e)}")
                    # Send error to client
                    await websocket.send_text(json.dumps({
                        "error": "Audio processing failed",
                        "response": f"Error processing audio: {str(e)}"
                    }))
                    # Skip further processing for this message
                    continue
            else:
                user_input = data.get("text", "").strip()

            if not user_input:
                await websocket.send_text(json.dumps({
                    "response": "Please enter a message.",
                    "extracted_keywords": []
                }))
                continue

            # Save user message to database with transaction handling
            try:
                # Create a new session to avoid transaction issues

                new_db = SessionLocal()

                try:
                    new_db.add(ChatMessage(
                        chat_message_id=str(uuid.uuid4()),
                        sender_id=patient_id,
                        receiver_id=AI_DOCTOR_ID,
                        message_text=user_input,
                        timestamp=datetime.now()
                    ))
                    new_db.commit()
                    print("✅ Saved user message to database")
                except Exception as inner_error:
                    new_db.rollback()  # Roll back the transaction on error
                    print(f"❌ Error saving user message to database: {str(inner_error)}")
                finally:
                    new_db.close()
            except Exception as db_error:
                print(f"❌ Error creating new database session: {str(db_error)}")
                # Continue anyway - we can still try to generate a response

            # No direct responses - let the LLM handle everything

            # Initialize default values for message_text and extracted_keywords
            message_text = "I'm sorry, but I encountered an error processing your request."
            extracted_keywords = []

            # Debug chat history
            print(f"Current chat history ({len(chat_history)} messages):")
            for i, msg in enumerate(chat_history[-4:]):  # Show last 4 messages for brevity
                print(f"  {i+1}. {msg.get('role')}: {msg.get('message')[:50]}...")

            # Process context and generate response in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                # Task 1: Create conversation summary and retrieve relevant documents
                def process_context():
                    # Create a conversation summary for better context retrieval
                    conversation_summary = create_conversation_summary(chat_history, user_input)
                    print(f"Created conversation summary: {conversation_summary}")

                    # Use the summary for retrieval if it's a follow-up question, otherwise use the original input
                    retrieval_query = conversation_summary if len(chat_history) > 0 else user_input
                    print(f"Using retrieval query: {retrieval_query}")

                    # Log the difference between original query and enhanced query
                    if retrieval_query != user_input:
                        print("=== QUERY ENHANCEMENT ===")
                        print(f"Original: {user_input}")
                        print(f"Enhanced: {retrieval_query}")
                        print("=========================")

                    # Get top-k relevant documents
                    relevant_docs = retriever.get_relevant_documents(retrieval_query)

                    # Log the number of documents retrieved
                    print(f"Retrieved {len(relevant_docs)} relevant documents")

                    # Extract the context (text content from the chunks)
                    contexts = [doc.page_content for doc in relevant_docs]
                    context = "\n\n".join(contexts)

                    # Log a preview of the retrieved context
                    context_preview = context[:200] + "..." if len(context) > 200 else context
                    print(f"Context preview: {context_preview}")

                    return context

                # Task 2: Prepare patient data for the agent
                def prepare_patient_data():
                    # Use the patient data we fetched earlier
                    return patient_data

                # Execute both tasks in parallel
                context_future = executor.submit(process_context)
                patient_data_future = executor.submit(prepare_patient_data)

                # Get results
                context = context_future.result()
                patient_data_result = patient_data_future.result()

            # Run the agent with extensive error handling
            try:
                print(f"Invoking agent with input: {user_input}")
                # print(f"Chat history length: {len(chat_history)}")

                # Add patient_id to the system message
                # system_message = {"role": "system", "content": f"patient_id: {patient_id}"}
                # full_history = chat_history + [system_message]

                # Invoke the agent with database session
                result = agent_executor.invoke(
                    user_input=user_input,
                    patient_id=patient_id,
                    dsm_content=context
                    # db=db  # Pass the database session to the agent
                )

                print(f"Agent result type: {type(result)}")
                print(f"Agent result keys: {result.keys() if isinstance(result, dict) else 'Not a dict'}")

                # Parse the agent's response
                if isinstance(result, dict) and "output" in result:
                    response_text = result["output"]
                    print(f"Got response text of length: {len(response_text)}")

                    # Try to extract JSON from the response
                    try:
                        # SmartAgent.invoke returns a JSON string in the "output" field
                        # So we need to parse it directly
                        print("Parsing JSON from SmartAgent output...")
                        parsed_response = json.loads(response_text)

                        # If parsing succeeds but doesn't have the expected structure,
                        # create a default structure
                        if not isinstance(parsed_response, dict) or "response" not in parsed_response:
                            print("Parsed JSON doesn't have expected structure, creating default")
                            parsed_response = {
                                "response": str(parsed_response),
                                "extracted_keywords": []
                            }
                    except json.JSONDecodeError:
                        # If direct parsing fails, try alternative methods
                        try:
                            # Check if the response is in markdown code blocks
                            import re
                            json_match = re.search(r"```json\s*(.*?)\s*```", response_text, re.DOTALL)
                            if json_match:
                                print("Found JSON in markdown code block, parsing...")
                                parsed_response = json.loads(json_match.group(1))
                            else:
                                # If no JSON found, create a default structure
                                print("No JSON found in response, creating default structure")
                                parsed_response = {
                                    "response": response_text,
                                    "extracted_keywords": []
                                }
                        except Exception as inner_e:
                            print(f"Error in alternative parsing: {str(inner_e)}")
                            parsed_response = {
                                "response": response_text,
                                "extracted_keywords": []
                            }
                else:
                    print(f"Unexpected result format: {result}")
                    # Use default values instead of raising an error
                    message_text = "I'm sorry, but I received an unexpected response format."
                    extracted_keywords = []
                    # Skip JSON parsing and continue with the default values
                    parsed_response = {
                        "response": message_text,
                        "extracted_keywords": extracted_keywords
                    }

                message_text = parsed_response.get("response", "")
                extracted_keywords = parsed_response.get("extracted_keywords", [])
                print(f"Final message text length: {len(message_text)}")
                print(f"Extracted keywords: {extracted_keywords}")
            except Exception as e:
                import traceback
                print(f"❌ Agent error: {str(e)}")
                print(f"Traceback: {traceback.format_exc()}")
                # Use default values instead of raising the error
                message_text = f"I'm sorry, but I encountered an error: {str(e)}"
                extracted_keywords = []

            # Save AI response to database with transaction handling
            try:
                # Create a new session to avoid transaction issues

                new_db = SessionLocal()

                try:
                    new_db.add(ChatMessage(
                        chat_message_id=str(uuid.uuid4()),
                        sender_id=AI_DOCTOR_ID,
                        receiver_id=patient_id,
                        message_text=message_text,
                        extracted_keywords=json.dumps(extracted_keywords),
                        timestamp=datetime.now()
                    ))
                    new_db.commit()
                    print("✅ Saved AI response to database")
                except Exception as inner_error:
                    new_db.rollback()  # Roll back the transaction on error
                    print(f"❌ Error saving AI response to database: {str(inner_error)}")
                finally:
                    new_db.close()
            except Exception as db_error:
                print(f"❌ Error creating new database session: {str(db_error)}")
                # Continue anyway - we can still send the response to the client

            # Update chat history
            chat_history.append({"role": "patient", "message": user_input, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
            chat_history.append({"role": "doctor", "message": message_text, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})

            # Keep chat history limited to last 10 exchanges (20 messages)
            if len(chat_history) > 20:
                chat_history = chat_history[-20:]

            print(f"Updated chat history. Current length: {len(chat_history)}")

            # Send response to client
            await websocket.send_text(json.dumps({
                "response": message_text,
                "extracted_keywords": extracted_keywords
            }))

    except WebSocketDisconnect:
        print(f"❌ Patient {patient_id} disconnected from smart chat.")
    except Exception as e:
        import traceback
        error_message = f"Error on line {sys.exc_info()[-1].tb_lineno}: {str(e)}"
        print(f"\n\n=== WEBSOCKET ERROR ===\n\n")
        print(f"❌ {error_message}")
        print(f"Traceback: {traceback.format_exc()}")
        print(f"\n\n=== END WEBSOCKET ERROR ===\n\n")

        # Set default values for message_text and extracted_keywords
        message_text = f"An error occurred: {str(e)}"
        extracted_keywords = []

        try:
            await websocket.send_text(json.dumps({
                "response": message_text,
                "extracted_keywords": extracted_keywords
            }))
        except Exception as ws_error:
            print(f"Error sending error message to websocket: {str(ws_error)}")
    finally:
        await websocket.close()
        print(f"✅ Smart chat WebSocket closed for patient {patient_id}.")

# HTTP endpoint to check if a patient exists
@smart_chat_router.get("/validate-patient/{patient_id}")
async def validate_patient(patient_id: str, db: Session = Depends(get_db), authorization: str = Header(default=None)):
    print(f"Validating patient ID: {patient_id}")

    # Check for authorization header
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Extract token
    token = authorization.split(" ")[1]

    # Get static JWT token from environment
    from os import getenv
    from dotenv import load_dotenv

    # Load environment variables
    load_dotenv()

    # Get static JWT token
    STATIC_JWT_TOKEN = getenv("STATIC_JWT_TOKEN")

    # Validate token (simple check for static token)
    if token != STATIC_JWT_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        # Create a new session to avoid transaction issues
        new_db = SessionLocal()

        try:
            patient = new_db.query(Patient).filter(Patient.patient_id == patient_id).first()
            if not patient:
                print(f"Patient not found: {patient_id}")
                new_db.close()
                raise HTTPException(status_code=404, detail="Patient not found")
            print(f"Patient found: {patient.first_name} {patient.last_name}")
            result = {"valid": True, "patient_name": f"{patient.first_name} {patient.last_name}"}
            new_db.close()
            return result
        except Exception as inner_e:
            new_db.close()
            raise inner_e
    except Exception as e:
        print(f"Error validating patient: {str(e)}")
        raise

