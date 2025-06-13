import os
import uuid
import json
import base64
import asyncio
import concurrent.futures
from datetime import datetime
from dotenv import load_dotenv
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
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

# Load environment variables
load_dotenv()
ai_audio_smart_chat_router = APIRouter()

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
AI_DOCTOR_ID = "00000000-0000-0000-0000-000000000000"

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

        # Create embeddings and vector store
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vector_store = FAISS.from_texts(chunks, embeddings)

        # Save the vector store
        os.makedirs(HNSW_INDEX_DIR, exist_ok=True)
        vector_store.save_local(HNSW_INDEX_DIR)
        print("✅ Created and saved new FAISS HNSW index")
        return vector_store
    except Exception as e:
        print(f"❌ Error initializing HNSW vector store: {str(e)}")
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

# Cache for patient data
patient_data_cache = {}

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
        print(f"Error retrieving information: {str(e)}")
        return []

def fetch_patient_data_parallel(patient_id: str, db: Session):
    """Fetch patient data in parallel for better performance."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Define tasks for each data type
        tasks = {
            "patient": executor.submit(lambda: db.query(Patient).filter(Patient.patient_id == patient_id).first()),
            "onboarding": executor.submit(lambda: db.query(OnboardingQuestion).filter(OnboardingQuestion.patient_id == patient_id).all()),
            "diary": executor.submit(lambda: db.query(DiaryEntry).filter(DiaryEntry.patient_id == patient_id).order_by(desc(DiaryEntry.entry_date)).limit(5).all()),
            "emotions": executor.submit(lambda: db.query(EmotionAnalysis).filter(EmotionAnalysis.patient_id == patient_id).order_by(desc(EmotionAnalysis.analysis_date)).limit(5).all()),
            "medical": executor.submit(lambda: db.query(MedicalHistory).filter(MedicalHistory.patient_id == patient_id).all()),
            "prescriptions": executor.submit(lambda: db.query(Prescription).filter(Prescription.patient_id == patient_id).all())
        }

        # Collect results
        results = {}
        for key, task in tasks.items():
            try:
                results[key] = task.result()
            except Exception as e:
                print(f"Error fetching {key} data: {str(e)}")
                results[key] = None

        return results

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

def create_conversation_summary(chat_history, current_question):
    """Create a summary of the conversation for better context retrieval."""
    if len(chat_history) == 0:
        return current_question

    # If we have OpenAI available, use it to create a better summary
    if OPENAI_CLIENT:
        try:
            # Format the chat history
            formatted_history = ""
            for msg in chat_history[-5:]:  # Use last 5 messages for context
                role = msg.get("role", "unknown")
                message = msg.get("message", "")
                formatted_history += f"{role}: {message}\n"

            # Create the prompt
            prompt = f"""
            Here is a conversation between a patient and an AI doctor:

            {formatted_history}

            Current question from patient: {current_question}

            Based on this conversation, create a search query that would help find relevant information to answer the patient's current question. The query should capture the main topic and any specific details that would be useful for retrieval.
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
    return SmartAgent(db)

def fetch_chat_history(patient_id: str, db: Session):
    """Fetch chat history for a patient."""
    try:
        messages = db.query(ChatMessage).filter(
            ((ChatMessage.sender_id == patient_id) & (ChatMessage.receiver_id == AI_DOCTOR_ID)) |
            ((ChatMessage.sender_id == AI_DOCTOR_ID) & (ChatMessage.receiver_id == patient_id))
        ).order_by(ChatMessage.createdAt.desc()).limit(10).all()

        chat_history = []
        for msg in messages:
            role = "patient" if msg.sender_id == patient_id else "doctor"
            chat_history.append({
                "role": role,
                "message": msg.message_text,
                "timestamp": msg.createdAt.strftime("%Y-%m-%d %H:%M:%S") if msg.createdAt else "Unknown"
            })

        # Reverse to get chronological order
        chat_history.reverse()
        return chat_history
    except Exception as e:
        print(f"Error fetching chat history: {str(e)}")
        return []

def generate_audio_from_text(text: str, voice: str = "alloy"):
    """Generate audio from text using OpenAI's text-to-speech API.

    Args:
        text (str): The text to convert to speech
        voice (str): The voice to use. Options: alloy, echo, fable, onyx, nova, shimmer

    Returns:
        str: Base64 encoded audio data or None if generation failed
    """
    if not OPENAI_CLIENT:
        print("OpenAI client not initialized. Cannot generate audio.")
        return None

    # Validate voice parameter
    valid_voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
    if voice not in valid_voices:
        print(f"Invalid voice: {voice}. Using default voice 'alloy'.")
        voice = "alloy"

    try:
        print(f"Generating audio with voice: {voice}")
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
        print(f"Error generating audio: {str(e)}")
        return None

# WebSocket endpoint for smart chat with AI and audio
@ai_audio_smart_chat_router.websocket("/audio-smart-chat/{patient_id}")
async def audio_smart_chat_with_ai(websocket: WebSocket, patient_id: str, db: Session = Depends(get_db)):
    print(f"\n\n=== NEW WEBSOCKET CONNECTION REQUEST ===")
    print(f"Patient ID: {patient_id}")
    print(f"Query parameters: {websocket.query_params}")
    print(f"Headers: {websocket.headers}")
    print(f"=== END CONNECTION REQUEST ===\n\n")
    """
    WebSocket endpoint for smart chat with AI with audio support.

    Args:
        websocket (WebSocket): WebSocket connection.
        patient_id (str): Patient ID.
        db (Session): Database session.

    Input formats:
        text (str): Text input.
        {
            "text": "Hello, how are you?",
        }
        audio (str): Audio input.
        {
            "audio": "base64 encoded audio data",
        }

    Output format:
        {
            "response": "AI response text",
            "audio": "base64 encoded audio data",
            "extracted_keywords": ["keyword1", "keyword2", ...]
        }
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

    # Create the agent
    agent_executor = create_agent(db)
    print(f"Created agent for patient: {patient_id}")

    # Initialize chat history
    chat_history = fetch_chat_history(patient_id, db)
    print(f"Fetched chat history for patient: {patient_id}. {len(chat_history)} messages found.")

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            data = json.loads(data)
            print(f"Received message from patient {patient_id}: {data}")

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

                    print(f"Transcribed audio: {user_input}")

                    # Wait a moment to ensure the transcription is displayed before processing
                    await asyncio.sleep(0.5)
                except Exception as e:
                    print(f"❌ Error processing audio: {str(e)}")
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

            print(f"Processing input from patient {patient_id}: {user_input}")

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

                    # Retrieve relevant information
                    relevant_docs = retrieve_relevant_information(retrieval_query)
                    print(f"Retrieved {len(relevant_docs)} relevant documents")

                    # Combine the retrieved documents into a single context string
                    context = "\n\n".join(relevant_docs)
                    return context

                # Task 2: Get patient data
                def get_patient_context():
                    # Get patient data
                    patient_data = get_cached_patient_data(patient_id, db)
                    print(f"Got patient data for {patient_id}")
                    return patient_data

                # Execute both tasks in parallel
                context_future = executor.submit(process_context)
                patient_data_future = executor.submit(get_patient_context)

                # Wait for both tasks to complete
                context = context_future.result()
                patient_data = patient_data_future.result()

            # Run the agent with extensive error handling
            try:
                print(f"Invoking agent with input: {user_input}")

                # Invoke the agent with database session
                result = agent_executor.invoke(
                    user_input=user_input,
                    patient_id=patient_id,
                    dsm_content=context
                )

                print(f"Agent result type: {type(result)}")
                print(f"Agent result keys: {result.keys() if isinstance(result, dict) else 'Not a dict'}")

                # Extract the response and keywords
                if isinstance(result, dict) and "response" in result:
                    message_text = result["response"]
                    extracted_keywords = result.get("extracted_keywords", [])
                else:
                    message_text = str(result)
                    extracted_keywords = []

                print(f"Generated response: {message_text}")
                print(f"Extracted keywords: {extracted_keywords}")

            except Exception as e:
                print(f"❌ Error running agent: {str(e)}")
                message_text = f"I'm sorry, I encountered an error while processing your request. Please try again or contact support if the issue persists."
                extracted_keywords = []

            # Get the voice preference from the data
            voice = data.get("voice", "alloy")
            print(f"Using voice: {voice} for response")

            # Generate audio from the response text with the selected voice
            audio_base64 = generate_audio_from_text(message_text, voice)

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

            # Create a unique response ID to help client detect duplicates
            response_id = str(uuid.uuid4())

            # Send response to client with audio
            response_data = {
                "response": message_text,
                "extracted_keywords": extracted_keywords,
                "response_id": response_id  # Add a unique ID to each response
            }

            # Add audio if available
            if audio_base64:
                response_data["audio"] = audio_base64

            # Send the response directly, not wrapped in another object
            await websocket.send_text(json.dumps(response_data))

    except WebSocketDisconnect:
        print(f"WebSocket disconnected for patient: {patient_id}")
    except Exception as e:
        print(f"❌ Error in WebSocket connection: {str(e)}")
        try:
            await websocket.send_text(json.dumps({
                "error": "Server error",
                "response": f"An unexpected error occurred: {str(e)}"
            }))
        except:
            pass  # If we can't send the error, just log it
