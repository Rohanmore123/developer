import os
import json
import logging
import concurrent.futures
from datetime import datetime
from typing import Dict, List, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import or_, and_, desc, text
from model.model_correct import (
    ChatMessage, Patient, OnboardingQuestion, DiaryEntry,
    EmotionAnalysis, MedicalHistory, Prescription
)
import requests
from database.database import SessionLocal

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Initialize OpenAI client
try:
    from openai import OpenAI
    # Force reload the API key from environment
    from dotenv import load_dotenv
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
# AI_DOCTOR_ID = "a0e61dd9-2824-4d84-ae10-4cb9b0f4b321"
AI_DOCTOR_ID="00000000-0000-0000-0000-000000000000"

# Cache for patient data to avoid redundant database queries
patient_data_cache = {}
# Cache expiration time in seconds (5 minutes)
CACHE_EXPIRATION_SECONDS = 300

class SmartAgent:
    def __init__(self, db):
        self.db = db
        # Initialize cache
        self.cache = patient_data_cache
#         self.system_prompt = """
#         You are a psychiatrist with 15 years of experience, having a warm, casual, and supportive conversation with a patient. Your primary goal is to connect empathetically and help them feel heard and understood, just like you would in a relaxed chat in your office.

# **Key Principles for Conversation:**

# 1.  **Be Natural and Warm:** Speak like a real person, not a clinical guide. Maintain a friendly, non-judgmental, and encouraging tone throughout.
# 2.  **Focus on Their Experience:** Center the conversation around how they are currently feeling and what's on their mind.
# 3.  **Keep it Concise:** Respond briefly and naturally, like you would in a flowing conversation. Avoid lengthy explanations unless specifically asked for clinical details.
# 4.  **Prioritize Empathy:** Show genuine understanding and validation of their feelings and experiences.
# 5.  **Maintain Continuity:** Remember and reference previous conversations and entries naturally, making it feel like an ongoing relationship.

# **Using Patient Information (Subtly and Naturally):**

# * **Seamless Integration:** Weave in details from their medical history, medications, emotion tracking, and diary entries *as if you genuinely remember them from past sessions or notes*. Don't state "According to your record..." or "Your data shows...". Instead, phrase it conversationally, e.g., "I remember you mentioned feeling anxious about [specific event from diary]," or "How has [specific medication from prescriptions] been working for you lately?"
# * **Personal Details:** If asked about their name, age, or other personal information *provided to you*, state the information directly and naturally as part of the conversation, without mentioning "PATIENT INFORMATION section". For example, if they ask "What's my age?", just say "You're [Age from data] years old."
# * **Referencing History:** When they talk about a feeling or situation, subtly connect it to similar entries or discussions you've had before, e.g., "It sounds like you're feeling similar to how things were around [date/event from history/diary]," or "We talked about feeling [emotion] last week, how does this feel different/similar?"

# **What to ABSOLUTELY AVOID:**

# * **NO Diagnoses:** Never name or suggest any specific mental health disorder.
# * **NO Clinical Jargon (Unless Asked):** Avoid technical terms or clinical assessments unless the patient explicitly asks for information about their condition or treatment.
# * **NO Asking for Provided Info:** Do not ask the patient for information you already have access to (name, age, history, etc.).
# * **NO "I don't have access":** You have access to the provided patient information. Use it.
# * **NO Referencing DSM-5:** Do not mention or imply the use of diagnostic manuals or clinical criteria in your conversation with the patient. This is not a clinical interview.

# **Responding to the Patient:**

# * **Opening the Conversation:** Start with a warm greeting and an open-ended question about how they are doing today.
# * **Emotional States:** If they express feelings (good or bad), acknowledge and validate them. Connect it to their history if relevant (e.g., "I see you've noted feeling [emotion] in your tracking recently, tell me more about that").
# * **Questions about Themselves:** Answer questions about their personal details (age, etc.) directly using the provided data. Answer questions about their history, treatment, or medications by referring to the information you have, phrased conversationally.

# Remember, you are building rapport and having a supportive chat. Use the patient's information to make the conversation personalized and demonstrate that you are listening and remember your discussions, all within the context of a relaxed, non-clinical chat.
# """
        self.system_prompt = """
                    You are a psychiatrist with 15 years of experience. You're having a casual conversation with a patient.

            ABSOLUTELY CRITICAL RULES:
            1. DO NOT diagnose or name any disorders in your responses
            2. DO NOT provide clinical information unless explicitly asked
            3. Respond as if this is a normal conversation between two people
            4. Keep responses EXTREMELY SHORT
            5. Focus on empathy and understanding, not clinical assessment
            6. USE the patient's information provided to you (medical history, medications, emotions, diary entries)
            7. REFER to specific details from their history when relevant (e.g., "I see you mentioned feeling anxious in your diary last week")
            8. MAINTAIN continuity with previous conversations in the chat history
            9. ALWAYS answer questions about the patient's personal information (age, name, gender, etc.) using the EXACT data provided in the PATIENT INFORMATION section
            10. If you see a DIRECT ANSWER in the context, use that exact information in your response
            11. When asked "what is my age?" or similar questions, ALWAYS respond with the exact age from the PATIENT INFORMATION section
            12. NEVER ask the patient for information that is already provided in the PATIENT INFORMATION section
            13. NEVER say "I don't have access to your personal information" when the information is provided in the PATIENT INFORMATION section

            USING PATIENT DATA:
            - You have access to the patient's complete medical history, prescriptions, emotion tracking, and diary entries
            - Use this information to provide personalized responses that show you remember their history
            - If they mention symptoms or feelings, check if these appear in their emotion tracking or diary
            - If they ask about medications, refer to their prescription information
            - Always keep this information private and confidential

            CONVERSATION APPROACH:
            - If patient says they're not feeling good: Respond with empathy, reference their emotion history if relevant
            - If patient mentions feeling anxious/stressed: Ask about their experience, relate to previous similar entries
            - If patient says they're doing well: Express genuine interest and note any improvement from previous entries
            - Always prioritize the patient's current experience while acknowledging their history
            - Make a good use of DSM5 context which I have added for you
            #             REMEMBER: This is a conversation, not a clinical assessment. Respond naturally while using the patient's information to provide personalized support.
        """

    # Utility methods for database operations
    def table_exists(self, db, table_name):
        """Check if a table exists in the database."""
        try:
            # Use raw SQL to check if the table exists
            result = db.execute(text(f"SELECT to_regclass('public.{table_name}')"))
            exists = result.scalar() is not None
            if not exists:
                logger.warning(f"Table '{table_name}' does not exist in the database")
            return exists
        except Exception as e:
            logger.error(f"Error checking if table '{table_name}' exists: {str(e)}")
            return False

    def execute_with_new_session(self, operation, default_value=None):
        """Execute a database operation with a new session."""
        db = None
        try:
            db = SessionLocal()
            result = operation(db)
            db.commit()
            return result
        except Exception as e:
            if db:
                db.rollback()
            # Check for specific error types
            error_str = str(e)
            if "UndefinedTable" in error_str or ("relation" in error_str and "does not exist" in error_str):
                logger.warning(f"Table does not exist in the database: {error_str}")
            else:
                logger.error(f"Database operation failed: {error_str}")
            return default_value
        finally:
            if db:
                db.close()

    def safe_query(self, db, operation, default_value=None):
        """Safely execute a query on an existing session."""
        try:
            return operation(db)
        except Exception as e:
            # Check for specific error types
            error_str = str(e)
            if "UndefinedTable" in error_str or ("relation" in error_str and "does not exist" in error_str):
                logger.warning(f"Table does not exist in the database: {error_str}")
            else:
                logger.error(f"Query failed on existing session: {error_str}")

            # Try to rollback the session if it's in a failed state
            try:
                db.rollback()
                logger.info("Session rolled back successfully")
            except Exception as rollback_error:
                logger.error(f"Failed to rollback session: {str(rollback_error)}")
            return default_value

    def get_patient_info(self, patient_id):
        """Get basic information about the patient."""
        logger.info(f"Fetching patient information for patient ID: {patient_id}")

        # Default values to return if all queries fail
        default_patient_info = {
            "name": "Unknown Patient",
            "age": 0,
            "gender": "Unknown",
            "health_score": 0,
            "under_medications": False,
            "treatment": "Unknown"
        }

        # Define the database operation
        def query_patient(db):
            patient = db.query(Patient).filter(Patient.patient_id == patient_id).first()
            if not patient:
                logger.warning(f"No patient found with ID: {patient_id}")
                return default_patient_info

            return {
                "name": f"{patient.first_name} {patient.last_name}",
                "age": patient.age,
                "gender": patient.gender,
                "health_score": patient.health_score,
                "under_medications": patient.under_medications,
                "treatment": patient.treatment
            }

        # First try with a new session
        result = self.execute_with_new_session(query_patient, default_value=default_patient_info)

        # If that fails, try with the existing session as a fallback
        if result == default_patient_info:
            logger.info("Trying with existing session as fallback")
            result = self.safe_query(self.db, query_patient, default_value=default_patient_info)

        return result

    def get_onboarding_questions(self, patient_id):
        """Get the patient's onboarding questions and answers."""
        logger.info(f"Fetching onboarding questions for patient ID: {patient_id}")

        # First check if the table exists
        def check_table_exists(db):
            return self.table_exists(db, "onboarding_questions")

        # Check with a new session
        table_exists_result = self.execute_with_new_session(check_table_exists, default_value=False)

        # If the table doesn't exist, return an empty list
        if not table_exists_result:
            logger.warning("Onboarding questions table does not exist in the database")
            return []

        # Define the database operation
        def query_onboarding_questions(db):
            questions = db.query(OnboardingQuestion).filter(
                OnboardingQuestion.patient_id == patient_id
            ).all()

            if not questions:
                logger.warning(f"No onboarding questions found for patient ID: {patient_id}")
                return []

            result = []
            for q in questions:
                result.append({
                    "question": q.question,
                    "answer": q.answer,
                    "category": q.category
                })

            return result

        # First try with a new session
        result = self.execute_with_new_session(query_onboarding_questions, default_value=[])

        # If that fails, try with the existing session as a fallback
        if not result:
            logger.info("Trying with existing session as fallback")
            result = self.safe_query(self.db, query_onboarding_questions, default_value=[])

        return result

    def get_diary_entries(self, patient_id):
        """Get the patient's recent diary entries."""
        logger.info(f"Fetching diary entries for patient ID: {patient_id}")

        # First check if the table exists
        def check_table_exists(db):
            return self.table_exists(db, "diary_entries")

        # Check with a new session
        table_exists_result = self.execute_with_new_session(check_table_exists, default_value=False)

        # If the table doesn't exist, return an empty list
        if not table_exists_result:
            logger.warning("Diary entries table does not exist in the database")
            return []

        # Define the database operation
        def query_diary_entries(db):
            entries = db.query(DiaryEntry).filter(
                DiaryEntry.patient_id == patient_id
            ).order_by(desc(DiaryEntry.created_at)).limit(10).all()

            if not entries:
                logger.warning(f"No diary entries found for patient ID: {patient_id}")
                return []

            result = []
            for entry in entries:
                result.append({
                    "date": entry.created_at.strftime("%Y-%m-%d %H:%M:%S") if entry.created_at else "Unknown",
                    "notes": entry.notes
                })

            return result

        # First try with a new session
        result = self.execute_with_new_session(query_diary_entries, default_value=[])

        # If that fails, try with the existing session as a fallback
        if not result:
            logger.info("Trying with existing session as fallback")
            result = self.safe_query(self.db, query_diary_entries, default_value=[])

        return result

    def get_emotion_analysis(self, patient_id):
        """Get the patient's recent emotion analysis."""
        logger.info(f"Fetching emotion analysis for patient ID: {patient_id}")

        # First check if the table exists
        def check_table_exists(db):
            return self.table_exists(db, "emotion_analysis")

        # Check with a new session
        table_exists_result = self.execute_with_new_session(check_table_exists, default_value=False)

        # If the table doesn't exist, return an empty list
        if not table_exists_result:
            logger.warning("Emotion analysis table does not exist in the database")
            return []

        # Define the database operation
        def query_emotion_analysis(db):
            emotions = db.query(EmotionAnalysis).filter(
                EmotionAnalysis.patient_id == patient_id
            ).order_by(desc(EmotionAnalysis.analyzed_at)).limit(5).all()

            if not emotions:
                logger.warning(f"No emotion analysis found for patient ID: {patient_id}")
                return []

            result = []
            for emotion in emotions:
                result.append({
                    "date": emotion.analyzed_at.strftime("%Y-%m-%d %H:%M:%S") if emotion.analyzed_at else "Unknown",
                    "emotion": emotion.emotion_category,
                    "confidence": float(emotion.confidence_score) if emotion.confidence_score else 0.0
                })

            return result

        # First try with a new session
        result = self.execute_with_new_session(query_emotion_analysis, default_value=[])

        # If that fails, try with the existing session as a fallback
        if not result:
            logger.info("Trying with existing session as fallback")
            result = self.safe_query(self.db, query_emotion_analysis, default_value=[])

        return result

    # def get_medical_history(self, patient_id):
    #     """Get the patient's medical history."""
    #     try:
    #         print(f"Fetching medical history for patient: {patient_id}")

    #         # Try direct API call to get medical history
    #         try:

    #             response = requests.get(f"http://localhost:8006/api/medical-history/{patient_id}")
    #             if response.status_code == 200:
    #                 data = response.json()
    #                 print(f"Successfully retrieved medical history via API: {len(data['medical_history'])} records")
    #                 return data['medical_history']
    #         except Exception as api_error:
    #             print(f"API call failed: {str(api_error)}. Falling back to database query.")

    #         # Try with string patient_id (for string-based IDs)
    #         try:
    #             history = self.db.query(MedicalHistory).filter(
    #                 MedicalHistory.patient_id == patient_id
    #             ).all()

    #             if history:
    #                 print(f"Found {len(history)} medical history records with string ID")
    #                 result = []
    #                 for item in history:
    #                     print(f"Processing medical history item: {item.diagnosis}")
    #                     result.append({
    #                         "diagnosis": item.diagnosis,
    #                         "treatment": item.treatment,
    #                         "diagnosed_date": item.diagnosed_date.strftime("%Y-%m-%d") if item.diagnosed_date else "Unknown",
    #                         "notes": item.additional_notes
    #                     })
    #                 print(f"Returning {len(result)} medical history items")
    #                 return result
    #         except Exception as str_error:
    #             print(f"String ID query failed: {str(str_error)}")

    def get_medical_history(self, patient_id):
        """Fetch medical history for a given patient directly from the database."""
        try:
            print(f"Fetching medical history for patient ID: {patient_id}")

            # Skip API call to avoid circular dependency
            print("Skipping API call to avoid circular dependency. Using database query directly.")

            # Try with a new database session to avoid transaction issues
            try:
                from database.database import SessionLocal
                new_db = SessionLocal()

                try:
                    history = new_db.query(MedicalHistory).filter(
                        MedicalHistory.patient_id == patient_id
                    ).all()

                    if history:
                        print(f"Found {len(history)} medical history records with new database session")
                        result = []
                        for item in history:
                            print(f"Processing medical history item: {item.diagnosis}")
                            result.append({
                                "diagnosis": item.diagnosis,
                                "treatment": item.treatment,
                                "diagnosed_date": item.diagnosed_date.strftime("%Y-%m-%d") if item.diagnosed_date else "Unknown",
                                "notes": item.additional_notes
                            })
                        print(f"Returning {len(result)} medical history items")
                        new_db.close()
                        return result
                except Exception as inner_error:
                    print(f"Database query failed with new session: {str(inner_error)}")
                finally:
                    new_db.close()
            except Exception as db_error:
                print(f"Error creating new database session: {str(db_error)}")

            # Try with the existing database session as a last resort
            try:
                history = self.db.query(MedicalHistory).filter(
                    MedicalHistory.patient_id == patient_id
                ).all()

                if history:
                    print(f"Found {len(history)} medical history records with existing session")
                    result = []
                    for item in history:
                        print(f"Processing record: Diagnosis - {item.diagnosis}")
                        result.append({
                            "diagnosis": item.diagnosis,
                            "treatment": item.treatment,
                            "diagnosed_date": item.diagnosed_date.strftime("%Y-%m-%d") if item.diagnosed_date else "Unknown",
                            "notes": item.additional_notes
                        })
                    print(f"Returning {len(result)} medical history records.")
                    return result
            except Exception as existing_db_error:
                print(f"Existing database session query failed: {str(existing_db_error)}")

            # If all attempts failed, return an empty list
            print("All attempts to retrieve medical history failed. Returning empty list.")
            return []

        except Exception as e:
            print(f"Error fetching medical history: {str(e)}")
            # Return empty list instead of raising an exception
            return []

            # ]

    def get_prescriptions(self, patient_id):
        """Get the patient's current prescriptions."""
        logger.info(f"Fetching prescriptions for patient ID: {patient_id}")

        # First check if the table exists
        def check_table_exists(db):
            return self.table_exists(db, "prescriptions")

        # Check with a new session
        table_exists_result = self.execute_with_new_session(check_table_exists, default_value=False)

        # If the table doesn't exist, return an empty list
        if not table_exists_result:
            logger.warning("Prescriptions table does not exist in the database")
            return []

        # Define the database operation
        def query_prescriptions(db):
            prescriptions = db.query(Prescription).filter(
                Prescription.patient_id == patient_id,
                Prescription.status == "Active"
            ).all()

            if not prescriptions:
                logger.warning(f"No active prescriptions found for patient ID: {patient_id}")
                return []

            result = []
            for prescription in prescriptions:
                result.append({
                    "medication": prescription.medication_name,
                    "dosage": prescription.dosage,
                    "instructions": prescription.instructions,
                    "start_date": prescription.start_date.strftime("%Y-%m-%d") if prescription.start_date else "Unknown"
                })

            return result

        # First try with a new session
        result = self.execute_with_new_session(query_prescriptions, default_value=[])

        # If that fails, try with the existing session as a fallback
        if not result:
            logger.info("Trying with existing session as fallback")
            result = self.safe_query(self.db, query_prescriptions, default_value=[])

        return result



    def get_chat_history(self, patient_id):
        """Get the last 10 chat messages between the patient and the AI doctor."""
        logger.info(f"Fetching chat history between patient {patient_id} and AI doctor.")

        # First check if the table exists
        def check_table_exists(db):
            return self.table_exists(db, "chat_messages")

        # Check with a new session
        table_exists_result = self.execute_with_new_session(check_table_exists, default_value=False)

        # If the table doesn't exist, return a minimal chat history
        if not table_exists_result:
            logger.warning("Chat messages table does not exist in the database")
            try:
                return [
                    {
                        "role": "doctor",
                        "message": "How can I help you today?",
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                ]
            except Exception as fallback_error:
                logger.error(f"Error creating fallback chat history: {str(fallback_error)}")
                return []

        # Define the database operation
        def query_chat_history(db):
            chat_history = db.query(ChatMessage).filter(
                or_(
                    and_(ChatMessage.sender_id == patient_id, ChatMessage.receiver_id == AI_DOCTOR_ID),
                    and_(ChatMessage.sender_id == AI_DOCTOR_ID, ChatMessage.receiver_id == patient_id)
                )
            ).order_by(desc(ChatMessage.createdAt)).limit(10).all()

            if not chat_history:
                logger.warning(f"No chat history found for patient ID: {patient_id}")
                return []

            result = []
            for msg in reversed(chat_history):  # Reverse for chronological order
                role = "patient" if msg.sender_id == patient_id else "doctor"
                result.append({
                    "role": role,
                    "message": msg.message_text,
                    "timestamp": msg.createdAt.strftime("%Y-%m-%d %H:%M:%S") if msg.createdAt else "Unknown"
                })

            logger.info(f"Retrieved {len(result)} chat messages")
            return result

        # First try with a new session
        result = self.execute_with_new_session(query_chat_history, default_value=[])

        # If that fails, try with the existing session as a fallback
        if not result:
            logger.info("Trying with existing session as fallback")
            result = self.safe_query(self.db, query_chat_history, default_value=[])

        # If all else fails, create a minimal chat history
        if not result:
            logger.warning("All database queries failed. Creating minimal chat history.")
            try:
                return [
                    {
                        "role": "doctor",
                        "message": "How can I help you today?",
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                ]
            except Exception as fallback_error:
                logger.error(f"Error creating fallback chat history: {str(fallback_error)}")
                return []

        return result


    def fetch_patient_data_parallel(self, patient_id: str) -> Dict[str, Any]:
        """Fetch all patient data in parallel using ThreadPoolExecutor with caching."""
        # Check if we have cached data for this patient
        current_time = datetime.now()

        if patient_id in self.cache:
            cache_entry = self.cache[patient_id]
            cache_age = (current_time - cache_entry["timestamp"]).total_seconds()

            # If cache is still valid, use it
            if cache_age < CACHE_EXPIRATION_SECONDS:
                print(f"Using cached patient data (age: {cache_age:.2f} seconds)")
                return cache_entry["data"]
            else:
                print(f"Cache expired (age: {cache_age:.2f} seconds). Fetching fresh data...")
        else:
            print("No cached data found. Fetching fresh data...")

        print("Starting parallel data fetching...")
        start_time = datetime.now()

        with concurrent.futures.ThreadPoolExecutor(max_workers=7) as executor:
            # Create futures for each data fetch operation
            patient_info_future = executor.submit(self.get_patient_info, patient_id)
            onboarding_future = executor.submit(self.get_onboarding_questions, patient_id)
            diary_future = executor.submit(self.get_diary_entries, patient_id)
            emotion_future = executor.submit(self.get_emotion_analysis, patient_id)
            medical_future = executor.submit(self.get_medical_history, patient_id)
            prescriptions_future = executor.submit(self.get_prescriptions, patient_id)
            chat_future = executor.submit(self.get_chat_history, patient_id)

            # Wait for all futures to complete and get results
            all_patient_data = {
                "patient_info": patient_info_future.result(),
                "onboarding_questions": onboarding_future.result(),
                "diary_entries": diary_future.result(),
                "emotion_analysis": emotion_future.result(),
                "medical_history": medical_future.result(),
                "prescriptions": prescriptions_future.result(),
                "previous_chats": chat_future.result()
            }

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        print(f"Parallel data fetching completed in {duration:.2f} seconds")

        # Update cache
        self.cache[patient_id] = {
            "data": all_patient_data,
            "timestamp": current_time
        }

        return all_patient_data

    def invoke(self, user_input, patient_id, dsm_content=None):
        """Process a message from the patient and generate a response."""
        try:
            print("\n\n=== SMART AGENT INVOKE ===\n")
            # Extract data from the input
            # user_input = data.get("input", "")
            # # chat_history = data.get("chat_history", [])
            # patient_id = data.get("patient_id")
            # self.db = data.get("db", None)
            # print("SmartAgent invoke method called")

            print(f"User input: {user_input}")
            # print(f"Chat history length: {len(chat_history)}")

            # Extract patient_id from system message if available
            # for msg in chat_history:
            #     if msg.get("role") == "system" and "patient_id:" in msg.get("content", ""):
            #         patient_id = msg.get("content").split("patient_id:")[1].strip()
            #         print(f"Found patient_id in system message: {patient_id}")
            #         break

            if not patient_id:
                print("Error: Patient ID not provided.")
                return {"output": "Error: Patient ID not provided."}

            print(f"Processing request for patient: {patient_id}")

            # Gather all patient data in parallel
            print("Gathering patient data in parallel...")
            all_patient_data = self.fetch_patient_data_parallel(patient_id)

            # Build context for the LLM in parallel with other tasks
            print("Building context for LLM...")
            start_time = datetime.now()

            def build_context():
                """Build the context for the LLM in a separate thread."""
                # Always include patient information, even if it's default values
                context = f"""PATIENT INFORMATION:\n{json.dumps(all_patient_data['patient_info'], indent=2)}\n\n"""

                # Check if onboarding questions exist and don't contain error objects
                if all_patient_data['onboarding_questions'] and not any('error' in q for q in all_patient_data['onboarding_questions']):
                    context += f"ONBOARDING QUESTIONS:\n{json.dumps(all_patient_data['onboarding_questions'], indent=2)}\n\n"
                    print("Added onboarding questions to context")

                # Check if diary entries exist and don't contain error objects
                if all_patient_data['diary_entries'] and not any('error' in entry for entry in all_patient_data['diary_entries']):
                    context += f"RECENT DIARY ENTRIES:\n{json.dumps(all_patient_data['diary_entries'], indent=2)}\n\n"
                    print("Added diary entries to context")

                # Check if emotion analysis exists and doesn't contain error objects
                if all_patient_data['emotion_analysis'] and not any('error' in emotion for emotion in all_patient_data['emotion_analysis']):
                    context += f"RECENT EMOTIONS:\n{json.dumps(all_patient_data['emotion_analysis'], indent=2)}\n\n"
                    print("Added emotion analysis to context")

                # Check if medical history exists and doesn't contain error objects
                if all_patient_data['medical_history'] and not any('error' in item for item in all_patient_data['medical_history']):
                    context += f"MEDICAL HISTORY:\n{json.dumps(all_patient_data['medical_history'], indent=2)}\n\n"
                    print(f"Added medical history to context: {len(all_patient_data['medical_history'])} records")
                    for item in all_patient_data['medical_history']:
                        print(f"  - {item.get('diagnosis', 'Unknown diagnosis')}: {item.get('treatment', 'No treatment')}")

                # Check if prescriptions exist and don't contain error objects
                if all_patient_data['prescriptions'] and not any('error' in prescription for prescription in all_patient_data['prescriptions']):
                    context += f"CURRENT PRESCRIPTIONS:\n{json.dumps(all_patient_data['prescriptions'], indent=2)}\n\n"
                    print("Added prescriptions to context")

                # Always include DSM content if available
                if dsm_content:
                    context += f"DSM5 CONTENT related to patient's question:\n{dsm_content}\n\n"
                    print("Added DSM content to context")

                # Check if previous chats exist and don't contain error objects
                if all_patient_data['previous_chats'] and not any('error' in chat for chat in all_patient_data['previous_chats']):
                    context += f"PREVIOUS CHATS:\n{json.dumps(all_patient_data['previous_chats'], indent=2)}\n\n"
                    print("Added previous chats to context")

                # Add the user's current question
                context += f"\n**PATIENT QUESTION:** {user_input}\n\n"

                # Add instructions for the response
                context += """Based on the above information, provide a helpful, empathetic response to the patient.
                Keep your response conversational, brief, and focused on the patient's needs.
                If no specific information is available, provide general guidance and support.
                ! important:: extract only keywords that are related to mood or emotionsfrom PATIENT QUESTION and store it as a extracted_keywords if they are not available in the PATIENT QUESTION then extracted_keywords = []
                ❗Reply ONLY in this JSON structure:
                {
                  "response": "short, empathetic reply",
                  "extracted_keywords": ["keyword1", "keyword2"]
                }
                Only output the JSON (no explanation, no markdown).
                """

                return context

            # Use ThreadPoolExecutor to build context in parallel
            with concurrent.futures.ThreadPoolExecutor() as executor:
                context_future = executor.submit(build_context)

                # While context is being built, we can do other preparation work here if needed
                # For example, we could prepare the OpenAI client or do other tasks

                # Get the context when it's ready
                context = context_future.result()

            end_time = datetime.now()
            context_duration = (end_time - start_time).total_seconds()
            print(f"Context building completed in {context_duration:.2f} seconds")

            # Use OpenAI to generate a response
            if OPENAI_CLIENT:
                logger.info("Generating response using OpenAI...")
                start_time = datetime.now()

                # Generate the response using the context
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": context}
                ]

                logger.info(f"Sending {len(messages)} messages to OpenAI")
                logger.info(f"Context length: {len(context)} characters")

                # Force reload the API key from environment
                from dotenv import load_dotenv
                load_dotenv(override=True)
                api_key = os.getenv("OPENAI_API_KEY")
                logger.info(f"Using API key for chat completion: {api_key[:10]}...{api_key[-5:]}")

                # Create a new OpenAI client with the reloaded API key
                openai_client = OpenAI(api_key=api_key)

                # Make the API call
                response_completion = openai_client.chat.completions.create(
                    model="gpt-4o-2024-11-20",
                    messages=messages,
                    max_tokens=250,
                    temperature=0.2
                )

                end_time = datetime.now()
                api_duration = (end_time - start_time).total_seconds()
                print(f"OpenAI API call completed in {api_duration:.2f} seconds")

                response = response_completion.choices[0].message.content.strip()
                ai_response = response.strip('```json').strip('```').strip().replace('{\n','{').replace('\n}','}').replace(",\n",",").replace('\n','###')
                ai_response=ai_response.strip().replace('###', '')
                print(f"Generated response: {ai_response}")
                if not ai_response:
                    print("🔥 GPT returned empty response.")
                    message_text = "I'm sorry, can you say that again?"
                    extracted_keywords = []
                else:
                    try:
                        parsed = json.loads(ai_response)
                        # message_text = parsed.get("response", "").encode('utf-8').decode('unicode_escape')
                        # extracted_keywords = parsed.get("extracted_keywords", [])
                        message_text = parsed.get("response", "")
                        extracted_keywords = parsed.get("extracted_keywords", [])
                    except json.JSONDecodeError as e:
                        print("🔥 JSON parsing failed:", e)
                        print("⚠️ GPT said:", repr(ai_response))
                        message_text = "Sorry, I didn’t quite get that. Could you rephrase?"
                        extracted_keywords = []

                # Return the response directly without wrapping it in an "output" field
                result = {
                    "response": message_text,
                    "extracted_keywords": extracted_keywords
                }
                print("\n=== END SMART AGENT INVOKE ===\n\n")
                return result
            else:
                print("Error: OpenAI client not initialized")
                return {
                    "response": "I'm sorry, but I'm having trouble connecting to my knowledge base right now. Please try again later.",
                    "extracted_keywords": []
                }
        except Exception as e:
            import traceback
            print(f"Error in SmartAgent.invoke: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            return {
                "response": f"I'm sorry, but I encountered an error: {str(e)}",
                "extracted_keywords": []
            }
