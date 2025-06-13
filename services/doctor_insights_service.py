from typing import Dict, List, Any, Optional
import logging
import json
import threading
import concurrent.futures
from datetime import datetime, timedelta
import random
from sqlalchemy.orm import Session
from sqlalchemy import desc, and_, or_, text
from fastapi import APIRouter, Depends, HTTPException, Request, Header
from pydantic import BaseModel

from database.database import get_db
from model.model_correct import Patient, Doctor, Appointment, Prescription, DiaryEntry, EmotionAnalysis, ChatMessage, MyDiary, MedicalHistory
from auth.dependencies import get_current_active_user
import jwt
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from openai import OpenAI
from dotenv import load_dotenv
import os
from utils.cache_utils import with_selective_cache

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize OpenAI client
try:
    OPENAI_CLIENT = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
    if not OPENAI_CLIENT:
        logger.warning("OpenAI client could not be initialized. Check your API key.")
except ImportError:
    logger.warning("OpenAI package not installed. Some features may not work.")
    OPENAI_CLIENT = None

router = APIRouter()

# Create a custom security scheme
security = HTTPBearer(auto_error=False)

# JWT secret key - should match the one used in auth service
JWT_SECRET_KEY = "your-secret-key"  # This should be loaded from environment variables in production
# For testing, we'll use a hardcoded key that matches the static token
STATIC_JWT_SECRET = "static-jwt-secret-for-testing"

# Helper function to get user from token
async def get_current_user_from_token(request: Request = None, credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """Get the current user from the token."""
    # First try to get token from Authorization header
    token = None
    if credentials:
        token = credentials.credentials

    # If no token in header, try to get it from query parameters
    if not token and request:
        token = request.query_params.get("token")
        logger.info(f"Token from query params: {token}")

    # If still no token, try to get it from cookies
    if not token and request:
        token = request.cookies.get("token")

    # If no token found, return None
    if not token:
        logger.warning("No token found in request")
        return None

    try:
        # Decode the token
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=["HS256"])
        logger.info(f"Successfully decoded token with verification: {payload}")
        return payload
    except jwt.PyJWTError as e:
        logger.warning(f"JWT verification error: {str(e)}")
        # For now, let's accept any token for testing
        try:
            # Try to decode without verification
            payload = jwt.decode(token, options={"verify_signature": False})
            logger.info(f"Successfully decoded token without verification: {payload}")
            return payload
        except Exception as e:
            logger.error(f"Failed to decode token: {str(e)}")
            return None

# Helper function to get user role from token
def get_current_user_role(current_user: dict) -> str:
    """Extract the role from the current user token."""
    if not current_user:
        return "anonymous"

    # Check if roles is in the token
    roles = current_user.get("roles", "")

    # Parse roles (comma-separated string)
    if isinstance(roles, str):
        role_list = [r.strip().lower() for r in roles.split(",")]
        if "admin" in role_list:
            return "admin"
        elif "doctor" in role_list:
            return "doctor"
        elif "patient" in role_list:
            return "patient"

    # Default role
    return "user"

# Models for response
class InsightItem(BaseModel):
    title: str
    description: str
    type: str  # "chart", "text", "alert", etc.
    data: Optional[Dict[str, Any]] = None
    priority: int = 0  # Higher number means higher priority
    confidence: Optional[str] = "medium"  # low, medium, high
    evidence: Optional[str] = None  # Supporting evidence for the insight
    suggested_action: Optional[str] = None  # Suggested action for the doctor

class InsightsResponse(BaseModel):
    insights: List[InsightItem]
    generated_at: datetime

# Function to get patient basic information
def get_patient_basic_info(patient_id: str, db: Session) -> Dict[str, Any]:
    """Get basic patient information."""
    try:
        patient = db.query(Patient).filter(Patient.patient_id == patient_id).first()
        if not patient:
            logger.warning(f"Patient {patient_id} not found")
            return {}

        # Format patient basic info
        patient_info = {}
        try:
            # Basic demographics
            first_name = patient.first_name or ""
            last_name = patient.last_name or ""
            name = f"{first_name} {last_name}".strip()

            # Calculate age from DOB with error handling
            age = "Unknown"
            try:
                if patient.dob:
                    today = datetime.now()
                    dob = patient.dob
                    age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
            except Exception as age_error:
                logger.error(f"Error calculating age: {str(age_error)}")

            patient_info = {
                "name": name,
                "age": age,
                "gender": patient.gender or "Unknown",
                "health_score": patient.health_score or 0,
                "under_medications": patient.under_medications or False
            }
        except Exception as e:
            logger.error(f"Error formatting patient info: {str(e)}")

        return patient_info
    except Exception as e:
        logger.error(f"Error getting patient basic info: {str(e)}")
        return {}

# Function to get medical history
def get_medical_history(patient_id: str, db: Session) -> List[Dict[str, Any]]:
    """Get patient medical history."""
    try:
        history_records = db.query(MedicalHistory).filter(
            MedicalHistory.patient_id == patient_id
        ).order_by(MedicalHistory.created_at.desc() if hasattr(MedicalHistory, 'created_at') else MedicalHistory.medical_history_id.desc()).limit(10).all()

        medical_history = []
        for record in history_records:
            history_item = {
                "diagnosis": record.diagnosis if hasattr(record, 'diagnosis') else "Unknown",
                "date": record.created_at.strftime("%Y-%m-%d") if hasattr(record, 'created_at') and record.created_at else "Unknown",
                "notes": record.additional_notes if hasattr(record, 'additional_notes') else "",
                "treatment": record.treatment if hasattr(record, 'treatment') else ""
            }
            medical_history.append(history_item)
        return medical_history
    except Exception as e:
        logger.error(f"Error getting medical history: {str(e)}")
        return []

# Function to get prescriptions
def get_prescriptions(patient_id: str, db: Session) -> List[Dict[str, Any]]:
    """Get patient prescriptions."""
    try:
        rx_records = db.query(Prescription).filter(
            Prescription.patient_id == patient_id
        ).all()

        prescriptions = []
        for rx in rx_records:
            rx_item = {
                "medication": rx.medication_name if hasattr(rx, 'medication_name') else "Unknown",
                "dosage": rx.dosage if hasattr(rx, 'dosage') else "",
                "frequency": rx.frequency if hasattr(rx, 'frequency') else "",
                "status": rx.status if hasattr(rx, 'status') else "Unknown"
            }
            prescriptions.append(rx_item)
        return prescriptions
    except Exception as e:
        logger.error(f"Error getting prescriptions: {str(e)}")
        return []

# Function to get emotion analyses
def get_emotion_analyses(patient_id: str, db: Session) -> List[Dict[str, Any]]:
    """Get patient emotion analyses."""
    try:
        emotion_records = db.query(EmotionAnalysis).filter(
            EmotionAnalysis.patient_id == patient_id
        ).order_by(EmotionAnalysis.analyzed_at.desc() if hasattr(EmotionAnalysis, 'analyzed_at') else EmotionAnalysis.emotion_id.desc()).limit(30).all()

        emotions = []
        for record in emotion_records:
            emotion_item = {
                "emotion": record.emotion_category if hasattr(record, 'emotion_category') else "Unknown",
                "confidence": float(record.confidence_score) if hasattr(record, 'confidence_score') else 0.0,
                "date": record.analyzed_at.strftime("%Y-%m-%d") if hasattr(record, 'analyzed_at') and record.analyzed_at else "Unknown"
            }
            emotions.append(emotion_item)
        return emotions
    except Exception as e:
        logger.error(f"Error getting emotion analyses: {str(e)}")
        return []

# Function to get diary entries
def get_diary_entries(patient_id: str, db: Session) -> List[Dict[str, Any]]:
    """Get patient diary entries."""
    try:
        diary_entries = []
        # First try DiaryEntry model
        try:
            diary_records = db.query(DiaryEntry).filter(
                DiaryEntry.patient_id == patient_id
            ).order_by(DiaryEntry.created_at.desc()).limit(15).all()

            for entry in diary_records:
                diary_item = {
                    "content": entry.content if hasattr(entry, 'content') else "",
                    "date": entry.created_at.strftime("%Y-%m-%d") if hasattr(entry, 'created_at') and entry.created_at else "Unknown"
                }
                diary_entries.append(diary_item)
        except Exception:
            # Fallback to MyDiary model
            diary_records = db.query(MyDiary).filter(
                MyDiary.patient_id == patient_id
            ).order_by(MyDiary.created_at.desc()).limit(15).all()

            for entry in diary_records:
                diary_item = {
                    "content": entry.notes if hasattr(entry, 'notes') else "",
                    "date": entry.created_at.strftime("%Y-%m-%d") if hasattr(entry, 'created_at') and entry.created_at else "Unknown"
                }
                diary_entries.append(diary_item)
        return diary_entries
    except Exception as e:
        logger.error(f"Error getting diary entries: {str(e)}")
        return []

# Function to get appointments
def get_appointments(patient_id: str, db: Session) -> Dict[str, List[Dict[str, Any]]]:
    """Get patient appointments (both past and upcoming)."""
    try:
        past_appointments = []
        upcoming_appointments = []
        current_date = datetime.now().strftime("%Y-%m-%d")

        # Past appointments
        past_appt_stmt = text(f"""
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
                patient_id = '{patient_id}'
                AND appointment_date < '{current_date}'
            ORDER BY
                appointment_date DESC
            LIMIT 10
        """)

        # Execute raw SQL
        past_result = db.execute(past_appt_stmt)
        past_records = past_result.fetchall()

        for appt in past_records:
            appt_date = "Unknown"
            try:
                if appt.appointment_date:
                    if isinstance(appt.appointment_date, str):
                        appt_date = appt.appointment_date
                    else:
                        appt_date = appt.appointment_date.strftime("%Y-%m-%d")
            except:
                pass

            past_appointments.append({
                "date": appt_date,
                "status": appt.status if appt.status else "Unknown",
                "notes": appt.notes if appt.notes else "",
                "reason": appt.visit_reason if hasattr(appt, 'visit_reason') and appt.visit_reason else "General checkup"
            })

        # Upcoming appointments
        upcoming_appt_stmt = text(f"""
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
                patient_id = '{patient_id}'
                AND appointment_date >= '{current_date}'
            ORDER BY
                appointment_date ASC
            LIMIT 5
        """)

        # Execute raw SQL
        upcoming_result = db.execute(upcoming_appt_stmt)
        upcoming_records = upcoming_result.fetchall()

        for appt in upcoming_records:
            appt_date = "Unknown"
            try:
                if appt.appointment_date:
                    if isinstance(appt.appointment_date, str):
                        appt_date = appt.appointment_date
                    else:
                        appt_date = appt.appointment_date.strftime("%Y-%m-%d")
            except:
                pass

            upcoming_appointments.append({
                "date": appt_date,
                "status": appt.status if appt.status else "Unknown",
                "notes": appt.notes if appt.notes else "",
                "reason": appt.visit_reason if hasattr(appt, 'visit_reason') and appt.visit_reason else "General checkup"
            })

        return {
            "past_appointments": past_appointments,
            "upcoming_appointments": upcoming_appointments
        }
    except Exception as e:
        logger.error(f"Error getting appointments: {str(e)}")
        return {"past_appointments": [], "upcoming_appointments": []}

# Function to get chat history
def get_chat_history(patient_id: str, db: Session) -> List[Dict[str, Any]]:
    """Get chat history for a patient."""
    try:
        # Get the last 30 messages
        messages = db.query(ChatMessage).filter(
            (ChatMessage.sender_id == patient_id) | (ChatMessage.receiver_id == patient_id)
        ).order_by(desc(ChatMessage.createdAt)).limit(30).all()

        # Reverse to get chronological order
        messages.reverse()

        # Format messages
        chat_history = []
        for msg in messages:
            role = "patient" if msg.sender_id == patient_id else "doctor"
            chat_history.append({
                "role": role,
                "content": msg.message_text,
                "timestamp": msg.createdAt.strftime("%Y-%m-%d %H:%M:%S") if msg.createdAt else "Unknown"
            })

        return chat_history
    except Exception as e:
        logger.error(f"Error getting chat history: {str(e)}")
        return []

# Function to generate emotional and sentiment trends summary
def generate_emotional_trends_summary(emotions: List[Dict[str, Any]], chat_history: List[Dict[str, Any]]) -> str:
    """Generate a summary of emotional and sentiment trends."""
    try:
        if not emotions:
            return "No emotional data available."

        # Count emotion frequencies
        emotion_counts = {}
        for emotion in emotions:
            dominant = emotion.get("emotion", "").lower()
            if dominant:
                emotion_counts[dominant] = emotion_counts.get(dominant, 0) + 1

        # Sort emotions by frequency
        sorted_emotions = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)

        # Extract emotion trends
        emotion_trend_summary = []
        for emotion, count in sorted_emotions[:5]:  # Top 5 emotions
            percentage = int((count / len(emotions)) * 100)
            emotion_trend_summary.append(f"{emotion}: {count} occurrences ({percentage}%)")

        # Analyze chat content for sentiment
        patient_messages = [msg["content"] for msg in chat_history if msg["role"] == "patient"]

        # Simple sentiment analysis based on keywords
        positive_words = ["happy", "good", "great", "better", "improve", "joy", "grateful", "thankful"]
        negative_words = ["sad", "bad", "worse", "difficult", "hard", "pain", "struggle", "anxious", "worried", "stress"]

        positive_count = 0
        negative_count = 0

        for message in patient_messages:
            message_lower = message.lower()
            positive_count += sum(1 for word in positive_words if word in message_lower)
            negative_count += sum(1 for word in negative_words if word in message_lower)

        # Determine overall sentiment
        sentiment = "neutral"
        if positive_count > negative_count * 1.5:
            sentiment = "predominantly positive"
        elif negative_count > positive_count * 1.5:
            sentiment = "predominantly negative"
        elif positive_count > negative_count:
            sentiment = "slightly positive"
        elif negative_count > positive_count:
            sentiment = "slightly negative"

        # Create summary
        summary = f"Emotional Analysis Summary:\n"
        summary += f"- Overall sentiment in patient communication is {sentiment}.\n"
        summary += f"- Dominant emotions detected: {', '.join(emotion_trend_summary[:3])}.\n"

        # Add time-based trends if available
        if len(emotions) >= 10:
            recent_emotions = emotions[:10]  # Most recent 10
            recent_emotion_counts = {}
            for emotion in recent_emotions:
                dominant = emotion.get("emotion", "").lower()
                if dominant:
                    recent_emotion_counts[dominant] = recent_emotion_counts.get(dominant, 0) + 1

            sorted_recent = sorted(recent_emotion_counts.items(), key=lambda x: x[1], reverse=True)
            if sorted_recent:
                summary += f"- Recent emotional trend shows {sorted_recent[0][0]} as the most frequent emotion.\n"

        return summary
    except Exception as e:
        logger.error(f"Error generating emotional trends summary: {str(e)}")
        return "Error generating emotional trends summary."

# Function to generate mood and symptom progression summary
def generate_mood_symptom_summary(diary_entries: List[Dict[str, Any]], medical_history: List[Dict[str, Any]]) -> str:
    """Generate a summary of mood and symptom progression."""
    try:
        if not diary_entries and not medical_history:
            return "No mood or symptom data available."

        summary = f"Mood and Symptom Progression Summary:\n"

        # Analyze diary entries for mood indicators
        if diary_entries:
            # Keywords for different moods
            mood_keywords = {
                "depressed": ["depressed", "depression", "sad", "down", "hopeless", "empty"],
                "anxious": ["anxious", "anxiety", "worried", "nervous", "panic", "fear", "scared"],
                "irritable": ["irritable", "angry", "frustrated", "annoyed", "upset"],
                "positive": ["happy", "good", "great", "better", "joy", "hopeful", "optimistic"],
                "tired": ["tired", "exhausted", "fatigue", "no energy", "low energy"],
                "stressed": ["stressed", "overwhelmed", "pressure", "too much"]
            }

            # Count mood mentions
            mood_counts = {mood: 0 for mood in mood_keywords}

            for entry in diary_entries:
                content = entry.get("content", "").lower()
                for mood, keywords in mood_keywords.items():
                    for keyword in keywords:
                        if keyword in content:
                            mood_counts[mood] += 1
                            break

            # Sort moods by frequency
            sorted_moods = sorted(mood_counts.items(), key=lambda x: x[1], reverse=True)

            # Add mood trends to summary
            if sorted_moods[0][1] > 0:
                summary += f"- Most frequently expressed mood: {sorted_moods[0][0]} (mentioned in {sorted_moods[0][1]} entries).\n"

                # Check for secondary mood if present
                if sorted_moods[1][1] > 0:
                    summary += f"- Secondary mood: {sorted_moods[1][0]} (mentioned in {sorted_moods[1][1]} entries).\n"

            # Check for sleep issues
            sleep_keywords = ["sleep", "insomnia", "nightmare", "tired", "fatigue", "rest"]
            sleep_mentions = sum(1 for entry in diary_entries if any(keyword in entry.get("content", "").lower() for keyword in sleep_keywords))

            if sleep_mentions > 0:
                sleep_percentage = int((sleep_mentions / len(diary_entries)) * 100)
                summary += f"- Sleep issues mentioned in {sleep_percentage}% of diary entries.\n"

        # Analyze medical history for symptom patterns
        if medical_history:
            # Extract diagnoses
            diagnoses = [record.get("diagnosis", "").lower() for record in medical_history if record.get("diagnosis")]

            # Count diagnosis frequencies
            diagnosis_counts = {}
            for diagnosis in diagnoses:
                diagnosis_counts[diagnosis] = diagnosis_counts.get(diagnosis, 0) + 1

            # Add diagnosis information to summary
            if diagnosis_counts:
                sorted_diagnoses = sorted(diagnosis_counts.items(), key=lambda x: x[1], reverse=True)
                summary += f"- Primary diagnosis: {sorted_diagnoses[0][0]}.\n"

                # Check for comorbidities
                if len(sorted_diagnoses) > 1:
                    comorbidities = [d[0] for d in sorted_diagnoses[1:]]
                    summary += f"- Comorbidities: {', '.join(comorbidities[:3])}.\n"

        return summary
    except Exception as e:
        logger.error(f"Error generating mood and symptom summary: {str(e)}")
        return "Error generating mood and symptom summary."

# Function to generate behavioral adherence and routines summary
def generate_behavioral_summary(appointments: Dict[str, List[Dict[str, Any]]], diary_entries: List[Dict[str, Any]]) -> str:
    """Generate a summary of behavioral adherence and routines."""
    try:
        past_appointments = appointments.get("past_appointments", [])
        upcoming_appointments = appointments.get("upcoming_appointments", [])

        if not past_appointments and not diary_entries:
            return "No behavioral data available."

        summary = f"Behavioral Adherence and Routines Summary:\n"

        # Analyze appointment adherence
        if past_appointments:
            missed_appointments = sum(1 for appt in past_appointments if appt.get("status", "").lower() in ["missed", "cancelled"])
            total_past = len(past_appointments)

            if total_past > 0:
                adherence_rate = int(((total_past - missed_appointments) / total_past) * 100)
                summary += f"- Appointment adherence rate: {adherence_rate}% ({total_past-missed_appointments}/{total_past} appointments attended).\n"

                if adherence_rate < 70:
                    summary += f"- Low appointment adherence may indicate treatment engagement issues.\n"

        # Analyze diary entries for routine indicators
        if diary_entries:
            # Keywords for different routines
            routine_keywords = {
                "exercise": ["exercise", "workout", "gym", "run", "walk", "physical activity"],
                "medication": ["medication", "medicine", "pill", "dose", "prescription", "forgot to take"],
                "sleep": ["sleep", "bed", "woke up", "insomnia", "tired", "rest"],
                "diet": ["eat", "food", "meal", "diet", "nutrition", "hungry"],
                "social": ["friend", "family", "people", "social", "talk", "conversation"]
            }

            # Count routine mentions
            routine_counts = {routine: 0 for routine in routine_keywords}

            for entry in diary_entries:
                content = entry.get("content", "").lower()
                for routine, keywords in routine_keywords.items():
                    for keyword in keywords:
                        if keyword in content:
                            routine_counts[routine] += 1
                            break

            # Add routine information to summary
            mentioned_routines = [routine for routine, count in routine_counts.items() if count > 0]
            if mentioned_routines:
                summary += f"- Routines mentioned in diary: {', '.join(mentioned_routines)}.\n"

                # Check for medication adherence issues
                if routine_counts["medication"] > 0:
                    med_adherence_issues = sum(1 for entry in diary_entries if any(keyword in entry.get("content", "").lower() for keyword in ["forgot", "missed", "skip", "didn't take"]))
                    if med_adherence_issues > 0:
                        summary += f"- Potential medication adherence issues detected in {med_adherence_issues} diary entries.\n"

        return summary
    except Exception as e:
        logger.error(f"Error generating behavioral summary: {str(e)}")
        return "Error generating behavioral summary."

# Function to generate medication and side effect monitoring summary
def generate_medication_summary(prescriptions: List[Dict[str, Any]], chat_history: List[Dict[str, Any]], diary_entries: List[Dict[str, Any]]) -> str:
    """Generate a summary of medication adherence and side effects."""
    try:
        if not prescriptions:
            return "No medication data available."

        summary = f"Medication and Side Effect Summary:\n"

        # List current medications
        current_medications = [f"{rx.get('medication', 'Unknown')} {rx.get('dosage', '')}" for rx in prescriptions if rx.get('status', '').lower() == 'active']
        if current_medications:
            summary += f"- Current medications: {', '.join(current_medications)}.\n"

        # Extract medication mentions from chat and diary
        medication_mentions = []

        # Check chat history
        for msg in chat_history:
            if msg["role"] == "patient":
                content = msg["content"].lower()
                if any(med_term in content for med_term in ["medication", "medicine", "pill", "drug", "dose", "prescription"]):
                    medication_mentions.append({
                        "source": "chat",
                        "content": msg["content"],
                        "timestamp": msg["timestamp"]
                    })

        # Check diary entries
        for entry in diary_entries:
            content = entry.get("content", "").lower()
            if any(med_term in content for med_term in ["medication", "medicine", "pill", "drug", "dose", "prescription"]):
                medication_mentions.append({
                    "source": "diary",
                    "content": entry["content"],
                    "timestamp": entry.get("date", "Unknown")
                })

        # Analyze for side effects
        side_effect_keywords = ["side effect", "reaction", "nausea", "headache", "dizzy", "tired", "fatigue",
                               "insomnia", "sleep", "rash", "itch", "stomach", "appetite", "weight", "pain"]

        side_effect_mentions = []
        for mention in medication_mentions:
            content = mention["content"].lower()
            if any(se_term in content for se_term in side_effect_keywords):
                side_effect_mentions.append(mention)

        # Add side effect information to summary
        if side_effect_mentions:
            summary += f"- Potential medication side effects mentioned: {len(side_effect_mentions)} instances.\n"

            # Include a recent example if available
            if side_effect_mentions:
                recent_mention = side_effect_mentions[0]
                summary += f"- Recent side effect mention: \"{recent_mention['content'][:100]}...\"\n"

        # Check for adherence issues
        adherence_keywords = ["forgot", "missed", "skip", "didn't take", "haven't taken", "stop", "stopped"]
        adherence_issues = []

        for mention in medication_mentions:
            content = mention["content"].lower()
            if any(adh_term in content for adh_term in adherence_keywords):
                adherence_issues.append(mention)

        # Add adherence information to summary
        if adherence_issues:
            summary += f"- Potential medication adherence issues detected: {len(adherence_issues)} instances.\n"

        return summary
    except Exception as e:
        logger.error(f"Error generating medication summary: {str(e)}")
        return "Error generating medication summary."

# Function to generate cognitive patterns and thinking errors summary
def generate_cognitive_patterns_summary(chat_history: List[Dict[str, Any]], diary_entries: List[Dict[str, Any]]) -> str:
    """Generate a summary of cognitive patterns and thinking errors."""
    try:
        if not chat_history and not diary_entries:
            return "No cognitive pattern data available."

        summary = f"Cognitive Patterns and Thinking Errors Summary:\n"

        # Combine patient messages from chat and diary
        patient_texts = []
        for msg in chat_history:
            if msg["role"] == "patient":
                patient_texts.append(msg["content"])

        for entry in diary_entries:
            patient_texts.append(entry.get("content", ""))

        if not patient_texts:
            return "Insufficient data for cognitive pattern analysis."

        # Define cognitive distortion patterns
        cognitive_distortions = {
            "all-or-nothing": ["always", "never", "every time", "completely", "totally", "nothing", "everything"],
            "overgeneralization": ["everyone", "no one", "all people", "nobody", "everybody"],
            "catastrophizing": ["terrible", "awful", "disaster", "horrible", "worst", "can't stand", "unbearable"],
            "emotional reasoning": ["feel like a failure", "feel worthless", "feel hopeless", "feel like"],
            "should statements": ["should ", "must ", "have to ", "ought to "],
            "personalization": ["my fault", "blame myself", "because of me", "my responsibility"],
            "mental filtering": ["can't see anything good", "only see the bad", "nothing good"],
            "disqualifying the positive": ["doesn't count", "doesn't matter", "not important", "yeah but"],
            "jumping to conclusions": ["know what they're thinking", "they think I'm", "they don't like", "will fail", "won't work"],
            "labeling": ["i'm a failure", "i'm worthless", "i'm useless", "i'm a loser", "i'm stupid"]
        }

        # Count distortion occurrences
        distortion_counts = {distortion: 0 for distortion in cognitive_distortions}

        for text in patient_texts:
            text_lower = text.lower()
            for distortion, keywords in cognitive_distortions.items():
                for keyword in keywords:
                    if keyword in text_lower:
                        distortion_counts[distortion] += 1
                        break

        # Sort distortions by frequency
        sorted_distortions = sorted(distortion_counts.items(), key=lambda x: x[1], reverse=True)

        # Add distortion information to summary
        detected_distortions = [distortion for distortion, count in sorted_distortions if count > 0]
        if detected_distortions:
            summary += f"- Cognitive distortion patterns detected: {', '.join(detected_distortions[:3])}.\n"

            # Add details for top distortions
            for distortion, count in sorted_distortions[:2]:
                if count > 0:
                    summary += f"- {distortion.capitalize()} thinking pattern occurred approximately {count} times.\n"
        else:
            summary += "- No significant cognitive distortion patterns detected.\n"

        # Check for rumination patterns (repeated negative thoughts)
        rumination_keywords = ["keep thinking about", "can't stop thinking", "obsessing", "dwelling", "ruminating", "over and over"]
        rumination_count = sum(1 for text in patient_texts if any(keyword in text.lower() for keyword in rumination_keywords))

        if rumination_count > 0:
            summary += f"- Potential rumination patterns detected in {rumination_count} instances.\n"

        return summary
    except Exception as e:
        logger.error(f"Error generating cognitive patterns summary: {str(e)}")
        return "Error generating cognitive patterns summary."

# Function to generate doctor insights using GPT
def generate_doctor_insights(patient_data: Dict[str, Any], summaries: Dict[str, str]) -> List[Dict[str, Any]]:
    """Generate doctor insights using GPT based on patient data and summaries."""
    try:
        # Check if we have the necessary data to generate insights
        if not patient_data.get("patient_info") and not summaries:
            logger.warning("Insufficient data to generate doctor insights")
            return [{
                "title": "Insufficient Patient Data",
                "description": "There is not enough patient data available to generate meaningful clinical insights.",
                "category": "System Message",
                "priority": 5,
                "confidence": "high",
                "evidence": "Missing patient information and clinical summaries",
                "suggested_action": "Collect more patient data before proceeding with analysis"
            }]

        # Optimize the prompt by focusing on the most relevant information
        # Create a more concise prompt for GPT to reduce token usage
        patient_info = patient_data.get("patient_info", {})
        patient_name = f"{patient_info.get('name', 'the patient')}"
        patient_age = patient_info.get('age', 'Unknown')
        patient_gender = patient_info.get('gender', 'Unknown')

        # Create a more structured and concise prompt
        prompt = f"""
You are an expert clinical decision support system generating insights for {patient_name}, a {patient_age}-year-old {patient_gender}.

CLINICAL SUMMARIES:
1. EMOTIONAL TRENDS: {summaries.get("emotional_trends", "No emotional trends data available.")}
2. MOOD & SYMPTOMS: {summaries.get("mood_symptoms", "No mood and symptom data available.")}
3. BEHAVIORAL: {summaries.get("behavioral", "No behavioral data available.")}
4. MEDICATION: {summaries.get("medication", "No medication data available.")}
5. COGNITIVE PATTERNS: {summaries.get("cognitive_patterns", "No cognitive patterns data available.")}

KEY MEDICAL HISTORY:
{json.dumps(patient_data.get("medical_history", [])[:3], indent=2)}

CURRENT MEDICATIONS:
{json.dumps(patient_data.get("prescriptions", [])[:5], indent=2)}

Generate exactly 5 clinical insights in the following JSON format:
[
  {{
    "title": "Brief, clinically relevant title",
    "description": "Detailed explanation with specific clinical recommendations (2-3 sentences)",
    "category": "One of: Mental Health Assessment, Treatment Recommendation, Risk Assessment, Medication Management, Behavioral Patterns, Therapy Selection",
    "priority": "A number from 1-5, where 5 is highest priority and requires immediate attention",
    "confidence": "One of: low, medium, high - indicating confidence level in this insight",
    "evidence": "Brief reference to specific data points supporting this insight",
    "suggested_action": "Concrete next step for the clinician to take"
  }},
  ...
]

Focus on:
1. Mental health assessment: Provide specific diagnostic considerations based on symptoms and emotional patterns
2. Treatment recommendations: Suggest specific evidence-based therapies (e.g., CBT, DBT, ACT, psychodynamic) that would be most appropriate
3. Risk factors: Identify specific risk factors that require monitoring or intervention
4. Medication considerations: Provide specific medication recommendations or adjustments if appropriate
5. Therapy focus areas: Identify specific therapeutic targets or goals based on the patient's symptoms and behaviors

Your insights must be clinically precise, evidence-based, specific to this patient, actionable, and prioritized by clinical urgency.
For therapy recommendations, specify the exact type of therapy and why it would be beneficial. For medication recommendations, consider symptoms, history, and any side effects or adherence issues.
"""

        # Call GPT API
        if not OPENAI_CLIENT:
            logger.error("OpenAI client not initialized")
            return [{
                "title": "Error Generating Insights",
                "description": "OpenAI client not initialized. Please check your API key.",
                "category": "System Error",
                "priority": 3,
                "confidence": "low",
                "evidence": "Error in processing",
                "suggested_action": "Review raw patient data manually"
            }]

        try:
            # Log the start of the API call for performance tracking
            api_call_start = datetime.now()
            logger.info("Starting OpenAI API call for doctor insights")

            # Make the API call with the new client format - using the same model as smart_agent.py
            response = OPENAI_CLIENT.chat.completions.create(
                model="gpt-4o-2024-11-20",  # Use the same model as in smart_agent.py
                messages=[
                    {"role": "system", "content": "You are a clinical decision support system that provides evidence-based insights to mental health professionals. You have expertise in psychiatry, psychology, and evidence-based treatments including CBT, DBT, ACT, psychodynamic therapy, and psychopharmacology. You provide specific, actionable recommendations based on patient data."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,  # Lower temperature for more consistent responses (matching smart_agent.py)
                max_tokens=800,  # Reduced token limit for faster response while still allowing detailed insights
                top_p=0.95,       # Slightly more diverse responses
                presence_penalty=0.1,  # Slight penalty for repetition
                frequency_penalty=0.1  # Slight penalty for frequent tokens
            )

            # Extract the response
            gpt_response = response.choices[0].message.content.strip()

            # Log API call duration
            api_call_duration = (datetime.now() - api_call_start).total_seconds()
            logger.info(f"OpenAI API call completed in {api_call_duration:.2f} seconds")
        except Exception as api_error:
            logger.error(f"OpenAI API error: {str(api_error)}")
            return [{
                "title": "Error Generating Insights",
                "description": f"Error calling OpenAI API: {str(api_error)}",
                "category": "System Error",
                "priority": 3,
                "confidence": "low",
                "evidence": "Error in processing",
                "suggested_action": "Review raw patient data manually"
            }]

        # Parse the JSON response with performance tracking
        try:
            parse_start_time = datetime.now()
            logger.info("Starting JSON parsing")

            # Extract JSON from the response (in case there's any extra text)
            import re
            json_match = re.search(r'\[[\s\S]*\]', gpt_response)
            if json_match:
                json_str = json_match.group(0)
                insights = json.loads(json_str)
            else:
                insights = json.loads(gpt_response)

            parse_time = (datetime.now() - parse_start_time).total_seconds()
            logger.info(f"JSON parsing completed in {parse_time:.2f} seconds")

            # Validate the insights format more efficiently
            validation_start_time = datetime.now()
            validated_insights = []

            # Process all insights at once with a list comprehension for better performance
            validated_insights = [
                {
                    "title": insight.get("title", "Clinical Insight"),
                    "description": insight.get("description", ""),
                    "category": insight.get("category", "Clinical Assessment"),
                    "priority": int(insight.get("priority", 3)),
                    "confidence": insight.get("confidence", "medium"),
                    "evidence": insight.get("evidence", "Based on patient data analysis"),
                    "suggested_action": insight.get("suggested_action", "Discuss with patient")
                }
                for insight in insights
                if isinstance(insight, dict) and "title" in insight and "description" in insight
            ]

            validation_time = (datetime.now() - validation_start_time).total_seconds()
            logger.info(f"Insight validation completed in {validation_time:.2f} seconds")

            # Return at most 5 insights, prioritizing by priority level
            sorted_insights = sorted(validated_insights, key=lambda x: x["priority"], reverse=True)
            return sorted_insights[:5]

        except json.JSONDecodeError as json_error:
            logger.error(f"Error parsing GPT response as JSON: {str(json_error)}")
            logger.error(f"Raw response: {gpt_response}")
            # Return an error insight
            return [{
                "title": "Error Generating Insights",
                "description": "There was an error generating clinical insights. Please try again or review the raw patient data.",
                "category": "System Error",
                "priority": 3,
                "confidence": "low",
                "evidence": "Error in processing",
                "suggested_action": "Review raw patient data manually"
            }]

    except Exception as e:
        logger.error(f"Error generating doctor insights: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        # Return an error insight
        return [{
            "title": "Error Generating Insights",
            "description": f"There was an error generating clinical insights: {str(e)}",
            "category": "System Error",
            "priority": 3,
            "confidence": "low",
            "evidence": "Error in processing",
            "suggested_action": "Review raw patient data manually"
        }]

# Function to generate doctor insights with data collection
def generate_doctor_insights_internal(
    patient_id: str,
    db: Session,
    user_role: str
) -> InsightsResponse:
    """Internal function to generate doctor insights with caching."""
    # Check if user is a doctor or admin
    if user_role not in ["doctor", "admin"]:
        raise HTTPException(status_code=403, detail="Only doctors can access these insights")

    # Check if patient exists
    patient = db.query(Patient).filter(Patient.patient_id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    # Collect all patient data
    patient_data = {}

    # Use a single ThreadPoolExecutor for all operations to reduce overhead
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        # Start time for performance tracking
        start_time = datetime.now()
        logger.info(f"Starting data collection for patient {patient_id}")

        # Create futures for each data fetch operation using a dictionary for better organization
        futures = {
            "patient_info": executor.submit(get_patient_basic_info, patient_id, db),
            "medical_history": executor.submit(get_medical_history, patient_id, db),
            "prescriptions": executor.submit(get_prescriptions, patient_id, db),
            "emotions": executor.submit(get_emotion_analyses, patient_id, db),
            "diary_entries": executor.submit(get_diary_entries, patient_id, db),
            "appointments": executor.submit(get_appointments, patient_id, db),
            "chat_history": executor.submit(get_chat_history, patient_id, db)
        }

        # Collect results as they complete with error handling
        for key, future in futures.items():
            try:
                patient_data[key] = future.result()
            except Exception as e:
                logger.error(f"Error fetching {key}: {str(e)}")
                # Provide empty default values to prevent downstream errors
                if key == "patient_info":
                    patient_data[key] = {}
                elif key == "appointments":
                    patient_data[key] = {"past_appointments": [], "upcoming_appointments": []}
                else:
                    patient_data[key] = []

        data_collection_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Data collection completed in {data_collection_time:.2f} seconds")

        # Generate summaries in parallel
        logger.info("Starting summary generation")
        summary_start_time = datetime.now()

        # Create futures for each summary generation using a dictionary
        summary_futures = {
            "emotional_trends": executor.submit(
                generate_emotional_trends_summary,
                patient_data["emotions"],
                patient_data["chat_history"]
            ),
            "mood_symptoms": executor.submit(
                generate_mood_symptom_summary,
                patient_data["diary_entries"],
                patient_data["medical_history"]
            ),
            "behavioral": executor.submit(
                generate_behavioral_summary,
                patient_data["appointments"],
                patient_data["diary_entries"]
            ),
            "medication": executor.submit(
                generate_medication_summary,
                patient_data["prescriptions"],
                patient_data["chat_history"],
                patient_data["diary_entries"]
            ),
            "cognitive_patterns": executor.submit(
                generate_cognitive_patterns_summary,
                patient_data["chat_history"],
                patient_data["diary_entries"]
            )
        }

        # Collect summary results with error handling
        summaries = {}
        for key, future in summary_futures.items():
            try:
                summaries[key] = future.result()
            except Exception as e:
                logger.error(f"Error generating {key} summary: {str(e)}")
                summaries[key] = f"Unable to generate {key.replace('_', ' ')} summary due to an error."

        summary_time = (datetime.now() - summary_start_time).total_seconds()
        logger.info(f"Summary generation completed in {summary_time:.2f} seconds")

        # Generate insights using GPT
        logger.info("Starting GPT insights generation")
        gpt_start_time = datetime.now()

        # Submit GPT insight generation to the thread pool
        gpt_future = executor.submit(generate_doctor_insights, patient_data, summaries)
        gpt_insights = gpt_future.result()

        gpt_time = (datetime.now() - gpt_start_time).total_seconds()
        logger.info(f"GPT insights generation completed in {gpt_time:.2f} seconds")

        # Format insights for response
        insights = []
        for insight in gpt_insights:
            insights.append(InsightItem(
                title=insight["title"],
                description=insight["description"],
                type="clinical_insight",
                priority=insight["priority"],
                confidence=insight["confidence"],
                evidence=insight["evidence"],
                suggested_action=insight["suggested_action"]
            ))

        # Sort insights by priority (highest first)
        insights.sort(key=lambda x: x.priority, reverse=True)

        total_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Total insight generation completed in {total_time:.2f} seconds")

        return InsightsResponse(
            insights=insights,
            generated_at=datetime.now()
        )

# Apply caching to the internal function - only use patient_id and user_role for the cache key
cached_generate_doctor_insights = with_selective_cache("doctor_insights", [0, 2])(generate_doctor_insights_internal)

# Doctor insights endpoint
@router.get("/doctor/patient/{patient_id}", response_model=InsightsResponse)
async def get_doctor_insights_for_patient(
    patient_id: str,
    db: Session = Depends(get_db),
    current_user: Dict = Depends(get_current_active_user)
):
    """Get clinical insights about a patient for a doctor."""
    try:
        # Check if user is a doctor or admin
        user_role = get_current_user_role(current_user)
        logger.info(f"User role: {user_role}")
        logger.info(f"Current user: {current_user}")

        # Use cached function to generate insights
        logger.info(f"Calling cached function with patient_id={patient_id}, user_role={user_role}")
        result = cached_generate_doctor_insights(patient_id, db, user_role)
        logger.info(f"Cache function returned successfully")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating doctor insights: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error generating insights: {str(e)}")
