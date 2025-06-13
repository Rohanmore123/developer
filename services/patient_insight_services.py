from typing import Dict, List, Any, Optional
import logging
import json
import concurrent.futures
from datetime import datetime, timedelta
import random
from sqlalchemy.orm import Session
from sqlalchemy import desc, and_, or_, text
from fastapi import APIRouter, Depends, HTTPException, Request
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

    # If still no token, try to get it from cookies
    if not token and request:
        token = request.cookies.get("token")

    # If no token found, return None
    if not token:
        return None

    try:
        # Decode the token
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.PyJWTError:
        # For now, let's accept any token for testing
        try:
            # Try to decode without verification
            payload = jwt.decode(token, options={"verify_signature": False})
            return payload
        except:
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
class PatientInsightItem(BaseModel):
    title: str
    description: str
    type: str  # "progress", "recommendation", "milestone", "reminder", etc.
    data: Optional[Dict[str, Any]] = None
    priority: int = 0  # Higher number means higher priority
    action_steps: Optional[List[str]] = None  # Concrete steps the patient can take

class PatientInsightsResponse(BaseModel):
    insights: List[PatientInsightItem]
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

# Function to generate treatment progress summary
def generate_treatment_progress_summary(medical_history: List[Dict[str, Any]], prescriptions: List[Dict[str, Any]], appointments: Dict[str, List[Dict[str, Any]]]) -> str:
    """Generate a summary of the patient's treatment progress."""
    try:
        if not medical_history and not prescriptions and not appointments.get("past_appointments", []):
            return "No treatment data available to analyze progress."

        summary = "Treatment Progress Summary:\n"

        # Analyze medical history for treatment progression
        if medical_history:
            # Sort by date (most recent first)
            sorted_history = sorted(
                [h for h in medical_history if h.get("date") and h.get("date") != "Unknown"],
                key=lambda x: x.get("date", ""),
                reverse=True
            )

            if sorted_history:
                most_recent = sorted_history[0]
                summary += f"- Most recent diagnosis: {most_recent.get('diagnosis', 'Unknown')} (as of {most_recent.get('date', 'Unknown')}).\n"

                if len(sorted_history) > 1:
                    # Look for treatment changes over time
                    treatments = [h.get("treatment", "") for h in sorted_history if h.get("treatment")]
                    if len(set(treatments)) > 1:
                        summary += f"- Your treatment plan has been adjusted {len(set(treatments))} times.\n"
                    else:
                        summary += "- Your treatment plan has remained consistent.\n"

        # Analyze prescriptions
        active_prescriptions = [p for p in prescriptions if p.get("status", "").lower() == "active"]
        if active_prescriptions:
            summary += f"- You are currently on {len(active_prescriptions)} medication(s).\n"

        # Analyze past appointments
        past_appointments = appointments.get("past_appointments", [])
        if past_appointments:
            attended_count = sum(1 for appt in past_appointments if appt.get("status", "").lower() not in ["missed", "cancelled"])
            total_count = len(past_appointments)

            if total_count > 0:
                attendance_rate = (attended_count / total_count) * 100
                summary += f"- You've attended {attended_count} out of {total_count} scheduled appointments ({int(attendance_rate)}%).\n"

                # Check for appointment frequency
                if total_count >= 3:
                    summary += "- Regular check-ups are an important part of your treatment plan.\n"

        # Analyze upcoming appointments
        upcoming_appointments = appointments.get("upcoming_appointments", [])
        if upcoming_appointments:
            next_appointment = upcoming_appointments[0]
            summary += f"- Your next appointment is scheduled for {next_appointment.get('date', 'Unknown')}.\n"

        return summary
    except Exception as e:
        logger.error(f"Error generating treatment progress summary: {str(e)}")
        return "Unable to analyze treatment progress due to an error."

# Function to generate emotional wellbeing summary
def generate_emotional_wellbeing_summary(emotions: List[Dict[str, Any]], diary_entries: List[Dict[str, Any]]) -> str:
    """Generate a summary of the patient's emotional wellbeing."""
    try:
        if not emotions and not diary_entries:
            return "No emotional wellbeing data available to analyze."

        summary = "Emotional Wellbeing Summary:\n"

        # Analyze emotions
        if emotions:
            # Count emotion frequencies
            emotion_counts = {}
            for emotion in emotions:
                emotion_name = emotion.get("emotion", "").lower()
                if emotion_name:
                    emotion_counts[emotion_name] = emotion_counts.get(emotion_name, 0) + 1

            # Get dominant emotions
            sorted_emotions = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)

            if sorted_emotions:
                dominant_emotion = sorted_emotions[0][0]
                dominant_count = sorted_emotions[0][1]
                dominant_percentage = int((dominant_count / len(emotions)) * 100)

                summary += f"- Your most frequent emotion has been '{dominant_emotion}' ({dominant_percentage}% of recorded emotions).\n"

                # Check for emotional trends
                if len(emotions) >= 10:
                    recent_emotions = emotions[:10]  # Most recent 10
                    recent_emotion_counts = {}
                    for emotion in recent_emotions:
                        emotion_name = emotion.get("emotion", "").lower()
                        if emotion_name:
                            recent_emotion_counts[emotion_name] = recent_emotion_counts.get(emotion_name, 0) + 1

                    sorted_recent = sorted(recent_emotion_counts.items(), key=lambda x: x[1], reverse=True)
                    if sorted_recent and sorted_recent[0][0] != dominant_emotion:
                        summary += f"- Recently, '{sorted_recent[0][0]}' has become more prominent in your emotional state.\n"

                # Categorize emotions as positive/negative
                positive_emotions = ["happy", "joy", "excited", "calm", "relaxed", "content", "grateful"]
                negative_emotions = ["sad", "angry", "anxious", "stressed", "depressed", "worried", "fearful"]

                positive_count = sum(emotion_counts.get(e, 0) for e in positive_emotions)
                negative_count = sum(emotion_counts.get(e, 0) for e in negative_emotions)

                if positive_count > negative_count:
                    ratio = positive_count / max(negative_count, 1)
                    summary += f"- Overall, your emotional state has been more positive than negative (ratio {ratio:.1f}:1).\n"
                elif negative_count > positive_count:
                    ratio = negative_count / max(positive_count, 1)
                    summary += f"- You've experienced more challenging emotions than positive ones (ratio {ratio:.1f}:1).\n"

        # Analyze diary entries for emotional content
        if diary_entries:
            # Keywords for different emotional states
            emotional_keywords = {
                "positive": ["happy", "joy", "grateful", "excited", "good", "better", "improvement", "progress"],
                "negative": ["sad", "depressed", "anxious", "worried", "stress", "difficult", "hard", "struggle"],
                "neutral": ["okay", "fine", "normal", "average", "usual"]
            }

            # Count emotional mentions in diary
            emotional_mentions = {category: 0 for category in emotional_keywords}

            for entry in diary_entries:
                content = entry.get("content", "").lower()
                for category, keywords in emotional_keywords.items():
                    for keyword in keywords:
                        if keyword in content:
                            emotional_mentions[category] += 1
                            break

            # Determine dominant emotional tone in diary
            sorted_mentions = sorted(emotional_mentions.items(), key=lambda x: x[1], reverse=True)
            if sorted_mentions and sorted_mentions[0][1] > 0:
                summary += f"- Your diary entries reflect a predominantly {sorted_mentions[0][0]} emotional tone.\n"

        return summary
    except Exception as e:
        logger.error(f"Error generating emotional wellbeing summary: {str(e)}")
        return "Unable to analyze emotional wellbeing due to an error."

# Function to generate lifestyle and habits summary
def generate_lifestyle_summary(diary_entries: List[Dict[str, Any]], chat_history: List[Dict[str, Any]]) -> str:
    """Generate a summary of the patient's lifestyle and habits."""
    try:
        if not diary_entries and not chat_history:
            return "No lifestyle data available to analyze."

        summary = "Lifestyle and Habits Summary:\n"

        # Combine text from diary entries and patient chat messages
        all_texts = []
        for entry in diary_entries:
            all_texts.append(entry.get("content", "").lower())

        for msg in chat_history:
            if msg.get("role") == "patient":
                all_texts.append(msg.get("content", "").lower())

        if not all_texts:
            return "Insufficient data to analyze lifestyle patterns."

        # Keywords for different lifestyle aspects
        lifestyle_aspects = {
            "sleep": ["sleep", "insomnia", "tired", "rest", "nap", "woke up", "bed"],
            "exercise": ["exercise", "workout", "gym", "run", "walk", "jog", "physical activity", "active"],
            "diet": ["eat", "food", "meal", "diet", "nutrition", "hungry", "appetite"],
            "stress": ["stress", "overwhelm", "pressure", "relax", "meditation", "mindfulness"],
            "social": ["friend", "family", "social", "people", "talk", "conversation", "lonely", "isolation"]
        }

        # Count mentions of each lifestyle aspect
        aspect_mentions = {aspect: 0 for aspect in lifestyle_aspects}

        for text in all_texts:
            for aspect, keywords in lifestyle_aspects.items():
                for keyword in keywords:
                    if keyword in text:
                        aspect_mentions[aspect] += 1
                        break

        # Sort aspects by mention frequency
        sorted_aspects = sorted(aspect_mentions.items(), key=lambda x: x[1], reverse=True)

        # Add most mentioned aspects to summary
        mentioned_aspects = [aspect for aspect, count in sorted_aspects if count > 0]
        if mentioned_aspects:
            top_aspects = mentioned_aspects[:3]
            summary += f"- You frequently mention {', '.join(top_aspects)} in your entries and conversations.\n"

        # Check for specific lifestyle patterns

        # Sleep patterns
        sleep_texts = [text for text in all_texts if any(keyword in text for keyword in lifestyle_aspects["sleep"])]
        if sleep_texts:
            sleep_issues = any(issue in " ".join(sleep_texts) for issue in ["trouble", "problem", "can't sleep", "difficult", "insomnia", "wake up"])
            if sleep_issues:
                summary += "- You've mentioned having some challenges with sleep quality or duration.\n"
            else:
                summary += "- Sleep appears to be a topic you track regularly.\n"

        # Exercise patterns
        exercise_texts = [text for text in all_texts if any(keyword in text for keyword in lifestyle_aspects["exercise"])]
        if exercise_texts:
            exercise_positive = any(positive in " ".join(exercise_texts) for positive in ["good", "great", "enjoy", "love", "regular", "routine"])
            if exercise_positive:
                summary += "- You seem to have a positive relationship with physical activity.\n"
            else:
                summary += "- You've mentioned physical activity, though it may not be a consistent routine.\n"

        # Diet patterns
        diet_texts = [text for text in all_texts if any(keyword in text for keyword in lifestyle_aspects["diet"])]
        if diet_texts:
            diet_focus = any(focus in " ".join(diet_texts) for focus in ["healthy", "nutrition", "balanced", "vegetables", "fruits"])
            if diet_focus:
                summary += "- You appear to be mindful about nutrition and healthy eating.\n"

        return summary
    except Exception as e:
        logger.error(f"Error generating lifestyle summary: {str(e)}")
        return "Unable to analyze lifestyle patterns due to an error."

# Function to generate medication adherence summary
def generate_medication_summary(prescriptions: List[Dict[str, Any]], diary_entries: List[Dict[str, Any]], chat_history: List[Dict[str, Any]]) -> str:
    """Generate a summary of the patient's medication adherence."""
    try:
        if not prescriptions:
            return "No medication data available to analyze."

        summary = "Medication Summary:\n"

        # Count active medications
        active_medications = [p for p in prescriptions if p.get("status", "").lower() == "active"]
        if active_medications:
            summary += f"- You are currently prescribed {len(active_medications)} medication(s).\n"

            # List medications
            med_list = [f"{med.get('medication', 'Unknown')} ({med.get('dosage', '')} {med.get('frequency', '')})"
                       for med in active_medications]
            summary += f"- Your current medications: {', '.join(med_list)}.\n"

        # Combine text from diary entries and patient chat messages
        all_texts = []
        for entry in diary_entries:
            all_texts.append(entry.get("content", "").lower())

        for msg in chat_history:
            if msg.get("role") == "patient":
                all_texts.append(msg.get("content", "").lower())

        # Check for medication adherence mentions
        adherence_keywords = {
            "missed": ["missed", "forgot", "skip", "didn't take", "haven't taken"],
            "side_effects": ["side effect", "reaction", "nausea", "headache", "dizzy", "tired"],
            "adherence": ["took", "taking", "remember", "regular", "routine", "schedule"]
        }

        adherence_mentions = {category: 0 for category in adherence_keywords}

        for text in all_texts:
            for category, keywords in adherence_keywords.items():
                for keyword in keywords:
                    if keyword in text:
                        adherence_mentions[category] += 1
                        break

        # Add adherence insights
        if adherence_mentions["missed"] > 0:
            summary += "- You've mentioned occasionally missing medication doses.\n"

        if adherence_mentions["side_effects"] > 0:
            summary += "- You've discussed experiencing some medication side effects.\n"

        if adherence_mentions["adherence"] > adherence_mentions["missed"] * 2:
            summary += "- Overall, you appear to be consistent with your medication routine.\n"

        return summary
    except Exception as e:
        logger.error(f"Error generating medication summary: {str(e)}")
        return "Unable to analyze medication adherence due to an error."

# Function to generate patient insights using GPT
def generate_patient_insights(patient_data: Dict[str, Any], summaries: Dict[str, str]) -> List[Dict[str, Any]]:
    """Generate patient insights using GPT based on patient data and summaries."""
    try:
        # Check if we have the necessary data to generate insights
        if not patient_data.get("patient_info") and not summaries:
            logger.warning("Insufficient data to generate patient insights")
            return []

        # Optimize the prompt by focusing on the most relevant information
        # Create a more concise prompt for GPT to reduce token usage
        patient_info = patient_data.get("patient_info", {})
        patient_name = f"{patient_info.get('name', 'the patient')}"
        patient_age = patient_info.get('age', 'Unknown')
        patient_gender = patient_info.get('gender', 'Unknown')

        # Create a more structured and concise prompt
        prompt = f"""
You are an empathetic health coach generating personalized insights for {patient_name}, a {patient_age}-year-old {patient_gender}.

SUMMARIES:
1. TREATMENT: {summaries.get("treatment_progress", "No treatment progress data available.")}
2. EMOTIONAL: {summaries.get("emotional_wellbeing", "No emotional wellbeing data available.")}
3. LIFESTYLE: {summaries.get("lifestyle", "No lifestyle data available.")}
4. MEDICATION: {summaries.get("medication", "No medication data available.")}

KEY MEDICAL HISTORY:
{json.dumps(patient_data.get("medical_history", [])[:3], indent=2)}

CURRENT MEDICATIONS:
{json.dumps(patient_data.get("prescriptions", [])[:5], indent=2)}

Generate exactly 5 personalized insights in the following JSON format:
[
  {{
    "title": "Brief, encouraging title",
    "description": "Detailed explanation with specific recommendations (2-3 sentences)",
    "type": "One of: Progress, Recommendation, Milestone, Reminder, Encouragement",
    "priority": "A number from 1-5, where 5 is highest priority",
    "action_steps": ["Specific action 1", "Specific action 2", "Specific action 3"]
  }},
  ...
]

Focus on:
1. Treatment progress: Help the patient understand how their treatment is going and what to expect
2. Emotional wellbeing: Provide insights on emotional patterns and suggestions for emotional health
3. Lifestyle improvements: Offer practical suggestions for healthy habits based on their current patterns
4. Medication management: Help them understand and adhere to their medication regimen
5. Motivation and encouragement: Highlight positive trends and provide encouragement

Your insights must be empathetic, evidence-based, specific to this patient, actionable, and prioritized.
For each insight, provide 2-3 specific action steps the patient can take.
"""

        # Call GPT API
        if not OPENAI_CLIENT:
            logger.error("OpenAI client not initialized")
            return []

        try:
            # Log the start of the API call for performance tracking
            api_call_start = datetime.now()
            logger.info("Starting OpenAI API call for patient insights")

            # Make the API call with the new client format - using the same model as smart_agent.py
            response = OPENAI_CLIENT.chat.completions.create(
                model="gpt-4o-2024-11-20",  # Use the same model as in smart_agent.py
                messages=[
                    {"role": "system", "content": "You are an empathetic health coach that provides personalized insights to patients. You focus on being supportive, practical, and encouraging. You provide specific, actionable recommendations based on patient data."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,  # Lower temperature for more consistent responses (matching smart_agent.py)
                max_tokens=1500,  # Reduced token limit for faster response while still allowing detailed insights
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
            return []

        # Parse the JSON response
        try:
            # Extract JSON from the response (in case there's any extra text)
            import re
            json_match = re.search(r'\[[\s\S]*\]', gpt_response)
            if json_match:
                json_str = json_match.group(0)
                insights = json.loads(json_str)
            else:
                insights = json.loads(gpt_response)

            # Validate the insights format more efficiently
            validated_insights = []
            for insight in insights:
                if isinstance(insight, dict) and "title" in insight and "description" in insight:
                    # Use a dictionary comprehension for more efficient field validation
                    validated_insight = {
                        "title": insight.get("title", "Health Insight"),
                        "description": insight.get("description", ""),
                        "type": insight.get("type", "Recommendation"),
                        "priority": int(insight.get("priority", 3)),
                        "action_steps": insight.get("action_steps", [])
                    }
                    validated_insights.append(validated_insight)

            # Return at most 5 insights, prioritizing by priority level
            sorted_insights = sorted(validated_insights, key=lambda x: x["priority"], reverse=True)
            return sorted_insights[:5]

        except json.JSONDecodeError as json_error:
            logger.error(f"Error parsing GPT response as JSON: {str(json_error)}")
            logger.error(f"Raw response: {gpt_response}")
            return []

    except Exception as e:
        logger.error(f"Error generating patient insights: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return []

# Function to generate patient insights with data collection
def generate_patient_treatment_insights_internal(
    patient_id: str,
    db: Session,
    user_role: str,
    user_id: str
) -> PatientInsightsResponse:
    """Internal function to generate patient insights with caching."""
    # Get patient info
    patient = db.query(Patient).filter(Patient.patient_id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    # Only allow access if the user is this patient, a doctor, or an admin
    if user_role not in ["admin", "doctor"] and patient.user_id != user_id:
        raise HTTPException(status_code=403, detail="Not authorized to access this patient's insights")

    # Collect all patient data
    patient_data = {}

    # Use a single ThreadPoolExecutor for all operations to reduce overhead
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        # Start time for performance tracking
        start_time = datetime.now()
        logger.info(f"Starting data collection for patient {patient_id}")

        # Create futures for each data fetch operation
        futures = {
            "patient_info": executor.submit(get_patient_basic_info, patient_id, db),
            "medical_history": executor.submit(get_medical_history, patient_id, db),
            "prescriptions": executor.submit(get_prescriptions, patient_id, db),
            "emotions": executor.submit(get_emotion_analyses, patient_id, db),
            "diary_entries": executor.submit(get_diary_entries, patient_id, db),
            "appointments": executor.submit(get_appointments, patient_id, db),
            "chat_history": executor.submit(get_chat_history, patient_id, db)
        }

        # Collect results as they complete
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

        summary_futures = {
            "treatment_progress": executor.submit(
                generate_treatment_progress_summary,
                patient_data["medical_history"],
                patient_data["prescriptions"],
                patient_data["appointments"]
            ),
            "emotional_wellbeing": executor.submit(
                generate_emotional_wellbeing_summary,
                patient_data["emotions"],
                patient_data["diary_entries"]
            ),
            "lifestyle": executor.submit(
                generate_lifestyle_summary,
                patient_data["diary_entries"],
                patient_data["chat_history"]
            ),
            "medication": executor.submit(
                generate_medication_summary,
                patient_data["prescriptions"],
                patient_data["diary_entries"],
                patient_data["chat_history"]
            )
        }

        # Collect summary results
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
        gpt_future = executor.submit(generate_patient_insights, patient_data, summaries)
        gpt_insights = gpt_future.result()

        gpt_time = (datetime.now() - gpt_start_time).total_seconds()
        logger.info(f"GPT insights generation completed in {gpt_time:.2f} seconds")

        # Format insights for response
        insights = []
        for insight in gpt_insights:
            insights.append(PatientInsightItem(
                title=insight["title"],
                description=insight["description"],
                type=insight["type"].lower(),
                priority=insight["priority"],
                action_steps=insight["action_steps"]
            ))

        # Sort insights by priority (highest first)
        insights.sort(key=lambda x: x.priority, reverse=True)

        total_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Total insight generation completed in {total_time:.2f} seconds")

        return PatientInsightsResponse(
            insights=insights,
            generated_at=datetime.now()
        )

# Apply caching to the internal function - only use patient_id and user_role for the cache key
cached_generate_patient_insights = with_selective_cache("patient_treatment_insights", [0, 2, 3])(generate_patient_treatment_insights_internal)

# Patient insights endpoint
@router.get("/patient/{patient_id}", response_model=PatientInsightsResponse)
async def get_patient_treatment_insights(
    patient_id: str,
    db: Session = Depends(get_db),
    current_user: Dict = Depends(get_current_active_user)
):
    """Get personalized treatment insights for a patient."""
    try:
        # Check if user is the patient or an admin/doctor
        user_role = get_current_user_role(current_user)
        user_id = current_user.get("sub", "")

        # Use cached function to generate insights
        return cached_generate_patient_insights(patient_id, db, user_role, user_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating patient insights: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error generating insights: {str(e)}")
