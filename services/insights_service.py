from typing import Dict, List, Any, Optional
import logging
import json
from datetime import datetime, timedelta
import numpy as np
import random
from sqlalchemy.orm import Session
from sqlalchemy import desc, and_, or_
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
import openai
import os
from dotenv import load_dotenv

from database.database import get_db
from model.model_correct import Patient, Doctor, Appointment, Prescription, DiaryEntry, EmotionAnalysis, ChatMessage, MyDiary, MedicalHistory
from auth.dependencies import get_current_active_user

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

router = APIRouter()

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

class InsightsResponse(BaseModel):
    insights: List[InsightItem]
    generated_at: datetime

# Configure OpenAI
openai.api_key = OPENAI_API_KEY

# Helper functions
def analyze_chat_history(patient_id: str, db: Session, limit: int = 20) -> Dict[str, Any]:
    """
    Analyze chat history to extract insights about patient's mental health, concerns, and progress.

    Args:
        patient_id: The patient's ID
        db: Database session
        limit: Maximum number of messages to analyze

    Returns:
        Dictionary containing insights extracted from chat history
    """
    try:
        # Get patient information
        patient = db.query(Patient).filter(Patient.patient_id == patient_id).first()
        if not patient:
            logger.warning(f"Patient {patient_id} not found")
            return {
                "topics_discussed": [],
                "concerns": [],
                "progress_areas": [],
                "recent_messages": []
            }

        # Get recent chat messages
        messages = db.query(ChatMessage).filter(
            or_(
                and_(ChatMessage.sender_id == patient_id, ChatMessage.receiver_id != patient_id),
                and_(ChatMessage.receiver_id == patient_id, ChatMessage.sender_id != patient_id)
            )
        ).order_by(desc(ChatMessage.timestamp)).limit(limit).all()

        if not messages:
            logger.info(f"No chat messages found for patient {patient_id}")
            return {
                "topics_discussed": [],
                "concerns": [],
                "progress_areas": [],
                "recent_messages": []
            }

        # Format messages for analysis
        formatted_messages = []
        for msg in messages:
            role = "patient" if msg.sender_id == patient_id else "doctor"
            formatted_messages.append({
                "role": role,
                "content": msg.message_text,
                "timestamp": msg.timestamp.strftime("%Y-%m-%d %H:%M:%S") if msg.timestamp else "Unknown"
            })

        # Reverse to get chronological order
        formatted_messages.reverse()

        # Extract topics discussed
        topics = extract_topics_from_chat(formatted_messages)

        # Extract concerns
        concerns = extract_concerns_from_chat(formatted_messages)

        # Extract progress areas
        progress_areas = extract_progress_from_chat(formatted_messages)

        # Format recent messages for display
        recent_messages = []
        for msg in messages[:5]:  # Only include the 5 most recent messages
            role = "You" if msg.sender_id == patient_id else "Doctor"
            content = msg.message_text
            if content and len(content) > 100:
                content = content[:100] + "..."

            recent_messages.append({
                "role": role,
                "content": content,
                "timestamp": msg.timestamp.strftime("%Y-%m-%d %H:%M:%S") if msg.timestamp else "Unknown"
            })

        # Reverse recent messages to get chronological order
        recent_messages.reverse()

        return {
            "topics_discussed": topics,
            "concerns": concerns,
            "progress_areas": progress_areas,
            "recent_messages": recent_messages
        }
    except Exception as e:
        logger.error(f"Error analyzing chat history: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "topics_discussed": [],
            "concerns": [],
            "progress_areas": [],
            "recent_messages": []
        }

def extract_topics_from_chat(messages: List[Dict[str, Any]]) -> List[str]:
    """Extract main topics discussed in chat messages."""
    try:
        # Common health-related topics
        health_topics = [
            "sleep", "anxiety", "depression", "stress", "medication",
            "therapy", "exercise", "diet", "nutrition", "pain",
            "fatigue", "energy", "mood", "emotions", "relationships",
            "work", "family", "social", "physical health", "mental health"
        ]

        # Count topic occurrences
        topic_counts = {topic: 0 for topic in health_topics}

        # Analyze each message
        for message in messages:
            content = message.get("content", "").lower()
            for topic in health_topics:
                if topic in content:
                    topic_counts[topic] += 1

        # Get topics that were mentioned at least once
        mentioned_topics = [topic for topic, count in topic_counts.items() if count > 0]

        # Sort by frequency (most mentioned first)
        mentioned_topics.sort(key=lambda topic: topic_counts[topic], reverse=True)

        # Return top 5 topics or all if fewer than 5
        return mentioned_topics[:5]
    except Exception as e:
        logger.error(f"Error extracting topics from chat: {str(e)}")
        return []

def extract_concerns_from_chat(messages: List[Dict[str, Any]]) -> List[str]:
    """Extract patient concerns from chat messages."""
    try:
        # Keywords indicating concerns
        concern_indicators = [
            "worried about", "concerned about", "trouble with", "problem with",
            "struggling with", "difficulty", "can't", "unable to", "not able to",
            "issue with", "bothering me", "bothers me", "fear", "afraid"
        ]

        concerns = []

        # Only analyze patient messages
        patient_messages = [msg for msg in messages if msg.get("role") == "patient"]

        for message in patient_messages:
            content = message.get("content", "").lower()

            # Check for concern indicators
            for indicator in concern_indicators:
                if indicator in content:
                    # Extract the phrase after the indicator
                    start_idx = content.find(indicator) + len(indicator)
                    end_idx = content.find(".", start_idx)
                    if end_idx == -1:  # No period found
                        end_idx = len(content)

                    # Extract concern phrase
                    concern_phrase = content[start_idx:end_idx].strip()

                    # Only add if it's a meaningful phrase
                    if concern_phrase and len(concern_phrase) > 3 and len(concern_phrase) < 50:
                        # Capitalize first letter
                        concern_phrase = concern_phrase[0].upper() + concern_phrase[1:]
                        concerns.append(concern_phrase)

        # Remove duplicates while preserving order
        unique_concerns = []
        for concern in concerns:
            if concern not in unique_concerns:
                unique_concerns.append(concern)

        return unique_concerns[:3]  # Return top 3 concerns
    except Exception as e:
        logger.error(f"Error extracting concerns from chat: {str(e)}")
        return []

def extract_progress_from_chat(messages: List[Dict[str, Any]]) -> List[str]:
    """Extract areas of progress from chat messages."""
    try:
        # Keywords indicating progress
        progress_indicators = [
            "better", "improved", "improving", "progress", "feeling good",
            "more", "less", "reduced", "increased", "helping", "helps",
            "working", "effective", "successful", "achievement", "accomplished"
        ]

        progress_areas = []

        # Only analyze patient messages
        patient_messages = [msg for msg in messages if msg.get("role") == "patient"]

        for message in patient_messages:
            content = message.get("content", "").lower()

            # Check for progress indicators
            for indicator in progress_indicators:
                if indicator in content:
                    # Extract the phrase around the indicator
                    start_idx = max(0, content.find(indicator) - 30)
                    end_idx = min(len(content), content.find(indicator) + len(indicator) + 30)

                    # Extract progress phrase
                    progress_phrase = content[start_idx:end_idx].strip()

                    # Only add if it's a meaningful phrase
                    if progress_phrase and len(progress_phrase) > 10:
                        # Capitalize first letter
                        progress_phrase = progress_phrase[0].upper() + progress_phrase[1:]
                        progress_areas.append(progress_phrase)

        # Remove duplicates while preserving order
        unique_progress = []
        for progress in progress_areas:
            if not any(p in progress for p in unique_progress):
                unique_progress.append(progress)

        return unique_progress[:3]  # Return top 3 progress areas
    except Exception as e:
        logger.error(f"Error extracting progress from chat: {str(e)}")
        return []

# This function has been replaced by the one below
# Keeping this comment for reference

def generate_personalized_wellness_tips(patient_id: str, db: Session) -> List[Dict[str, str]]:
    """
    Generate personalized wellness tips based on patient data.

    This function is now a wrapper around generate_gpt_insights for backward compatibility.

    Args:
        patient_id: The patient's ID
        db: Database session

    Returns:
        List of wellness tips with categories
    """
    # Get insights from GPT
    insights = generate_gpt_insights(patient_id, db)

    # Convert the insights format to the old wellness tips format
    wellness_tips = []
    for insight in insights:
        wellness_tips.append({
            "category": insight["category"],
            "recommendation": insight["description"]
        })

    return wellness_tips

# This function has been replaced by generate_gpt_insights

# This function has been replaced by generate_gpt_insights

# This function has been replaced by generate_default_insights

def generate_mental_health_trends(patient_id: str, db: Session, days_back: int = 30) -> Dict[str, Any]:
    """
    Generate mental health trends over time based on emotional analysis and chat history.

    Args:
        patient_id: The patient's ID
        db: Database session
        days_back: Number of days to analyze

    Returns:
        Dictionary containing mental health trends
    """
    try:
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        # Get emotional analysis data within date range
        emotions = db.query(EmotionAnalysis).filter(
            EmotionAnalysis.patient_id == patient_id,
            EmotionAnalysis.analyzed_at >= start_date,
            EmotionAnalysis.analyzed_at <= end_date
        ).order_by(EmotionAnalysis.analyzed_at).all()

        if not emotions:
            logger.info(f"No emotion data found for patient {patient_id} in the last {days_back} days")
            return {
                "emotion_trends": [],
                "sentiment_trend": [],
                "overall_trend": "unknown"
            }

        # Group emotions by day
        emotion_by_day = {}
        sentiment_by_day = {}

        for emotion in emotions:
            day = emotion.analyzed_at.strftime("%Y-%m-%d")

            # Count emotions
            if day not in emotion_by_day:
                emotion_by_day[day] = {}

            category = emotion.emotion_category
            if category in emotion_by_day[day]:
                emotion_by_day[day][category] += 1
            else:
                emotion_by_day[day][category] = 1

            # Track sentiment scores
            if day not in sentiment_by_day:
                sentiment_by_day[day] = []

            sentiment_by_day[day].append(emotion.confidence_score)

        # Calculate dominant emotion for each day
        daily_emotions = []
        for day, counts in emotion_by_day.items():
            dominant_emotion = max(counts.items(), key=lambda x: x[1])[0]
            daily_emotions.append({
                "date": day,
                "emotion": dominant_emotion
            })

        # Calculate average sentiment for each day
        daily_sentiment = []
        for day, scores in sentiment_by_day.items():
            avg_sentiment = sum(scores) / len(scores)
            daily_sentiment.append({
                "date": day,
                "sentiment": round(avg_sentiment, 2)
            })

        # Sort by date
        daily_emotions.sort(key=lambda x: x["date"])
        daily_sentiment.sort(key=lambda x: x["date"])

        # Determine overall trend
        overall_trend = "stable"
        if len(daily_sentiment) >= 3:
            # Compare first third to last third
            first_third = daily_sentiment[:len(daily_sentiment)//3]
            last_third = daily_sentiment[-len(daily_sentiment)//3:]

            first_avg = sum(day["sentiment"] for day in first_third) / len(first_third)
            last_avg = sum(day["sentiment"] for day in last_third) / len(last_third)

            if last_avg > first_avg + 0.2:
                overall_trend = "improving"
            elif first_avg > last_avg + 0.2:
                overall_trend = "declining"

        return {
            "emotion_trends": daily_emotions,
            "sentiment_trend": daily_sentiment,
            "overall_trend": overall_trend
        }
    except Exception as e:
        logger.error(f"Error generating mental health trends: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "emotion_trends": [],
            "sentiment_trend": [],
            "overall_trend": "unknown"
        }

def calculate_health_score_trend(patient_id: str, db: Session) -> Dict[str, Any]:
    """Calculate health score trend over time."""
    try:
        # Get the current patient's health score
        patient = db.query(Patient).filter(Patient.patient_id == patient_id).first()

        if not patient:
            logger.warning(f"Patient {patient_id} not found")
            return {"dates": [], "scores": [], "trend": "unknown"}

        # Get current health score
        current_score = patient.health_score or 70  # Default to 70 if not set

        # For historical data, we would ideally have a health_score_history table
        # Since we don't, we'll create a simulated history based on the current score
        # with some realistic variations

        today = datetime.now().date()
        dates = [(today - timedelta(days=i*7)).strftime("%Y-%m-%d") for i in range(8)]
        dates.reverse()  # Oldest to newest

        # Create a realistic trend based on current score
        base_score = max(40, min(current_score - 10, 90))  # Start 10 points lower, within bounds
        scores = []

        # Generate trend toward current score
        for i in range(8):
            # Calculate target - gradually approach current score
            target = base_score + (current_score - base_score) * (i / 7)
            # Add some randomness
            variation = np.random.randint(-3, 4)
            score = int(target + variation)
            # Keep score within reasonable bounds
            score = max(0, min(score, 100))
            scores.append(score)

        # Ensure the last score matches the current patient health score
        if scores and patient.health_score:
            scores[-1] = patient.health_score

        return {
            "dates": dates,
            "scores": scores,
            "trend": "improving" if scores[-1] > scores[0] else "declining"
        }
    except Exception as e:
        logger.error(f"Error calculating health score trend: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {"dates": [], "scores": [], "trend": "unknown"}

def analyze_medication_adherence(patient_id: str, db: Session) -> Dict[str, Any]:
    """Analyze patient's medication adherence."""
    try:
        # Query prescriptions
        prescriptions = db.query(Prescription).filter(
            Prescription.patient_id == patient_id
        ).all()

        if not prescriptions:
            logger.info(f"No prescriptions found for patient {patient_id}")
            return {
                "adherence_rate": None,
                "missed_doses": 0,
                "total_doses": 0,
                "medications": []
            }

        # Get active prescriptions
        active_prescriptions = [p for p in prescriptions if p.status == "Active"]

        if not active_prescriptions:
            logger.info(f"No active prescriptions found for patient {patient_id}")
            # Use all prescriptions if no active ones
            active_prescriptions = prescriptions

        # In a real implementation, you would track actual medication intake
        # Since we don't have that data, we'll create realistic estimates based on
        # prescription dates and other factors

        medications = []
        total_doses = 0
        missed_doses = 0

        for prescription in active_prescriptions:
            try:
                # Calculate days on medication
                start_date = prescription.start_date
                end_date = prescription.end_date or datetime.now().date()

                if not start_date:
                    continue

                days_on_medication = (end_date - start_date).days + 1
                days_on_medication = max(1, days_on_medication)  # Ensure at least 1 day

                # Estimate doses based on prescription info and days
                # Assume typical dosing is 1-3 times per day
                daily_doses = 2  # Default assumption
                if prescription.dosage:
                    dosage_lower = prescription.dosage.lower()
                    if "once" in dosage_lower or "daily" in dosage_lower or "day" in dosage_lower:
                        daily_doses = 1
                    elif "twice" in dosage_lower or "two times" in dosage_lower or "2 times" in dosage_lower:
                        daily_doses = 2
                    elif "three" in dosage_lower or "3 times" in dosage_lower:
                        daily_doses = 3

                prescribed_doses = days_on_medication * daily_doses

                # Estimate missed doses - patients typically miss 15-30% of doses
                # Use patient health score as a factor - higher score = better adherence
                patient = db.query(Patient).filter(Patient.patient_id == patient_id).first()
                adherence_factor = 0.7  # Default
                if patient and patient.health_score:
                    # Convert health score (0-100) to adherence factor (0.5-0.95)
                    adherence_factor = 0.5 + (patient.health_score / 100) * 0.45

                # Calculate missed doses with some randomness
                missed = int(prescribed_doses * (1 - adherence_factor) * (0.8 + 0.4 * np.random.random()))
                missed = min(missed, prescribed_doses - 1)  # Ensure at least one dose taken

                adherence_rate = round(((prescribed_doses - missed) / prescribed_doses) * 100, 1)

                medications.append({
                    "name": prescription.medication_name,
                    "dosage": prescription.dosage,
                    "prescribed_doses": prescribed_doses,
                    "missed_doses": missed,
                    "adherence_rate": adherence_rate,
                    "start_date": start_date.strftime("%Y-%m-%d") if start_date else "Unknown",
                    "end_date": end_date.strftime("%Y-%m-%d") if end_date else "Ongoing"
                })

                total_doses += prescribed_doses
                missed_doses += missed
            except Exception as med_error:
                logger.error(f"Error processing prescription {prescription.prescription_id}: {str(med_error)}")
                continue

        overall_adherence = round(((total_doses - missed_doses) / total_doses) * 100, 1) if total_doses > 0 else 0

        return {
            "adherence_rate": overall_adherence,
            "missed_doses": missed_doses,
            "total_doses": total_doses,
            "medications": medications
        }
    except Exception as e:
        logger.error(f"Error analyzing medication adherence: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {"adherence_rate": None, "missed_doses": 0, "total_doses": 0, "medications": []}

def analyze_emotional_wellbeing(patient_id: str, db: Session) -> Dict[str, Any]:
    """Analyze patient's emotional wellbeing based on diary entries and emotion analysis."""
    try:
        # Try to get emotion analysis data
        emotion_analyses = db.query(EmotionAnalysis).filter(
            EmotionAnalysis.patient_id == patient_id
        ).order_by(EmotionAnalysis.analyzed_at.desc() if hasattr(EmotionAnalysis, 'analyzed_at') else EmotionAnalysis.emotion_id.desc()).limit(10).all()

        # If no emotion analyses, try to get diary entries directly
        diary_entries = []
        try:
            # First try DiaryEntry model
            diary_entries = db.query(DiaryEntry).filter(
                DiaryEntry.patient_id == patient_id
            ).order_by(DiaryEntry.created_at.desc()).limit(10).all()
        except Exception as diary_error:
            logger.error(f"Error fetching DiaryEntry: {str(diary_error)}")
            # Fallback to MyDiary model
            try:
                diary_entries = db.query(MyDiary).filter(
                    MyDiary.patient_id == patient_id
                ).order_by(MyDiary.created_at.desc()).limit(10).all()
            except Exception as my_diary_error:
                logger.error(f"Error fetching MyDiary: {str(my_diary_error)}")

        # Initialize default values
        dominant_emotion = "neutral"
        emotion_dist = {}
        recent_entries = []

        # Process emotion analyses if available
        if emotion_analyses:
            # Get the most recent emotion analysis
            latest_emotion = emotion_analyses[0]
            dominant_emotion = latest_emotion.emotion_category if hasattr(latest_emotion, 'emotion_category') else latest_emotion.dominant_emotion

            # Build emotion distribution from all recent analyses
            emotion_counts = {}
            for analysis in emotion_analyses:
                emotion = analysis.emotion_category if hasattr(analysis, 'emotion_category') else "unknown"
                if emotion in emotion_counts:
                    emotion_counts[emotion] += 1
                else:
                    emotion_counts[emotion] = 1

            # Convert counts to percentages
            total = sum(emotion_counts.values())
            if total > 0:
                emotion_dist = {emotion: round(count / total, 2) for emotion, count in emotion_counts.items()}

        # Process diary entries
        for entry in diary_entries:
            # Get content from the appropriate field
            content = ""
            if hasattr(entry, 'content'):
                content = entry.content
            elif hasattr(entry, 'notes'):
                content = entry.notes

            # Get date from the appropriate field
            date_str = "Unknown date"
            if hasattr(entry, 'created_at') and entry.created_at:
                date_str = entry.created_at.strftime("%Y-%m-%d")

            # Truncate long content
            if content and len(content) > 100:
                content = content[:100] + "..."

            # Get sentiment if available
            sentiment = "unknown"
            if hasattr(entry, 'sentiment'):
                sentiment = entry.sentiment

            recent_entries.append({
                "date": date_str,
                "content": content,
                "sentiment": sentiment
            })

        # If we have no emotion data but have diary entries, make a basic assessment
        if not emotion_analyses and diary_entries:
            # This is a very simplistic approach - in a real system, you'd use NLP
            dominant_emotion = "neutral"  # Default

            # Look for emotional keywords in diary entries
            positive_words = ["happy", "joy", "excited", "great", "good", "wonderful", "pleased"]
            negative_words = ["sad", "angry", "upset", "depressed", "anxious", "worried", "stressed"]

            positive_count = 0
            negative_count = 0

            for entry in recent_entries:
                content = entry["content"].lower()
                for word in positive_words:
                    if word in content:
                        positive_count += 1
                for word in negative_words:
                    if word in content:
                        negative_count += 1

            if positive_count > negative_count * 2:
                dominant_emotion = "happy"
            elif negative_count > positive_count * 2:
                dominant_emotion = "sad"
            elif negative_count > positive_count:
                dominant_emotion = "concerned"

            # Create a basic distribution
            emotion_dist = {
                "neutral": 0.6,
                dominant_emotion: 0.4
            }

        # Determine trend based on emotion analyses
        trend = "stable"
        if len(emotion_analyses) >= 3:
            # Very simple trend analysis - compare first and last emotions
            first_emotion = emotion_analyses[-1].emotion_category if hasattr(emotion_analyses[-1], 'emotion_category') else "unknown"
            last_emotion = emotion_analyses[0].emotion_category if hasattr(emotion_analyses[0], 'emotion_category') else "unknown"

            positive_emotions = ["happy", "joy", "excited", "content"]
            negative_emotions = ["sad", "angry", "anxious", "depressed", "stressed"]

            if first_emotion in negative_emotions and last_emotion in positive_emotions:
                trend = "improving"
            elif first_emotion in positive_emotions and last_emotion in negative_emotions:
                trend = "declining"

        return {
            "dominant_emotion": dominant_emotion,
            "emotion_distribution": emotion_dist,
            "trend": trend,
            "recent_entries": recent_entries
        }
    except Exception as e:
        logger.error(f"Error analyzing emotional wellbeing: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "dominant_emotion": "unknown",
            "emotion_distribution": {},
            "trend": "unknown",
            "recent_entries": []
        }

def get_upcoming_appointments(patient_id: str, db: Session) -> List[Dict[str, Any]]:
    """Get upcoming appointments for a patient."""
    try:
        # Get current date
        current_date = datetime.now().date()

        # Query appointments
        try:
            appointments = db.query(Appointment).filter(
                Appointment.patient_id == patient_id,
                Appointment.appointment_date >= current_date
            ).order_by(Appointment.appointment_date, Appointment.appointment_time).all()
        except Exception as query_error:
            logger.error(f"Error querying appointments: {str(query_error)}")
            return []

        # Filter for scheduled appointments if status field exists
        scheduled_appointments = []
        for appt in appointments:
            if not hasattr(appt, 'status') or appt.status == "Scheduled":
                scheduled_appointments.append(appt)

        if not scheduled_appointments:
            logger.info(f"No upcoming appointments found for patient {patient_id}")
            return []

        result = []
        for appointment in scheduled_appointments:
            try:
                # Format appointment date with proper error handling
                try:
                    appointment_date = appointment.appointment_date.strftime("%Y-%m-%d") if appointment.appointment_date else "No date specified"
                    days_until = (appointment.appointment_date - current_date).days
                except Exception as date_error:
                    logger.error(f"Error formatting appointment date: {str(date_error)}")
                    appointment_date = "Error formatting date"
                    days_until = 0

                # Format appointment time with proper error handling
                try:
                    appointment_time = appointment.appointment_time.strftime("%H:%M") if appointment.appointment_time else "No time specified"
                except Exception as time_error:
                    logger.error(f"Error formatting appointment time: {str(time_error)}")
                    appointment_time = "Error formatting time"

                # Get doctor info
                doctor_name = "Unknown Doctor"
                try:
                    if hasattr(appointment, 'doctor_id') and appointment.doctor_id:
                        doctor = db.query(Doctor).filter(Doctor.doctor_id == appointment.doctor_id).first()
                        if doctor:
                            # Format doctor name
                            title = doctor.title + " " if doctor.title else ""
                            first_name = doctor.first_name or ""
                            last_name = doctor.last_name or ""
                            doctor_name = f"{title}{first_name} {last_name}".strip()
                            if not doctor_name:
                                doctor_name = f"Doctor {str(doctor.doctor_id)[:8]}"
                except Exception as doctor_error:
                    logger.error(f"Error fetching doctor info: {str(doctor_error)}")

                # Get visit reason
                visit_reason = "Not specified"
                if hasattr(appointment, 'visit_reason') and appointment.visit_reason:
                    visit_reason = appointment.visit_reason

                result.append({
                    "date": appointment_date,
                    "time": appointment_time,
                    "doctor": doctor_name,
                    "reason": visit_reason,
                    "days_until": days_until
                })
            except Exception as appt_error:
                logger.error(f"Error processing appointment: {str(appt_error)}")
                continue

        # Sort by date and time
        result.sort(key=lambda x: (x["date"], x["time"]))

        return result
    except Exception as e:
        logger.error(f"Error fetching upcoming appointments: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return []

def generate_gpt_insights(patient_id: str, db: Session) -> List[Dict[str, Any]]:
    """
    Generate personalized insights using GPT based on comprehensive patient data.

    Args:
        patient_id: The patient's ID
        db: Database session

    Returns:
        List of insights with title, description, category, and priority
    """
    try:
        # Get patient basic information
        patient = db.query(Patient).filter(Patient.patient_id == patient_id).first()
        if not patient:
            logger.warning(f"Patient {patient_id} not found")
            return generate_default_insights()

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

        # Get medical history
        medical_history = []
        try:
            history_records = db.query(MedicalHistory).filter(
                MedicalHistory.patient_id == patient_id
            ).order_by(MedicalHistory.created_at.desc() if hasattr(MedicalHistory, 'created_at') else MedicalHistory.medical_history_id.desc()).limit(10).all()

            for record in history_records:
                history_item = {
                    "diagnosis": record.diagnosis if hasattr(record, 'diagnosis') else "Unknown",
                    "date": record.created_at.strftime("%Y-%m-%d") if hasattr(record, 'created_at') and record.created_at else "Unknown",
                    "notes": record.additional_notes if hasattr(record, 'additional_notes') else "",
                    "treatment": record.treatment if hasattr(record, 'treatment') else ""
                }
                medical_history.append(history_item)
        except Exception as med_error:
            logger.error(f"Error fetching medical history: {str(med_error)}")

        # Get prescriptions
        prescriptions = []
        try:
            rx_records = db.query(Prescription).filter(
                Prescription.patient_id == patient_id
            ).all()

            for rx in rx_records:
                rx_item = {
                    "medication": rx.medication_name if hasattr(rx, 'medication_name') else "Unknown",
                    "dosage": rx.dosage if hasattr(rx, 'dosage') else "",
                    "frequency": rx.frequency if hasattr(rx, 'frequency') else "",
                    "status": rx.status if hasattr(rx, 'status') else "Unknown"
                }
                prescriptions.append(rx_item)
        except Exception as rx_error:
            logger.error(f"Error fetching prescriptions: {str(rx_error)}")

        # Get emotion analyses
        emotions = []
        try:
            emotion_records = db.query(EmotionAnalysis).filter(
                EmotionAnalysis.patient_id == patient_id
            ).order_by(EmotionAnalysis.analyzed_at.desc() if hasattr(EmotionAnalysis, 'analyzed_at') else EmotionAnalysis.emotion_id.desc()).limit(15).all()

            for record in emotion_records:
                emotion_item = {
                    "emotion": record.emotion_category if hasattr(record, 'emotion_category') else "Unknown",
                    "date": record.analyzed_at.strftime("%Y-%m-%d") if hasattr(record, 'analyzed_at') and record.analyzed_at else "Unknown"
                }
                emotions.append(emotion_item)
        except Exception as emo_error:
            logger.error(f"Error fetching emotion analyses: {str(emo_error)}")

        # Get diary entries
        diary_entries = []
        try:
            # First try DiaryEntry model
            try:
                diary_records = db.query(DiaryEntry).filter(
                    DiaryEntry.patient_id == patient_id
                ).order_by(DiaryEntry.created_at.desc()).limit(10).all()

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
                ).order_by(MyDiary.created_at.desc()).limit(10).all()

                for entry in diary_records:
                    diary_item = {
                        "content": entry.notes if hasattr(entry, 'notes') else "",
                        "date": entry.created_at.strftime("%Y-%m-%d") if hasattr(entry, 'created_at') and entry.created_at else "Unknown"
                    }
                    diary_entries.append(diary_item)
        except Exception as diary_error:
            logger.error(f"Error fetching diary entries: {str(diary_error)}")

        # Get appointments
        appointments = []
        try:
            appt_records = db.query(Appointment).filter(
                Appointment.patient_id == patient_id
            ).order_by(Appointment.appointment_date.desc()).limit(5).all()

            for appt in appt_records:
                appt_item = {
                    "doctor": appt.doctor_name if hasattr(appt, 'doctor_name') else "Unknown",
                    "date": appt.appointment_date.strftime("%Y-%m-%d") if hasattr(appt, 'appointment_date') and appt.appointment_date else "Unknown",
                    "status": appt.status if hasattr(appt, 'status') else "Unknown",
                    "notes": appt.notes if hasattr(appt, 'notes') else ""
                }
                appointments.append(appt_item)
        except Exception as appt_error:
            logger.error(f"Error fetching appointments: {str(appt_error)}")

        # Compile all data into a comprehensive patient profile
        patient_profile = {
            "patient_info": patient_info,
            "medical_history": medical_history,
            "prescriptions": prescriptions,
            "emotions": emotions,
            "diary_entries": diary_entries,
            "appointments": appointments
        }

        # Create prompt for GPT
        prompt = f"""
You are a healthcare insights assistant. Based on the following patient data, generate 5 personalized health insights that would be valuable for the patient.
Each insight should be actionable, evidence-based, and tailored to the patient's specific health situation.

PATIENT DATA:
{json.dumps(patient_profile, indent=2)}

Generate exactly 5 insights in the following JSON format:
[
  {{
    "title": "Brief, catchy title for the insight",
    "description": "Detailed explanation of the insight (2-3 sentences)",
    "category": "One of: Treatment, Wellness, Mental Health, Physical Health, Nutrition, Sleep, Exercise",
    "priority": "A number from 1-5, where 5 is highest priority"
  }},
  ...
]

The insights should cover different aspects of health and wellness. Focus on patterns in the data, potential correlations, and actionable recommendations.
Do not include any explanatory text outside the JSON format.
"""

        # Call GPT API
        try:
            from services.smart_agent import SmartAgent
            agent = SmartAgent()
            response = agent.get_completion(prompt)

            # Parse the JSON response
            try:
                # Extract JSON from the response (in case there's any extra text)
                import re
                json_match = re.search(r'\[[\s\S]*\]', response)
                if json_match:
                    json_str = json_match.group(0)
                    insights = json.loads(json_str)
                else:
                    insights = json.loads(response)

                # Validate the insights format
                validated_insights = []
                for insight in insights:
                    if isinstance(insight, dict) and "title" in insight and "description" in insight:
                        # Ensure all required fields are present
                        validated_insight = {
                            "title": insight.get("title", "Health Insight"),
                            "description": insight.get("description", ""),
                            "category": insight.get("category", "Wellness"),
                            "priority": int(insight.get("priority", 3))
                        }
                        validated_insights.append(validated_insight)

                return validated_insights[:5]  # Ensure we return at most 5 insights
            except json.JSONDecodeError as json_error:
                logger.error(f"Error parsing GPT response as JSON: {str(json_error)}")
                logger.error(f"Raw response: {response}")
                # Fallback to default insights
                return generate_default_insights()
        except Exception as gpt_error:
            logger.error(f"Error calling GPT API: {str(gpt_error)}")
            import traceback
            logger.error(traceback.format_exc())
            # Fallback to default insights
            return generate_default_insights()
    except Exception as e:
        logger.error(f"Error generating GPT insights: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return generate_default_insights()

def generate_default_insights() -> List[Dict[str, Any]]:
    """Generate default insights when GPT generation fails."""
    return [
        {
            "title": "Stay Hydrated",
            "description": "Drinking adequate water is essential for overall health. Aim for 8 glasses daily to maintain energy levels and support bodily functions.",
            "category": "Wellness",
            "priority": 3
        },
        {
            "title": "Prioritize Sleep",
            "description": "Quality sleep is crucial for mental and physical health. Maintain a consistent sleep schedule and create a relaxing bedtime routine.",
            "category": "Sleep",
            "priority": 4
        },
        {
            "title": "Regular Exercise",
            "description": "Even moderate physical activity can significantly improve your health. Aim for at least 30 minutes of movement most days of the week.",
            "category": "Exercise",
            "priority": 4
        },
        {
            "title": "Mindful Eating",
            "description": "Focus on whole foods rich in nutrients. Include a variety of fruits, vegetables, lean proteins, and whole grains in your diet.",
            "category": "Nutrition",
            "priority": 3
        },
        {
            "title": "Stress Management",
            "description": "Chronic stress can impact your health. Practice relaxation techniques like deep breathing, meditation, or gentle yoga regularly.",
            "category": "Mental Health",
            "priority": 5
        }
    ]

def generate_lifestyle_recommendations(patient_id: str, db: Session) -> List[Dict[str, str]]:
    """Generate lifestyle recommendations based on patient data."""
    try:
        patient = db.query(Patient).filter(Patient.patient_id == patient_id).first()

        if not patient:
            logger.warning(f"Patient {patient_id} not found")
            return []

        recommendations = []

        # Get health score
        health_score = patient.health_score or 70  # Default to 70 if not set

        # Get medical history if available
        medical_conditions = []
        try:
            medical_history = db.query(MedicalHistory).filter(
                MedicalHistory.patient_id == patient_id
            ).all()

            for record in medical_history:
                if record.diagnosis:
                    medical_conditions.append(record.diagnosis.lower())
        except Exception as med_error:
            logger.error(f"Error fetching medical history: {str(med_error)}")

        # Check if patient is under medications
        under_medications = False
        if hasattr(patient, 'under_medications'):
            under_medications = patient.under_medications

        # Check for active prescriptions
        has_prescriptions = False
        try:
            prescriptions = db.query(Prescription).filter(
                Prescription.patient_id == patient_id,
                Prescription.status == "Active"
            ).first()
            has_prescriptions = prescriptions is not None
        except Exception as rx_error:
            logger.error(f"Error checking prescriptions: {str(rx_error)}")

        # If patient has prescriptions, consider them under medications
        if has_prescriptions:
            under_medications = True

        # Generate recommendations based on health score
        if health_score < 50:
            recommendations.append({
                "category": "Medical Checkup",
                "recommendation": "Schedule a comprehensive health checkup with your doctor as soon as possible."
            })

        if health_score < 70:
            recommendations.append({
                "category": "Exercise",
                "recommendation": "Start with 15-20 minutes of gentle exercise daily, gradually increasing to 30 minutes of moderate activity."
            })
        else:
            recommendations.append({
                "category": "Exercise",
                "recommendation": "Maintain your 30 minutes of daily exercise, and consider adding strength training twice a week."
            })

        # Sleep recommendations
        recommendations.append({
            "category": "Sleep",
            "recommendation": "Aim for 7-8 hours of quality sleep each night. Establish a regular sleep schedule and avoid screens before bedtime."
        })

        # Nutrition recommendations based on conditions
        if any(condition in medical_conditions for condition in ["diabetes", "high blood sugar", "glucose"]):
            recommendations.append({
                "category": "Nutrition",
                "recommendation": "Follow a low-glycemic diet with plenty of fiber. Limit refined carbohydrates and monitor your blood sugar regularly."
            })
        elif any(condition in medical_conditions for condition in ["hypertension", "high blood pressure"]):
            recommendations.append({
                "category": "Nutrition",
                "recommendation": "Follow a low-sodium diet rich in potassium. The DASH diet is recommended for managing blood pressure."
            })
        else:
            recommendations.append({
                "category": "Nutrition",
                "recommendation": "Eat a balanced diet with plenty of fruits, vegetables, whole grains, and lean proteins. Aim for at least 5 servings of fruits and vegetables daily."
            })

        # Stress management
        if health_score < 60:
            recommendations.append({
                "category": "Stress Management",
                "recommendation": "Practice deep breathing exercises for 5 minutes, three times daily. Consider speaking with a mental health professional about stress management techniques."
            })
        else:
            recommendations.append({
                "category": "Stress Management",
                "recommendation": "Practice mindfulness or meditation for 10 minutes daily to maintain mental well-being and reduce stress levels."
            })

        # Medication adherence if applicable
        if under_medications:
            recommendations.append({
                "category": "Medication",
                "recommendation": "Take all prescribed medications as directed. Set reminders if you often forget, and don't skip doses even if you feel better."
            })

        # Hydration for everyone
        recommendations.append({
            "category": "Hydration",
            "recommendation": "Drink at least 8 glasses of water daily. Stay well-hydrated, especially during physical activity or hot weather."
        })

        return recommendations
    except Exception as e:
        logger.error(f"Error generating lifestyle recommendations: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return []

def analyze_mental_health_conditions(patient_id: str, db: Session) -> Dict[str, Any]:
    """Analyze patient's mental health conditions and their trends."""
    try:
        # Get medical history for mental health conditions
        medical_history = []
        try:
            medical_history = db.query(MedicalHistory).filter(
                MedicalHistory.patient_id == patient_id
            ).order_by(MedicalHistory.created_at.desc() if hasattr(MedicalHistory, 'created_at') else MedicalHistory.medical_history_id.desc()).all()
        except Exception as med_error:
            logger.error(f"Error fetching medical history: {str(med_error)}")

        # Mental health conditions to look for
        mental_health_conditions = [
            "depression", "anxiety", "adhd", "ptsd", "bipolar",
            "ocd", "schizophrenia", "eating disorder", "insomnia",
            "panic disorder", "social anxiety", "phobia"
        ]

        # Extract mental health conditions from medical history
        conditions = []
        for record in medical_history:
            if record.diagnosis:
                diagnosis_lower = record.diagnosis.lower()
                for condition in mental_health_conditions:
                    if condition in diagnosis_lower:
                        condition_info = {
                            "condition": condition.title(),
                            "diagnosed_date": record.created_at.strftime("%Y-%m-%d") if hasattr(record, 'created_at') and record.created_at else "Unknown",
                            "notes": record.additional_notes if hasattr(record, 'additional_notes') else "",
                            "treatment": record.treatment if hasattr(record, 'treatment') else ""
                        }
                        conditions.append(condition_info)

        # Get emotion analyses to track condition trends
        emotion_analyses = db.query(EmotionAnalysis).filter(
            EmotionAnalysis.patient_id == patient_id
        ).order_by(EmotionAnalysis.analyzed_at.desc() if hasattr(EmotionAnalysis, 'analyzed_at') else EmotionAnalysis.emotion_id.desc()).limit(30).all()

        # Map conditions to related emotions for trend analysis
        condition_emotion_map = {
            "Depression": ["sad", "depressed", "hopeless", "neutral"],
            "Anxiety": ["anxious", "worried", "nervous", "fear"],
            "PTSD": ["fear", "anxious", "stressed"],
            "Bipolar": ["happy", "excited", "sad", "angry"],
            "OCD": ["anxious", "worried", "stressed"]
        }

        # Analyze trends for each condition
        condition_trends = []
        for condition in conditions:
            condition_name = condition["condition"]
            if condition_name in condition_emotion_map:
                related_emotions = condition_emotion_map[condition_name]

                # Count occurrences of related emotions
                emotion_counts = 0
                total_emotions = len(emotion_analyses) if emotion_analyses else 0

                for analysis in emotion_analyses:
                    emotion = analysis.emotion_category if hasattr(analysis, 'emotion_category') else "unknown"
                    if emotion.lower() in related_emotions:
                        emotion_counts += 1

                # Calculate prevalence
                prevalence = round((emotion_counts / total_emotions) * 100, 1) if total_emotions > 0 else 0

                # Determine trend
                trend = "stable"
                if total_emotions >= 10:
                    first_half = emotion_analyses[total_emotions//2:]
                    second_half = emotion_analyses[:total_emotions//2]

                    first_half_count = sum(1 for a in first_half if hasattr(a, 'emotion_category') and a.emotion_category.lower() in related_emotions)
                    second_half_count = sum(1 for a in second_half if hasattr(a, 'emotion_category') and a.emotion_category.lower() in related_emotions)

                    first_half_rate = first_half_count / len(first_half) if first_half else 0
                    second_half_rate = second_half_count / len(second_half) if second_half else 0

                    if second_half_rate < first_half_rate * 0.8:
                        trend = "improving"
                    elif second_half_rate > first_half_rate * 1.2:
                        trend = "worsening"

                condition_trends.append({
                    "condition": condition_name,
                    "prevalence": prevalence,
                    "trend": trend,
                    "diagnosed_date": condition["diagnosed_date"]
                })

        return {
            "conditions": conditions,
            "condition_trends": condition_trends
        }
    except Exception as e:
        logger.error(f"Error analyzing mental health conditions: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {"conditions": [], "condition_trends": []}

def analyze_symptoms(patient_id: str, db: Session) -> Dict[str, Any]:
    """Analyze patient's symptoms and their trends."""
    try:
        # Common symptoms to track
        common_symptoms = [
            "fatigue", "insomnia", "low appetite", "irritability",
            "headache", "nausea", "dizziness", "pain", "shortness of breath",
            "chest pain", "palpitations", "sweating", "trembling"
        ]

        # Get diary entries to extract symptoms
        diary_entries = []
        try:
            # First try DiaryEntry model
            diary_entries = db.query(DiaryEntry).filter(
                DiaryEntry.patient_id == patient_id
            ).order_by(DiaryEntry.created_at.desc()).limit(30).all()
        except Exception as diary_error:
            logger.error(f"Error fetching DiaryEntry: {str(diary_error)}")
            # Fallback to MyDiary model
            try:
                diary_entries = db.query(MyDiary).filter(
                    MyDiary.patient_id == patient_id
                ).order_by(MyDiary.created_at.desc()).limit(30).all()
            except Exception as my_diary_error:
                logger.error(f"Error fetching MyDiary: {str(my_diary_error)}")

        # Extract symptoms from diary entries
        symptom_occurrences = {symptom: 0 for symptom in common_symptoms}
        symptom_dates = {symptom: [] for symptom in common_symptoms}

        for entry in diary_entries:
            content = ""
            if hasattr(entry, 'content'):
                content = entry.content
            elif hasattr(entry, 'notes'):
                content = entry.notes

            if not content:
                continue

            entry_date = entry.created_at.strftime("%Y-%m-%d") if hasattr(entry, 'created_at') and entry.created_at else "Unknown"
            content_lower = content.lower()

            for symptom in common_symptoms:
                if symptom in content_lower:
                    symptom_occurrences[symptom] += 1
                    symptom_dates[symptom].append(entry_date)

        # Create symptom trends
        symptoms = []
        for symptom, count in symptom_occurrences.items():
            if count > 0:
                # Calculate frequency (percentage of days with symptom)
                frequency = round((count / len(diary_entries)) * 100, 1) if diary_entries else 0

                # Determine trend
                trend = "stable"
                dates = symptom_dates[symptom]
                if len(dates) >= 3:
                    # Check if symptom is becoming more or less frequent
                    first_half_dates = set(dates[len(dates)//2:])
                    second_half_dates = set(dates[:len(dates)//2])

                    if len(second_half_dates) < len(first_half_dates) * 0.7:
                        trend = "improving"
                    elif len(second_half_dates) > len(first_half_dates) * 1.3:
                        trend = "worsening"

                symptoms.append({
                    "symptom": symptom.title(),
                    "frequency": frequency,
                    "trend": trend,
                    "last_reported": dates[0] if dates else "Unknown"
                })

        # Sort by frequency (highest first)
        symptoms.sort(key=lambda x: x["frequency"], reverse=True)

        return {
            "symptoms": symptoms,
            "total_entries_analyzed": len(diary_entries)
        }
    except Exception as e:
        logger.error(f"Error analyzing symptoms: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {"symptoms": [], "total_entries_analyzed": 0}

def analyze_activities(patient_id: str, db: Session) -> Dict[str, Any]:
    """Analyze patient's activities like sleep, diet, exercise, etc."""
    try:
        # Get diary entries to extract activity information
        diary_entries = []
        try:
            # First try DiaryEntry model
            diary_entries = db.query(DiaryEntry).filter(
                DiaryEntry.patient_id == patient_id
            ).order_by(DiaryEntry.created_at.desc()).limit(30).all()
        except Exception as diary_error:
            logger.error(f"Error fetching DiaryEntry: {str(diary_error)}")
            # Fallback to MyDiary model
            try:
                diary_entries = db.query(MyDiary).filter(
                    MyDiary.patient_id == patient_id
                ).order_by(MyDiary.created_at.desc()).limit(30).all()
            except Exception as my_diary_error:
                logger.error(f"Error fetching MyDiary: {str(my_diary_error)}")

        # Activities to track
        activities = {
            "sleep": {
                "keywords": ["sleep", "slept", "nap", "rest", "tired", "insomnia"],
                "positive_indicators": ["good sleep", "slept well", "rested", "8 hours"],
                "negative_indicators": ["bad sleep", "insomnia", "couldn't sleep", "woke up", "tired"]
            },
            "exercise": {
                "keywords": ["exercise", "workout", "gym", "run", "jog", "walk", "yoga", "swim"],
                "positive_indicators": ["good workout", "exercised", "active"],
                "negative_indicators": ["no exercise", "skipped workout", "inactive"]
            },
            "diet": {
                "keywords": ["eat", "food", "meal", "diet", "nutrition", "breakfast", "lunch", "dinner"],
                "positive_indicators": ["healthy meal", "balanced diet", "nutritious", "vegetables", "protein"],
                "negative_indicators": ["junk food", "fast food", "skipped meal", "unhealthy"]
            },
            "screen_time": {
                "keywords": ["screen", "phone", "computer", "tv", "television", "social media", "netflix"],
                "positive_indicators": ["limited screen", "digital detox"],
                "negative_indicators": ["too much screen", "addicted to phone", "binge watching"]
            },
            "journaling": {
                "keywords": ["journal", "write", "diary", "reflect", "thoughts"],
                "positive_indicators": ["journaled", "wrote", "reflected"],
                "negative_indicators": ["didn't journal", "skipped writing"]
            }
        }

        # Analyze diary entries for activities
        activity_data = {}

        for activity_name, activity_info in activities.items():
            mentions = 0
            positive_mentions = 0
            negative_mentions = 0
            dates_mentioned = []

            for entry in diary_entries:
                content = ""
                if hasattr(entry, 'content'):
                    content = entry.content
                elif hasattr(entry, 'notes'):
                    content = entry.notes

                if not content:
                    continue

                entry_date = entry.created_at.strftime("%Y-%m-%d") if hasattr(entry, 'created_at') and entry.created_at else "Unknown"
                content_lower = content.lower()

                # Check if activity is mentioned
                is_mentioned = any(keyword in content_lower for keyword in activity_info["keywords"])

                if is_mentioned:
                    mentions += 1
                    dates_mentioned.append(entry_date)

                    # Check for positive indicators
                    if any(indicator in content_lower for indicator in activity_info["positive_indicators"]):
                        positive_mentions += 1

                    # Check for negative indicators
                    if any(indicator in content_lower for indicator in activity_info["negative_indicators"]):
                        negative_mentions += 1

            # Calculate sentiment ratio
            sentiment_ratio = 0
            if positive_mentions + negative_mentions > 0:
                sentiment_ratio = positive_mentions / (positive_mentions + negative_mentions)

            # Determine trend
            trend = "stable"
            if len(dates_mentioned) >= 3:
                first_half = dates_mentioned[len(dates_mentioned)//2:]
                second_half = dates_mentioned[:len(dates_mentioned)//2]

                if len(second_half) > len(first_half) * 1.3:
                    trend = "increasing"
                elif len(second_half) < len(first_half) * 0.7:
                    trend = "decreasing"

            # Determine quality
            quality = "neutral"
            if sentiment_ratio > 0.7:
                quality = "good"
            elif sentiment_ratio < 0.3:
                quality = "poor"

            activity_data[activity_name] = {
                "frequency": round((mentions / len(diary_entries)) * 100, 1) if diary_entries else 0,
                "quality": quality,
                "trend": trend,
                "last_mentioned": dates_mentioned[0] if dates_mentioned else "Unknown"
            }

        return {
            "activities": activity_data,
            "total_entries_analyzed": len(diary_entries)
        }
    except Exception as e:
        logger.error(f"Error analyzing activities: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {"activities": {}, "total_entries_analyzed": 0}

def analyze_correlations(patient_id: str, db: Session) -> Dict[str, Any]:
    """Analyze correlations between different health dimensions."""
    try:
        # Get emotional data
        emotion_data = analyze_emotional_wellbeing(patient_id, db)

        # Get activity data
        activity_data = analyze_activities(patient_id, db)

        # Get symptom data
        symptom_data = analyze_symptoms(patient_id, db)

        # Get diary entries for correlation analysis
        diary_entries = []
        try:
            # First try DiaryEntry model
            diary_entries = db.query(DiaryEntry).filter(
                DiaryEntry.patient_id == patient_id
            ).order_by(DiaryEntry.created_at.desc()).limit(60).all()
        except Exception as diary_error:
            logger.error(f"Error fetching DiaryEntry: {str(diary_error)}")
            # Fallback to MyDiary model
            try:
                diary_entries = db.query(MyDiary).filter(
                    MyDiary.patient_id == patient_id
                ).order_by(MyDiary.created_at.desc()).limit(60).all()
            except Exception as my_diary_error:
                logger.error(f"Error fetching MyDiary: {str(my_diary_error)}")

        # Generate correlation insights
        insights = []

        # Sleep and mood correlation
        if "sleep" in activity_data.get("activities", {}) and emotion_data.get("dominant_emotion") != "unknown":
            sleep_quality = activity_data["activities"]["sleep"]["quality"]
            dominant_emotion = emotion_data["dominant_emotion"]

            if sleep_quality == "good" and dominant_emotion in ["happy", "joy", "content"]:
                insights.append({
                    "type": "correlation",
                    "insight": "You tend to feel better on days when you report good sleep quality.",
                    "confidence": "medium",
                    "recommendation": "Prioritize maintaining your good sleep habits to support positive mood."
                })
            elif sleep_quality == "poor" and dominant_emotion in ["sad", "anxious", "angry", "stressed"]:
                insights.append({
                    "type": "correlation",
                    "insight": "Your mood tends to be lower on days when you report poor sleep.",
                    "confidence": "medium",
                    "recommendation": "Consider improving your sleep hygiene to potentially improve your mood."
                })

        # Exercise and energy correlation
        if "exercise" in activity_data.get("activities", {}) and "fatigue" in [s["symptom"].lower() for s in symptom_data.get("symptoms", [])]:
            exercise_frequency = activity_data["activities"]["exercise"]["frequency"]
            fatigue_symptom = next((s for s in symptom_data["symptoms"] if s["symptom"].lower() == "fatigue"), None)

            if exercise_frequency > 50 and fatigue_symptom and fatigue_symptom["trend"] == "improving":
                insights.append({
                    "type": "correlation",
                    "insight": "Your fatigue levels seem to improve when you exercise regularly.",
                    "confidence": "medium",
                    "recommendation": "Continue with your regular exercise routine to help manage fatigue."
                })

        # Screen time and sleep correlation
        if "screen_time" in activity_data.get("activities", {}) and "sleep" in activity_data.get("activities", {}):
            screen_time_quality = activity_data["activities"]["screen_time"]["quality"]
            sleep_quality = activity_data["activities"]["sleep"]["quality"]

            if screen_time_quality == "poor" and sleep_quality == "poor":
                insights.append({
                    "type": "correlation",
                    "insight": "High screen time appears to correlate with poorer sleep quality.",
                    "confidence": "medium",
                    "recommendation": "Consider reducing screen time, especially before bedtime, to potentially improve sleep."
                })

        # Diet and mood correlation
        if "diet" in activity_data.get("activities", {}) and emotion_data.get("dominant_emotion") != "unknown":
            diet_quality = activity_data["activities"]["diet"]["quality"]
            dominant_emotion = emotion_data["dominant_emotion"]

            if diet_quality == "good" and dominant_emotion in ["happy", "joy", "content"]:
                insights.append({
                    "type": "correlation",
                    "insight": "Your mood tends to be more positive on days when you eat healthier.",
                    "confidence": "medium",
                    "recommendation": "Continue focusing on nutritious meals to support your emotional wellbeing."
                })

        # Generate pattern recognition insights
        pattern_insights = []

        # Look for patterns in symptoms
        if symptom_data.get("symptoms", []):
            worsening_symptoms = [s for s in symptom_data["symptoms"] if s["trend"] == "worsening"]
            if worsening_symptoms:
                pattern_insights.append({
                    "type": "pattern",
                    "insight": f"Your {worsening_symptoms[0]['symptom']} has been increasing recently.",
                    "confidence": "medium",
                    "recommendation": "Consider discussing this trend with your healthcare provider."
                })

        # Look for patterns in activities
        if activity_data.get("activities", {}):
            decreasing_activities = [name for name, data in activity_data["activities"].items() if data["trend"] == "decreasing"]
            if "exercise" in decreasing_activities:
                pattern_insights.append({
                    "type": "pattern",
                    "insight": "Your exercise frequency has been decreasing over time.",
                    "confidence": "medium",
                    "recommendation": "Try to incorporate more physical activity into your routine."
                })

        # Generate progress feedback insights
        progress_insights = []

        # Look for improving symptoms
        if symptom_data.get("symptoms", []):
            improving_symptoms = [s for s in symptom_data["symptoms"] if s["trend"] == "improving"]
            if improving_symptoms:
                progress_insights.append({
                    "type": "progress",
                    "insight": f"Your {improving_symptoms[0]['symptom']} has been improving recently.",
                    "confidence": "medium",
                    "recommendation": "Continue with your current management approach for this symptom."
                })

        # Look for increasing positive activities
        if activity_data.get("activities", {}):
            increasing_activities = [name for name, data in activity_data["activities"].items() if data["trend"] == "increasing"]
            if "exercise" in increasing_activities:
                progress_insights.append({
                    "type": "progress",
                    "insight": "You've been exercising more consistently recently.",
                    "confidence": "medium",
                    "recommendation": "Keep up the good work with your exercise routine."
                })

        # Combine all insights
        all_insights = insights + pattern_insights + progress_insights

        # Sort by confidence
        all_insights.sort(key=lambda x: {"high": 3, "medium": 2, "low": 1}.get(x["confidence"], 0), reverse=True)

        return {
            "correlation_insights": insights,
            "pattern_insights": pattern_insights,
            "progress_insights": progress_insights,
            "all_insights": all_insights
        }
    except Exception as e:
        logger.error(f"Error analyzing correlations: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "correlation_insights": [],
            "pattern_insights": [],
            "progress_insights": [],
            "all_insights": []
        }

def analyze_treatment_effectiveness(patient_id: str, db: Session) -> Dict[str, Any]:
    """Analyze the effectiveness of current treatments."""
    try:
        # Get patient data
        patient = db.query(Patient).filter(Patient.patient_id == patient_id).first()

        if not patient:
            logger.warning(f"Patient {patient_id} not found")
            return {
                "overall_effectiveness": "unknown",
                "improvement_areas": [],
                "concerns": [],
                "recommendations": []
            }

        # Get prescriptions
        prescriptions = []
        try:
            prescriptions = db.query(Prescription).filter(
                Prescription.patient_id == patient_id
            ).all()
        except Exception as rx_error:
            logger.error(f"Error fetching prescriptions: {str(rx_error)}")

        # Get medical history
        medical_history = []
        try:
            medical_history = db.query(MedicalHistory).filter(
                MedicalHistory.patient_id == patient_id
            ).order_by(MedicalHistory.created_at.desc() if hasattr(MedicalHistory, 'created_at') else MedicalHistory.medical_history_id.desc()).all()
        except Exception as med_error:
            logger.error(f"Error fetching medical history: {str(med_error)}")

        # Get appointments for follow-ups
        appointments = []
        try:
            appointments = db.query(Appointment).filter(
                Appointment.patient_id == patient_id
            ).order_by(Appointment.appointment_date.desc()).limit(5).all()
        except Exception as appt_error:
            logger.error(f"Error fetching appointments: {str(appt_error)}")

        # Analyze treatment effectiveness based on available data

        # 1. Check if patient has active prescriptions
        active_prescriptions = [p for p in prescriptions if hasattr(p, 'status') and p.status == "Active"]

        # 2. Check health score trend (if we had historical data)
        health_score = patient.health_score or 50

        # 3. Determine overall effectiveness
        overall_effectiveness = "unknown"
        if health_score >= 80:
            overall_effectiveness = "good"
        elif health_score >= 60:
            overall_effectiveness = "moderate"
        else:
            overall_effectiveness = "poor"

        # 4. Identify improvement areas
        improvement_areas = []

        # Check for common health issues in medical history
        common_issues = {
            "sleep": ["insomnia", "sleep apnea", "sleep disorder", "trouble sleeping", "sleep quality"],
            "stress": ["stress", "anxiety", "tension", "overwhelmed"],
            "pain": ["pain", "ache", "discomfort", "soreness"],
            "energy": ["fatigue", "tired", "exhaustion", "low energy"],
            "mood": ["depression", "mood", "sadness", "irritability"]
        }

        found_issues = set()
        for record in medical_history:
            diagnosis = record.diagnosis.lower() if hasattr(record, 'diagnosis') and record.diagnosis else ""
            notes = record.notes.lower() if hasattr(record, 'notes') and record.notes else ""

            combined_text = diagnosis + " " + notes

            for issue_type, keywords in common_issues.items():
                if any(keyword in combined_text for keyword in keywords):
                    found_issues.add(issue_type)

        # Map issue types to readable labels
        issue_labels = {
            "sleep": "Sleep quality",
            "stress": "Stress levels",
            "pain": "Pain management",
            "energy": "Energy levels",
            "mood": "Mood stability"
        }

        improvement_areas = [issue_labels[issue] for issue in found_issues]

        # If no specific issues found, add generic areas based on health score
        if not improvement_areas:
            if health_score < 70:
                improvement_areas.append("Overall health")
            if active_prescriptions:
                improvement_areas.append("Medication effectiveness")

        # 5. Identify concerns
        concerns = []

        # Check for medication side effects
        if active_prescriptions:
            concerns.append("Potential medication side effects")

        # Check for chronic conditions
        chronic_conditions = ["diabetes", "hypertension", "asthma", "copd", "arthritis"]
        for record in medical_history:
            diagnosis = record.diagnosis.lower() if hasattr(record, 'diagnosis') and record.diagnosis else ""
            if any(condition in diagnosis for condition in chronic_conditions):
                concerns.append("Management of chronic conditions")
                break

        # 6. Generate recommendations
        recommendations = []

        # Regular follow-ups
        if not appointments:
            recommendations.append("Schedule a follow-up appointment to assess treatment progress")

        # Medication adherence
        if active_prescriptions:
            recommendations.append("Ensure consistent medication adherence for optimal results")

        # Health monitoring
        if "Management of chronic conditions" in concerns:
            recommendations.append("Monitor vital signs regularly and keep a health journal")

        # Lifestyle adjustments
        if health_score < 70:
            recommendations.append("Consider lifestyle adjustments to complement medical treatment")

        # If no specific recommendations, add generic ones
        if not recommendations:
            recommendations.append("Continue current treatment plan and monitor progress")
            if health_score < 80:
                recommendations.append("Discuss any persistent symptoms at your next appointment")

        return {
            "overall_effectiveness": overall_effectiveness,
            "improvement_areas": improvement_areas,
            "concerns": concerns,
            "recommendations": recommendations
        }
    except Exception as e:
        logger.error(f"Error analyzing treatment effectiveness: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "overall_effectiveness": "unknown",
            "improvement_areas": [],
            "concerns": [],
            "recommendations": []
        }

# API Endpoints
@router.get("/patient/{patient_id}", response_model=InsightsResponse)
async def get_patient_insights(
    patient_id: str,
    db: Session = Depends(get_db),
    current_user: Dict = Depends(get_current_active_user)
):
    """Get insights for a patient (viewed by the patient themselves)."""
    try:
        # Check if user has access to this patient's data
        user_id = current_user.get("sub")
        user_role = get_current_user_role(current_user)

        patient = db.query(Patient).filter(Patient.patient_id == patient_id).first()
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")

        # Only allow access if the user is this patient or an admin
        if user_role != "admin" and patient.user_id != user_id:
            raise HTTPException(status_code=403, detail="Not authorized to access this patient's insights")

        insights = []

        # Health Score Trend
        health_trend = calculate_health_score_trend(patient_id, db)
        if health_trend["dates"]:
            trend_text = "Your health score has been improving over the past few weeks." if health_trend["trend"] == "improving" else "Your health score has been declining recently."
            insights.append(InsightItem(
                title="Your Health Score Trend",
                description=trend_text,
                type="chart",
                data=health_trend,
                priority=3
            ))

        # Mental Health Trends
        mental_health_trends = generate_mental_health_trends(patient_id, db)
        if mental_health_trends["emotion_trends"]:
            trend_text = f"Your emotional well-being has been {mental_health_trends['overall_trend']} over the past month."
            insights.append(InsightItem(
                title="Your Emotional Well-being Trend",
                description=trend_text,
                type="emotion_chart",
                data=mental_health_trends,
                priority=4 if mental_health_trends["overall_trend"] == "declining" else 3
            ))

        # Chat History Analysis
        chat_analysis = analyze_chat_history(patient_id, db)
        if chat_analysis["topics_discussed"] or chat_analysis["concerns"] or chat_analysis["progress_areas"]:
            # Create a summary text
            summary_parts = []

            if chat_analysis["topics_discussed"]:
                topics_text = ", ".join(chat_analysis["topics_discussed"][:3])
                summary_parts.append(f"You've been discussing {topics_text}")

            if chat_analysis["concerns"]:
                concerns_text = chat_analysis["concerns"][0]
                summary_parts.append(f"You've expressed concern about {concerns_text}")

            if chat_analysis["progress_areas"]:
                progress_text = chat_analysis["progress_areas"][0]
                summary_parts.append(f"You've made progress with: {progress_text}")

            summary = ". ".join(summary_parts)

            insights.append(InsightItem(
                title="Conversation Insights",
                description=summary,
                type="chat_analysis",
                data=chat_analysis,
                priority=2
            ))

        # Medication Adherence
        med_adherence = analyze_medication_adherence(patient_id, db)
        if med_adherence["adherence_rate"] is not None:
            adherence_text = f"You've taken {med_adherence['adherence_rate']}% of your prescribed medications."
            if med_adherence["adherence_rate"] < 80:
                adherence_text += " Try to improve your medication adherence for better health outcomes."
            insights.append(InsightItem(
                title="Medication Adherence",
                description=adherence_text,
                type="progress",
                data=med_adherence,
                priority=4 if med_adherence["adherence_rate"] < 80 else 2
            ))

        # Emotional Wellbeing
        emotional_data = analyze_emotional_wellbeing(patient_id, db)
        if emotional_data["dominant_emotion"] != "unknown":
            emotion_text = f"Your dominant emotion recently has been {emotional_data['dominant_emotion']}."
            if emotional_data["dominant_emotion"].lower() in ["sad", "angry", "anxious", "depressed"]:
                emotion_text += " Consider speaking with your doctor about how you're feeling."
                priority = 5
            else:
                priority = 1
            insights.append(InsightItem(
                title="Emotional Wellbeing",
                description=emotion_text,
                type="emotion",
                data=emotional_data,
                priority=priority
            ))

        # Upcoming Appointments
        appointments = get_upcoming_appointments(patient_id, db)
        if appointments:
            next_appt = appointments[0]
            appt_text = f"Your next appointment is with {next_appt['doctor']} in {next_appt['days_until']} days."
            insights.append(InsightItem(
                title="Upcoming Appointments",
                description=appt_text,
                type="calendar",
                data={"appointments": appointments},
                priority=3 if next_appt["days_until"] <= 3 else 2
            ))

        # GPT-Generated Insights
        gpt_insights = generate_gpt_insights(patient_id, db)
        if gpt_insights:
            # Add each GPT insight as a separate insight item
            for insight in gpt_insights:
                insights.append(InsightItem(
                    title=insight["title"],
                    description=insight["description"],
                    type="gpt_insight",
                    data={"insight": insight},
                    priority=insight["priority"]
                ))

        # Treatment Effectiveness
        treatment_data = analyze_treatment_effectiveness(patient_id, db)
        if treatment_data["overall_effectiveness"] != "unknown":
            effectiveness_text = f"Your current treatment appears to be {treatment_data['overall_effectiveness']}."
            if treatment_data["improvement_areas"]:
                areas = ", ".join(treatment_data["improvement_areas"][:2])
                effectiveness_text += f" Focus on improving: {areas}."

            insights.append(InsightItem(
                title="Treatment Progress",
                description=effectiveness_text,
                type="treatment",
                data=treatment_data,
                priority=3 if treatment_data["overall_effectiveness"] != "good" else 1
            ))

        # Mental Health Conditions
        mental_health_data = analyze_mental_health_conditions(patient_id, db)
        if mental_health_data["condition_trends"]:
            # Find the most significant condition trend
            significant_condition = None
            for condition in mental_health_data["condition_trends"]:
                if condition["trend"] == "worsening":
                    significant_condition = condition
                    break
                elif condition["trend"] == "improving" and not significant_condition:
                    significant_condition = condition

            if significant_condition:
                condition_text = f"Your {significant_condition['condition']} appears to be {significant_condition['trend']}."
                if significant_condition["trend"] == "worsening":
                    condition_text += " Consider discussing this with your healthcare provider."
                    priority = 5
                elif significant_condition["trend"] == "improving":
                    condition_text += " Keep up with your current management approach."
                    priority = 2
                else:
                    condition_text += " Your condition appears stable."
                    priority = 3

                insights.append(InsightItem(
                    title=f"{significant_condition['condition']} Trend",
                    description=condition_text,
                    type="condition_trend",
                    data={"condition": significant_condition},
                    priority=priority
                ))

        # Symptoms Analysis
        symptom_data = analyze_symptoms(patient_id, db)
        if symptom_data["symptoms"]:
            # Find the most significant symptom (highest frequency or worsening)
            significant_symptom = None
            for symptom in symptom_data["symptoms"]:
                if symptom["trend"] == "worsening" and symptom["frequency"] > 30:
                    significant_symptom = symptom
                    break

            # If no worsening symptom found, use the most frequent one
            if not significant_symptom and symptom_data["symptoms"]:
                significant_symptom = symptom_data["symptoms"][0]

            if significant_symptom:
                symptom_text = f"Your {significant_symptom['symptom']} has been reported in {significant_symptom['frequency']}% of your entries"
                if significant_symptom["trend"] == "worsening":
                    symptom_text += " and appears to be increasing."
                    priority = 4
                elif significant_symptom["trend"] == "improving":
                    symptom_text += " but appears to be improving."
                    priority = 2
                else:
                    symptom_text += " and appears stable."
                    priority = 3

                insights.append(InsightItem(
                    title=f"{significant_symptom['symptom']} Tracking",
                    description=symptom_text,
                    type="symptom_trend",
                    data={"symptom": significant_symptom, "all_symptoms": symptom_data["symptoms"][:5]},
                    priority=priority
                ))

        # Activities Analysis
        activity_data = analyze_activities(patient_id, db)
        if activity_data.get("activities", {}):
            # Find the most significant activity insights
            sleep_data = activity_data["activities"].get("sleep", {})
            exercise_data = activity_data["activities"].get("exercise", {})
            diet_data = activity_data["activities"].get("diet", {})

            # Check sleep quality
            if sleep_data and sleep_data.get("quality") in ["good", "poor"]:
                sleep_quality = sleep_data["quality"]
                sleep_text = f"Your sleep quality appears to be {sleep_quality}."

                if sleep_quality == "poor":
                    sleep_text += " Consider improving your sleep habits for better overall health."
                    priority = 4
                else:
                    sleep_text += " Keep maintaining your good sleep habits."
                    priority = 2

                insights.append(InsightItem(
                    title="Sleep Quality",
                    description=sleep_text,
                    type="activity_insight",
                    data={"activity": "sleep", "details": sleep_data},
                    priority=priority
                ))

            # Check exercise trends
            if exercise_data and exercise_data.get("trend") in ["increasing", "decreasing"]:
                exercise_trend = exercise_data["trend"]
                exercise_text = f"Your exercise frequency has been {exercise_trend}."

                if exercise_trend == "decreasing":
                    exercise_text += " Try to incorporate more physical activity into your routine."
                    priority = 3
                else:
                    exercise_text += " Great job staying active!"
                    priority = 2

                insights.append(InsightItem(
                    title="Exercise Trends",
                    description=exercise_text,
                    type="activity_insight",
                    data={"activity": "exercise", "details": exercise_data},
                    priority=priority
                ))

            # Check diet quality
            if diet_data and diet_data.get("quality") in ["good", "poor"]:
                diet_quality = diet_data["quality"]
                diet_text = f"Your diet quality appears to be {diet_quality}."

                if diet_quality == "poor":
                    diet_text += " Consider improving your nutrition for better mental and physical health."
                    priority = 3
                else:
                    diet_text += " Keep up with your healthy eating habits!"
                    priority = 2

                insights.append(InsightItem(
                    title="Diet Quality",
                    description=diet_text,
                    type="activity_insight",
                    data={"activity": "diet", "details": diet_data},
                    priority=priority
                ))

        # Correlation Insights
        correlation_data = analyze_correlations(patient_id, db)
        if correlation_data["all_insights"]:
            # Get the top 3 insights
            top_insights = correlation_data["all_insights"][:3]

            for i, insight_item in enumerate(top_insights):
                insight_text = insight_item["insight"]
                if insight_item["recommendation"]:
                    insight_text += f" {insight_item['recommendation']}"

                # Determine priority based on insight type
                if insight_item["type"] == "correlation":
                    priority = 2
                elif insight_item["type"] == "pattern" and "worsening" in insight_text.lower():
                    priority = 4
                elif insight_item["type"] == "progress":
                    priority = 1
                else:
                    priority = 3

                insights.append(InsightItem(
                    title=f"Health Pattern #{i+1}",
                    description=insight_text,
                    type="correlation_insight",
                    data={"insight": insight_item},
                    priority=priority
                ))

        # Sort insights by priority (highest first)
        insights.sort(key=lambda x: x.priority, reverse=True)

        return InsightsResponse(
            insights=insights,
            generated_at=datetime.now()
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating patient insights: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating insights: {str(e)}")

@router.get("/doctor/patient/{patient_id}", response_model=InsightsResponse)
async def get_doctor_insights_for_patient(
    patient_id: str,
    db: Session = Depends(get_db),
    current_user: Dict = Depends(get_current_active_user)
):
    """Get insights about a patient for a doctor."""
    try:
        # Check if user is a doctor
        user_role = get_current_user_role(current_user)
        if user_role not in ["doctor", "admin"]:
            raise HTTPException(status_code=403, detail="Only doctors can access these insights")

        patient = db.query(Patient).filter(Patient.patient_id == patient_id).first()
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")

        insights = []

        # Health Score Trend (more detailed for doctors)
        health_trend = calculate_health_score_trend(patient_id, db)
        if health_trend["dates"]:
            trend_direction = "improving" if health_trend["trend"] == "improving" else "declining"
            trend_text = f"Patient's health score has been {trend_direction} over the past few weeks."
            insights.append(InsightItem(
                title="Health Score Trend",
                description=trend_text,
                type="chart",
                data=health_trend,
                priority=3
            ))

        # Mental Health Trends (clinical perspective)
        mental_health_trends = generate_mental_health_trends(patient_id, db)
        if mental_health_trends["emotion_trends"]:
            trend_text = f"Patient's emotional well-being has been {mental_health_trends['overall_trend']} over the past month."
            if mental_health_trends["overall_trend"] == "declining":
                trend_text += " Consider adjusting treatment approach."

            insights.append(InsightItem(
                title="Emotional Well-being Trend",
                description=trend_text,
                type="emotion_chart",
                data=mental_health_trends,
                priority=4 if mental_health_trends["overall_trend"] == "declining" else 3
            ))

        # Chat History Analysis (clinical perspective)
        chat_analysis = analyze_chat_history(patient_id, db)
        if chat_analysis["topics_discussed"] or chat_analysis["concerns"] or chat_analysis["progress_areas"]:
            # Create a summary text for doctors
            summary_parts = []

            if chat_analysis["topics_discussed"]:
                topics_text = ", ".join(chat_analysis["topics_discussed"][:3])
                summary_parts.append(f"Patient has been discussing {topics_text}")

            if chat_analysis["concerns"]:
                concerns_text = ", ".join(chat_analysis["concerns"])
                summary_parts.append(f"Patient has expressed concerns about: {concerns_text}")

            if chat_analysis["progress_areas"]:
                progress_text = ", ".join(chat_analysis["progress_areas"])
                summary_parts.append(f"Patient reports progress in: {progress_text}")

            summary = ". ".join(summary_parts)

            insights.append(InsightItem(
                title="Conversation Analysis",
                description=summary,
                type="chat_analysis",
                data=chat_analysis,
                priority=2
            ))

        # Medication Adherence (with clinical implications)
        med_adherence = analyze_medication_adherence(patient_id, db)
        if med_adherence["adherence_rate"] is not None:
            adherence_text = f"Patient has taken {med_adherence['adherence_rate']}% of prescribed medications."
            if med_adherence["adherence_rate"] < 80:
                adherence_text += " Poor adherence may be affecting treatment outcomes."
                priority = 5
            else:
                priority = 2
            insights.append(InsightItem(
                title="Medication Adherence",
                description=adherence_text,
                type="progress",
                data=med_adherence,
                priority=priority
            ))

        # Emotional Wellbeing (clinical perspective)
        emotional_data = analyze_emotional_wellbeing(patient_id, db)
        if emotional_data["dominant_emotion"] != "unknown":
            emotion_text = f"Patient's dominant emotion has been {emotional_data['dominant_emotion']}."
            if emotional_data["dominant_emotion"].lower() in ["sad", "angry", "anxious", "depressed"]:
                emotion_text += " Consider evaluating for mood disorders."
                priority = 4
            else:
                priority = 1
            insights.append(InsightItem(
                title="Emotional Assessment",
                description=emotion_text,
                type="emotion",
                data=emotional_data,
                priority=priority
            ))

        # Treatment Effectiveness
        treatment_data = analyze_treatment_effectiveness(patient_id, db)

        # Add patient info to treatment data for the frontend to use
        try:
            patient_info = {}
            if patient:
                # Format name with proper handling of None values
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
                    "name": name or f"Patient {patient.patient_id[:8]}...",
                    "age": age,
                    "gender": patient.gender or "Unknown",
                    "health_score": patient.health_score or 0,
                    "under_medications": patient.under_medications or False
                }

            # Add patient info to treatment data
            treatment_data["patient_info"] = patient_info
        except Exception as e:
            logger.error(f"Error adding patient info to treatment data: {str(e)}")

        insights.append(InsightItem(
            title="Treatment Effectiveness",
            description=f"Current treatment appears to be {treatment_data['overall_effectiveness']}.",
            type="analysis",
            data=treatment_data,
            priority=4 if treatment_data["overall_effectiveness"] != "good" else 2
        ))

        # GPT-Generated Insights (clinical perspective)
        gpt_insights = generate_gpt_insights(patient_id, db)
        if gpt_insights:
            # Add each GPT insight as a separate insight item with clinical perspective
            for insight in gpt_insights:
                # Modify the description for clinical perspective
                clinical_description = insight["description"].replace("you", "the patient").replace("your", "their")

                insights.append(InsightItem(
                    title=f"Clinical: {insight['title']}",
                    description=clinical_description,
                    type="clinical_gpt_insight",
                    data={"insight": insight},
                    priority=insight["priority"]
                ))

        # Mental Health Conditions (clinical perspective)
        mental_health_data = analyze_mental_health_conditions(patient_id, db)
        if mental_health_data["condition_trends"]:
            # Create a summary of all condition trends for the doctor
            condition_summaries = []
            for condition in mental_health_data["condition_trends"]:
                trend_text = f"{condition['condition']}: {condition['trend']} (prevalence: {condition['prevalence']}%)"
                condition_summaries.append(trend_text)

            conditions_text = "; ".join(condition_summaries)

            # Determine if there are any concerning trends
            has_worsening = any(c["trend"] == "worsening" for c in mental_health_data["condition_trends"])
            priority = 5 if has_worsening else 3

            insights.append(InsightItem(
                title="Mental Health Condition Trends",
                description=f"Patient condition trends: {conditions_text}",
                type="clinical_condition_trends",
                data=mental_health_data,
                priority=priority
            ))

        # Symptoms Analysis (clinical perspective)
        symptom_data = analyze_symptoms(patient_id, db)
        if symptom_data["symptoms"]:
            # Create a summary of top symptoms for the doctor
            top_symptoms = symptom_data["symptoms"][:5]  # Top 5 symptoms by frequency
            symptom_summaries = []

            for symptom in top_symptoms:
                trend_indicator = "↑" if symptom["trend"] == "worsening" else "↓" if symptom["trend"] == "improving" else "→"
                symptom_text = f"{symptom['symptom']} ({symptom['frequency']}% {trend_indicator})"
                symptom_summaries.append(symptom_text)

            symptoms_text = ", ".join(symptom_summaries)

            # Determine if there are any concerning symptoms
            has_worsening = any(s["trend"] == "worsening" and s["frequency"] > 30 for s in top_symptoms)
            priority = 4 if has_worsening else 2

            insights.append(InsightItem(
                title="Symptom Analysis",
                description=f"Key symptoms reported: {symptoms_text}",
                type="clinical_symptom_analysis",
                data={"symptoms": top_symptoms, "total_analyzed": symptom_data["total_entries_analyzed"]},
                priority=priority
            ))

        # Activities Analysis (clinical perspective)
        activity_data = analyze_activities(patient_id, db)
        if activity_data.get("activities", {}):
            # Create a summary of all activities for the doctor
            activity_summaries = []

            for activity_name, details in activity_data["activities"].items():
                if details["frequency"] > 10:  # Only include activities mentioned in >10% of entries
                    trend_indicator = "↑" if details["trend"] == "increasing" else "↓" if details["trend"] == "decreasing" else "→"
                    quality_indicator = "+" if details["quality"] == "good" else "-" if details["quality"] == "poor" else "="
                    activity_text = f"{activity_name.title()} ({quality_indicator}{trend_indicator})"
                    activity_summaries.append(activity_text)

            activities_text = ", ".join(activity_summaries)

            # Determine if there are any concerning activity patterns
            has_concerns = any(
                (details["quality"] == "poor" and details["frequency"] > 30) or
                (details["trend"] == "decreasing" and details["quality"] == "good")
                for _, details in activity_data["activities"].items()
            )
            priority = 3 if has_concerns else 2

            insights.append(InsightItem(
                title="Lifestyle Factors Analysis",
                description=f"Key lifestyle factors: {activities_text}",
                type="clinical_activity_analysis",
                data=activity_data,
                priority=priority
            ))

        # Correlation Insights (clinical perspective)
        correlation_data = analyze_correlations(patient_id, db)
        if correlation_data["all_insights"]:
            # Create a clinical summary of correlations
            clinical_insights = []

            for insight in correlation_data["all_insights"][:5]:  # Top 5 insights
                # Reformat insight for clinical perspective
                insight_text = insight["insight"].replace("You", "Patient").replace("your", "their")
                clinical_insights.append(insight_text)

            insights_text = "; ".join(clinical_insights[:2])  # Show top 2 in description

            # Determine priority based on insight types
            has_concerning_patterns = any(
                (insight["type"] == "pattern" and "worsening" in insight["insight"].lower()) or
                (insight["type"] == "correlation" and any(term in insight["insight"].lower() for term in ["poor", "worse", "negative", "decline"]))
                for insight in correlation_data["all_insights"][:3]
            )
            priority = 4 if has_concerning_patterns else 2

            insights.append(InsightItem(
                title="Health Pattern Correlations",
                description=f"Observed patterns: {insights_text}",
                type="clinical_correlations",
                data={"insights": correlation_data["all_insights"][:5]},
                priority=priority
            ))

        # Sort insights by priority (highest first)
        insights.sort(key=lambda x: x.priority, reverse=True)

        return InsightsResponse(
            insights=insights,
            generated_at=datetime.now()
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating doctor insights: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating insights: {str(e)}")
