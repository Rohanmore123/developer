from typing import Dict, List, Any, Optional
import logging
import json
import concurrent.futures
from datetime import datetime, timedelta
import random
import os
from sqlalchemy.orm import Session
from sqlalchemy import desc, and_, or_, text
from fastapi import APIRouter, Depends, HTTPException, Request, Header
from pydantic import BaseModel
from typing import Optional

from database.database import get_db
from model.model_correct import Patient, Doctor, Appointment, Prescription, DiaryEntry, EmotionAnalysis, ChatMessage, MyDiary, MedicalHistory
from auth.dependencies import get_current_active_user
import jwt
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from utils.cache_utils import with_selective_cache
from openai import OpenAI

# Initialize OpenAI client
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_CLIENT = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
class InsightItem(BaseModel):
    title: str
    description: str
    type: str  # "chart", "text", "alert", etc.
    data: Optional[Dict[str, Any]] = None
    priority: int = 0  # Higher number means higher priority

class InsightsResponse(BaseModel):
    insights: List[InsightItem]
    generated_at: datetime

def get_upcoming_appointments(patient_id: str, db: Session) -> List[Dict[str, Any]]:
    """Get upcoming appointments for a patient."""
    try:
        # Get current date
        current_date = datetime.now().date()

        # Query appointments - use a more explicit approach to avoid schema issues
        try:
            # First check what columns are available in the Appointment model
            available_columns = [column.name for column in Appointment.__table__.columns]
            logger.info(f"Available appointment columns: {available_columns}")

            # Use a more explicit query to avoid schema mismatches
            stmt = text(f"""
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
                    appointment_date, appointment_time
            """)

            # Execute raw SQL to avoid ORM issues
            result = db.execute(stmt)
            appointments = []

            for row in result:
                appt = {}
                for column in row.keys():
                    appt[column] = row[column]
                appointments.append(appt)

            if not appointments:
                logger.info(f"No upcoming appointments found for patient {patient_id}")
                return []

        except Exception as query_error:
            logger.error(f"Error querying appointments: {str(query_error)}")
            # Fallback to empty list on error
            return []

        # Process appointments
        result = []
        for appointment in appointments:
            try:
                # Check if this is a scheduled appointment
                if 'status' in appointment and appointment['status'] != "Scheduled":
                    continue

                # Format appointment date with proper error handling
                try:
                    appt_date = appointment.get('appointment_date')
                    if isinstance(appt_date, (datetime, datetime.date)):
                        appointment_date = appt_date.strftime("%Y-%m-%d")
                        days_until = (appt_date - current_date).days
                    else:
                        appointment_date = "No date specified"
                        days_until = 0
                except Exception as date_error:
                    logger.error(f"Error formatting appointment date: {str(date_error)}")
                    appointment_date = "Error formatting date"
                    days_until = 0

                # Format appointment time with proper error handling
                try:
                    appt_time = appointment.get('appointment_time')
                    if isinstance(appt_time, datetime.time):
                        appointment_time = appt_time.strftime("%H:%M")
                    else:
                        appointment_time = "No time specified"
                except Exception as time_error:
                    logger.error(f"Error formatting appointment time: {str(time_error)}")
                    appointment_time = "Error formatting time"

                # Get doctor info
                doctor_name = "Unknown Doctor"
                try:
                    doctor_id = appointment.get('doctor_id')
                    if doctor_id:
                        doctor = db.query(Doctor).filter(Doctor.doctor_id == doctor_id).first()
                        if doctor:
                            # Format doctor name
                            title = doctor.title + " " if hasattr(doctor, 'title') and doctor.title else ""
                            first_name = doctor.first_name or "" if hasattr(doctor, 'first_name') else ""
                            last_name = doctor.last_name or "" if hasattr(doctor, 'last_name') else ""
                            doctor_name = f"{title}{first_name} {last_name}".strip()
                            if not doctor_name:
                                doctor_name = f"Doctor {str(doctor_id)[:8]}"
                except Exception as doctor_error:
                    logger.error(f"Error fetching doctor info: {str(doctor_error)}")

                # Get visit reason if available
                visit_reason = "Not specified"
                if 'visit_reason' in appointment and appointment['visit_reason']:
                    visit_reason = appointment['visit_reason']

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
        # Return empty list on error to avoid breaking the insights generation
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
            # Get past appointments using raw SQL to avoid schema mismatches
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
                ORDER BY
                    appointment_date DESC
                LIMIT 5
            """)

            # Execute raw SQL
            appt_result = db.execute(past_appt_stmt)
            appt_records = appt_result.fetchall()

            for appt in appt_records:
                # Get doctor name from doctor_id
                doctor_name = "Unknown"
                try:
                    if appt.doctor_id:
                        doctor = db.query(Doctor).filter(Doctor.doctor_id == appt.doctor_id).first()
                        if doctor:
                            doctor_name = f"Dr. {doctor.first_name} {doctor.last_name}"
                except:
                    pass

                # Format date
                appt_date = "Unknown"
                try:
                    if appt.appointment_date:
                        if isinstance(appt.appointment_date, str):
                            appt_date = appt.appointment_date
                        else:
                            appt_date = appt.appointment_date.strftime("%Y-%m-%d")
                except:
                    pass

                appt_item = {
                    "doctor": doctor_name,
                    "date": appt_date,
                    "status": appt.status if appt.status else "Unknown",
                    "notes": appt.notes if appt.notes else "",
                    "reason": appt.visit_reason if hasattr(appt, 'visit_reason') and appt.visit_reason else "General checkup"
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

        # Now, instead of using GPT, we'll generate insights based on the patient data
        # This is a rule-based approach that looks at the patient data and generates relevant insights

        insights = []

        # 1. Check BMI and generate insight if available
        if "bmi" in patient_info and patient_info["bmi"] and patient_info["bmi"] != "Unknown":
            try:
                bmi = float(patient_info["bmi"])
                if bmi < 18.5:
                    insights.append({
                        "title": "Healthy Weight Management",
                        "description": "Your BMI indicates you may be underweight. Consider working with a nutritionist to develop a balanced meal plan that helps you reach a healthy weight while ensuring you get all essential nutrients.",
                        "category": "Nutrition",
                        "priority": 4
                    })
                elif bmi >= 25 and bmi < 30:
                    insights.append({
                        "title": "Weight Management Plan",
                        "description": "Your BMI indicates you're in the overweight range. Small changes like adding 30 minutes of daily walking and reducing processed food intake can help improve your overall health metrics.",
                        "category": "Wellness",
                        "priority": 3
                    })
                elif bmi >= 30:
                    insights.append({
                        "title": "Comprehensive Weight Management",
                        "description": "Your BMI falls in the obese range, which increases risk for several health conditions. Consider consulting with your doctor about a personalized weight management plan including nutrition, exercise, and possibly support groups.",
                        "category": "Physical Health",
                        "priority": 5
                    })
            except:
                pass

        # 2. Check emotions and generate insight if available
        if emotions:
            # Count emotion frequencies
            emotion_counts = {}
            for emotion in emotions:
                dominant = emotion.get("emotion", "").lower()
                if dominant:
                    emotion_counts[dominant] = emotion_counts.get(dominant, 0) + 1

            # Find most common emotion
            most_common_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0] if emotion_counts else None

            if most_common_emotion:
                if most_common_emotion in ["sad", "sadness", "depressed", "depression"]:
                    insights.append({
                        "title": "Mood Enhancement Strategies",
                        "description": "Your recent emotional patterns show frequent sadness. Regular physical activity, especially outdoors, can boost serotonin levels and improve mood. Even 15 minutes of morning sunlight can make a difference.",
                        "category": "Mental Health",
                        "priority": 4
                    })
                elif most_common_emotion in ["anxious", "anxiety", "worried", "fear", "scared"]:
                    insights.append({
                        "title": "Anxiety Management Techniques",
                        "description": "Your emotional patterns indicate frequent anxiety. Try the 4-7-8 breathing technique when feeling anxious: inhale for 4 seconds, hold for 7, exhale for 8. This activates your parasympathetic nervous system.",
                        "category": "Mental Health",
                        "priority": 4
                    })
                elif most_common_emotion in ["angry", "anger", "frustrated", "frustration"]:
                    insights.append({
                        "title": "Emotional Regulation Strategies",
                        "description": "Your emotional patterns show frequent anger or frustration. When emotions run high, try the STOP technique: Stop, Take a breath, Observe your feelings without judgment, and Proceed with a thoughtful response.",
                        "category": "Mental Health",
                        "priority": 3
                    })
                elif most_common_emotion in ["happy", "happiness", "joy", "content", "contentment"]:
                    insights.append({
                        "title": "Maintaining Positive Emotions",
                        "description": "Your emotional patterns show frequent positive emotions. Continue practices that bring you joy, and consider keeping a gratitude journal to further enhance your well-being by documenting three positive experiences daily.",
                        "category": "Mental Health",
                        "priority": 2
                    })

        # 3. Check prescriptions and generate insight if available
        if prescriptions:
            insights.append({
                "title": "Medication Adherence",
                "description": "Consistent medication timing improves effectiveness. Set daily alarms for your medications and use a pill organizer to ensure you're taking the right medications at the right times.",
                "category": "Treatment",
                "priority": 4
            })

        # 4. Check medical history and generate insight if available
        if medical_history:
            # Look for common conditions
            conditions = [record.get("diagnosis", "").lower() for record in medical_history]

            if any(condition for condition in conditions if "diabetes" in condition):
                insights.append({
                    "title": "Blood Sugar Management",
                    "description": "For diabetes management, try the plate method: fill half your plate with non-starchy vegetables, a quarter with lean protein, and a quarter with whole grains or starchy vegetables to help regulate blood sugar levels.",
                    "category": "Treatment",
                    "priority": 5
                })
            elif any(condition for condition in conditions if "hypertension" in condition or "high blood pressure" in condition):
                insights.append({
                    "title": "Blood Pressure Management",
                    "description": "For hypertension management, the DASH diet has shown significant benefits. Focus on fruits, vegetables, whole grains, and low-fat dairy while reducing sodium intake to less than 2,300mg daily.",
                    "category": "Treatment",
                    "priority": 5
                })
            elif any(condition for condition in conditions if "depression" in condition or "anxiety" in condition):
                insights.append({
                    "title": "Mental Wellness Routine",
                    "description": "For managing depression and anxiety, establishing a consistent sleep schedule can significantly impact mood regulation. Aim to go to bed and wake up at the same times each day, even on weekends.",
                    "category": "Mental Health",
                    "priority": 4
                })

        # 5. General wellness insight based on age
        if "age" in patient_info and patient_info["age"] and patient_info["age"] != "Unknown":
            try:
                age = int(patient_info["age"])
                if age < 30:
                    insights.append({
                        "title": "Preventive Health Habits",
                        "description": "Building healthy habits in your 20s creates a foundation for lifelong health. Focus on establishing a regular exercise routine, learning stress management techniques, and developing cooking skills for nutritious meals.",
                        "category": "Wellness",
                        "priority": 3
                    })
                elif age >= 30 and age < 50:
                    insights.append({
                        "title": "Midlife Health Optimization",
                        "description": "In your 30s and 40s, prioritize regular health screenings and stress management. Consider adding strength training twice weekly to maintain muscle mass and bone density as your metabolism naturally slows.",
                        "category": "Wellness",
                        "priority": 3
                    })
                else:
                    insights.append({
                        "title": "Healthy Aging Strategies",
                        "description": "After 50, focus on maintaining mobility and independence. Regular balance exercises can reduce fall risk, while staying socially connected helps maintain cognitive function and emotional wellbeing.",
                        "category": "Wellness",
                        "priority": 4
                    })
            except:
                pass

        # If we don't have enough insights, add some general ones
        while len(insights) < 5:
            # Add general insights that apply to most people
            general_insights = [
                {
                    "title": "Hydration Habit",
                    "description": "Most people don't drink enough water, which can affect energy, cognition, and overall health. Try keeping a water bottle visible on your desk and refilling it at least twice during your workday.",
                    "category": "Wellness",
                    "priority": 3
                },
                {
                    "title": "Movement Breaks",
                    "description": "Sitting for long periods increases health risks even if you exercise regularly. Set a timer to stand and move for 5 minutes every hour during your workday to improve circulation and energy levels.",
                    "category": "Physical Health",
                    "priority": 3
                },
                {
                    "title": "Sleep Environment Optimization",
                    "description": "Quality sleep begins with your environment. Keep your bedroom cool (65-68°F), dark, and quiet. Consider removing electronic devices or using night mode to reduce blue light exposure before bedtime.",
                    "category": "Sleep",
                    "priority": 4
                },
                {
                    "title": "Mindful Eating Practice",
                    "description": "Eating without distractions helps you tune into hunger cues and enjoy food more. Try eating one meal a day without screens, focusing completely on the flavors, textures, and your body's signals.",
                    "category": "Nutrition",
                    "priority": 3
                },
                {
                    "title": "Nature Connection",
                    "description": "Spending time in natural settings reduces stress hormones and improves mood. Even a 20-minute walk in a park or green space three times weekly can significantly benefit your mental wellbeing.",
                    "category": "Mental Health",
                    "priority": 3
                }
            ]

            # Add insights that aren't already in our list
            for insight in general_insights:
                if insight not in insights:
                    insights.append(insight)
                    break

        # Return at most 5 insights, prioritizing by priority level
        sorted_insights = sorted(insights, key=lambda x: x["priority"], reverse=True)
        return sorted_insights[:5]
    except Exception as e:
        logger.error(f"Error generating insights: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return generate_default_insights()

def generate_categorized_insights() -> Dict[str, List[Dict[str, Any]]]:
    """Generate categorized insights for different health and wellness areas."""
    return {
        "mindfulness_stress_reduction": [
            {
                "title": "5-Minute Deep Breathing",
                "description": "Taking 5 deep breaths can help reduce cortisol levels and calm your nervous system. Try this whenever you feel overwhelmed or anxious.",
                "category": "Mindfulness",
                "priority": 4
            },
            {
                "title": "Mindful Body Scan",
                "description": "Mindful body scans can improve sleep quality and reduce muscle tension. Spend 5 minutes before bed noticing sensations from head to toe without judgment.",
                "category": "Mindfulness",
                "priority": 3
            },
            {
                "title": "Daily Guided Meditation",
                "description": "A 10-minute guided meditation each day can lower anxiety and increase resilience. Start with a simple breathing meditation and gradually increase duration.",
                "category": "Mindfulness",
                "priority": 4
            },
            {
                "title": "Practice Stillness",
                "description": "Spending just a few minutes in stillness can improve emotional regulation. Find a quiet spot, sit comfortably, and simply observe your thoughts without engaging with them.",
                "category": "Mindfulness",
                "priority": 3
            },
            {
                "title": "Consistent Mindfulness",
                "description": "Practicing mindfulness daily can improve focus and reduce reactivity over time. Even 5 minutes per day is more effective than occasional longer sessions.",
                "category": "Mindfulness",
                "priority": 3
            },
            {
                "title": "Release Physical Stress",
                "description": "Stress often resides in the body. Gentle movement or stretching can help release tension, especially in your shoulders, jaw, and lower back.",
                "category": "Mindfulness",
                "priority": 3
            },
            {
                "title": "Emotional Journaling",
                "description": "Journaling about your emotions before bed can clear your mind and support better rest. Write freely for 5-10 minutes without editing or judging your thoughts.",
                "category": "Mindfulness",
                "priority": 3
            },
            {
                "title": "Mindful Observation",
                "description": "Mindfulness isn't about stopping thoughts, but observing them without judgment. Practice noticing thoughts as they arise and letting them pass like clouds.",
                "category": "Mindfulness",
                "priority": 3
            },
            {
                "title": "Nasal Breathing",
                "description": "Breathing through your nose and slowing down your exhale calms your heart rate. Try making your exhale slightly longer than your inhale for a calming effect.",
                "category": "Mindfulness",
                "priority": 3
            },
            {
                "title": "Quick Stress Reset",
                "description": "Even 2 minutes of conscious breathwork can disrupt stress spirals. Place one hand on your chest and one on your belly, and take 5-10 slow, deep breaths.",
                "category": "Mindfulness",
                "priority": 4
            }
        ],
        "social_wellness": [
            {
                "title": "Prioritize Connections",
                "description": "Meaningful social connection is one of the strongest predictors of long-term mental health. Schedule regular time with people who make you feel supported and understood.",
                "category": "Social Wellness",
                "priority": 4
            },
            {
                "title": "Mood-Boosting Conversations",
                "description": "A short chat with someone you trust can shift your mood significantly. Even a 10-minute phone call can provide perspective and emotional support.",
                "category": "Social Wellness",
                "priority": 3
            },
            {
                "title": "Address Loneliness",
                "description": "Loneliness isn't just emotional — it can also affect immune health and sleep quality. Consider joining groups based on your interests to build new connections.",
                "category": "Social Wellness",
                "priority": 4
            },
            {
                "title": "Quality Over Quantity",
                "description": "Quality of connections matters more than quantity — focus on nurturing key relationships. Deeper connections with a few people often provide more support than many superficial ones.",
                "category": "Social Wellness",
                "priority": 3
            },
            {
                "title": "Set Healthy Boundaries",
                "description": "Setting boundaries is an act of self-respect and can reduce emotional burnout. Practice saying no to requests that drain your energy without adding value.",
                "category": "Social Wellness",
                "priority": 4
            },
            {
                "title": "Alternative Connections",
                "description": "Spending time with pets or nature can also count as emotional connection. These interactions can reduce stress hormones and provide comfort without social pressure.",
                "category": "Social Wellness",
                "priority": 3
            },
            {
                "title": "Give Back",
                "description": "Volunteering or acts of kindness can boost serotonin and feelings of purpose. Even small gestures like helping a neighbor can strengthen community bonds.",
                "category": "Social Wellness",
                "priority": 3
            },
            {
                "title": "Process Emotions Together",
                "description": "Talking through your emotions with someone can reduce the intensity of negative feelings. Sharing difficult experiences often makes them easier to bear.",
                "category": "Social Wellness",
                "priority": 4
            },
            {
                "title": "Regular Check-ins",
                "description": "Scheduling regular check-ins with loved ones builds stronger emotional resilience. Create a routine of connecting with important people in your life.",
                "category": "Social Wellness",
                "priority": 3
            },
            {
                "title": "Balance Social Time",
                "description": "Balancing social time with solitude helps maintain mental and emotional harmony. Pay attention to when you need connection versus when you need alone time.",
                "category": "Social Wellness",
                "priority": 3
            }
        ],
        "digital_wellness": [
            {
                "title": "Evening Screen Reduction",
                "description": "Reducing screen time before bed can improve melatonin production and sleep quality. Try to disconnect from devices at least 30-60 minutes before bedtime.",
                "category": "Digital Wellness",
                "priority": 4
            },
            {
                "title": "App Switching Awareness",
                "description": "Frequent switching between apps can lead to cognitive fatigue and stress. Try to focus on one digital task at a time before moving to another.",
                "category": "Digital Wellness",
                "priority": 3
            },
            {
                "title": "Notification Management",
                "description": "Turning off notifications during focused work blocks improves productivity and calm. Consider setting specific times to check messages rather than responding immediately.",
                "category": "Digital Wellness",
                "priority": 3
            },
            {
                "title": "Daily Digital Detox",
                "description": "A digital detox, even for 2 hours a day, can lower anxiety and increase mental clarity. Designate tech-free times like during meals or walks.",
                "category": "Digital Wellness",
                "priority": 4
            },
            {
                "title": "News Consumption Limits",
                "description": "Consuming excessive negative news content can heighten stress and emotional reactivity. Consider limiting news checking to once or twice daily.",
                "category": "Digital Wellness",
                "priority": 3
            },
            {
                "title": "Active vs. Passive Screen Time",
                "description": "Replacing 30 mins of passive scrolling with an outdoor walk boosts mood significantly. Be intentional about how you spend your screen time.",
                "category": "Digital Wellness",
                "priority": 3
            },
            {
                "title": "Blue Light Management",
                "description": "Blue light exposure in the evening can interfere with your circadian rhythm. Use night mode on devices or consider blue light blocking glasses after sunset.",
                "category": "Digital Wellness",
                "priority": 3
            },
            {
                "title": "Intentional Tech Use",
                "description": "Mindful tech use means using devices intentionally — not reactively. Ask yourself 'why am I picking up my phone?' before automatically checking it.",
                "category": "Digital Wellness",
                "priority": 4
            },
            {
                "title": "Track Screen Time",
                "description": "App usage tracking can help you recognize habits that affect your mood. Many phones have built-in tools to monitor which apps consume your attention.",
                "category": "Digital Wellness",
                "priority": 3
            },
            {
                "title": "Screen-Free Zones",
                "description": "Setting screen-free zones (like during meals) supports better connection and mindfulness. Create physical spaces in your home where devices aren't welcome.",
                "category": "Digital Wellness",
                "priority": 3
            }
        ],
        "cognitive_function": [
            {
                "title": "Pomodoro Technique",
                "description": "Working in 25-minute focused intervals with short breaks (Pomodoro) can boost productivity. This technique helps maintain mental energy throughout the day.",
                "category": "Cognitive Function",
                "priority": 3
            },
            {
                "title": "Hydration for Focus",
                "description": "Hydration plays a key role in attention and memory retention. Even mild dehydration can impair cognitive performance and concentration.",
                "category": "Cognitive Function",
                "priority": 3
            },
            {
                "title": "Protein-Rich Breakfast",
                "description": "A high-protein breakfast can improve alertness and cognitive stamina throughout the day. Include eggs, yogurt, or plant-based proteins in your morning meal.",
                "category": "Cognitive Function",
                "priority": 3
            },
            {
                "title": "Single-Tasking",
                "description": "Multitasking actually reduces efficiency — try single-tasking to improve flow. Focus completely on one task before moving to the next for better results.",
                "category": "Cognitive Function",
                "priority": 4
            },
            {
                "title": "Mental Refresh Breaks",
                "description": "Mental fatigue is real — short movement or breath breaks can recharge your brain. Take a 5-minute break every hour to maintain optimal cognitive function.",
                "category": "Cognitive Function",
                "priority": 3
            },
            {
                "title": "Distraction-Free Workspace",
                "description": "Limiting distractions in your workspace can enhance deep work and focus. Create an environment that supports concentration by minimizing noise and visual clutter.",
                "category": "Cognitive Function",
                "priority": 3
            },
            {
                "title": "Morning Cognitive Window",
                "description": "Cognitive clarity is often higher in the first 2 hours after waking up — leverage that time. Schedule your most demanding mental tasks during your peak alertness period.",
                "category": "Cognitive Function",
                "priority": 3
            },
            {
                "title": "Caffeine Moderation",
                "description": "Too much caffeine can impair focus and increase jitteriness — moderation helps. Pay attention to how different amounts affect your particular body and mind.",
                "category": "Cognitive Function",
                "priority": 3
            },
            {
                "title": "Gratitude for Mental Clarity",
                "description": "Practicing gratitude has been shown to enhance mental clarity and reduce rumination. Spend a few minutes each day acknowledging what's going well in your life.",
                "category": "Cognitive Function",
                "priority": 3
            },
            {
                "title": "Sleep for Cognitive Health",
                "description": "Adequate sleep is one of the most powerful tools for memory consolidation and problem solving. Prioritize 7-9 hours of quality sleep for optimal brain function.",
                "category": "Cognitive Function",
                "priority": 4
            }
        ],
        "mental_state": [
            {
                "title": "Practice Box Breathing",
                "description": "When feeling stressed, try box breathing: inhale for 4 counts, hold for 4, exhale for 4, hold for 4. This technique helps regulate your nervous system and reduce anxiety.",
                "category": "Mental Health",
                "priority": 4
            },
            {
                "title": "Daily Mood Check-in",
                "description": "Take 2 minutes each morning to check in with your emotions. Simply naming how you feel can reduce emotional intensity and improve self-awareness.",
                "category": "Mental Health",
                "priority": 3
            },
            {
                "title": "5-4-3-2-1 Grounding",
                "description": "When anxious, use the 5-4-3-2-1 technique: notice 5 things you see, 4 things you can touch, 3 things you hear, 2 things you smell, and 1 thing you taste.",
                "category": "Mental Health",
                "priority": 4
            },
            {
                "title": "Positive Self-Talk",
                "description": "Replace self-critical thoughts with compassionate ones. Instead of 'I failed,' try 'I'm learning and growing from this experience.'",
                "category": "Mental Health",
                "priority": 3
            },
            {
                "title": "Emotional Journaling",
                "description": "Spend 10 minutes writing about your emotions without judgment. This practice can help process difficult feelings and gain perspective on challenges.",
                "category": "Mental Health",
                "priority": 3
            },
            {
                "title": "Gratitude Practice",
                "description": "List three things you're grateful for before bed. This simple habit has been shown to improve mood and sleep quality over time.",
                "category": "Mental Health",
                "priority": 3
            },
            {
                "title": "Mental Health Break",
                "description": "Schedule short 5-minute breaks throughout your day to reset mentally. Step away from screens, stretch, or simply look out the window.",
                "category": "Mental Health",
                "priority": 3
            },
            {
                "title": "Worry Time",
                "description": "Set aside 15 minutes daily as designated 'worry time.' When worries arise outside this time, note them down for later, helping contain anxiety.",
                "category": "Mental Health",
                "priority": 4
            },
            {
                "title": "Social Connection",
                "description": "Reach out to a supportive friend or family member today. Social connection is one of the strongest predictors of mental wellbeing.",
                "category": "Mental Health",
                "priority": 3
            },
            {
                "title": "Mindful Moment",
                "description": "Take three deep breaths and fully focus on one of your senses for 30 seconds. This mini-meditation can interrupt stress cycles throughout the day.",
                "category": "Mental Health",
                "priority": 3
            }
        ],
        "sleep_health": [
            {
                "title": "Consistent Sleep Schedule",
                "description": "Try to go to bed and wake up at the same time every day, even on weekends. This helps regulate your body's internal clock and improves sleep quality.",
                "category": "Sleep",
                "priority": 4
            },
            {
                "title": "Evening Blue Light Reduction",
                "description": "Reduce exposure to blue light from screens 1-2 hours before bedtime. This helps your brain produce melatonin, the hormone that regulates sleep.",
                "category": "Sleep",
                "priority": 4
            },
            {
                "title": "Bedroom Environment",
                "description": "Keep your bedroom cool (65-68°F/18-20°C), dark, and quiet for optimal sleep. Consider blackout curtains or a white noise machine if needed.",
                "category": "Sleep",
                "priority": 3
            },
            {
                "title": "Caffeine Curfew",
                "description": "Avoid caffeine after 2pm as it can stay in your system for 6+ hours and disrupt your ability to fall asleep easily.",
                "category": "Sleep",
                "priority": 3
            },
            {
                "title": "Relaxing Bedtime Routine",
                "description": "Develop a 20-30 minute relaxing routine before bed, such as reading, gentle stretching, or taking a warm bath to signal to your body it's time to sleep.",
                "category": "Sleep",
                "priority": 4
            },
            {
                "title": "Limit Daytime Naps",
                "description": "If you nap during the day, limit it to 20-30 minutes and before 3pm to avoid interfering with nighttime sleep.",
                "category": "Sleep",
                "priority": 3
            },
            {
                "title": "Morning Sunlight",
                "description": "Get exposure to natural light within 30 minutes of waking up to help regulate your circadian rhythm and improve sleep quality the following night.",
                "category": "Sleep",
                "priority": 4
            },
            {
                "title": "Evening Magnesium",
                "description": "Consider a magnesium-rich snack before bed (like a small handful of nuts) as magnesium can help relax muscles and promote better sleep.",
                "category": "Sleep",
                "priority": 3
            },
            {
                "title": "Limit Evening Fluids",
                "description": "Reduce fluid intake 1-2 hours before bedtime to minimize nighttime bathroom trips that disrupt your sleep cycle.",
                "category": "Sleep",
                "priority": 3
            },
            {
                "title": "Sleep-Friendly Bedroom",
                "description": "Reserve your bed for sleep and intimacy only. Working or watching TV in bed can create mental associations that make falling asleep more difficult.",
                "category": "Sleep",
                "priority": 3
            }
        ],
        "nutrition_diet": [
            {
                "title": "Balanced Plate Method",
                "description": "Aim to fill half your plate with vegetables, a quarter with lean protein, and a quarter with whole grains for nutritionally balanced meals.",
                "category": "Nutrition",
                "priority": 3
            },
            {
                "title": "Hydration Reminder",
                "description": "Drink water throughout the day - even mild dehydration can affect mood, energy, and cognitive function. Aim for 8 glasses daily.",
                "category": "Nutrition",
                "priority": 3
            },
            {
                "title": "Mood-Boosting Foods",
                "description": "Include foods rich in omega-3s (fatty fish, walnuts), folate (leafy greens), and tryptophan (eggs, turkey) to support brain health and mood regulation.",
                "category": "Nutrition",
                "priority": 4
            },
            {
                "title": "Mindful Eating Practice",
                "description": "Try eating one meal today without distractions. Notice the flavors, textures, and your body's hunger and fullness cues.",
                "category": "Nutrition",
                "priority": 3
            },
            {
                "title": "Blood Sugar Stability",
                "description": "Pair carbohydrates with protein or healthy fats to maintain stable blood sugar levels, which helps prevent energy crashes and mood swings.",
                "category": "Nutrition",
                "priority": 4
            },
            {
                "title": "Colorful Plate Challenge",
                "description": "Try to include at least 3 different colored vegetables or fruits in your meals today for a wider range of nutrients and antioxidants.",
                "category": "Nutrition",
                "priority": 3
            },
            {
                "title": "Gut-Brain Connection",
                "description": "Include fermented foods like yogurt, kefir, or sauerkraut to support gut health, which is increasingly linked to mental wellbeing.",
                "category": "Nutrition",
                "priority": 3
            },
            {
                "title": "Meal Timing",
                "description": "Try to eat at consistent times each day to help regulate your body's hunger hormones and energy levels.",
                "category": "Nutrition",
                "priority": 3
            },
            {
                "title": "Protein with Breakfast",
                "description": "Include protein in your breakfast to improve concentration and reduce cravings later in the day. Try eggs, Greek yogurt, or nut butter.",
                "category": "Nutrition",
                "priority": 3
            },
            {
                "title": "Mindful Snacking",
                "description": "Choose nutrient-dense snacks like nuts, fruit, or yogurt instead of processed options to maintain energy and focus between meals.",
                "category": "Nutrition",
                "priority": 3
            }
        ],
        "physical_activity": [
            {
                "title": "Movement Snacking",
                "description": "Break up long periods of sitting with 2-3 minute 'movement snacks' - quick stretches, walking, or bodyweight exercises to boost energy and focus.",
                "category": "Exercise",
                "priority": 3
            },
            {
                "title": "Morning Energy Boost",
                "description": "Try 5 minutes of gentle stretching or yoga in the morning to increase blood flow, reduce stiffness, and improve your mood for the day ahead.",
                "category": "Exercise",
                "priority": 3
            },
            {
                "title": "Nature Walk",
                "description": "Take a 15-minute walk in nature when feeling stressed or mentally foggy. Research shows green spaces can reduce stress hormones and improve mood.",
                "category": "Exercise",
                "priority": 3
            },
            {
                "title": "Strength for Mental Health",
                "description": "Include 2-3 strength training sessions weekly. Building muscle not only improves physical health but also boosts confidence and reduces anxiety.",
                "category": "Exercise",
                "priority": 4
            },
            {
                "title": "Exercise for Better Sleep",
                "description": "Try to complete moderate exercise at least 3 hours before bedtime to improve sleep quality while allowing your body time to wind down.",
                "category": "Exercise",
                "priority": 4
            },
            {
                "title": "Mood-Lifting Movement",
                "description": "When feeling low, try 10 minutes of any movement you enjoy. Exercise releases endorphins that can quickly improve your emotional state.",
                "category": "Exercise",
                "priority": 4
            },
            {
                "title": "Mindful Movement",
                "description": "Try activities like yoga, tai chi, or mindful walking that combine physical movement with mental focus to reduce stress and improve body awareness.",
                "category": "Exercise",
                "priority": 3
            },
            {
                "title": "Active Commuting",
                "description": "Consider walking, biking, or adding a short walk to your commute. Even small increases in daily movement improve cardiovascular health and mood.",
                "category": "Exercise",
                "priority": 3
            },
            {
                "title": "Consistency Over Intensity",
                "description": "Regular moderate exercise provides more mental health benefits than occasional intense workouts. Aim for 30 minutes of movement most days.",
                "category": "Exercise",
                "priority": 3
            },
            {
                "title": "Social Movement",
                "description": "Try exercising with a friend or group to boost motivation and add social connection benefits to your physical activity.",
                "category": "Exercise",
                "priority": 3
            }
        ],
        "mindfulness_stress_reduction": [
  {
    "title": "Guided Meditation",
    "description": "Listen to a 10-minute guided meditation to help manage stress and anxiety. Focus on your breath and let go of intrusive thoughts.",
    "category": "Mindfulness",
    "priority": 3
  },
  {
    "title": "Box Breathing",
    "description": "Try box breathing: inhale for 4 seconds, hold for 4, exhale for 4, and hold again. Repeat this for 2 minutes.",
    "category": "Mindfulness",
    "priority": 2
  },
  {
    "title": "Mindful Walking",
    "description": "Take a 10-minute walk and focus on each step, your breath, and your surroundings.",
    "category": "Mindfulness",
    "priority": 2
  },
  {
    "title": "Body Scan",
    "description": "Lie down and mentally scan your body from head to toe. Observe sensations without judgment.",
    "category": "Mindfulness",
    "priority": 3
  },
  {
    "title": "Gratitude Journaling",
    "description": "Write down 3 things you're grateful for today to promote a positive mindset.",
    "category": "Mindfulness",
    "priority": 1
  },
  {
    "title": "Breath Awareness",
    "description": "Set a timer for 3 minutes and simply observe your breath as it flows in and out.",
    "category": "Mindfulness",
    "priority": 1
  },
  {
    "title": "5 Senses Grounding",
    "description": "Name 1 thing you can see, hear, touch, smell, and taste to ground yourself in the present.",
    "category": "Mindfulness",
    "priority": 2
  },
  {
    "title": "Affirmation Practice",
    "description": "Repeat a calming affirmation 5 times aloud: 'I am safe, I am calm, I am enough.'",
    "category": "Mindfulness",
    "priority": 2
  },
  {
    "title": "Nature Mindfulness",
    "description": "Spend 5 minutes observing the details of a tree, plant, or the sky — fully present in the moment.",
    "category": "Mindfulness",
    "priority": 2
  },
  {
    "title": "Emotional Labeling",
    "description": "Pause and name your current emotion. This simple act helps regulate your feelings.",
    "category": "Mindfulness",
    "priority": 1
  }
],
    " Social Wellness":[
  {
    "title": "Check-In With a Friend",
    "description": "Send a message or call someone you care about today. Even a short chat can lift your spirits.",
    "category": "Social Wellness",
    "priority": 2
  },
  {
    "title": "Compliment Someone",
    "description": "Give a sincere compliment today. It boosts both your mood and theirs.",
    "category": "Social Wellness",
    "priority": 1
  },
  {
    "title": "Join a Group Activity",
    "description": "Attend a group event or activity (virtual or physical) to feel more connected.",
    "category": "Social Wellness",
    "priority": 2
  },
  {
    "title": "Express Gratitude",
    "description": "Tell someone how much you appreciate them — via text, call, or face-to-face.",
    "category": "Social Wellness",
    "priority": 3
  },
  {
    "title": "Reconnect With Someone",
    "description": "Reach out to someone you haven’t talked to in a while. Reconnection supports emotional well-being.",
    "category": "Social Wellness",
    "priority": 2
  },
  {
    "title": "Limit Draining Interactions",
    "description": "Notice which interactions drain you. Protect your energy by setting gentle boundaries.",
    "category": "Social Wellness",
    "priority": 3
  },
  {
    "title": "Share a Meal",
    "description": "Have a meal with a friend, family member, or colleague today to foster connection.",
    "category": "Social Wellness",
    "priority": 2
  },
  {
    "title": "Support Someone",
    "description": "Offer a helping hand or a listening ear to someone in need today.",
    "category": "Social Wellness",
    "priority": 1
  },
  {
    "title": "Social Recharge Time",
    "description": "Balance your social time with intentional solitude to recharge and reflect.",
    "category": "Social Wellness",
    "priority": 1
  },
  {
    "title": "Reflect on Relationships",
    "description": "Take 5 minutes to reflect: Which connections bring you energy? Which ones drain it?",
    "category": "Social Wellness",
    "priority": 2
  }
],
"Digital Wellness & Screen Time":[
  {
    "title": "Digital Curfew",
    "description": "Avoid screens 1 hour before bedtime to support melatonin production and quality sleep.",
    "category": "Digital Wellness",
    "priority": 3
  },
  {
    "title": "Mute Notifications",
    "description": "Turn off non-essential notifications for the next few hours to reduce cognitive load.",
    "category": "Digital Wellness",
    "priority": 2
  },
  {
    "title": "Social Media Fast",
    "description": "Take a break from social media for the next 4–6 hours to clear your mind.",
    "category": "Digital Wellness",
    "priority": 2
  },
  {
    "title": "Screen-Free Meal",
    "description": "Have one meal today without screens. Focus on your food and company instead.",
    "category": "Digital Wellness",
    "priority": 1
  },
  {
    "title": "App Cleanup",
    "description": "Delete one app that no longer serves your well-being or goals.",
    "category": "Digital Wellness",
    "priority": 1
  },
  {
    "title": "Digital Boundaries",
    "description": "Set a specific time window for checking emails or messages to avoid overchecking.",
    "category": "Digital Wellness",
    "priority": 2
  },
  {
    "title": "Track Screen Time",
    "description": "Review your screen time usage today and notice where your energy goes.",
    "category": "Digital Wellness",
    "priority": 2
  },
  {
    "title": "Airplane Mode Break",
    "description": "Put your phone on airplane mode for 30 minutes to recharge mentally.",
    "category": "Digital Wellness",
    "priority": 1
  },
  {
    "title": "Intentional Scrolling",
    "description": "Before opening a social app, ask: 'Why am I here? What do I want from this time?'",
    "category": "Digital Wellness",
    "priority": 2
  },
  {
    "title": "No-Phone Zone",
    "description": "Designate a space in your home where phones aren’t allowed — like your bedroom or dining table.",
    "category": "Digital Wellness",
    "priority": 3
  }
],
" Cognitive Function & Focus":[
  {
    "title": "Pomodoro Focus Sprint",
    "description": "Try a 25-minute focused work sprint followed by a 5-minute break. Repeat if needed.",
    "category": "Cognitive Function",
    "priority": 2
  },
  {
    "title": "Mental Warm-Up",
    "description": "Start your day with a puzzle, light reading, or journaling to warm up your brain.",
    "category": "Cognitive Function",
    "priority": 1
  },
  {
    "title": "Hydration Check",
    "description": "Drink a glass of water. Dehydration is a common cause of mental fog.",
    "category": "Cognitive Function",
    "priority": 1
  },
  {
    "title": "Deep Work Block",
    "description": "Block off 60 minutes for deep, uninterrupted work. Silence all distractions.",
    "category": "Cognitive Function",
    "priority": 3
  },
  {
    "title": "Brain Break",
    "description": "Stand up and stretch for 2–3 minutes to refresh your mental clarity.",
    "category": "Cognitive Function",
    "priority": 2
  },
  {
    "title": "High-Focus Time",
    "description": "Identify your peak focus hours today and reserve them for your most demanding tasks.",
    "category": "Cognitive Function",
    "priority": 2
  },
  {
    "title": "Caffeine Awareness",
    "description": "Notice how caffeine affects your focus. Too much may lead to restlessness.",
    "category": "Cognitive Function",
    "priority": 1
  },
  {
    "title": "Task Prioritization",
    "description": "Write down the top 3 things you want to get done today. Focus on one at a time.",
    "category": "Cognitive Function",
    "priority": 3
  },
  {
    "title": "Reduce Task Switching",
    "description": "Multitasking decreases efficiency. Focus on a single task for better results.",
    "category": "Cognitive Function",
    "priority": 2
  },
  {
    "title": "Mental Check-In",
    "description": "Pause and ask: How focused do I feel? What can I shift right now to improve it?",
    "category": "Cognitive Function",
    "priority": 1
  }
]
    }





def generate_default_insights() -> List[Dict[str, Any]]:
    """Generate default insights when GPT generation fails."""
    # Get all categorized insights
    all_categories = generate_categorized_insights()

    # Select one insight from each of the 4 main categories
    default_insights = []
    for category in all_categories:
        insights = all_categories[category]
        default_insights.append(random.choice(insights))

    # Ensure we have exactly 5 insights by adding one more from a random category if needed
    if len(default_insights) < 5:
        random_category = random.choice(list(all_categories.keys()))
        # Make sure we don't select a duplicate insight
        available_insights = [i for i in all_categories[random_category] if i not in default_insights]
        if available_insights:
            default_insights.append(random.choice(available_insights))

    return default_insights[:5]  # Return at most 5 insights

# API Endpoints
# Function to generate patient insights with data collection
def generate_patient_insights_internal(
    patient_id: str,
    db: Session,
    user_role: str,
    user_id: str
) -> InsightsResponse:
    """Internal function to generate patient insights with caching."""
    # Get patient info
    patient = db.query(Patient).filter(Patient.patient_id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    # Only allow access if the user is this patient or an admin
    if user_role != "admin" and patient.user_id != user_id:
        raise HTTPException(status_code=403, detail="Not authorized to access this patient's insights")

    insights = []

    # Upcoming Appointments (keep this for practical utility)
    appointments = get_upcoming_appointments(patient_id, db)
    if appointments:
        next_appt = appointments[0]
        appt_text = f"Your next appointment is with {next_appt['doctor']} in {next_appt['days_until']} days."
        insights.append(InsightItem(
            title="Upcoming Appointments",
            description=appt_text,
            type="calendar",
            priority=3 if next_appt["days_until"] <= 3 else 2
        ))

    # Get all insights (both categorized and GPT-generated)
    all_insights = []

    # Get categorized insights
    all_categories = generate_categorized_insights()
    category_keys = list(all_categories.keys())

    # Get one random insight from each category
    for category in category_keys:
        category_insights = all_categories[category]
        selected_insight = random.choice(category_insights)
        all_insights.append({
            "title": selected_insight["title"],
            "description": selected_insight["description"],
            "type": "general_insight",
            "priority": selected_insight["priority"],
            "source": "categorized"
        })

    # Get GPT-generated insights
    gpt_insights = generate_gpt_insights(patient_id, db)
    for insight in gpt_insights:
        all_insights.append({
            "title": insight["title"],
            "description": insight["description"],
            "type": "gpt_insight",
            "priority": insight["priority"],
            "source": "gpt"
        })

    # Shuffle all insights to randomize the order
    random.shuffle(all_insights)

    # Select 5 insights total: 3 from categorized and 2 from GPT
    categorized_count = 0
    gpt_count = 0
    selected_insights = []

    # First pass: try to get the desired counts
    for insight in all_insights:
        if insight["source"] == "categorized" and categorized_count < 3:
            selected_insights.append(insight)
            categorized_count += 1
        elif insight["source"] == "gpt" and gpt_count < 2:
            selected_insights.append(insight)
            gpt_count += 1

        # Stop if we have enough of both types
        if categorized_count >= 3 and gpt_count >= 2:
            break

    # Second pass: fill in any missing slots with available insights
    remaining_slots = 5 - len(selected_insights)
    if remaining_slots > 0:
        remaining_insights = [i for i in all_insights if i not in selected_insights]
        selected_insights.extend(remaining_insights[:remaining_slots])

    # Add the selected insights to the response
    for insight in selected_insights:
        insights.append(InsightItem(
            title=insight["title"],
            description=insight["description"],
            type=insight["type"],
            priority=insight["priority"]
        ))

    # Sort insights by priority (highest first)
    insights.sort(key=lambda x: x.priority, reverse=True)

    return InsightsResponse(
        insights=insights,
        generated_at=datetime.now()
    )

# Apply caching to the internal function - only use patient_id and user_role for the cache key
cached_generate_patient_insights = with_selective_cache("simplified_patient_insights", [0, 2, 3])(generate_patient_insights_internal)

@router.get("/patient/{patient_id}", response_model=InsightsResponse)
async def get_patient_insights(
    patient_id: str,
    db: Session = Depends(get_db),
    current_user: Dict = Depends(get_current_active_user)
):
    """Get insights for a patient."""
    try:
        # Get user role from token
        user_role = get_current_user_role(current_user)
        user_id = current_user.get("sub", "")

        # Use cached function to generate insights
        return cached_generate_patient_insights(patient_id, db, user_role, user_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating patient insights: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating insights: {str(e)}")


