from fastapi import APIRouter, Depends, HTTPException, status
from apscheduler.schedulers.background import BackgroundScheduler
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_
from datetime import datetime, timezone, timedelta
import uuid
import requests
import logging
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from database.database import get_db, SessionLocal
from model.model_correct import ChatMessage, EmotionAnalysis, Patient, UserEmotionInsights
import os
from dotenv import load_dotenv
from typing import List, Optional, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

HF_API_URL = os.getenv("HF_API_URL")
HF_API_KEY = os.getenv("HF_API_KEY")
emotion_router = APIRouter()
analyzer = SentimentIntensityAnalyzer()
HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}

def get_emotion(text):
    """
    Get emotion from text using Hugging Face API

    Args:
        text (str): The text to analyze

    Returns:
        tuple: (emotion, confidence) or fallback to VADER-based emotion if API call fails
    """
    try:
        # Skip empty or very short texts
        if not text or len(text) < 5:
            logger.warning(f"Text too short for emotion analysis: {text}")
            # Use VADER for very short texts instead of defaulting to neutral
            return get_vader_emotion(text)

        # Add timeout to prevent hanging
        response = requests.post(HF_API_URL, headers=HEADERS, json={"inputs": text}, timeout=10)

        # Check if response is successful
        if response.status_code != 200:
            logger.error(f"API error: {response.status_code} - {response.text}")
            # Fallback to VADER-based emotion
            return get_vader_emotion(text)

        result = response.json()

        # Validate response format
        if not result or not isinstance(result, list) or len(result) == 0 or len(result[0]) == 0:
            logger.error(f"Invalid API response format: {result}")
            # Fallback to VADER-based emotion
            return get_vader_emotion(text)

        return result[0][0]["label"], round(result[0][0]["score"], 2)

    except requests.RequestException as e:
        logger.error(f"Request error in get_emotion: {str(e)}")
        return get_vader_emotion(text)
    except ValueError as e:
        logger.error(f"JSON parsing error in get_emotion: {str(e)}")
        return get_vader_emotion(text)
    except Exception as e:
        logger.error(f"Unexpected error in get_emotion: {str(e)}")
        return get_vader_emotion(text)

def get_vader_emotion(text):
    """
    Get emotion from text using VADER sentiment analysis

    Args:
        text (str): The text to analyze

    Returns:
        tuple: (emotion, confidence)
    """
    try:
        # Get sentiment scores
        sentiment_scores = analyzer.polarity_scores(text)
        compound_score = sentiment_scores["compound"]

        # Map sentiment score to emotion with narrower neutral range
        if compound_score >= 0.5:
            emotion = "joy"
        elif compound_score >= 0.2:
            emotion = "optimism"
        elif compound_score >= 0.05:
            emotion = "surprise"
        elif compound_score > -0.05:
            emotion = "neutral"  # Narrower range for neutral
        elif compound_score >= -0.2:
            emotion = "worry"
        elif compound_score >= -0.5:
            emotion = "sadness"
        else:
            emotion = "anger"

        # Return emotion and confidence
        return emotion, round(compound_score, 2)
    except Exception as e:
        logger.error(f"Error in get_vader_emotion: {str(e)}")
        # Only use neutral as a last resort
        return "neutral", 0.0

def analyze_emotions(days_back=7, skip_api_calls=False):
    """
    Analyze emotions in chat messages with parallelization for improved performance

    Args:
        days_back (int): Number of days to look back for unanalyzed messages
        skip_api_calls (bool): Skip Hugging Face API calls and use VADER sentiment only

    Returns:
        dict: Summary of analysis results
    """
    import concurrent.futures
    import threading
    from functools import partial

    logger.info("Starting parallel emotion analysis...")
    db = None

    try:
        # Create a new database session
        db = SessionLocal()

        # Check if required tables exist
        try:
            # Test query to check if tables exist
            db.query(Patient).limit(1).all()
            db.query(ChatMessage).limit(1).all()
            db.query(EmotionAnalysis).limit(1).all()
        except Exception as table_error:
            logger.error(f"Database tables not found: {str(table_error)}")
            return {
                "error": "Database tables not found",
                "details": str(table_error),
                "status": "failed"
            }

        # Get the cutoff date for analysis
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_back)
        logger.info(f"Analyzing messages from the last {days_back} days (after {cutoff_date})")

        # Get all patient IDs (no limit)
        try:
            patient_ids = db.query(Patient.patient_id).distinct().all()
            logger.info(f"Found {len(patient_ids)} patients to analyze")
        except Exception as patient_error:
            logger.error(f"Error getting patient IDs: {str(patient_error)}")
            return {
                "error": "Error getting patient IDs",
                "details": str(patient_error),
                "status": "failed"
            }

        # Thread-local storage for database sessions
        thread_local = threading.local()

        # Function to get a thread-local database session
        def get_thread_db():
            if not hasattr(thread_local, "db"):
                thread_local.db = SessionLocal()
            return thread_local.db

        # Process a single message
        def process_message(msg, skip_api_calls):
            thread_db = get_thread_db()
            try:
                # Skip messages without text
                if not msg.message_text:
                    logger.warning(f"Skipping message {msg.chat_message_id} - no text content")
                    return {"status": "skipped", "reason": "no_text"}

                # Get emotion (skip API call if requested)
                if skip_api_calls:
                    # Use VADER sentiment only
                    sentiment_scores = analyzer.polarity_scores(msg.message_text)
                    compound_score = sentiment_scores["compound"]

                    # Map sentiment score to emotion with narrower neutral range
                    if compound_score >= 0.5:
                        emotion = "joy"
                    elif compound_score >= 0.2:
                        emotion = "optimism"
                    elif compound_score >= 0.05:
                        emotion = "surprise"
                    elif compound_score > -0.05:
                        emotion = "neutral"  # Narrower range for neutral
                    elif compound_score >= -0.2:
                        emotion = "worry"
                    elif compound_score >= -0.5:
                        emotion = "sadness"
                    else:
                        emotion = "anger"

                    # Use compound score as sentiment
                    sentiment = round(compound_score, 2)
                    logger.debug(f"Using VADER sentiment: {emotion} with score: {sentiment}")
                else:
                    # Use Hugging Face API
                    emotion, confidence = get_emotion(msg.message_text)

                    # Log the emotion and confidence
                    logger.debug(f"Detected emotion: {emotion} with confidence: {confidence}")

                    # Get sentiment score (using VADER for consistency)
                    sentiment = round(analyzer.polarity_scores(msg.message_text)["compound"], 2)

                # Create emotion analysis record
                emotion_analysis = EmotionAnalysis(
                    emotion_id=str(uuid.uuid4()),
                    chat_message_id=msg.chat_message_id,
                    patient_id=msg.sender_id,
                    emotion_category=emotion,
                    confidence_score=sentiment,
                    analyzed_at=datetime.now(timezone.utc)
                )

                thread_db.add(emotion_analysis)
                thread_db.commit()

                return {"status": "success", "emotion": emotion, "sentiment": sentiment}

            except Exception as e:
                thread_db.rollback()
                logger.error(f"Error processing message {msg.chat_message_id}: {str(e)}")
                return {"status": "failed", "error": str(e)}

        # Process a single patient
        def process_patient(patient_id_tuple):
            thread_db = get_thread_db()
            patient_results = {
                "total_messages": 0,
                "analyzed_messages": 0,
                "skipped_messages": 0,
                "failed_messages": 0
            }

            try:
                patient_id = patient_id_tuple[0]
                patient_id_str = str(patient_id)

                # Find messages that haven't been analyzed yet
                analyzed_message_ids = thread_db.query(EmotionAnalysis.chat_message_id).filter(
                    EmotionAnalysis.patient_id == patient_id_str
                ).subquery()

                # Get recent messages that haven't been analyzed yet (no limit per patient)
                messages = thread_db.query(ChatMessage).filter(
                    ChatMessage.sender_id == patient_id_str,
                    ChatMessage.timestamp >= cutoff_date,
                    ~ChatMessage.chat_message_id.in_(analyzed_message_ids)
                ).all()

                patient_results["total_messages"] = len(messages)
                logger.info(f"Found {len(messages)} unanalyzed messages for patient {patient_id_str}")

                # Process messages in parallel using ThreadPoolExecutor
                if messages:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                        process_func = partial(process_message, skip_api_calls=skip_api_calls)
                        results = list(executor.map(process_func, messages))

                    # Count results
                    for result in results:
                        if result["status"] == "success":
                            patient_results["analyzed_messages"] += 1
                        elif result["status"] == "skipped":
                            patient_results["skipped_messages"] += 1
                        else:
                            patient_results["failed_messages"] += 1

                return patient_results

            except Exception as e:
                logger.error(f"Error processing patient {patient_id_tuple[0]}: {str(e)}")
                return {
                    "total_messages": 0,
                    "analyzed_messages": 0,
                    "skipped_messages": 0,
                    "failed_messages": 1,
                    "error": str(e)
                }

        # Process all patients in parallel using ThreadPoolExecutor
        total_results = {
            "total_messages": 0,
            "analyzed_messages": 0,
            "skipped_messages": 0,
            "failed_messages": 0,
            "total_patients_processed": len(patient_ids),
            "patients_with_errors": 0
        }

        # Use ThreadPoolExecutor for patients
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(32, len(patient_ids))) as executor:
            patient_results = list(executor.map(process_patient, patient_ids))

        # Aggregate results
        for result in patient_results:
            total_results["total_messages"] += result.get("total_messages", 0)
            total_results["analyzed_messages"] += result.get("analyzed_messages", 0)
            total_results["skipped_messages"] += result.get("skipped_messages", 0)
            total_results["failed_messages"] += result.get("failed_messages", 0)
            if "error" in result:
                total_results["patients_with_errors"] += 1

        # Add status
        total_results["status"] = "success"
        total_results["api_calls_skipped"] = skip_api_calls

        logger.info(f"Emotion analysis completed: {total_results}")
        return total_results

    except Exception as e:
        logger.error(f"Error in analyze_emotions: {str(e)}")
        if db:
            db.rollback()
        return {
            "error": str(e),
            "type": type(e).__name__,
            "status": "failed"
        }

    finally:
        if db:
            db.close()

# ✅ Run Analysis Daily
scheduler = BackgroundScheduler(timezone="Asia/Kolkata")
# Schedule daily emotion analysis at midnight with no limits
scheduler.add_job(analyze_emotions, "cron", hour=0, kwargs={"days_back": 7, "skip_api_calls": False})
scheduler.start()

@emotion_router.get("/run-analysis")
def run_analysis(days_back: int = 7, skip_api_calls: bool = False):
    """
    Run emotion analysis on chat messages with parallelization for improved performance

    Args:
        days_back (int): Number of days to look back for unanalyzed messages
        skip_api_calls (bool): Skip Hugging Face API calls and use VADER sentiment only

    Returns:
        dict: Summary of analysis results
    """
    import concurrent.futures
    import threading
    from functools import partial

    try:
        # Validate days_back parameter
        if days_back < 1 or days_back > 365:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="days_back must be between 1 and 365"
            )

        # Check if Hugging Face API is configured
        if not HF_API_URL or not HF_API_KEY:
            logger.warning("Hugging Face API not configured. Using fallback emotion detection.")
            skip_api_calls = True

        # Log the start of analysis
        logger.info(f"Starting parallel emotion analysis for the last {days_back} days (skip_api_calls: {skip_api_calls})")

        # Create a database session
        db = SessionLocal()

        try:
            # Check if required tables exist
            try:
                # Test query to check if tables exist
                db.query(Patient).limit(1).all()
                db.query(ChatMessage).limit(1).all()
                db.query(EmotionAnalysis).limit(1).all()
            except Exception as table_error:
                logger.error(f"Database tables not found: {str(table_error)}")
                return {
                    "error": "Database tables not found",
                    "details": str(table_error),
                    "status": "failed"
                }

            # Get the cutoff date for analysis
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_back)

            # Get all patient IDs (no limit)
            patient_ids = db.query(Patient.patient_id).distinct().all()
            logger.info(f"Found {len(patient_ids)} patients to analyze")

            # Thread-local storage for database sessions
            thread_local = threading.local()

            # Function to get a thread-local database session
            def get_thread_db():
                if not hasattr(thread_local, "db"):
                    thread_local.db = SessionLocal()
                return thread_local.db

            # Process a single message
            def process_message(msg, skip_api_calls):
                thread_db = get_thread_db()
                try:
                    # Skip messages without text
                    if not msg.message_text:
                        logger.warning(f"Skipping message {msg.chat_message_id} - no text content")
                        return {"status": "skipped", "reason": "no_text"}

                    # Get emotion (skip API call if requested)
                    if skip_api_calls:
                        # Use VADER sentiment only
                        sentiment_scores = analyzer.polarity_scores(msg.message_text)
                        compound_score = sentiment_scores["compound"]

                        # Map sentiment score to emotion with narrower neutral range
                        if compound_score >= 0.5:
                            emotion = "joy"
                        elif compound_score >= 0.2:
                            emotion = "optimism"
                        elif compound_score >= 0.05:
                            emotion = "surprise"
                        elif compound_score > -0.05:
                            emotion = "neutral"  # Narrower range for neutral
                        elif compound_score >= -0.2:
                            emotion = "worry"
                        elif compound_score >= -0.5:
                            emotion = "sadness"
                        else:
                            emotion = "anger"

                        # Use absolute value of compound score as confidence
                        sentiment = round(compound_score, 2)
                    else:
                        # Use Hugging Face API
                        emotion, _ = get_emotion(msg.message_text)

                        # Still use VADER for sentiment score for consistency
                        sentiment = round(analyzer.polarity_scores(msg.message_text)["compound"], 2)

                    # Create emotion analysis record
                    emotion_analysis = EmotionAnalysis(
                        emotion_id=str(uuid.uuid4()),
                        chat_message_id=msg.chat_message_id,
                        patient_id=msg.sender_id,
                        emotion_category=emotion,
                        confidence_score=sentiment,
                        analyzed_at=datetime.now(timezone.utc)
                    )

                    thread_db.add(emotion_analysis)
                    thread_db.commit()

                    return {"status": "success", "emotion": emotion, "sentiment": sentiment}

                except Exception as e:
                    thread_db.rollback()
                    logger.error(f"Error processing message {msg.chat_message_id}: {str(e)}")
                    return {"status": "failed", "error": str(e)}

            # Process a single patient
            def process_patient(patient_id_tuple, cutoff_date, skip_api_calls):
                thread_db = get_thread_db()
                patient_results = {
                    "total_messages": 0,
                    "analyzed_messages": 0,
                    "skipped_messages": 0,
                    "failed_messages": 0
                }

                try:
                    patient_id = patient_id_tuple[0]
                    patient_id_str = str(patient_id)

                    # Find messages that haven't been analyzed yet
                    analyzed_message_ids = thread_db.query(EmotionAnalysis.chat_message_id).filter(
                        EmotionAnalysis.patient_id == patient_id_str
                    ).subquery()

                    # Get recent messages that haven't been analyzed yet (no limit per patient)
                    messages = thread_db.query(ChatMessage).filter(
                        ChatMessage.sender_id == patient_id_str,
                        ChatMessage.timestamp >= cutoff_date,
                        ~ChatMessage.chat_message_id.in_(analyzed_message_ids)
                    ).all()

                    patient_results["total_messages"] = len(messages)
                    logger.info(f"Found {len(messages)} unanalyzed messages for patient {patient_id_str}")

                    # Process messages in parallel using ThreadPoolExecutor
                    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                        process_func = partial(process_message, skip_api_calls=skip_api_calls)
                        results = list(executor.map(process_func, messages))

                    # Count results
                    for result in results:
                        if result["status"] == "success":
                            patient_results["analyzed_messages"] += 1
                        elif result["status"] == "skipped":
                            patient_results["skipped_messages"] += 1
                        else:
                            patient_results["failed_messages"] += 1

                    return patient_results

                except Exception as e:
                    logger.error(f"Error processing patient {patient_id_tuple[0]}: {str(e)}")
                    return {
                        "total_messages": 0,
                        "analyzed_messages": 0,
                        "skipped_messages": 0,
                        "failed_messages": 1,
                        "error": str(e)
                    }
                finally:
                    # Don't close the thread_db here, as it's reused across patients
                    pass

            # Process all patients in parallel using ProcessPoolExecutor
            total_results = {
                "total_messages": 0,
                "analyzed_messages": 0,
                "skipped_messages": 0,
                "failed_messages": 0,
                "total_patients_processed": len(patient_ids),
                "patients_with_errors": 0
            }

            # Use ThreadPoolExecutor for patients (ProcessPoolExecutor can have issues with SQLAlchemy)
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(32, len(patient_ids))) as executor:
                process_patient_func = partial(process_patient, cutoff_date=cutoff_date, skip_api_calls=skip_api_calls)
                patient_results = list(executor.map(process_patient_func, patient_ids))

            # Aggregate results
            for result in patient_results:
                total_results["total_messages"] += result.get("total_messages", 0)
                total_results["analyzed_messages"] += result.get("analyzed_messages", 0)
                total_results["skipped_messages"] += result.get("skipped_messages", 0)
                total_results["failed_messages"] += result.get("failed_messages", 0)
                if "error" in result:
                    total_results["patients_with_errors"] += 1

            # Add status
            total_results["status"] = "success"

            logger.info(f"Emotion analysis completed: {total_results}")

            # Return results
            return {
                "message": "Emotion analysis completed successfully",
                "days_analyzed": days_back,
                "api_calls_skipped": skip_api_calls,
                "results": total_results
            }

        except Exception as e:
            logger.error(f"Error in run_analysis: {str(e)}")
            db.rollback()
            raise

        finally:
            db.close()

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error in run_analysis: {str(e)}")
        import traceback
        traceback.print_exc()

        # Return a more informative error message
        error_detail = {
            "error": str(e),
            "type": type(e).__name__,
            "message": "An error occurred while running emotion analysis. Check server logs for details."
        }

        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error_detail)


@emotion_router.get("/patient/{patient_id}")
def get_patient_emotions(patient_id: str, days_back: int = 30, db: Session = Depends(get_db)):
    """
    Get emotion analysis for a specific patient

    Args:
        patient_id (str): The patient ID
        days_back (int): Number of days to look back
        db (Session): Database session

    Returns:
        dict: Patient emotion analysis
    """
    try:
        # Validate days_back parameter
        if days_back < 1 or days_back > 365:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="days_back must be between 1 and 365"
            )

        # Validate patient_id
        patient = db.query(Patient).filter(Patient.patient_id == patient_id).first()
        if not patient:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Patient with ID {patient_id} not found"
            )

        # Get the cutoff date
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_back)

        # Get emotion analysis for this patient
        emotions = db.query(EmotionAnalysis).filter(
            EmotionAnalysis.patient_id == str(patient_id),
            EmotionAnalysis.analyzed_at >= cutoff_date
        ).order_by(EmotionAnalysis.analyzed_at.desc()).all()

        # Count emotions by category
        emotion_counts = {}
        for emotion in emotions:
            category = emotion.emotion_category
            emotion_counts[category] = emotion_counts.get(category, 0) + 1

        # Log the emotion distribution for debugging
        logger.info(f"Emotion distribution for patient {patient_id}: {emotion_counts}")

        # Get the dominant emotion
        if emotion_counts:
            # Sort emotions by count (highest first)
            sorted_emotions = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)

            # If the dominant emotion is neutral and we have other emotions, use the second highest
            if sorted_emotions[0][0] == "neutral" and len(sorted_emotions) > 1:
                dominant_emotion = sorted_emotions[1][0]
                logger.info(f"Dominant emotion was neutral, using second highest: {dominant_emotion}")
            else:
                dominant_emotion = sorted_emotions[0][0]
        else:
            dominant_emotion = "unknown"

        # Calculate average sentiment
        total_sentiment = sum(emotion.confidence_score for emotion in emotions)
        avg_sentiment = round(total_sentiment / len(emotions), 2) if emotions else 0

        # Get recent messages with emotions
        recent_messages = []
        for emotion in emotions[:10]:  # Get the 10 most recent
            message = db.query(ChatMessage).filter(ChatMessage.chat_message_id == emotion.chat_message_id).first()
            if message:
                recent_messages.append({
                    "message_id": message.chat_message_id,
                    "text": message.message_text[:100] + "..." if len(message.message_text) > 100 else message.message_text,
                    "timestamp": message.timestamp.isoformat() if message.timestamp else None,
                    "emotion": emotion.emotion_category,
                    "sentiment": emotion.confidence_score
                })

        # Return the analysis
        return {
            "patient_id": patient_id,
            "patient_name": f"{patient.first_name} {patient.last_name}",
            "days_analyzed": days_back,
            "total_messages_analyzed": len(emotions),
            "dominant_emotion": dominant_emotion,
            "average_sentiment": avg_sentiment,
            "emotion_distribution": emotion_counts,
            "recent_messages": recent_messages
        }

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error in get_patient_emotions: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
