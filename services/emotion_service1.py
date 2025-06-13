import concurrent.futures
import time
import uuid
import requests
import logging
import os
import traceback
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Any, Tuple
from collections import defaultdict

from fastapi import APIRouter, Depends, HTTPException, status, Query
from apscheduler.schedulers.background import BackgroundScheduler
from sqlalchemy.orm import Session
from sqlalchemy import select, text  # Using select and text from sqlalchemy
from dotenv import load_dotenv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Assuming these are correctly defined elsewhere
from database.database import get_db, SessionLocal
from model.model_correct import ChatMessage, EmotionAnalysis, Patient  # Only using these models

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()
HF_API_URL = os.getenv("HF_API_URL")
HF_API_KEY = os.getenv("HF_API_KEY")
# How many messages to process before committing to DB / logging progress
DEFAULT_BATCH_SIZE = 200
# How many concurrent API calls to make
MAX_WORKERS = 10 # Adjust based on API rate limits and server resources

# --- Globals ---
emotion_router = APIRouter()
analyzer = SentimentIntensityAnalyzer()
HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"} if HF_API_KEY else {}
scheduler = BackgroundScheduler(timezone="Asia/Kolkata") # Or your preferred timezone


# --- Emotion Detection Logic ---

def get_vader_emotion(text: Optional[str]) -> Tuple[str, float]:
    """ VADER fallback: returns emotion category and compound score. """
    if not text:
        return "neutral", 0.0
    try:
        sentiment_scores = analyzer.polarity_scores(text)
        compound_score = sentiment_scores["compound"]

        # Map sentiment score to emotion
        if compound_score >= 0.5: emotion = "joy"
        elif compound_score >= 0.2: emotion = "optimism"
        elif compound_score >= 0.05: emotion = "surprise" # VADER often flags neutral slightly positive
        elif compound_score > -0.05: emotion = "neutral"
        elif compound_score >= -0.2: emotion = "worry"
        elif compound_score >= -0.5: emotion = "sadness"
        else: emotion = "anger" # VADER often flags neutral slightly negative

        return emotion, round(compound_score, 2)
    except Exception as e:
        logger.error(f"Error in get_vader_emotion for text '{text[:50]}...': {str(e)}")
        return "neutral", 0.0

def get_emotion_hf(text: str) -> Tuple[str, float]:
    """ Get emotion from Hugging Face API. Falls back to VADER on error. """
    # Basic validation
    if not text or len(text) < 5:
        # logger.warning(f"Text too short for HF API, using VADER: '{text}'")
        return get_vader_emotion(text)

    if not HF_API_URL or not HEADERS.get("Authorization"):
         logger.warning("HF API not configured, falling back to VADER.")
         return get_vader_emotion(text)

    try:
        response = requests.post(HF_API_URL, headers=HEADERS, json={"inputs": text}, timeout=15) # Increased timeout
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        result = response.json()

        # Validate response format (adjust based on your specific model's output)
        if isinstance(result, list) and result and isinstance(result[0], list) and result[0]:
            # Assuming standard classification output: [[{'label': '...', 'score': ...}, ...]]
            top_emotion = result[0][0]
            if "label" in top_emotion and "score" in top_emotion:
                 # Use VADER score for consistency in sentiment value? Or HF score?
                 # Let's use VADER score for consistency for now. HF label + VADER score.
                 _, vader_score = get_vader_emotion(text)  # Use underscore to indicate unused variable
                 # Return HF primary emotion label, and VADER compound score
                 return top_emotion["label"], vader_score
            else:
                 logger.error(f"Invalid API response format (missing keys): {result}")
                 return get_vader_emotion(text)
        else:
            logger.error(f"Unexpected API response format: {result}")
            return get_vader_emotion(text)

    except requests.Timeout:
         logger.error(f"Timeout error calling HF API for text: '{text[:50]}...'")
         return get_vader_emotion(text)
    except requests.RequestException as e:
        logger.error(f"Request error calling HF API: {str(e)}")
        return get_vader_emotion(text)
    except ValueError as e: # Includes JSON decoding errors
        logger.error(f"JSON parsing error from HF API: {str(e)}")
        return get_vader_emotion(text)
    except Exception as e:
        logger.error(f"Unexpected error in get_emotion_hf: {str(e)}")
        return get_vader_emotion(text)

# --- Core Analysis Task Function ---

def _process_message_for_emotion(message: ChatMessage, skip_api_calls: bool) -> Optional[EmotionAnalysis]:
    """Processes a single message to get emotion and returns an EmotionAnalysis object (unsaved)."""
    if not message.message_text:
        # logger.warning(f"Skipping message {message.chat_message_id} - no text content")
        return None # Indicate skipped

    try:
        if skip_api_calls:
            emotion_category, confidence_score = get_vader_emotion(message.message_text)
        else:
            # Use HF API (which has VADER fallback internally)
            emotion_category, confidence_score = get_emotion_hf(message.message_text)

        return EmotionAnalysis(
            emotion_id=str(uuid.uuid4()),
            chat_message_id=message.chat_message_id,
            patient_id=message.sender_id, # Assuming sender_id is the patient_id string
            emotion_category=emotion_category,
            confidence_score=confidence_score, # Using VADER compound score
            analyzed_at=datetime.now(timezone.utc)
        )
    except Exception as e:
         logger.error(f"Error creating EmotionAnalysis for msg {message.chat_message_id}: {e}", exc_info=True)
         return None # Indicate failure


def run_emotion_analysis_job(days_back: int = 7, batch_size: int = DEFAULT_BATCH_SIZE, skip_api_calls: bool = False):
    """
    Core job to analyze emotions for unanalyzed messages within the lookback period.
    Processes in batches using parallel API calls.
    """
    logger.info(f"Starting emotion analysis job. Days back: {days_back}, Batch size: {batch_size}, Skip API: {skip_api_calls}")
    start_time = time.time()
    db: Optional[Session] = None
    total_analyzed = 0
    total_skipped = 0
    total_failed = 0
    total_processed_messages = 0
    processed_batches = 0

    # Define a function to create a fresh database session
    def get_fresh_db_session():
        try:
            # Close any existing session to ensure we get a fresh connection
            if 'db' in locals() and db is not None:
                try:
                    db.close()
                except Exception:
                    pass  # Ignore errors when closing

            # Create a new session
            new_db = SessionLocal()

            # Test the connection with a simple query
            new_db.execute(text("SELECT 1")).scalar()
            logger.info("Successfully established database connection")
            return new_db
        except Exception as e:
            logger.error(f"Error creating database session: {e}", exc_info=True)
            if 'new_db' in locals() and new_db is not None:
                try:
                    new_db.close()
                except Exception:
                    pass
            raise

    try:
        # Get a fresh database connection
        db = SessionLocal()

        # 1. Find all distinct patient_ids who sent messages recently
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_back)
        logger.info(f"Analyzing messages since {cutoff_date}")

        # Distinct sender IDs from recent messages - with retry logic
        max_retries = 3
        retry_count = 0
        patient_ids = []

        while retry_count < max_retries:
            try:
                # Include messages with NULL createdAt as they might be older messages
                patient_ids_query = select(ChatMessage.sender_id).filter(
                    (ChatMessage.createdAt >= cutoff_date) | (ChatMessage.createdAt.is_(None))
                ).distinct()
                patient_ids = [row[0] for row in db.execute(patient_ids_query).all()]
                logger.info(f"Found {len(patient_ids)} distinct patients with recent messages.")
                break  # Success, exit the retry loop
            except Exception as e:
                retry_count += 1
                logger.warning(f"Database error on attempt {retry_count}/{max_retries}: {e}")
                if retry_count < max_retries:
                    # Get a fresh connection before retrying
                    try:
                        db.close()
                    except:
                        pass
                    db = get_fresh_db_session()
                    time.sleep(1)  # Brief pause before retry
                else:
                    logger.error("Max retries reached for patient_ids query")
                    raise  # Re-raise the exception after max retries

        if not patient_ids:
            logger.info("No patients found with recent messages to analyze.")
            return {"status": "success", "message": "No recent messages found to analyze."}

        # 2. Find all message IDs already analyzed for these patients - with retry logic
        max_retries = 3
        retry_count = 0
        analyzed_message_ids = set()

        while retry_count < max_retries:
            try:
                analyzed_ids_query = select(EmotionAnalysis.chat_message_id).filter(
                    EmotionAnalysis.patient_id.in_(patient_ids)
                    # Optional: Add timestamp filter here too if EmotionAnalysis table is huge
                    # EmotionAnalysis.analyzed_at >= cutoff_date - timedelta(days=1) # Example extra filter
                )
                analyzed_message_ids = {row[0] for row in db.execute(analyzed_ids_query).all()}
                logger.info(f"Found {len(analyzed_message_ids)} previously analyzed messages for these patients.")
                break  # Success, exit the retry loop
            except Exception as e:
                retry_count += 1
                logger.warning(f"Database error on attempt {retry_count}/{max_retries}: {e}")
                if retry_count < max_retries:
                    # Get a fresh connection before retrying
                    try:
                        db.close()
                    except:
                        pass
                    db = get_fresh_db_session()
                    time.sleep(1)  # Brief pause before retry
                else:
                    logger.error("Max retries reached for analyzed_ids_query")
                    raise  # Re-raise the exception after max retries

        # 3. Query for unanalyzed messages in batches
        # Include messages with NULL createdAt as they might be older unanalyzed messages
        messages_query = select(ChatMessage).filter(
            ChatMessage.sender_id.in_(patient_ids),
            (ChatMessage.createdAt >= cutoff_date) | (ChatMessage.createdAt.is_(None)),
            ~ChatMessage.chat_message_id.in_(analyzed_message_ids)
        ).order_by(ChatMessage.createdAt.asc().nulls_first()) # Process NULL timestamps first, then older messages

        # Use a loop with offset/limit for batching to avoid loading all messages into memory
        offset = 0
        while True:
            batch_start_time = time.time()
            logger.info(f"Fetching message batch: offset={offset}, limit={batch_size}")

            # Fetch message batch with retry logic
            max_retries = 3
            retry_count = 0
            message_batch = []

            while retry_count < max_retries:
                try:
                    message_batch = db.execute(messages_query.offset(offset).limit(batch_size)).scalars().all()
                    break  # Success, exit the retry loop
                except Exception as e:
                    retry_count += 1
                    logger.warning(f"Database error fetching batch (attempt {retry_count}/{max_retries}): {e}")
                    if retry_count < max_retries:
                        # Get a fresh connection before retrying
                        try:
                            db.close()
                        except:
                            pass
                        db = get_fresh_db_session()
                        time.sleep(1)  # Brief pause before retry
                    else:
                        logger.error("Max retries reached for batch fetch")
                        raise  # Re-raise the exception after max retries

            if not message_batch:
                logger.info("No more unanalyzed messages found.")
                break

            logger.info(f"Processing batch of {len(message_batch)} messages.")
            total_processed_messages += len(message_batch)
            new_analyses: List[EmotionAnalysis] = []
            futures = []

            # Use ThreadPoolExecutor for concurrent processing (API calls)
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                for msg in message_batch:
                     futures.append(executor.submit(_process_message_for_emotion, msg, skip_api_calls))

                for future in concurrent.futures.as_completed(futures):
                     try:
                         result = future.result()
                         if result is not None:
                             new_analyses.append(result)
                             total_analyzed += 1
                         else:
                             # _process_message_for_emotion returns None if skipped or failed internally
                             # We can't easily distinguish here without more complex return values
                             # Assume skipped for now, failures are logged internally
                             total_skipped += 1
                     except Exception as exc:
                         logger.error(f"Error getting result from parallel task: {exc}", exc_info=True)
                         total_failed += 1

            # 4. Bulk save the results for the batch
            if new_analyses:
                # Commit with retry logic
                max_retries = 3
                retry_count = 0
                commit_success = False

                while retry_count < max_retries and not commit_success:
                    try:
                        logger.info(f"Adding {len(new_analyses)} new emotion analyses to the session (attempt {retry_count + 1}/{max_retries}).")
                        # Use add_all which is generally safe and performs well
                        db.add_all(new_analyses)
                        # Alternatively, for potentially higher performance on some backends (like PostgreSQL with psycopg2):
                        # db.bulk_save_objects(new_analyses)
                        db.commit()
                        logger.info(f"Successfully committed batch {processed_batches + 1}.")
                        commit_success = True
                    except Exception as commit_error:
                        retry_count += 1
                        logger.warning(f"Database commit error (attempt {retry_count}/{max_retries}): {commit_error}")
                        try:
                            db.rollback()
                        except:
                            pass

                        if retry_count < max_retries:
                            # Get a fresh connection before retrying
                            try:
                                db.close()
                            except:
                                pass
                            db = get_fresh_db_session()
                            time.sleep(1)  # Brief pause before retry
                        else:
                            logger.error(f"Max retries reached for commit. Batch failed.", exc_info=True)
                            total_failed += len(new_analyses)  # Mark analyses in this batch as failed
                            total_analyzed -= len(new_analyses)

                if not commit_success:
                    logger.error(f"Failed to commit batch {processed_batches + 1} after {max_retries} attempts.")
            else:
                 logger.info("No new analyses generated in this batch.")

            processed_batches += 1
            offset += len(message_batch) # Move to the next batch offset
            batch_end_time = time.time()
            logger.info(f"Batch {processed_batches} finished in {batch_end_time - batch_start_time:.2f}s. "
                        f"Analyzed so far: {total_analyzed}, Skipped: {total_skipped}, Failed: {total_failed}")

            # Optional: Add a small sleep if hitting API rate limits aggressively
            # time.sleep(1)

        # --- End of while loop ---

    except Exception as e:
        logger.error(f"Critical error during emotion analysis job: {e}", exc_info=True)
        if db:
            db.rollback()
        # Log or return detailed error status
        return {
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "analyzed_messages": total_analyzed,
            "skipped_messages": total_skipped,
            "failed_messages": total_failed,
        }
    finally:
        if db:
            db.close()
        end_time = time.time()
        logger.info(f"Emotion analysis job finished in {end_time - start_time:.2f} seconds.")

    summary = {
        "status": "success",
        "total_patients_considered": len(patient_ids) if 'patient_ids' in locals() else 0,
        "total_messages_processed": total_processed_messages,
        "analyzed_messages": total_analyzed,
        "skipped_messages": total_skipped, # Includes messages with no text or internal processing errors
        "failed_messages": total_failed, # Failures during commit or getting future result
        "batches_processed": processed_batches,
        "duration_seconds": round(end_time - start_time, 2)
    }
    logger.info(f"Job Summary: {summary}")
    return summary


# --- Scheduled Job ---
# Schedule the job to run daily at midnight server time (Asia/Kolkata)
# Pass default arguments or configure as needed
try:
    # Check if job already exists to prevent duplicates on reload
    if not scheduler.get_job('daily_emotion_analysis'):
        scheduler.add_job(
            run_emotion_analysis_job,
            "cron",
            hour=0,
            minute=5, # Run at 00:05
            id='daily_emotion_analysis', # Add an ID
            replace_existing=True      # Replace if somehow exists with same ID
            # args=[7, DEFAULT_BATCH_SIZE, False] # Example if you need to pass args
        )
        logger.info("Scheduled daily emotion analysis job.")
    else:
         logger.info("Daily emotion analysis job already scheduled.")

    if not scheduler.running:
         scheduler.start()
         logger.info("Scheduler started.")
    else:
         logger.info("Scheduler already running.")

except Exception as e:
    logger.error(f"Failed to schedule emotion analysis job: {e}", exc_info=True)


# --- API Endpoints ---

@emotion_router.post("/run-analysis", status_code=status.HTTP_202_ACCEPTED)
async def trigger_analysis_endpoint(
    days_back: int = Query(7, ge=1, le=90, description="Number of past days to analyze messages from."),
    batch_size: int = Query(DEFAULT_BATCH_SIZE, ge=50, le=1000, description="Number of messages to process per batch."),
    skip_api_calls: bool = Query(False, description="Use VADER only (faster, less accurate) instead of Hugging Face API."),
    # db: Session = Depends(get_db) # Avoid using Depends here if running in background
):
    """
    Triggers the emotion analysis background job.
    Processes ALL unanalyzed messages within the specified days_back period in batches.
    """
    logger.info("Received request to trigger emotion analysis.")
    # Basic check for API config
    if not skip_api_calls and (not HF_API_URL or not HEADERS.get("Authorization")):
        logger.warning("Triggering analysis, but HF API not configured. Forcing skip_api_calls=True.")
        skip_api_calls = True

    # Option 1: Run synchronously (might timeout for long jobs)
    # result = run_emotion_analysis_job(days_back, batch_size, skip_api_calls)
    # return {"message": "Analysis finished.", "details": result}

    # Option 2: Run in background using APScheduler's ability to run jobs immediately
    # This prevents HTTP timeouts for long-running tasks.
    try:
        job = scheduler.add_job(
             run_emotion_analysis_job,
             trigger='date', # Run immediately
             args=[days_back, batch_size, skip_api_calls],
             id=f'manual_analysis_{uuid.uuid4()}', # Unique ID for this run
             replace_existing=False,
             misfire_grace_time=None # Run ASAP if missed
         )
        logger.info(f"Successfully submitted manual analysis job with ID: {job.id}")
        return {
            "message": "Emotion analysis job submitted successfully.",
            "job_id": job.id,
            "details": f"Processing messages from last {days_back} days, batch size {batch_size}, skipping API: {skip_api_calls}. Check server logs for progress."
            }
    except Exception as e:
         logger.error(f"Failed to submit manual analysis job: {e}", exc_info=True)
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to schedule analysis job.")


@emotion_router.get("/patient/{patient_id}")
def get_patient_emotions(patient_id: str, days_back: int = 30, db: Session = Depends(get_db)):
    """
    Get aggregated emotion analysis and recent messages for a specific patient.
    """
    logger.info(f"Fetching emotion data for patient {patient_id} for last {days_back} days.")
    if days_back < 1 or days_back > 365:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="days_back must be between 1 and 365")

    try:
        # Validate patient exists (handle both UUID and string formats)
        try:
            patient = db.query(Patient).filter(Patient.patient_id == patient_id).first()
        except Exception as e:
            logger.warning(f"Error querying patient with UUID format: {e}")
            # If UUID query fails, patient might not exist or ID format issue
            patient = None

        if not patient:
            # Check if we have any emotion analysis for this patient_id (as string)
            emotion_exists = db.query(EmotionAnalysis).filter(EmotionAnalysis.patient_id == patient_id).first()
            if not emotion_exists:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Patient with ID {patient_id} not found")
            # If emotion analysis exists, create a dummy patient object for response
            class DummyPatient:
                first_name = "Unknown"
                last_name = "Patient"
            patient = DummyPatient()

        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_back)

        # Optimized Query: Fetch emotions and join with messages in one go
        # Handle NULL createdAt values by including them and filtering by analyzed_at instead
        results = db.query(
            EmotionAnalysis,
            ChatMessage.message_text,
            ChatMessage.createdAt,
            EmotionAnalysis.analyzed_at
        ).join(
            ChatMessage, EmotionAnalysis.chat_message_id == ChatMessage.chat_message_id
        ).filter(
            EmotionAnalysis.patient_id == patient_id,
            # Use analyzed_at from EmotionAnalysis if createdAt is NULL
            (ChatMessage.createdAt >= cutoff_date) |
            ((ChatMessage.createdAt.is_(None)) & (EmotionAnalysis.analyzed_at >= cutoff_date))
        ).order_by(
            # Order by analyzed_at if createdAt is NULL, otherwise use createdAt
            ChatMessage.createdAt.desc().nulls_last(),
            EmotionAnalysis.analyzed_at.desc()
        ).all()

        if not results:
             logger.info(f"No emotion analysis found for patient {patient_id} in the last {days_back} days.")
             return {
                 "patient_id": patient_id,
                 "patient_name": f"{patient.first_name} {patient.last_name}",
                 "days_analyzed": days_back,
                 "total_messages_analyzed": 0,
                 "dominant_emotion": "unknown",
                 "average_sentiment": 0,
                 "mood_score": 50,  # Default mood score is 50 (neutral)
                 "emotion_distribution": {},
                 "recent_messages": []
             }

        total_sentiment = 0
        emotion_counts = defaultdict(int)
        recent_messages_output = []

        for i, (emotion_record, message_text, message_timestamp, analyzed_timestamp) in enumerate(results):
            category = emotion_record.emotion_category
            emotion_counts[category] += 1
            total_sentiment += emotion_record.confidence_score

            # Include text for the 10 most recent messages
            if i < 10:
                # Use createdAt if available, otherwise use analyzed_at for timestamp
                timestamp_to_use = message_timestamp if message_timestamp else analyzed_timestamp
                recent_messages_output.append({
                     "message_id": emotion_record.chat_message_id,
                     "text": message_text[:100] + "..." if message_text and len(message_text) > 100 else message_text,
                     "timestamp": timestamp_to_use.isoformat() if timestamp_to_use else None,
                     "emotion": category,
                     "sentiment": emotion_record.confidence_score
                 })

        # Determine dominant emotion (excluding neutral if possible)
        dominant_emotion = "unknown"
        if emotion_counts:
            sorted_emotions = sorted(emotion_counts.items(), key=lambda item: item[1], reverse=True)
            if sorted_emotions[0][0] == "neutral" and len(sorted_emotions) > 1:
                dominant_emotion = sorted_emotions[1][0]
            else:
                dominant_emotion = sorted_emotions[0][0]

        avg_sentiment = round(total_sentiment / len(results), 2) if results else 0

        # Calculate mood score: 50 + (average_score/2) * 100
        mood_score = round(50 + (avg_sentiment / 2) * 100, 2)

        # Ensure mood score is within 0-100 range
        mood_score = max(0, min(100, mood_score))

        logger.info(f"Found {len(results)} analyzed messages for patient {patient_id}. Dominant: {dominant_emotion}, Avg Sentiment: {avg_sentiment}, Mood Score: {mood_score}")

        return {
            "patient_id": patient_id,
            "patient_name": f"{patient.first_name} {patient.last_name}",
            "days_analyzed": days_back,
            "total_messages_analyzed": len(results),
            "dominant_emotion": dominant_emotion,
            "average_sentiment": avg_sentiment,
            "mood_score": mood_score,
            "emotion_distribution": dict(emotion_counts),
            "recent_messages": recent_messages_output
        }

    except HTTPException:
        raise # Re-raise validation errors etc.
    except Exception as e:
        logger.error(f"Error fetching emotions for patient {patient_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An error occurred while fetching patient emotions.")


@emotion_router.get("/patient/{patient_id}/daily-mood")
def get_patient_daily_mood(
    patient_id: str,
    days_back: int = Query(30, ge=1, le=365, description="Number of days to look back"),
    include_dates: bool = Query(True, description="Include dates in the response"),
    db: Session = Depends(get_db)
):
    """
    Get daily average mood values for a patient over a specified number of days.

    Returns an array of objects containing daily sentiment scores, dates, and days of the week.
    Each day's score is the average of all sentiment scores from that day's messages.
    If no data exists for a day, 0 is used as the sentiment score.

    Example response with include_dates=True:
    [
        {"date": "2023-05-01", "day": "Monday", "score": 0.1},
        {"date": "2023-05-02", "day": "Tuesday", "score": -0.2},
        ...
    ]

    Example response with include_dates=False (legacy format):
    [0.1, -0.2, 0.4, 0.5, 0, 0.1, -0.3]
    """
    logger.info(f"Fetching daily mood data for patient {patient_id} for last {days_back} days.")

    try:
        # Validate patient exists (handle both UUID and string formats)
        try:
            patient = db.query(Patient).filter(Patient.patient_id == patient_id).first()
        except Exception as e:
            logger.warning(f"Error querying patient with UUID format: {e}")
            patient = None

        if not patient:
            # Check if we have any emotion analysis for this patient_id (as string)
            emotion_exists = db.query(EmotionAnalysis).filter(EmotionAnalysis.patient_id == patient_id).first()
            if not emotion_exists:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Patient with ID {patient_id} not found")

        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_back)

        # Fetch emotions and join with messages
        # Handle NULL createdAt values by including them and filtering by analyzed_at instead
        results = db.query(
            EmotionAnalysis,
            ChatMessage.createdAt,
            EmotionAnalysis.analyzed_at
        ).join(
            ChatMessage, EmotionAnalysis.chat_message_id == ChatMessage.chat_message_id
        ).filter(
            EmotionAnalysis.patient_id == patient_id,
            # Use analyzed_at from EmotionAnalysis if createdAt is NULL
            (ChatMessage.createdAt >= cutoff_date) |
            ((ChatMessage.createdAt.is_(None)) & (EmotionAnalysis.analyzed_at >= cutoff_date))
        ).order_by(
            # Order by analyzed_at if createdAt is NULL, otherwise use createdAt
            ChatMessage.createdAt.asc().nulls_first(),
            EmotionAnalysis.analyzed_at.asc()
        ).all()

        # Group sentiment scores by day
        daily_sentiments = defaultdict(list)

        for emotion_record, created_at, analyzed_at in results:
            # Use createdAt if available, otherwise use analyzed_at
            timestamp_to_use = created_at if created_at else analyzed_at
            if timestamp_to_use:
                # Convert to date string in format YYYY-MM-DD
                day_key = timestamp_to_use.date().isoformat()
                daily_sentiments[day_key].append(float(emotion_record.confidence_score))

        # Get the date range for the requested period
        end_date = datetime.now(timezone.utc).date()
        start_date = end_date - timedelta(days=days_back-1)
        current_date = start_date

        if include_dates:
            # Return detailed format with dates and days
            daily_mood_data = []

            # Create entries for each day in the range
            while current_date <= end_date:
                day_key = current_date.isoformat()
                sentiments = daily_sentiments.get(day_key, [])

                # Calculate average sentiment for the day if data exists, otherwise use 0
                if sentiments:
                    avg_sentiment = round(sum(sentiments) / len(sentiments), 2)
                else:
                    avg_sentiment = 0  # No data for this day, use 0 as default

                # Get day of week name
                day_of_week = current_date.strftime("%A")  # Full day name (Monday, Tuesday, etc.)

                daily_mood_data.append({
                    "date": day_key,
                    "day": day_of_week,
                    "score": avg_sentiment
                })

                current_date += timedelta(days=1)

            logger.info(f"Generated daily mood data with dates for patient {patient_id} over {days_back} days.")
            return daily_mood_data
        else:
            # Return legacy format (just array of scores)
            daily_mood_values = []

            # Create entries for each day in the range
            while current_date <= end_date:
                day_key = current_date.isoformat()
                sentiments = daily_sentiments.get(day_key, [])

                # Calculate average sentiment for the day if data exists, otherwise use 0
                if sentiments:
                    avg_sentiment = round(sum(sentiments) / len(sentiments), 2)
                else:
                    avg_sentiment = 0  # No data for this day, use 0 as default

                daily_mood_values.append(avg_sentiment)
                current_date += timedelta(days=1)

            logger.info(f"Generated daily mood values (legacy format) for patient {patient_id} over {days_back} days.")
            return daily_mood_values

    except HTTPException:
        raise  # Re-raise validation errors
    except Exception as e:
        logger.error(f"Error fetching daily mood for patient {patient_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while fetching patient daily mood data."
        )