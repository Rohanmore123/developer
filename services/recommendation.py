import concurrent.futures
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Tuple
from sqlalchemy.orm import Session
from fastapi import APIRouter, Depends, HTTPException, Query
from database.database import get_db
from model.model_correct import Patient, Doctor, ChatMessage, Rating, DoctorAvailability

# Cache for database query results (not final recommendations)
query_cache = {}
CACHE_EXPIRATION_SECONDS = 300  # 5 minutes
MAX_DOCTORS_PER_QUERY = 100  # Limit number of doctors per query to improve performance

recommend_router = APIRouter()

def score_doctor(doctor, patient, past_messages, db):
    score = 0
    # Check if language matches
    if hasattr(doctor, 'language') and hasattr(patient, 'language') and doctor.language == patient.language:
        score += 5

    # Check if address contains similar country information
    if hasattr(doctor, 'address') and hasattr(patient, 'address') and doctor.address and patient.address:
        # Check if the addresses contain similar country/region information
        if doctor.address.lower() in patient.address.lower() or patient.address.lower() in doctor.address.lower():
            score += 4  # Significant bonus for country/region match

    # Chat Message Matching
    common_keywords = ["stress", "depression", "anxiety", "relationship", "trauma", "insomnia"]
    for message in past_messages:
        # Check if message_text exists and is not None
        if message.message_text and hasattr(doctor, 'specialization') and doctor.specialization:
            for keyword in common_keywords:
                if keyword in message.message_text.lower() and keyword in doctor.specialization.lower():
                    score += 5

    # Rating Score
    # Convert UUID to string for comparison with ratings table
    doctor_id_str = str(doctor.doctor_id)
    ratings = db.query(Rating).filter(Rating.doctor_id == doctor_id_str).all()
    if ratings:
        avg_rating = sum(r.rating for r in ratings) / len(ratings)
        score += avg_rating

    return round(score, 2)

def score_doctors_parallel(doctors, patient, past_messages, db):
    """Score multiple doctors in parallel using ThreadPoolExecutor."""
    start_time = datetime.now()
    print(f"Starting parallel scoring for {len(doctors)} doctors")

    results = []

    # Use ThreadPoolExecutor to score doctors in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(10, len(doctors))) as executor:
        # Create a list to store the futures
        futures = []

        # Submit scoring tasks for each doctor
        for doctor in doctors:
            future = executor.submit(score_doctor, doctor, patient, past_messages, db)
            futures.append((doctor, future))

        # Collect results as they complete
        for doctor, future in futures:
            score = future.result()
            results.append((doctor, score))

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"Parallel scoring completed in {duration:.2f} seconds for {len(doctors)} doctors")

    return results

def get_doctor_recommendations(patient_id: str, db: Session, filters: dict = None):
    # Start timing
    start_time = datetime.now()
    print(f"Starting doctor recommendations for patient {patient_id}")

    # Check if the patient exists
    patient = db.query(Patient).filter(Patient.patient_id == patient_id).first()
    print(f"Looking for patient with ID: {patient_id}")
    print(f"Found patient: {patient}")

    if not patient:
        print("Patient not found, returning empty list")
        return []  # Return empty list if no patient found

    # Use patient_id for messages
    patient_id_for_messages = str(patient.patient_id)
    print(f"Using patient_id_for_messages: {patient_id_for_messages}")

    # Get past messages for this patient
    past_messages = db.query(ChatMessage).filter(ChatMessage.sender_id == patient_id_for_messages).all()
    print(f"Found {len(past_messages)} past messages")

    # Step 1: Filtered Doctors
    filtered_query = db.query(Doctor)

    if filters:
        if "language" in filters:
            filtered_query = filtered_query.filter(Doctor.language == filters["language"])
        if "region" in filters:
            filtered_query = filtered_query.filter(Doctor.region == filters["region"])
        if "gender" in filters:
            filtered_query = filtered_query.filter(Doctor.gender == filters["gender"])
        if "specialization" in filters:
            filtered_query = filtered_query.filter(Doctor.specialization.ilike(f"%{filters['specialization']}%"))

    filtered_doctors = filtered_query.all()
    print(f"Found {len(filtered_doctors)} doctors matching filters")

    # Score filtered doctors in parallel
    filtered_scores = score_doctors_parallel(filtered_doctors, patient, past_messages, db)

    # Step 2: Rule-based Completion (if < 5 doctors found)
    if len(filtered_scores) < 5:
        filtered_ids = [doc.doctor_id for doc, _ in filtered_scores]
        remaining_doctors = db.query(Doctor).filter(~Doctor.doctor_id.in_(filtered_ids)).all()
        print(f"Need more doctors, found {len(remaining_doctors)} additional candidates")

        # Score remaining doctors in parallel
        remaining_scores = score_doctors_parallel(remaining_doctors, patient, past_messages, db)

        # Sort and pick top remaining doctors to fill the gap
        remaining_scores.sort(key=lambda x: x[1], reverse=True)
        doctors_needed = 5 - len(filtered_scores)
        filtered_scores += remaining_scores[:doctors_needed]

    # Final Deduplication & Packaging
    unique_doctors = {}
    for doctor, score in filtered_scores:
        if doctor.doctor_id not in unique_doctors:
            # Get ratings for the doctor - doctor_id in Rating table is a string
            doctor_id_str = str(doctor.doctor_id)
            ratings = db.query(Rating).filter(Rating.doctor_id == doctor_id_str).all()
            print(f"Looking for ratings with doctor_id: {doctor_id_str}")
            avg_rating = round(sum(r.rating for r in ratings) / len(ratings), 2) if ratings else "N/A"

            # Get name from the doctor object directly
            full_name = f"{doctor.first_name} {doctor.last_name}"
            gender = doctor.gender if doctor.gender else "N/A"

            unique_doctors[doctor.doctor_id] = {
                "doctor_id": doctor.doctor_id,
                "name": full_name,
                "gender": gender,
                "specialization": doctor.specialization,
                "language": doctor.language,
                "region": doctor.region,
                "average_rating": avg_rating,
                "score": score,
            }

    # Return top 5 recommendations
    final_recommendations = list(unique_doctors.values())[:5]
    return final_recommendations

@recommend_router.get("/test")
def test_recommendation_router():
    return {"message": "Recommendation router is working!"}

@recommend_router.get("/simple/{patient_id}")
def simple_recommendation(patient_id: str):
    try:
        # Just return a simple message with the patient ID
        return {"message": f"Received request for patient ID: {patient_id}"}
    except Exception as e:
        # Log the error for debugging
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in simple_recommendation: {str(e)}\n{error_details}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

def process_doctor_parallel(doctor, patient, min_rating, specialization, language, country, day_of_week, db, source="filter_match"):
    """Process a single doctor in parallel, including ratings and availability."""
    # Skip if doctor is None
    if not doctor:
        return None

    # Get ratings for this doctor using ORM
    doctor_id_str = str(doctor.doctor_id)
    ratings = db.query(Rating).filter(Rating.doctor_id == doctor_id_str).all()

    # Calculate average rating
    if ratings:
        avg_rating = sum(r.rating for r in ratings) / len(ratings)
        num_ratings = len(ratings)
    else:
        avg_rating = 0
        num_ratings = 0

    # Apply min_rating filter if provided
    if min_rating is not None and avg_rating < min_rating:
        return None

    # Calculate score based on source
    if source == "filter_match":
        score = 5.0 + avg_rating  # Base score + rating bonus
    elif source == "rule_based":
        score = 3.0 + avg_rating  # Lower base score for rule-based
    else:  # random
        score = 1.0 + avg_rating  # Lowest base score for random

    # Add language match bonus if patient exists
    if patient and patient.language and doctor.language and patient.language == doctor.language:
        if source == "rule_based":
            score += 3.0  # Higher language bonus for rule-based
        else:
            score += 2.0  # Standard language bonus

    # Add country match bonus if original filter had country
    if country and doctor.address and country.lower() in doctor.address.lower():
        if source == "rule_based":
            score += 2.5  # Country match is important for rule-based
        else:
            score += 1.5

    # Add specialization match bonus if original filter had specialization
    if specialization and doctor.specialization and specialization.lower() in doctor.specialization.lower():
        if source == "rule_based":
            score += 2.0
        else:
            score += 1.0

    # Get availability information
    availability_info = []
    if day_of_week:
        # Capitalize the first letter to match database format
        day_of_week_capitalized = day_of_week.capitalize()

        # Validate day of week
        valid_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        if day_of_week_capitalized in valid_days:
            # Query availability for this doctor on the specified day
            availabilities = db.query(DoctorAvailability).filter(
                DoctorAvailability.doctor_id == doctor.doctor_id,
                DoctorAvailability.day_of_week == day_of_week_capitalized
            ).all()

            # Format availability times
            for avail in availabilities:
                availability_info.append({
                    "day": avail.day_of_week,
                    "start_time": avail.start_time.strftime("%H:%M") if avail.start_time else None,
                    "end_time": avail.end_time.strftime("%H:%M") if avail.end_time else None
                })

    doctor_info = {
        "doctor_id": str(doctor.doctor_id),
        "name": f"{doctor.first_name} {doctor.last_name}",
        "specialization": doctor.specialization,
        "language": doctor.language,
        "gender": doctor.gender,
        "address": doctor.address,  # This includes country/region information
        "avg_rating": round(avg_rating, 2),
        "num_ratings": num_ratings,
        "score": round(score, 2),
        "availability": availability_info,
        "source": source
    }

    return doctor_info

def process_doctors_batch_parallel(doctors, patient, min_rating, specialization, language, country, day_of_week, db, source="filter_match"):
    """Process a batch of doctors in parallel."""
    start_time = datetime.now()
    print(f"Starting parallel processing for {len(doctors)} doctors with source '{source}'")

    # If no doctors to process, return empty list immediately
    if not doctors:
        print(f"No doctors to process for source '{source}'")
        return []

    results = []

    # Use ThreadPoolExecutor to process doctors in parallel
    # Ensure max_workers is at least 1
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(10, len(doctors)))) as executor:
        # Create a list to store the futures
        futures = []

        # Submit processing tasks for each doctor
        for doctor in doctors:
            future = executor.submit(
                process_doctor_parallel,
                doctor,
                patient,
                min_rating,
                specialization,
                language,
                country,
                day_of_week,
                db,
                source
            )
            futures.append(future)

        # Collect results as they complete
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:  # Only add non-None results
                results.append(result)

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"Parallel processing completed in {duration:.2f} seconds for {len(doctors)} doctors, got {len(results)} valid results")

    return results

@recommend_router.get("/recommend/{patient_id}")
def recommend_doctors(
    patient_id: str,
    specialization: str = Query(None, description="Filter by doctor specialization"),
    language: str = Query(None, description="Filter by doctor language"),
    min_rating: float = Query(None, description="Filter by minimum doctor rating", ge=0, le=5),
    gender: str = Query(None, description="Filter by doctor gender"),
    day_of_week: str = Query(None, description="Filter by day of week (Monday, Tuesday, etc.)"),
    country: str = Query(None, description="Filter by doctor's country"),
    db: Session = Depends(get_db)
):
    """
    Recommend doctors for a patient based on their profile and ratings.
    Always returns 10 doctors using a fallback mechanism:
    1. First try to get doctors matching all filters
    2. If fewer than 10 doctors are found, add doctors using rule-based scoring
    3. If still fewer than 10 doctors, add random doctors to reach 10 total

    Parameters:
    - patient_id: The ID of the patient
    - specialization: Filter by doctor specialization
    - language: Filter by doctor language
    - min_rating: Filter by minimum doctor rating
    - gender: Filter by doctor gender
    - day_of_week: Filter by day of week (Monday, Tuesday, etc.)
    - country: Filter by doctor's country

    Returns:
    - List of 10 recommended doctors with scores
    """
    try:
        # Start timing
        start_time = datetime.now()
        print(f"Starting doctor recommendations for patient {patient_id}")

        # Create a cache key based on the parameters (for database query results, not final recommendations)
        query_cache_key = f"{patient_id}_{specialization}_{language}_{min_rating}_{gender}_{day_of_week}_{country}"

        # Add a timestamp to ensure some randomness in recommendations
        # This ensures we don't return exactly the same recommendations every time
        import random
        random_seed = random.randint(1, 1000)

        # We'll use the cache for database query results to improve performance
        # but still allow for some randomness in the final recommendations
        current_time = datetime.now()
        cached_query_results = None

        if query_cache_key in query_cache:
            cache_entry = query_cache[query_cache_key]
            cache_age = (current_time - cache_entry["timestamp"]).total_seconds()

            # If cache is still valid, use the cached query results
            if cache_age < CACHE_EXPIRATION_SECONDS:
                print(f"Using cached query results (age: {cache_age:.2f} seconds)")
                cached_query_results = cache_entry["data"]
            else:
                print(f"Cache expired (age: {cache_age:.2f} seconds). Performing fresh database queries...")
        else:
            print("No cached query results found. Performing fresh database queries...")

        # Get patient for language matching
        patient = db.query(Patient).filter(Patient.patient_id == patient_id).first()

        # Track which doctors we've already processed
        processed_doctor_ids = set()
        final_result = []

        # STEP 1: Get doctors matching all filters
        # Build the query with filters
        query = db.query(Doctor)

        # Check if any filters are applied
        has_filters = any([specialization, language, gender, min_rating is not None, day_of_week, country])

        # Apply filters if provided
        if specialization:
            query = query.filter(Doctor.specialization.ilike(f"%{specialization}%"))
        if language:
            query = query.filter(Doctor.language == language)
        if gender:
            query = query.filter(Doctor.gender == gender)
        if country:
            # Check if the address field contains the country name
            query = query.filter(Doctor.address.ilike(f"%{country}%"))

        # Apply availability filter if provided
        if day_of_week:
            # Capitalize the first letter to match database format
            day_of_week_capitalized = day_of_week.capitalize()

            # Validate day of week
            valid_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            if day_of_week_capitalized in valid_days:
                print(f"Filtering for availability on {day_of_week_capitalized}")
                # Join with DoctorAvailability and filter by day of week
                query = query.join(DoctorAvailability).filter(DoctorAvailability.day_of_week == day_of_week_capitalized)
            else:
                print(f"Invalid day of week: {day_of_week}. Expected one of: {', '.join(valid_days)}")

        # If no filters are applied, limit to top 10 doctors to improve performance
        if not has_filters:
            doctors = query.limit(10).all()
            print(f"No filters applied, using top {len(doctors)} doctors")

            # Process doctors in parallel with source as "random" when no filters are applied
            # This maintains consistency with the original algorithm
            filtered_doctors = process_doctors_batch_parallel(
                doctors,
                patient,
                min_rating,
                specialization,
                language,
                country,
                day_of_week,
                db,
                "random"  # Use "random" as source when no filters are applied
            )
        else:
            # Optimize query by limiting the number of doctors
            # For day_of_week filter which requires a join, optimize the query
            if day_of_week:
                day_of_week_capitalized = day_of_week.capitalize()
                valid_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

                if day_of_week_capitalized in valid_days:
                    print(f"Optimizing query for day_of_week filter: {day_of_week_capitalized}")

                    # Get doctor IDs with this availability first (limited to improve performance)
                    available_doctor_ids = db.query(DoctorAvailability.doctor_id).filter(
                        DoctorAvailability.day_of_week == day_of_week_capitalized
                    ).distinct().limit(MAX_DOCTORS_PER_QUERY).all()

                    available_doctor_ids = [id[0] for id in available_doctor_ids]
                    print(f"Found {len(available_doctor_ids)} doctors with availability on {day_of_week_capitalized}")

                    # Then filter doctors by these IDs instead of using a join
                    query = db.query(Doctor).filter(Doctor.doctor_id.in_(available_doctor_ids))

                    # Apply other filters
                    if specialization:
                        query = query.filter(Doctor.specialization.ilike(f"%{specialization}%"))
                    if language:
                        query = query.filter(Doctor.language == language)
                    if gender:
                        query = query.filter(Doctor.gender == gender)
                    if country:
                        query = query.filter(Doctor.address.ilike(f"%{country}%"))

                    doctors = query.all()
                else:
                    # Limit the number of doctors to improve performance
                    doctors = query.limit(MAX_DOCTORS_PER_QUERY).all()
            else:
                # Limit the number of doctors to improve performance
                doctors = query.limit(MAX_DOCTORS_PER_QUERY).all()

            print(f"Found {len(doctors)} doctors matching all filters (limited to {MAX_DOCTORS_PER_QUERY})")

            # Process filtered doctors in parallel with source as "filter_match"
            filtered_doctors = process_doctors_batch_parallel(
                doctors,
                patient,
                min_rating,
                specialization,
                language,
                country,
                day_of_week,
                db,
                "filter_match"
            )

        # Add to final result and track processed doctors
        for doctor_info in filtered_doctors:
            final_result.append(doctor_info)
            processed_doctor_ids.add(doctor_info["doctor_id"])

        # Sort filtered doctors by score (highest first)
        final_result.sort(key=lambda x: x["score"], reverse=True)

        # STEP 2: If we don't have enough doctors, use rule-based scoring
        if len(final_result) < 10:
            print(f"Need {10 - len(final_result)} more doctors from rule-based scoring")

            # Get doctors that weren't in the filtered results
            rule_based_query = db.query(Doctor).filter(~Doctor.doctor_id.in_(processed_doctor_ids))

            # Apply partial filters for rule-based scoring
            if language:
                rule_based_query = rule_based_query.filter(Doctor.language == language)
            if country:
                rule_based_query = rule_based_query.filter(Doctor.address.ilike(f"%{country}%"))

            # Apply availability filter for rule-based scoring
            if day_of_week:
                day_of_week_capitalized = day_of_week.capitalize()
                valid_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                if day_of_week_capitalized in valid_days:
                    # Optimize the availability filter by getting IDs first
                    available_doctor_ids = db.query(DoctorAvailability.doctor_id).filter(
                        DoctorAvailability.day_of_week == day_of_week_capitalized
                    ).distinct().limit(MAX_DOCTORS_PER_QUERY).all()

                    available_doctor_ids = [id[0] for id in available_doctor_ids]
                    print(f"Found {len(available_doctor_ids)} doctors with availability on {day_of_week_capitalized} for rule-based scoring")

                    # Filter by these IDs instead of using a join
                    rule_based_query = rule_based_query.filter(Doctor.doctor_id.in_(available_doctor_ids))

            # Limit the number of doctors to improve performance
            rule_based_doctors = rule_based_query.limit(MAX_DOCTORS_PER_QUERY).all()
            print(f"Found {len(rule_based_doctors)} potential doctors for rule-based scoring (limited to {MAX_DOCTORS_PER_QUERY})")

            # Process rule-based doctors in parallel
            rule_based_results = process_doctors_batch_parallel(
                rule_based_doctors,
                patient,
                None,  # Don't apply min_rating filter for rule-based
                specialization,
                language,
                country,
                day_of_week,
                db,
                "rule_based"
            )

            # Sort rule-based doctors by score
            rule_based_results.sort(key=lambda x: x["score"], reverse=True)

            # Add only as many as needed to reach 10
            doctors_needed = 10 - len(final_result)
            for doctor_info in rule_based_results[:doctors_needed]:
                final_result.append(doctor_info)
                processed_doctor_ids.add(doctor_info["doctor_id"])

        # STEP 3: If we still don't have 10 doctors, add random doctors
        if len(final_result) < 10:
            print(f"Need {10 - len(final_result)} more random doctors")

            # Get random doctors that weren't already processed
            # Limit the query to improve performance
            random_doctors_query = db.query(Doctor).filter(~Doctor.doctor_id.in_(processed_doctor_ids))

            # Get a limited number of random doctors
            doctors_needed = 10 - len(final_result)
            random_doctors = random_doctors_query.limit(doctors_needed * 3).all()  # Get 3x as many as needed for randomness
            print(f"Found {len(random_doctors)} potential random doctors (limited query)")

            import random
            random.shuffle(random_doctors)  # Randomize the order

            # Take only as many as needed
            random_doctors = random_doctors[:doctors_needed * 2]  # Get twice as many as needed in case some are filtered out

            # Process random doctors in parallel
            random_results = process_doctors_batch_parallel(
                random_doctors,
                patient,
                None,  # Don't apply min_rating filter for random
                specialization,
                language,
                country,
                day_of_week,
                db,
                "random"
            )

            # Add random doctors to final result
            for doctor_info in random_results[:doctors_needed]:
                final_result.append(doctor_info)
                processed_doctor_ids.add(doctor_info["doctor_id"])

        # Final sort by score
        final_result.sort(key=lambda x: x["score"], reverse=True)

        # Make sure we're returning exactly 10 doctors
        if len(final_result) < 10:
            print(f"Warning: Only found {len(final_result)} doctors, which is less than the required 10")

        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        print(f"Recommendation process completed in {total_duration:.2f} seconds")
        print(f"Returning {len(final_result[:10])} doctors in total")

        # Cache the results before returning
        result = {"recommended_doctors": final_result[:10]}
        recommendation_cache[cache_key] = {
            "data": result,
            "timestamp": current_time
        }
        print(f"Cached recommendation results with key: {cache_key}")

        return result
    except Exception as e:
        # Log the error for debugging
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in recommend_doctors: {str(e)}\n{error_details}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
