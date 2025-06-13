import concurrent.futures
import time
import random
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict

from sqlalchemy.orm import Session
from sqlalchemy import func  # Required for potential future optimizations like count
from fastapi import APIRouter, Depends, HTTPException, Query

# Assuming these are defined in your project structure
from database.database import get_db
from model.model_correct import Patient, Doctor, ChatMessage, Rating, DoctorAvailability

# --- Configuration ---
# Removed caching configuration
MAX_DOCTORS_PER_QUERY = 1000    # Increased limit for DB query results per stage
PARALLEL_MAX_WORKERS = 10       # Max threads for parallel processing

recommend_router = APIRouter()

# --- Helper Function for Bulk Data Fetching ---

def get_bulk_doctor_data(
    doctor_ids_uuid: List[Any], # Assuming doctor_id in Doctor/DoctorAvailability is UUID
    db: Session,
    day_of_week: Optional[str] = None
) -> Tuple[Dict[str, List[Rating]], Dict[str, List[DoctorAvailability]]]:
    """
    Fetches ratings and availability for a list of doctor UUIDs in bulk.
    Uses STRING representation of doctor_id as keys in the returned maps
    for consistent lookup with Rating table (assuming Rating.doctor_id is string).
    """
    ratings_map = defaultdict(list)
    availability_map = defaultdict(list)

    if not doctor_ids_uuid:
        return ratings_map, availability_map

    # Convert UUIDs to strings for querying Rating table and for map keys
    doctor_ids_str = [str(uuid) for uuid in doctor_ids_uuid]

    # --- Bulk fetch ratings ---
    # Assuming Rating.doctor_id is stored as VARCHAR/TEXT matching the string UUID
    print(f"Bulk fetching ratings for {len(doctor_ids_str)} doctor IDs...")
    ratings_fetch_start = time.time()
    ratings = db.query(Rating).filter(Rating.doctor_id.in_(doctor_ids_str)).all()
    for r in ratings:
        # Use string doctor_id from Rating table directly as key
        ratings_map[r.doctor_id].append(r)
    ratings_fetch_end = time.time()
    print(f"Ratings fetch took {ratings_fetch_end - ratings_fetch_start:.4f}s")


    # --- Bulk fetch availability if needed ---
    if day_of_week:
        print(f"Bulk fetching availability for day '{day_of_week}'...")
        avail_fetch_start = time.time()
        day_of_week_capitalized = day_of_week.capitalize()
        valid_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

        if day_of_week_capitalized in valid_days:
            # Assuming DoctorAvailability.doctor_id is UUID, matching Doctor.doctor_id
            availabilities = db.query(DoctorAvailability).filter(
                DoctorAvailability.doctor_id.in_(doctor_ids_uuid), # Query with UUIDs
                DoctorAvailability.day_of_week == day_of_week_capitalized
            ).all()

            for avail in availabilities:
                # Convert UUID to string for map key consistency
                availability_map[str(avail.doctor_id)].append(avail)
        avail_fetch_end = time.time()
        print(f"Availability fetch took {avail_fetch_end - avail_fetch_start:.4f}s")

    return ratings_map, availability_map

# --- Optimized Doctor Processing Function (using pre-fetched data) ---

def process_doctor_parallel(
    doctor: Doctor,
    patient: Optional[Patient],
    min_rating: Optional[float], # Used only if source is 'filter_match'
    # Filters passed for scoring context:
    specialization_filter: Optional[str],
    language_filter: Optional[str],
    country_filter: Optional[str],
    day_of_week_filter: Optional[str], # Passed for context, data is from map
    # Pre-fetched data maps (keys are STRING doctor IDs):
    ratings_map: Dict[str, List[Rating]],
    availability_map: Dict[str, List[DoctorAvailability]],
    source: str # "filter_match", "rule_based", "random"
) -> Optional[Dict[str, Any]]:
    """
    Processes a single doctor using pre-fetched data. Returns None if filtered out.
    Calculates score based on matches and pre-fetched ratings.
    Formats availability from the pre-fetched map.
    """
    if not doctor:
        return None

    doctor_id_str = str(doctor.doctor_id) # Use consistent string ID for lookups
    doctor_ratings = ratings_map.get(doctor_id_str, []) # Get pre-fetched ratings

    # Calculate average rating
    if doctor_ratings:
        avg_rating = sum(r.rating for r in doctor_ratings) / len(doctor_ratings)
        num_ratings = len(doctor_ratings)
    else:
        avg_rating = 0
        num_ratings = 0

    # Apply min_rating filter ONLY if this doctor came from the initial filter match
    if source == "filter_match" and min_rating is not None and avg_rating < min_rating:
        # print(f"Doctor {doctor_id_str} filtered out by min_rating ({avg_rating:.2f} < {min_rating})")
        return None

    # --- Scoring Logic ---
    score = avg_rating # Base score is average rating

    # Add base score based on source
    if source == "filter_match":
        score += 5.0
    elif source == "rule_based":
        score += 3.0
    else: # random
        score += 1.0

    # Add language match bonus (using language_filter if provided)
    patient_lang = patient.language if patient else None
    if patient_lang and doctor.language and patient_lang == doctor.language:
        score += 3.0 if source == "rule_based" else 2.0 # Higher bonus for rule-based

    # Add country match bonus (using country_filter if provided)
    # Check both address and region fields
    address_match = doctor.address and country_filter and country_filter.lower() in doctor.address.lower()
    region_match = patient and patient.region and doctor.address and patient.region.lower() in doctor.address.lower()

    if address_match or region_match:
         score += 2.5 if source == "rule_based" else 1.5

    # Add specialization match bonus (using specialization_filter if provided)
    if specialization_filter and doctor.specialization and specialization_filter.lower() in doctor.specialization.lower():
         score += 2.0 if source == "rule_based" else 1.0

    # --- Get Availability Info from pre-fetched map ---
    availability_info = []
    doctor_availabilities = availability_map.get(doctor_id_str, []) # Lookup using string ID

    for avail in doctor_availabilities:
         availability_info.append({
            "day": avail.day_of_week,
            "start_time": avail.start_time.strftime("%H:%M") if avail.start_time else "N/A",
            "end_time": avail.end_time.strftime("%H:%M") if avail.end_time else "N/A"
        })

    # --- Assemble final doctor information ---
    doctor_info = {
        "doctor_id": doctor_id_str,
        "name": f"{doctor.first_name} {doctor.last_name}",
        "specialization": doctor.specialization if doctor.specialization else "N/A",
        "language": doctor.language if doctor.language else "N/A",
        "gender": doctor.gender if doctor.gender else "N/A",
        "address": doctor.address if doctor.address else "N/A",
        "avg_rating": round(avg_rating, 2),
        "num_ratings": num_ratings,
        "score": round(score, 2),
        "availability": availability_info, # Add formatted availability
        "source": source
    }

    # Add treatment information if available
    if hasattr(doctor, 'treatment') and doctor.treatment:
        # Handle treatment field which might be JSONB in the database
        if isinstance(doctor.treatment, dict):
            doctor_info["treatment"] = doctor.treatment
        else:
            # If it's a string or other type, include it as is
            doctor_info["treatment"] = doctor.treatment

    return doctor_info


# --- Optimized Parallel Batch Processing ---

def process_doctors_batch_parallel(
    doctors_to_process: List[Doctor],
    patient: Optional[Patient],
    min_rating: Optional[float], # Applied only if source is 'filter_match'
    # Original filters passed for context in scoring:
    specialization_filter: Optional[str],
    language_filter: Optional[str],
    country_filter: Optional[str],
    day_of_week_filter: Optional[str],
    db: Session,
    source: str # "filter_match", "rule_based", "random"
) -> List[Dict[str, Any]]:
    """
    Processes a batch of doctors in parallel using pre-fetched data.
    """
    start_time = time.time()
    num_doctors = len(doctors_to_process)
    print(f"Starting parallel processing for {num_doctors} doctors (source: '{source}')")

    if not doctors_to_process:
        print(f"No doctors to process for source '{source}'")
        return []

    # --- Step 1: Bulk Fetch Data ---
    doctor_ids_uuid = [doc.doctor_id for doc in doctors_to_process]
    ratings_map, availability_map = get_bulk_doctor_data(
        doctor_ids_uuid, db, day_of_week_filter
    )

    # --- Step 2: Process in Parallel ---
    results = []
    # Ensure max_workers is at least 1, even if num_doctors is small
    num_workers = max(1, min(PARALLEL_MAX_WORKERS, num_doctors))
    print(f"Using {num_workers} workers for parallel processing.")
    process_start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Create futures, passing pre-fetched maps to each task
        futures = [
            executor.submit(
                process_doctor_parallel,
                doctor,
                patient,
                min_rating, # Pass min_rating - it will only be applied if source is 'filter_match'
                specialization_filter,
                language_filter,
                country_filter,
                day_of_week_filter,
                ratings_map,
                availability_map,
                source
            ) for doctor in doctors_to_process
        ]

        # Collect results as they complete
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                if result:  # Only add non-None results (i.e., doctors not filtered out)
                    results.append(result)
            except Exception as exc:
                print(f"Error processing a doctor in parallel: {exc}")
                traceback.print_exc()


    process_end_time = time.time()
    total_duration = process_end_time - start_time
    processing_duration = process_end_time - process_start_time
    print(f"Parallel processing completed in {processing_duration:.4f}s "
          f"(Total batch time: {total_duration:.4f}s). Got {len(results)} valid doctor results.")

    return results


# --- Main Recommendation Endpoint ---

@recommend_router.get("/recommend/{patient_id}")
def recommend_doctors(
    patient_id: str,
    specialization: str = Query(None, description="Filter by doctor specialization (case-insensitive substring match)"),
    language: str = Query(None, description="Filter by exact doctor language"),
    min_rating: float = Query(None, description="Filter by minimum average doctor rating", ge=0, le=5),
    gender: str = Query(None, description="Filter by exact doctor gender"),
    day_of_week: str = Query(None, description="Filter by availability on day (e.g., Monday, Tuesday...)"),
    country: str = Query(None, description="Filter by country (case-insensitive substring match in address)"),
    db: Session = Depends(get_db)
):
    """
    Recommends **up to 10** doctors based on filters and scoring.

    Fallback Mechanism:
    1.  **Filter Match:** Tries to find doctors matching all provided filters.
    2.  **Rule-Based:** If < 10 found, adds doctors matching some key filters (language, country, availability) with slightly lower scores.
    3.  **Random:** If still < 10, adds randomly selected remaining doctors with the lowest base scores.

    Doctors are processed in parallel batches with pre-fetched ratings and availability
    for improved performance. Results are cached for 5 minutes.
    """
    request_start_time = time.time()
    print(f"\n--- Request received for patient {patient_id} at {datetime.now()} ---")
    print(f"Filters: spec='{specialization}', lang='{language}', rating>='{min_rating}', "
          f"gender='{gender}', day='{day_of_week}', country='{country}'")

    try:
        # --- Cache Check Removed ---
        # Caching has been removed to ensure fresh recommendations every time
        current_time = datetime.now()
        print("Generating fresh recommendations...")

        # --- Get Patient Info (optional, used for language matching) ---
        patient = db.query(Patient).filter(Patient.patient_id == patient_id).first()
        if not patient:
             print(f"WARN: Patient ID {patient_id} not found. Proceeding without patient-specific context.")
             # Allow proceeding, but patient-specific scoring (like language) won't apply
        else:
             # Log patient information for debugging
             print(f"Found patient: {patient.first_name} {patient.last_name}")
             print(f"Patient language: {patient.language}")
             print(f"Patient region: {patient.region}")

             # Handle JSONB fields
             if hasattr(patient, 'preferences') and patient.preferences:
                 print(f"Patient has preferences: {type(patient.preferences)}")
             if hasattr(patient, 'interests') and patient.interests:
                 print(f"Patient has interests: {type(patient.interests)}")
             if hasattr(patient, 'treatment') and patient.treatment:
                 print(f"Patient has treatment info: {type(patient.treatment)}")

        # --- Initialization ---
        processed_doctor_ids_str: Set[str] = set() # Store string IDs of processed doctors
        final_results: List[Dict[str, Any]] = []
        TARGET_RECOMMENDATIONS = 10

        # --- STEP 1: Filter Match ---
        print("\n--- Step 1: Filter Match ---")
        step1_start_time = time.time()
        filter_match_doctors: List[Doctor] = []

        # Check if any filters are applied
        has_filters = any([specialization, language, gender, country, day_of_week])
        print(f"Has filters: {has_filters}")

        # Flag to track if filters were applied but no doctors were found
        filters_applied_but_no_results = False

        # Build base query
        query = db.query(Doctor)

        # Apply primary filters
        if specialization:
            query = query.filter(Doctor.specialization.ilike(f"%{specialization}%"))
        if language:
            query = query.filter(Doctor.language == language)
        if gender:
            query = query.filter(Doctor.gender == gender)
        if country:
            query = query.filter(Doctor.address.ilike(f"%{country}%"))

        # Note: min_rating is applied after fetching the doctors, not in the database query

        # Apply availability filter (optimized)
        available_doctor_ids_step1: Optional[List[Any]] = None # Store UUIDs
        if day_of_week:
            day_of_week_capitalized = day_of_week.capitalize()
            valid_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            if day_of_week_capitalized in valid_days:
                print(f"Finding doctors available on {day_of_week_capitalized}...")
                avail_ids_query = db.query(DoctorAvailability.doctor_id).filter(
                    DoctorAvailability.day_of_week == day_of_week_capitalized
                ).distinct().limit(MAX_DOCTORS_PER_QUERY) # Use the configured limit

                available_doctor_ids_step1 = [id_tuple[0] for id_tuple in avail_ids_query.all()]
                print(f"Found {len(available_doctor_ids_step1)} potential doctor IDs available.")

                if not available_doctor_ids_step1:
                     print("No doctors found available for the specified day. Filter match step will yield no results.")
                     # Set query to yield no results if availability is mandatory and none found
                     query = query.filter(Doctor.doctor_id.in_([]))
                else:
                     # Filter the main query by these available IDs
                     query = query.filter(Doctor.doctor_id.in_(available_doctor_ids_step1))
            else:
                 print(f"WARN: Invalid day_of_week '{day_of_week}' provided.")
                 # Optionally, raise HTTPException(400, "Invalid day_of_week")

        # Limit results and execute query
        filter_match_doctors = query.limit(MAX_DOCTORS_PER_QUERY).all()
        print(f"Found {len(filter_match_doctors)} doctors matching initial filters (pre-rating check, limit: {MAX_DOCTORS_PER_QUERY}).")

        # Set flag if filters were applied but no doctors were found
        filters_applied_but_no_results = has_filters and not filter_match_doctors
        if filters_applied_but_no_results:
            print("Filters were applied but no doctors were found. Will use rule-based approach instead of random.")

        if filter_match_doctors:
            # Determine the appropriate source based on whether filters are applied
            source = "filter_match" if has_filters else "rule_based"
            print(f"Using source '{source}' for initial doctors")

            # Process these doctors, applying min_rating filter internally
            filtered_results = process_doctors_batch_parallel(
                doctors_to_process=filter_match_doctors,
                patient=patient,
                min_rating=min_rating, # Apply min_rating filter for this batch
                specialization_filter=specialization,
                language_filter=language,
                country_filter=country,
                day_of_week_filter=day_of_week,
                db=db,
                source=source  # Use rule_based when no filters are applied
            )
            # Add unique results
            for doc_info in filtered_results:
                doc_id_str = doc_info["doctor_id"]
                if doc_id_str not in processed_doctor_ids_str:
                    final_results.append(doc_info)
                    processed_doctor_ids_str.add(doc_id_str)

        step1_end_time = time.time()
        print(f"Step 1 completed in {step1_end_time - step1_start_time:.4f}s. Found {len(final_results)} doctors.")


        # --- STEP 2: Rule-Based Fallback ---
        if len(final_results) < TARGET_RECOMMENDATIONS:
            print("\n--- Step 2: Rule-Based Fallback ---")
            step2_start_time = time.time()
            needed = TARGET_RECOMMENDATIONS - len(final_results)
            print(f"Need {needed} more doctors.")

            # Query for doctors not already processed
            rule_based_query = db.query(Doctor).filter(~Doctor.doctor_id.in_(
                 # Convert string IDs back to UUIDs if Doctor.doctor_id is UUID
                 # This assumes your Doctor model uses UUID for doctor_id
                 # If it uses String, just pass processed_doctor_ids_str
                 [uid for uid in processed_doctor_ids_str] # Adjust if needed based on Doctor.doctor_id type
            ))

            # Apply *key* filters for rule-based matching
            if language: # Prioritize language match
                rule_based_query = rule_based_query.filter(Doctor.language == language)
            if country: # Prioritize country match
                rule_based_query = rule_based_query.filter(Doctor.address.ilike(f"%{country}%"))

            # Apply availability filter (optimized) - reuse logic from Step 1
            available_doctor_ids_step2: Optional[List[Any]] = None # Store UUIDs
            if day_of_week:
                # Re-fetch available IDs excluding already processed ones, or filter results later
                # Simpler: fetch available IDs again and filter the query
                 day_of_week_capitalized = day_of_week.capitalize()
                 valid_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                 if day_of_week_capitalized in valid_days:
                     print(f"Finding rule-based doctors available on {day_of_week_capitalized}...")
                     avail_ids_query_step2 = db.query(DoctorAvailability.doctor_id).filter(
                         DoctorAvailability.day_of_week == day_of_week_capitalized,
                         # Exclude already processed doctors from the availability check
                         ~DoctorAvailability.doctor_id.in_([uid for uid in processed_doctor_ids_str]) # Adjust type if needed
                     ).distinct().limit(MAX_DOCTORS_PER_QUERY)

                     available_doctor_ids_step2 = [id_tuple[0] for id_tuple in avail_ids_query_step2.all()]
                     print(f"Found {len(available_doctor_ids_step2)} potential rule-based doctor IDs available.")

                     if not available_doctor_ids_step2:
                          print("No additional rule-based doctors found available for the specified day.")
                          rule_based_query = rule_based_query.filter(Doctor.doctor_id.in_([]))
                     else:
                          rule_based_query = rule_based_query.filter(Doctor.doctor_id.in_(available_doctor_ids_step2))

            # Limit and fetch potential rule-based doctors
            rule_based_doctors = rule_based_query.limit(MAX_DOCTORS_PER_QUERY).all()
            print(f"Found {len(rule_based_doctors)} potential rule-based doctors (limit: {MAX_DOCTORS_PER_QUERY}).")

            if rule_based_doctors:
                 # Process rule-based doctors (min_rating filter is NOT applied here)
                 rule_based_results = process_doctors_batch_parallel(
                     doctors_to_process=rule_based_doctors,
                     patient=patient,
                     min_rating=None, # Do not apply min_rating filter for rule-based
                     specialization_filter=specialization, # Pass filters for scoring context
                     language_filter=language,
                     country_filter=country,
                     day_of_week_filter=day_of_week,
                     db=db,
                     source="rule_based"
                 )

                 # Sort by score and add needed amount
                 rule_based_results.sort(key=lambda x: x["score"], reverse=True)
                 added_count = 0
                 for doc_info in rule_based_results:
                     if added_count >= needed:
                         break
                     doc_id_str = doc_info["doctor_id"]
                     if doc_id_str not in processed_doctor_ids_str:
                         final_results.append(doc_info)
                         processed_doctor_ids_str.add(doc_id_str)
                         added_count += 1
                 print(f"Added {added_count} rule-based doctors.")

            step2_end_time = time.time()
            print(f"Step 2 completed in {step2_end_time - step2_start_time:.4f}s. Total doctors now: {len(final_results)}.")


        # --- STEP 3: Fallback (Rule-Based or Random) ---
        if len(final_results) < TARGET_RECOMMENDATIONS:
            # Determine whether to use rule-based or random approach
            # Use rule-based when:
            # 1. Filters were applied but no doctors were found, OR
            # 2. No filters were applied (as per previous change)
            fallback_source = "rule_based" if (filters_applied_but_no_results or not has_filters) else "random"

            print(f"\n--- Step 3: {fallback_source.title()} Fallback ---")
            step3_start_time = time.time()
            needed = TARGET_RECOMMENDATIONS - len(final_results)
            print(f"Need {needed} more doctors using {fallback_source} approach.")

            # For filters that don't match any doctors, we need to clear those filters for the fallback
            if filters_applied_but_no_results:
                print("Clearing non-matching filters for fallback query")
                # Query for any doctors not already processed, ignoring the original filters
                fallback_query = db.query(Doctor).filter(~Doctor.doctor_id.in_(
                    [uid for uid in processed_doctor_ids_str] # Adjust type if needed
                ))
            else:
                # Normal case - Query for doctors not already processed
                fallback_query = db.query(Doctor).filter(~Doctor.doctor_id.in_(
                    [uid for uid in processed_doctor_ids_str] # Adjust type if needed
                ))

            # Fetch a slightly larger pool than needed
            # Using limit is more efficient than fetching all for large tables
            potential_fallback_doctors = fallback_query.limit(needed * 3).all() # Fetch 3x needed
            print(f"Found {len(potential_fallback_doctors)} potential {fallback_source} doctors.")

            # For random approach, shuffle the list
            if fallback_source == "random":
                random.shuffle(potential_fallback_doctors)

            # Take only as many as needed (with some buffer)
            doctors_to_process = potential_fallback_doctors[:needed * 2] # Process 2x needed

            if doctors_to_process:
                 # Process doctors (min_rating filter is NOT applied)
                 fallback_results = process_doctors_batch_parallel(
                     doctors_to_process=doctors_to_process,
                     patient=patient,
                     min_rating=None, # Do not apply min_rating
                     specialization_filter=specialization, # Pass filters for scoring context
                     language_filter=language,
                     country_filter=country,
                     day_of_week_filter=day_of_week,
                     db=db,
                     source=fallback_source  # Use rule_based or random based on our determination
                 )

                 # Add needed amount
                 added_count = 0
                 for doc_info in fallback_results:
                     if added_count >= needed:
                         break
                     doc_id_str = doc_info["doctor_id"]
                     # Double check uniqueness although query should handle it
                     if doc_id_str not in processed_doctor_ids_str:
                         final_results.append(doc_info)
                         processed_doctor_ids_str.add(doc_id_str)
                         added_count += 1
                 print(f"Added {added_count} {fallback_source} doctors.")

            step3_end_time = time.time()
            print(f"Step 3 completed in {step3_end_time - step3_start_time:.4f}s. Total doctors now: {len(final_results)}.")


        # --- Finalize Results ---
        print("\n--- Finalizing Results ---")
        # Final sort by score (descending)
        final_results.sort(key=lambda x: x["score"], reverse=True)

        # Truncate to target number
        final_recommendations = final_results[:TARGET_RECOMMENDATIONS]

        if len(final_recommendations) < TARGET_RECOMMENDATIONS:
             print(f"WARN: Only found {len(final_recommendations)} doctors in total, less than target {TARGET_RECOMMENDATIONS}.")

        result_payload = {"recommended_doctors": final_recommendations}

        # --- Cache Result Removed ---
        # No caching to ensure fresh recommendations every time

        request_end_time = time.time()
        total_duration = request_end_time - request_start_time
        print(f"--- Recommendation request completed in {total_duration:.4f} seconds ---")

        return result_payload

    except Exception as e:
        # Log the error for debugging
        error_details = traceback.format_exc()
        print(f"FATAL ERROR in recommend_doctors for patient {patient_id}: {str(e)}\n{error_details}")
        # Raise HTTPException for FastAPI to handle
        raise HTTPException(status_code=500, detail=f"Internal server error processing recommendations.")


# --- Other Simple Endpoints (kept from original) ---

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
        error_details = traceback.format_exc()
        print(f"Error in simple_recommendation: {str(e)}\n{error_details}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# --- Removed old/unused functions ---
# Removed get_doctor_recommendations and score_doctor functions as their logic
# is now integrated into the main endpoint and the new parallel processing functions.