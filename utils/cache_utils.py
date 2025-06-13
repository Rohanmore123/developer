"""
Caching utilities for API responses.
"""
import time
from datetime import datetime, timedelta
import logging
from typing import Dict, Any, Optional, Callable
import threading

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cache dictionary: key -> (data, expiry_timestamp)
_CACHE: Dict[str, tuple] = {}
_CACHE_LOCK = threading.Lock()

def get_cache_key(prefix: str, *args) -> str:
    """
    Generate a cache key from prefix and arguments.

    Args:
        prefix: A prefix for the cache key (e.g., 'patient_insights')
        *args: Additional arguments to include in the key

    Returns:
        A string cache key
    """
    return f"{prefix}:{':'.join(str(arg) for arg in args)}"

def get_from_cache(key: str) -> Optional[Any]:
    """
    Get data from cache if it exists and is not expired.

    Args:
        key: The cache key

    Returns:
        The cached data or None if not found or expired
    """
    with _CACHE_LOCK:
        if key not in _CACHE:
            return None

        data, expiry = _CACHE[key]
        current_time = time.time()

        if current_time > expiry:
            # Cache expired, remove it
            del _CACHE[key]
            return None

        return data

def set_in_cache(key: str, data: Any, expiry_days: int = 1) -> None:
    """
    Store data in cache with expiry time.

    Args:
        key: The cache key
        data: The data to cache
        expiry_days: Number of days until the cache expires (default: 1)
    """
    # Calculate expiry timestamp
    expiry = time.time() + (expiry_days * 24 * 60 * 60)

    with _CACHE_LOCK:
        _CACHE[key] = (data, expiry)

    # Log cache operation
    expiry_time = datetime.fromtimestamp(expiry).strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f"Cached data with key '{key}' until {expiry_time}")

def clear_cache() -> None:
    """Clear all cached data."""
    with _CACHE_LOCK:
        _CACHE.clear()
    logger.info("Cache cleared")

def with_daily_cache(prefix: str) -> Callable:
    """
    Decorator to cache function results for one day.

    Args:
        prefix: A prefix for the cache key

    Returns:
        Decorated function with caching
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs) -> Any:
            # Generate cache key
            cache_key = get_cache_key(prefix, *args)

            # Try to get from cache
            cached_data = get_from_cache(cache_key)
            if cached_data is not None:
                logger.info(f"Cache hit for key '{cache_key}'")
                return cached_data

            # Cache miss, call the function
            logger.info(f"Cache miss for key '{cache_key}'")
            result = func(*args, **kwargs)

            # Store in cache
            set_in_cache(cache_key, result)

            return result
        return wrapper
    return decorator

def with_selective_cache(prefix: str, arg_indices: list) -> Callable:
    """
    Decorator to cache function results for one day, using only selected arguments for the cache key.
    Will not cache error responses to prevent error persistence.

    Args:
        prefix: A prefix for the cache key
        arg_indices: List of argument indices to include in the cache key

    Returns:
        Decorated function with caching
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs) -> Any:
            # Generate cache key using only selected arguments
            selected_args = [args[i] for i in arg_indices if i < len(args)]
            cache_key = get_cache_key(prefix, *selected_args)

            # Try to get from cache
            cached_data = get_from_cache(cache_key)
            if cached_data is not None:
                logger.info(f"Cache hit for key '{cache_key}'")

                # Check if cached data contains error insights
                if hasattr(cached_data, 'insights') and cached_data.insights:
                    # Check if any insight has "Error" in the title
                    has_error = any(
                        insight.title and "Error" in insight.title
                        for insight in cached_data.insights
                    )

                    if has_error:
                        logger.info(f"Cached data contains error insights. Ignoring cache for key '{cache_key}'")
                        # Don't return cached error, continue to regenerate
                    else:
                        # Return valid cached data
                        return cached_data
                else:
                    # Return cached data if it doesn't have insights attribute
                    return cached_data

            # Cache miss or error in cache, call the function
            logger.info(f"Cache miss for key '{cache_key}'")
            result = func(*args, **kwargs)

            # Only cache if result doesn't contain error insights
            should_cache = True
            if hasattr(result, 'insights') and result.insights:
                # Check if any insight has "Error" in the title
                has_error = any(
                    insight.title and "Error" in insight.title
                    for insight in result.insights
                )

                if has_error:
                    logger.info(f"Result contains error insights. Not caching for key '{cache_key}'")
                    should_cache = False

            # Store in cache if it's not an error response
            if should_cache:
                set_in_cache(cache_key, result)
                logger.info(f"Cached result for key '{cache_key}'")

            return result
        return wrapper
    return decorator
