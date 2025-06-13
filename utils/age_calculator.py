from datetime import datetime

def calculate_age(dob):
    """
    Calculate age based on date of birth.
    
    Args:
        dob: Date of birth as a datetime object
        
    Returns:
        int: Age in years
    """
    if not dob:
        return 0
        
    today = datetime.now()
    
    # Calculate age
    age = today.year - dob.year
    
    # Adjust age if birthday hasn't occurred yet this year
    if (today.month, today.day) < (dob.month, dob.day):
        age -= 1
        
    return age
