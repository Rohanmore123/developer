import jwt
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get JWT settings from environment variables
JWT_SECRET = os.getenv("JWT_SECRET")
if not JWT_SECRET:
    import logging
    logging.warning("JWT_SECRET environment variable not set! Using an insecure default for development only.")
    JWT_SECRET = "INSECURE_JWT_SECRET_FOR_DEVELOPMENT_ONLY_PLEASE_SET_ENV_VARIABLE"

JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
STATIC_JWT_TOKEN = os.getenv("STATIC_JWT_TOKEN")

# Security scheme for Swagger UI
security = HTTPBearer()

def verify_token(token: str):
    """
    Verify a JWT token and return the decoded payload
    """
    try:
        # If a static token is configured, check if the provided token matches
        if STATIC_JWT_TOKEN and token == STATIC_JWT_TOKEN:
            # For static tokens, we trust it and return a default payload
            return {
                "sub": "static-user",
                "email": "static@example.com",
                "roles": "admin,doctor,patient"
            }

        # Otherwise, decode and verify the token normally
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Get the current user from the JWT token
    """
    token = credentials.credentials
    payload = verify_token(token)
    return payload
