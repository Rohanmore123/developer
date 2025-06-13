from fastapi import APIRouter, HTTPException, status
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get static JWT token from environment variables
STATIC_JWT_TOKEN = os.getenv("STATIC_JWT_TOKEN")

auth_router = APIRouter(tags=["Authentication"])

@auth_router.get("/token")
async def get_static_token():
    """
    Return the static JWT token for authentication
    """
    if not STATIC_JWT_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Static JWT token not configured",
        )

    return {
        "access_token": STATIC_JWT_TOKEN,
        "token_type": "bearer",
        "user_id": "static-user",
        "email": "static@example.com",
        "roles": "admin,doctor,patient"
    }
