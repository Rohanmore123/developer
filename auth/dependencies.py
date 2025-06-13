from fastapi import Depends, HTTPException, status
from auth.jwt_handler import get_current_user
from typing import List, Optional

def get_current_active_user(current_user: dict = Depends(get_current_user)):
    """
    Get the current active user
    """
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return current_user

def has_role(required_roles: List[str]):
    """
    Check if the current user has the required role
    """
    def role_checker(current_user: dict = Depends(get_current_active_user)):
        user_roles = current_user.get("roles", "").split(",")
        for role in required_roles:
            if role in user_roles:
                return current_user
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions",
        )
    return role_checker
