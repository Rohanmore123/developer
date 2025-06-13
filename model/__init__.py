from .model_correct import (
    Base,
    engine,
    User,
    Patient,
    Doctor,
    DoctorAvailability,
    Appointment,
    Rating,
    ChatMessage,
    EmotionAnalysis,
    UserEmotionInsights,
    DiaryEntry,
)

__all__ = [
    "Base",
    "engine",
    "User",
    "Patient",
    "Doctor",
    "DoctorAvailability",
    "Appointment",
    "Rating",
    "ChatMessage",
    "EmotionAnalysis",
    "UserEmotionInsights",
    "DiaryEntry",
]



# # Create fresh schema
## Base.metadata.create_all(bind=engine)