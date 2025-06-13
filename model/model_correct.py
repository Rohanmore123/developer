import uuid
from datetime import datetime, date, time
from sqlalchemy import (
    create_engine, Column, String, Integer, Boolean, ForeignKey, Text,
    Date, Time, Enum, DateTime, Float, DECIMAL, Numeric, UUID
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.dialects.postgresql import JSONB
import enum
import os
from dotenv import load_dotenv

load_dotenv()

# Get database connection details from environment variables
username = os.getenv("postgres_username")
password = os.getenv("postgres_password")
host = os.getenv("postgres_host")
port = os.getenv("postgres_port")
database = os.getenv("postgres_database")

# Construct the database URL
DATABASE_URL = f"postgresql://{username}:{password}@{host}:{port}/{database}"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Enum for Gender
class GenderEnum(str, enum.Enum):
    M = "M"
    F = "F"
    Other = "Other"

# Enum for Appointment Status
class AppointmentStatusEnum(str, enum.Enum):
    Scheduled = "Scheduled"
    Completed = "Completed"
    Cancelled = "Cancelled"

# Enum for Prescription Status
class PrescriptionStatusEnum(str, enum.Enum):
    Active = "Active"
    Completed = "Completed"
    Discontinued = "Discontinued"

# Enum for Message Type
class MessageTypeEnum(str, enum.Enum):
    P = "P"  # Patient
    D = "D"  # Doctor
    S = "S"  # System

# ---------------- USERS ----------------
class User(Base):
    __tablename__ = "users"

    user_id = Column(UUID, primary_key=True)
    title = Column(String(50))
    first_name = Column(String(100), nullable=False)
    middle_name = Column(String(100))
    last_name = Column(String(100), nullable=False)
    gender = Column(String(5))
    email = Column(String(255), nullable=False)
    password_hash = Column(Text, nullable=False)
    roles = Column(String(100), nullable=False)
    profile_picture = Column(Text)
    is_active = Column(Boolean)
    last_login = Column(DateTime)
    is_deleted = Column(Boolean)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)

    patients = relationship("Patient", back_populates="user")
    doctors = relationship("Doctor", back_populates="user")

    def __repr__(self):
        return f"<User(user_id={self.user_id}, email={self.email})>"

# ---------------- PATIENTS ----------------
class Patient(Base):
    __tablename__ = "patients"

    patient_id = Column(UUID, primary_key=True)
    user_id = Column(UUID, ForeignKey("users.user_id"), nullable=False)
    title = Column(String(50))
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    dob = Column(DateTime(timezone=True), nullable=False)
    gender = Column(String(5), nullable=False)
    language = Column(String(50))
    religion = Column(String(50))  # Note: Column name is 'religion' in patients table
    address = Column(String(255))
    phone = Column(String(15))
    health_score = Column(Integer)
    under_medications = Column(Boolean)
    created_at = Column(DateTime(timezone=True))
    updated_at = Column(DateTime(timezone=True))
    region = Column(String(100))
    isOnboarded = Column(Boolean)
    onboardingCompletedAt = Column(DateTime(timezone=True))
    timezone = Column(String(50))
    preferences = Column(JSONB)
    interests = Column(JSONB)
    treatment = Column(JSONB)

    user = relationship("User", back_populates="patients")
    appointments = relationship("Appointment", back_populates="patient")
    medical_history = relationship("MedicalHistory", back_populates="patient")
    prescriptions = relationship("Prescription", back_populates="patient")
    diaries = relationship("MyDiary", back_populates="patient")
    timelines = relationship("MyTimeline", back_populates="patient")

    # No need for religion property since the column is already named 'religion'

    def __repr__(self):
        return f"<Patient(patient_id={self.patient_id}, name={self.first_name} {self.last_name})>"

# ---------------- DOCTORS ----------------
class Doctor(Base):
    __tablename__ = "doctors"

    doctor_id = Column(UUID, primary_key=True)
    user_id = Column(UUID, ForeignKey("users.user_id"), nullable=False)
    title = Column(String(50))
    first_name = Column(String(100), nullable=False)
    middle_name = Column(String(100))
    last_name = Column(String(100), nullable=False)
    dob = Column(Date, nullable=False)
    age = Column(Integer)
    gender = Column(String(5), nullable=False)
    language = Column(String(50))
    relligion = Column(String(50))  # Note: Column name has two 'l's in the database
    address = Column(Text)
    phone = Column(String(15))
    email = Column(String(255))
    interest = Column(Text)  # Note: Column name is singular in the database
    specialization = Column(Text)
    cosultation_fee = Column(Numeric(10, 2))  # Note: Column name is missing an 'n' in the database
    treatment = Column(Text, nullable=False)
    health_score = Column(Integer)
    under_medications = Column(Boolean)
    createdAt = Column(DateTime)  # Note: Column name uses camelCase in the database
    updatedAt = Column(DateTime)  # Note: Column name uses camelCase in the database

    user = relationship("User", back_populates="doctors")
    appointments = relationship("Appointment", back_populates="doctor")
    availability = relationship("DoctorAvailability", back_populates="doctor")
    prescriptions = relationship("Prescription", back_populates="doctor")
    medical_history = relationship("MedicalHistory", back_populates="doctor")

    # Add properties for backward compatibility with code that uses 'religion' and 'interests'
    @property
    def religion(self):
        return self.relligion

    @religion.setter
    def religion(self, value):
        self.relligion = value

    @property
    def interests(self):
        return self.interest

    @interests.setter
    def interests(self, value):
        self.interest = value

    @property
    def consultation_fee(self):
        return self.cosultation_fee

    @consultation_fee.setter
    def consultation_fee(self, value):
        self.cosultation_fee = value

    @property
    def created_at(self):
        return self.createdAt

    @created_at.setter
    def created_at(self, value):
        self.createdAt = value

    @property
    def updated_at(self):
        return self.updatedAt

    @updated_at.setter
    def updated_at(self, value):
        self.updatedAt = value

    def __repr__(self):
        return f"<Doctor(doctor_id={self.doctor_id}, name={self.first_name} {self.last_name})>"

# ---------------- APPOINTMENTS ----------------
class Appointment(Base):
    __tablename__ = "appointments"

    appointment_id = Column(UUID, primary_key=True)
    patient_id = Column(UUID, ForeignKey("patients.patient_id"))
    doctor_id = Column(UUID, ForeignKey("doctors.doctor_id"))
    appointment_date = Column(Date, nullable=False)
    appointment_time = Column(Time, nullable=False)
    visit_reason = Column(Text)
    consultation_type = Column(String(50))
    status = Column(String(9))
    notes = Column(Text)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)

    patient = relationship("Patient", back_populates="appointments")
    doctor = relationship("Doctor", back_populates="appointments")

    def __repr__(self):
        return f"<Appointment(appointment_id={self.appointment_id}, date={self.appointment_date}, time={self.appointment_time})>"

# ---------------- DOCTOR AVAILABILITY ----------------
class DoctorAvailability(Base):
    __tablename__ = "doctors_availability"

    availability_id = Column(UUID, primary_key=True)
    doctor_id = Column(UUID, ForeignKey("doctors.doctor_id"))
    day_of_week = Column(String(20), nullable=False)
    start_time = Column(Time, nullable=False)
    end_time = Column(Time, nullable=False)

    doctor = relationship("Doctor", back_populates="availability")

    def __repr__(self):
        return f"<DoctorAvailability(availability_id={self.availability_id}, day={self.day_of_week})>"

# ---------------- PRESCRIPTIONS ----------------
class Prescription(Base):
    __tablename__ = "prescriptions"

    prescription_id = Column(UUID, primary_key=True)
    patient_id = Column(UUID, ForeignKey("patients.patient_id"))
    doctor_id = Column(UUID, ForeignKey("doctors.doctor_id"))
    medication_name = Column(String(255), nullable=False)
    dosage = Column(String(100), nullable=False)
    instructions = Column(Text)
    start_date = Column(Date, nullable=False)
    end_date = Column(Date)
    status = Column(String(12))
    created_at = Column(DateTime)
    updated_at = Column(DateTime)

    patient = relationship("Patient", back_populates="prescriptions")
    doctor = relationship("Doctor", back_populates="prescriptions")

    def __repr__(self):
        return f"<Prescription(prescription_id={self.prescription_id}, medication={self.medication_name})>"

# ---------------- MEDICAL HISTORY ----------------
class MedicalHistory(Base):
    __tablename__ = "medicalhistory"

    history_id = Column(UUID, primary_key=True)
    patient_id = Column(UUID, ForeignKey("patients.patient_id"))
    diagnosis = Column(Text, nullable=False)
    treatment = Column(Text)
    diagnosed_date = Column(Date, nullable=False)
    doctor_id = Column(UUID, ForeignKey("doctors.doctor_id"))
    additional_notes = Column(Text)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)

    patient = relationship("Patient", back_populates="medical_history")
    doctor = relationship("Doctor", back_populates="medical_history")

    def __repr__(self):
        return f"<MedicalHistory(history_id={self.history_id}, diagnosis={self.diagnosis})>"

# ---------------- MY DIARY ----------------
class MyDiary(Base):
    __tablename__ = "mydiary"

    event_id = Column(UUID, primary_key=True)
    patient_id = Column(UUID, ForeignKey("patients.patient_id"))
    notes = Column(Text, nullable=False)
    message_type = Column(String(5), nullable=False)
    additional_notes = Column(Text)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)

    patient = relationship("Patient", back_populates="diaries")

    def __repr__(self):
        return f"<MyDiary(event_id={self.event_id}, patient_id={self.patient_id})>"

# ---------------- MY TIMELINE ----------------
class MyTimeline(Base):
    __tablename__ = "mytimeline"

    timeline_id = Column(UUID, primary_key=True)
    patient_id = Column(UUID, ForeignKey("patients.patient_id"))
    timeline_name = Column(String(200), nullable=False)
    timeline_description = Column(Text)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)

    patient = relationship("Patient", back_populates="timelines")
    events = relationship("MyTimelineEvent", back_populates="timeline")

    def __repr__(self):
        return f"<MyTimeline(timeline_id={self.timeline_id}, name={self.timeline_name})>"

# ---------------- MY TIMELINE EVENTS ----------------
class MyTimelineEvent(Base):
    __tablename__ = "mytimelineevents"

    event_id = Column(UUID, primary_key=True)
    timeline_id = Column(UUID, ForeignKey("mytimeline.timeline_id"))
    patient_id = Column(UUID, ForeignKey("patients.patient_id"))
    event_type = Column(String(50), nullable=False)
    event_name = Column(String(100), nullable=False)
    event_description = Column(Text)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)

    timeline = relationship("MyTimeline", back_populates="events")

    def __repr__(self):
        return f"<MyTimelineEvent(event_id={self.event_id}, name={self.event_name})>"

# ---------------- CHAT MESSAGES ----------------
class ChatMessage(Base):
    __tablename__ = "chat_messages"

    chat_message_id = Column(String, primary_key=True)
    sender_id = Column(String, nullable=False)
    receiver_id = Column(String, nullable=False)
    message_text = Column(Text, nullable=False)
    extracted_keywords = Column(Text)
    # Keep this for backward compatibility
    # user_id = Column(String)
    createdAt = Column(DateTime, nullable=False, default=datetime.now)  # Required non-null field in the database
    # updatedAt = Column(DateTime, nullable=False, default=datetime.now)  # Required non-null field in the database

    emotion_analysis = relationship("EmotionAnalysis", back_populates="chat_message")

    def __repr__(self):
        return f"<ChatMessage(chat_message_id={self.chat_message_id}, sender={self.sender_id})>"

class DiaryEntry(Base):
    __tablename__ = "diary_entries"

    event_id = Column(String, primary_key=True)
    patient_id = Column(String, ForeignKey("patients.patient_id"), nullable=False)
    notes = Column(Text, nullable=False)
    created_at = Column(DateTime)

    def __repr__(self):
        return f"<DiaryEntry(id={self.event_id}, patient_id={self.patient_id})>"

# ---------------- EMOTION ANALYSIS ----------------
class EmotionAnalysis(Base):
    __tablename__ = "emotion_analysis"

    emotion_id = Column(String, primary_key=True)
    chat_message_id = Column(String, ForeignKey("chat_messages.chat_message_id"))
    patient_id = Column(String, nullable=False)
    emotion_category = Column(String, nullable=False)
    confidence_score = Column(DECIMAL(5, 2), nullable=False)
    analyzed_at = Column(DateTime)

    chat_message = relationship("ChatMessage", back_populates="emotion_analysis")

    def __repr__(self):
        return f"<EmotionAnalysis(emotion_id={self.emotion_id}, emotion={self.emotion_category})>"

# ---------------- USER EMOTION INSIGHTS ----------------
class UserEmotionInsights(Base):
    __tablename__ = "user_emotion_insights"

    insight_id = Column(String, primary_key=True)
    patient_id = Column(String, nullable=False)
    date = Column(Date)
    dominant_emotion = Column(String, nullable=False)
    emotion_trend = Column(String)
    major_shifts = Column(String)

    def __repr__(self):
        return f"<UserEmotionInsights(insight_id={self.insight_id}, emotion={self.dominant_emotion})>"

# ---------------- RATINGS ----------------
class Rating(Base):
    __tablename__ = "ratings"

    rating_id = Column(String, primary_key=True)
    patient_id = Column(String, nullable=False)
    doctor_id = Column(String, nullable=False)  # This is a String, not a UUID
    rating = Column(Integer, nullable=False)
    review = Column(Text)
    created_at = Column(DateTime)

    def __repr__(self):
        return f"<Rating(rating_id={self.rating_id}, rating={self.rating})>"

# ---------------- ONBOARDING QUESTIONS ----------------
class OnboardingQuestion(Base):
    __tablename__ = "onboarding_questions"

    question_id = Column(UUID, primary_key=True, default=uuid.uuid4)
    patient_id = Column(String, nullable=False)
    question = Column(Text, nullable=False)
    answer = Column(Text)
    category = Column(String(100))  # Category of the question (e.g., medical history, preferences)
    timestamp = Column(DateTime, default=datetime.now)

    def __repr__(self):
        return f"<OnboardingQuestion(question_id={self.question_id}, patient_id={self.patient_id})>"

