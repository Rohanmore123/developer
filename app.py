from fastapi import FastAPI, Depends, HTTPException, APIRouter
from pydantic import BaseModel
from typing import Optional
import logging

from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# # Disable OpenAI and httpx logging to prevent excessive HTTP request logs
# logging.getLogger("openai").setLevel(logging.WARNING)
# logging.getLogger("httpx").setLevel(logging.WARNING)

# Initialize vector stores before importing service modules
logger.info("Initializing vector stores at application startup...")
import vector_store_initializer
vector_store_initializer.initialize_vector_stores()
logger.info("Vector store initialization complete.")

# Import service modules after vector stores are initialized
# from services.pdf_processing_optimized import optimized_pdf_router
from services.pdf_processing_specialized import specialized_pdf_router
from services.insights_service_simplified import router as insights_router
from services.doctor_insights_service import router as doctor_insights_router
from services.patient_insight_services import router as patient_treatment_insights_router
from services.chat_final import chat_final_router
from services.chat_final2 import chat_final_router
from services.emotion_service1 import emotion_router
from services.recommendation1 import recommend_router
from services.smart_chat_doc_service import smart_chat_doc_router

# Define empty routers for services that don't exist
from fastapi import APIRouter

# Import JWT authentication
from auth.auth_router import auth_router
from auth.dependencies import get_current_active_user

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum

import os

# Initialize FastAPI app
app = FastAPI(
    title="Mental Health AI Chatbot",
    description="API for Mental Health AI Chatbot with JWT Authentication",
    version="1.0.0"
)

# Initialize all vector stores at startup to avoid duplicate loading
# logger.info("Initializing all vector stores at application startup...")
# initialize_all_vector_stores()
# logger.info("Vector store initialization complete.")

# Define request models
class ChatRequest(BaseModel):
    patient_id: str
    message: str

class EmotionRequest(BaseModel):
    message: str

class RecommendationRequest(BaseModel):
    patient_id: str
    context: Optional[str] = None

class MedicalHistoryRequest(BaseModel):
    patient_id: str
    diagnosis: str
    treatment: str
    diagnosed_date: str
    additional_notes: Optional[str] = None

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include authentication router
app.include_router(auth_router, prefix="/auth", tags=["Authentication"])

# ✅ Include all API modules with JWT protection


# Public emotion router - no authentication required


# Protected emotion router - all endpoints require authentication
# Exclude the run-analysis endpoint which is handled by the public router
app.include_router(
    emotion_router,
    prefix="/emotion",
    tags=["Emotion Analysis"],
    dependencies=[Depends(get_current_active_user)]
)

app.include_router(
    recommend_router,
    prefix="/recommendation",
    tags=["Doctor Recommendation"],
    dependencies=[Depends(get_current_active_user)]
)




app.include_router(
    specialized_pdf_router,
    prefix="/pdf",  # Add prefix to match frontend expectations
    tags=["Specialized PDF Processing"]
)

# app.include_router(
#     optimized_pdf_router,
#     prefix="/pdf/optimized",  # Add prefix to match frontend expectations
#     tags=["Optimized PDF Processing"]
# )

app.include_router(
    insights_router,
    prefix="/insights",
    tags=["Patient Insights"],
    dependencies=[Depends(get_current_active_user)]
)

app.include_router(
    doctor_insights_router,
    prefix="/doctor-insights",
    tags=["Doctor Clinical Insights"],
    dependencies=[Depends(get_current_active_user)]
)

app.include_router(
    patient_treatment_insights_router,
    prefix="/treatment-insights",
    tags=["Patient Treatment Insights"],
    dependencies=[Depends(get_current_active_user)]
)

app.include_router(
    smart_chat_doc_router,
    tags=["Smart Chat for Doctors"]
    # Removed authentication dependency since we handle it in the WebSocket endpoint
)

app.include_router(
    chat_final_router,
    tags=["Multi-Persona Chat"]
    # Removed authentication dependency since we handle it in the WebSocket endpoint
)


@app.get("/")
def home():
    return {"message": "Mental Health AI Chatbot API Running ✅"}

@app.get("/health")
def health_check():
    """
    Health check endpoint that provides system status and component health.
    """
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
            "components": {
                "api": "healthy",
                "database": "unknown",
                "vector_stores": "unknown",
                "openai": "unknown",
                "s3": "unknown"
            },
            "uptime": "running",
            "environment": "production"
        }

        # Check database connection
        try:
            from database.database import SessionLocal
            db = SessionLocal()
            # Simple query to test database connection
            db.execute("SELECT 1")
            db.close()
            health_status["components"]["database"] = "healthy"
        except Exception as db_error:
            logger.warning(f"Database health check failed: {str(db_error)}")
            health_status["components"]["database"] = "unhealthy"
            health_status["status"] = "degraded"

        # Check vector stores
        try:
            import vector_store_initializer
            if hasattr(vector_store_initializer, 'vector_stores_initialized') and vector_store_initializer.vector_stores_initialized:
                health_status["components"]["vector_stores"] = "healthy"
            else:
                health_status["components"]["vector_stores"] = "initializing"
                health_status["status"] = "degraded"
        except Exception as vs_error:
            logger.warning(f"Vector store health check failed: {str(vs_error)}")
            health_status["components"]["vector_stores"] = "unhealthy"
            health_status["status"] = "degraded"

        # Check OpenAI API
        try:
            import os
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                health_status["components"]["openai"] = "configured"
            else:
                health_status["components"]["openai"] = "not_configured"
                health_status["status"] = "degraded"
        except Exception as openai_error:
            logger.warning(f"OpenAI health check failed: {str(openai_error)}")
            health_status["components"]["openai"] = "unhealthy"
            health_status["status"] = "degraded"

        # Check S3 connection
        try:
            import boto3
            import os

            s3_bucket = os.getenv("PDF_BUCKET_NAME", "prasha-health-pdf")
            aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
            aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
            aws_region = os.getenv("AWS_REGION", "us-east-1")

            if aws_access_key and aws_secret_key:
                s3_client = boto3.client(
                    's3',
                    aws_access_key_id=aws_access_key,
                    aws_secret_access_key=aws_secret_key,
                    region_name=aws_region
                )
                # Test S3 connection with a simple list_buckets call
                response = s3_client.list_buckets()
                buckets = [bucket['Name'] for bucket in response['Buckets']]

                if s3_bucket in buckets:
                    health_status["components"]["s3"] = "healthy"
                else:
                    health_status["components"]["s3"] = "bucket_not_found"
                    health_status["status"] = "degraded"
            else:
                health_status["components"]["s3"] = "not_configured"
                health_status["status"] = "degraded"
        except Exception as s3_error:
            logger.warning(f"S3 health check failed: {str(s3_error)}")
            health_status["components"]["s3"] = "unhealthy"
            health_status["status"] = "degraded"

        # Set overall status based on critical components
        critical_components = ["api", "database"]
        unhealthy_critical = [comp for comp in critical_components if health_status["components"][comp] == "unhealthy"]

        if unhealthy_critical:
            health_status["status"] = "unhealthy"
        elif health_status["status"] != "degraded":
            health_status["status"] = "healthy"

        return health_status

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "components": {
                "api": "unhealthy"
            }
        }





# ✅ Lambda Handler
# handler = Mangum(app)

# ✅ Mount Static Directory (containing HTML files)
app.mount("/static", StaticFiles(directory="static"), name="static")






# @app.get("/pdf-upload-optimized")
# def serve_pdf_upload_optimized():
#     return FileResponse(os.path.join("static", "pdf_upload_optimized.html"))


@app.get("/patient-insights")
def serve_patient_insights():
    return FileResponse(os.path.join("static", "patient_insights.html"))

@app.get("/doctor-insights")
def serve_doctor_insights():
    return FileResponse(os.path.join("static", "doctor_insights.html"))

@app.get("/treatment-journey")
def serve_patient_treatment_insights():
    return FileResponse(os.path.join("static", "treatment_journey.html"))



@app.get("/chat-final")
def serve_chat_final():
    return FileResponse(os.path.join("static", "chat_final.html"))

@app.get("/chat-final2")
def serve_chat_final():
    return FileResponse(os.path.join("static", "chat_final2.html"))

@app.get("/chat-doctor")
def serve_doctor_chat():
    return FileResponse(os.path.join("static", "chat_doctor.html"))

@app.get("/pdf-upload-specialized")
def serve_pdf_upload_specialized():
    return FileResponse(os.path.join("static", "pdf_upload_specialized.html"))