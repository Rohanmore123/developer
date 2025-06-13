from .emotion_service1 import emotion_router
from .recommendation1 import recommend_router
from .chat_final import chat_final_router
from .pdf_processing_optimized import optimized_pdf_router
# from .insights_service_simplified import insights_router

__all__ = ["emotion_router",
           "recommend_router",
           "chat_final_router",
           "optimized_pdf_router"
           # "insights_router",
           ]
