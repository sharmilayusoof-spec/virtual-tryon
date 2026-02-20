"""
Application configuration and settings
"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings"""
    
    # Application
    APP_NAME: str = "VTO Backend"
    VERSION: str = "1.0.0"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 5500
    
    # Storage
    UPLOAD_DIR: str = "storage/uploads"
    PROCESSED_DIR: str = "storage/processed"
    RESULTS_DIR: str = "storage/results"
    MAX_UPLOAD_SIZE: int = 10485760  # 10MB
    
    # ML Models
    POSE_MODEL_CONFIDENCE: float = 0.5
    SEGMENTATION_THRESHOLD: float = 0.5
    
    # CORS
    CORS_ORIGINS: list = ["*"]  # Allow all origins in development
    
    # Pixa API
    PIXA_API_KEY: Optional[str] = None
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

