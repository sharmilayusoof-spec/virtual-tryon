"""
Pydantic models for request/response validation
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = "healthy"
    timestamp: str
    version: str


class UploadResponse(BaseModel):
    """File upload response"""
    file_id: str
    filename: str
    size: int
    content_type: str
    upload_time: str


class TryOnRequest(BaseModel):
    """Try-on request (for reference)"""
    user_image_id: Optional[str] = None
    cloth_image_id: Optional[str] = None


class TryOnResponse(BaseModel):
    """Try-on response"""
    status: str
    image_url: str
    time_taken: float
    metadata: Optional[Dict[str, Any]] = None


class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    detail: str
    timestamp: str


class PoseDetectionResponse(BaseModel):
    """Pose detection response"""
    landmarks: List[Dict[str, float]]
    confidence: float
    pose_detected: bool
    metadata: Optional[Dict[str, Any]] = None


class StorageInfoResponse(BaseModel):
    """Storage information response"""
    total_files: int
    total_size: int
    uploads: int
    processed: int
    results: int

