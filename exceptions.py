"""
Custom exceptions for the application
"""
from fastapi import HTTPException, status


class VTOException(HTTPException):
    """Base exception for VTO application"""
    def __init__(self, detail: str, status_code: int = status.HTTP_400_BAD_REQUEST):
        super().__init__(status_code=status_code, detail=detail)


class FileValidationError(VTOException):
    """Raised when file validation fails"""
    def __init__(self, detail: str = "Invalid file"):
        super().__init__(detail=detail, status_code=status.HTTP_400_BAD_REQUEST)


class ImageProcessingError(VTOException):
    """Raised when image processing fails"""
    def __init__(self, detail: str = "Image processing failed"):
        super().__init__(detail=detail, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)


class PoseDetectionError(VTOException):
    """Raised when pose detection fails"""
    def __init__(self, detail: str = "Pose detection failed"):
        super().__init__(detail=detail, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY)


class StorageError(VTOException):
    """Raised when storage operations fail"""
    def __init__(self, detail: str = "Storage operation failed"):
        super().__init__(detail=detail, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

