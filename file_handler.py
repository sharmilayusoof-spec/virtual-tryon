"""
File handling utilities
"""
import os
import uuid
import aiofiles
from pathlib import Path
from typing import Optional
from fastapi import UploadFile
from app.core.config import settings
from app.core.exceptions import FileValidationError, StorageError


ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}
MAX_FILE_SIZE = settings.MAX_UPLOAD_SIZE


async def save_upload_file(file: UploadFile, directory: str) -> tuple[str, str]:
    """
    Save uploaded file to specified directory
    
    Args:
        file: Uploaded file
        directory: Target directory
        
    Returns:
        Tuple of (file_id, file_path)
    """
    # Validate file
    validate_file(file)
    
    # Generate unique file ID
    file_id = str(uuid.uuid4())
    file_ext = Path(file.filename or "").suffix.lower()
    filename = f"{file_id}{file_ext}"
    
    # Create directory if not exists
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    
    # Save file
    file_path = dir_path / filename
    try:
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
    except Exception as e:
        raise StorageError(f"Failed to save file: {str(e)}")
    
    return file_id, str(file_path)


def validate_file(file: UploadFile) -> None:
    """
    Validate uploaded file
    
    Args:
        file: Uploaded file
        
    Raises:
        FileValidationError: If validation fails
    """
    if not file.filename:
        raise FileValidationError("No filename provided")
    
    # Check extension
    file_ext = Path(file.filename or "").suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise FileValidationError(
            f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Check content type (more lenient - allow missing or generic types)
    if file.content_type and not (
        file.content_type.startswith('image/') or
        file.content_type == 'application/octet-stream'
    ):
        raise FileValidationError(f"File must be an image, got: {file.content_type}")


def get_file_path(file_id: str, directory: str) -> Optional[str]:
    """
    Get file path by ID
    
    Args:
        file_id: File ID
        directory: Directory to search
        
    Returns:
        File path if found, None otherwise
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        return None
    
    for ext in ALLOWED_EXTENSIONS:
        file_path = dir_path / f"{file_id}{ext}"
        if file_path.exists():
            return str(file_path)
    
    return None


def delete_file(file_path: str) -> bool:
    """
    Delete file
    
    Args:
        file_path: Path to file
        
    Returns:
        True if deleted, False otherwise
    """
    try:
        path = Path(file_path)
        if path.exists():
            path.unlink()
            return True
        return False
    except Exception:
        return False


def ensure_directories() -> None:
    """Ensure all required directories exist"""
    directories = [
        settings.UPLOAD_DIR,
        settings.PROCESSED_DIR,
        settings.RESULTS_DIR
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

