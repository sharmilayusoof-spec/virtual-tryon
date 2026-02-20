"""
Image processing utilities
"""
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional
from app.core.exceptions import ImageProcessingError


def load_image(image_path: str) -> np.ndarray:
    """
    Load image from file
    
    Args:
        image_path: Path to image
        
    Returns:
        Image as numpy array (BGR)
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ImageProcessingError(f"Failed to load image: {image_path}")
        return img
    except Exception as e:
        raise ImageProcessingError(f"Error loading image: {str(e)}")


def save_image(image: np.ndarray, output_path: str) -> None:
    """
    Save image to file
    
    Args:
        image: Image as numpy array
        output_path: Output file path
    """
    try:
        # Ensure directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, image)
    except Exception as e:
        raise ImageProcessingError(f"Error saving image: {str(e)}")


def resize_image(image: np.ndarray, target_size: Tuple[int, int], 
                 maintain_aspect: bool = True) -> np.ndarray:
    """
    Resize image
    
    Args:
        image: Input image
        target_size: Target (width, height)
        maintain_aspect: Whether to maintain aspect ratio
        
    Returns:
        Resized image
    """
    try:
        if maintain_aspect:
            h, w = image.shape[:2]
            target_w, target_h = target_size
            
            # Calculate scaling factor
            scale = min(target_w / w, target_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Resize
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Create canvas and center image
            canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            y_offset = (target_h - new_h) // 2
            x_offset = (target_w - new_w) // 2
            canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            
            return canvas
        else:
            return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    except Exception as e:
        raise ImageProcessingError(f"Error resizing image: {str(e)}")


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image to [0, 1] range
    
    Args:
        image: Input image
        
    Returns:
        Normalized image
    """
    return image.astype(np.float32) / 255.0


def denormalize_image(image: np.ndarray) -> np.ndarray:
    """
    Denormalize image from [0, 1] to [0, 255]
    
    Args:
        image: Normalized image
        
    Returns:
        Denormalized image
    """
    return (image * 255).astype(np.uint8)


def convert_to_rgb(image: np.ndarray) -> np.ndarray:
    """
    Convert BGR to RGB
    
    Args:
        image: BGR image
        
    Returns:
        RGB image
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def convert_to_bgr(image: np.ndarray) -> np.ndarray:
    """
    Convert RGB to BGR
    
    Args:
        image: RGB image
        
    Returns:
        BGR image
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def blend_images(base: np.ndarray, overlay: np.ndarray, 
                 alpha: float = 0.7) -> np.ndarray:
    """
    Blend two images
    
    Args:
        base: Base image
        overlay: Overlay image
        alpha: Blend factor (0-1)
        
    Returns:
        Blended image
    """
    try:
        # Ensure same size
        if base.shape != overlay.shape:
            overlay = cv2.resize(overlay, (base.shape[1], base.shape[0]))
        
        return cv2.addWeighted(base, 1 - alpha, overlay, alpha, 0)
    except Exception as e:
        raise ImageProcessingError(f"Error blending images: {str(e)}")

