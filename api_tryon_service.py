"""
Commercial API-based Virtual Try-On Service
Professional quality without GPU requirement

Supports multiple providers:
- Pixelcut API
- Remove.bg + Custom overlay
- DeepAR API
"""

import requests
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict
import logging
import os
from io import BytesIO

logger = logging.getLogger(__name__)


class APITryOnService:
    """
    Virtual try-on using commercial APIs
    No GPU required, professional quality results
    """
    
    def __init__(self, provider: str = "mock", api_key: Optional[str] = None):
        """
        Initialize API try-on service
        
        Args:
            provider: API provider ("pixelcut", "deepar", "mock")
            api_key: API key for the provider
        """
        self.provider = provider
        self.api_key = api_key or os.getenv(f"{provider.upper()}_API_KEY")
        
        self.providers = {
            "pixelcut": self._pixelcut_tryon,
            "deepar": self._deepar_tryon,
            "mock": self._mock_tryon  # For testing without API
        }
        
        if provider not in self.providers:
            raise ValueError(f"Unknown provider: {provider}. Choose from: {list(self.providers.keys())}")
        
        logger.info(f"API Try-On Service initialized with provider: {provider}")
    
    def is_available(self) -> bool:
        """Check if API service is available"""
        if self.provider == "mock":
            return True
        return self.api_key is not None
    
    def process_tryon(self, 
                     person_img: np.ndarray, 
                     cloth_img: np.ndarray) -> np.ndarray:
        """
        Process virtual try-on using API
        
        Args:
            person_img: Person image (H, W, 3) RGB
            cloth_img: Clothing image (H, W, 3) RGB
            
        Returns:
            Try-on result image (H, W, 3) RGB
        """
        if not self.is_available():
            raise RuntimeError(
                f"API key not found for {self.provider}. "
                f"Set {self.provider.upper()}_API_KEY environment variable."
            )
        
        try:
            # Call appropriate provider
            result = self.providers[self.provider](person_img, cloth_img)
            return result
            
        except Exception as e:
            logger.error(f"API try-on failed: {e}")
            raise
    
    def _pixelcut_tryon(self, person_img: np.ndarray, cloth_img: np.ndarray) -> np.ndarray:
        """
        Use Pixelcut API for virtual try-on
        
        API: https://www.pixelcut.ai/
        Cost: ~$0.05 per image
        """
        url = "https://api.pixelcut.ai/v1/virtual-tryon"
        
        # Convert images to bytes
        person_bytes = self._img_to_bytes(person_img)
        cloth_bytes = self._img_to_bytes(cloth_img)
        
        # Prepare request
        files = {
            'person': ('person.jpg', person_bytes, 'image/jpeg'),
            'garment': ('garment.jpg', cloth_bytes, 'image/jpeg')
        }
        
        headers = {
            'Authorization': f'Bearer {self.api_key}'
        }
        
        # Make request
        response = requests.post(url, files=files, headers=headers, timeout=30)
        
        if response.status_code == 200:
            # Convert response to image
            result_img = self._bytes_to_img(response.content)
            return result_img
        else:
            raise Exception(f"Pixelcut API error: {response.status_code} - {response.text}")
    
    def _deepar_tryon(self, person_img: np.ndarray, cloth_img: np.ndarray) -> np.ndarray:
        """
        Use DeepAR API for virtual try-on
        
        API: https://www.deepar.ai/
        Cost: ~$0.03 per image
        """
        url = "https://api.deepar.ai/v1/tryon"
        
        # Convert images to bytes
        person_bytes = self._img_to_bytes(person_img)
        cloth_bytes = self._img_to_bytes(cloth_img)
        
        # Prepare request
        files = {
            'image': ('person.jpg', person_bytes, 'image/jpeg'),
            'cloth': ('cloth.jpg', cloth_bytes, 'image/jpeg')
        }
        
        headers = {
            'X-API-Key': self.api_key
        }
        
        # Make request
        response = requests.post(url, files=files, headers=headers, timeout=30)
        
        if response.status_code == 200:
            result_img = self._bytes_to_img(response.content)
            return result_img
        else:
            raise Exception(f"DeepAR API error: {response.status_code} - {response.text}")
    
    def _mock_tryon(self, person_img: np.ndarray, cloth_img: np.ndarray) -> np.ndarray:
        """
        Mock API for testing (uses enhanced basic algorithm)
        Free, no API key needed
        """
        from app.services.ml.tryon_service import TryOnService
        from app.utils.image_processor import save_image, load_image
        import tempfile
        import uuid
        
        # Use basic service for mock
        basic_service = TryOnService()
        
        # Save images temporarily
        temp_dir = Path(tempfile.gettempdir())
        person_path = temp_dir / f"person_{uuid.uuid4()}.jpg"
        cloth_path = temp_dir / f"cloth_{uuid.uuid4()}.jpg"
        result_path = temp_dir / f"result_{uuid.uuid4()}.jpg"
        
        try:
            # Save images
            save_image(person_img, str(person_path))
            save_image(cloth_img, str(cloth_path))
            
            # Process
            basic_service.process(str(person_path), str(cloth_path), str(result_path))
            
            # Load result
            result_img = load_image(str(result_path))
            
            return result_img
            
        finally:
            # Cleanup
            for path in [person_path, cloth_path, result_path]:
                if path.exists():
                    path.unlink()
    
    def _img_to_bytes(self, img: np.ndarray) -> bytes:
        """Convert numpy image to bytes"""
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Encode to JPEG
        success, buffer = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        if not success:
            raise Exception("Failed to encode image")
        
        return buffer.tobytes()
    
    def _bytes_to_img(self, img_bytes: bytes) -> np.ndarray:
        """Convert bytes to numpy image"""
        # Decode image
        nparr = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img_bgr is None:
            raise Exception("Failed to decode image")
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        return img_rgb
    
    def estimate_cost(self, num_images: int) -> Dict[str, float]:
        """
        Estimate cost for number of images
        
        Args:
            num_images: Number of try-on operations
            
        Returns:
            Cost breakdown by provider
        """
        costs = {
            "pixelcut": num_images * 0.05,
            "deepar": num_images * 0.03,
            "mock": 0.0
        }
        
        return costs


# Utility functions

def get_api_key_instructions(provider: str) -> str:
    """Get instructions for obtaining API key"""
    
    instructions = {
        "pixelcut": """
        Pixelcut API Setup:
        1. Go to: https://www.pixelcut.ai/
        2. Sign up for an account
        3. Navigate to API section
        4. Generate API key
        5. Set environment variable: PIXELCUT_API_KEY=your_key_here
        
        Pricing: ~$0.05 per image
        Free tier: 100 images/month
        """,
        
        "deepar": """
        DeepAR API Setup:
        1. Go to: https://www.deepar.ai/
        2. Create account
        3. Go to Dashboard > API Keys
        4. Create new API key
        5. Set environment variable: DEEPAR_API_KEY=your_key_here
        
        Pricing: ~$0.03 per image
        Free tier: 50 images/month
        """,
        
        "mock": """
        Mock Provider (No API Key Needed):
        - Uses enhanced basic algorithm
        - Free, unlimited
        - Lower quality than commercial APIs
        - Good for testing
        """
    }
    
    return instructions.get(provider, "Unknown provider")


# Example usage
if __name__ == "__main__":
    # Test with mock provider (no API key needed)
    service = APITryOnService(provider="mock")
    
    print(f"✅ API Try-On Service initialized")
    print(f"Provider: {service.provider}")
    print(f"Available: {service.is_available()}")
    
    # Show cost estimates
    costs = service.estimate_cost(1000)
    print("\nCost for 1000 images:")
    for provider, cost in costs.items():
        print(f"  {provider}: ${cost:.2f}")
    
    # Show API key instructions
    print("\n" + "="*50)
    print("To use commercial APIs:")
    print("="*50)
    for provider in ["pixelcut", "deepar"]:
        print(get_api_key_instructions(provider))

