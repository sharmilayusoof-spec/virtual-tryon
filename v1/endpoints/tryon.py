"""
Try-on endpoints
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from datetime import datetime
import uuid
from pathlib import Path
from app.models.schemas import TryOnResponse
from app.services.ml.tryon_service import TryOnService
from app.services.ml.api_tryon_service import APITryOnService
from app.utils.file_handler import save_upload_file, validate_file
from app.utils.image_processor import load_image
from app.core.config import settings
from app.core.exceptions import ImageProcessingError, PoseDetectionError
import logging

logger = logging.getLogger(__name__)

router = APIRouter()
api_service = APITryOnService(provider="pixelcut")
tryon_service = TryOnService()
api_service = APITryOnService(provider="mock")  # Using mock for now (free)


@router.post("/process", response_model=TryOnResponse)
async def process_tryon(
    user_image: UploadFile = File(..., description="User photo"),
    cloth_image: UploadFile = File(..., description="Clothing image"),
    use_api: str = Form(default="false")
):
    """
    Process virtual try-on
    
    Upload user photo and clothing image to generate try-on result.
    Set use_api=true for better quality (may have costs with commercial APIs)
    """
    import time
    start_time = time.time()
    
    try:
        logger.info(f"=== Try-on Request Received ===")
        logger.info(f"User image: {user_image.filename if user_image else 'None'}, content_type: {user_image.content_type if user_image else 'None'}")
        logger.info(f"Cloth image: {cloth_image.filename if cloth_image else 'None'}, content_type: {cloth_image.content_type if cloth_image else 'None'}")
        logger.info(f"Use API: {use_api}")
        
        # Validate files
        validate_file(user_image)
        validate_file(cloth_image)
        
        # Save uploaded files
        user_id, user_path = await save_upload_file(user_image, settings.UPLOAD_DIR)
        cloth_id, cloth_path = await save_upload_file(cloth_image, settings.UPLOAD_DIR)
        
        # Generate output path
        result_id = str(uuid.uuid4())
        output_path = Path(settings.RESULTS_DIR) / f"tryon_result_{result_id}.jpg"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Choose algorithm
        algorithm_used = "basic"
        
        # Convert string to boolean
        use_api_bool = use_api.lower() in ('true', '1', 'yes')
        
        if use_api_bool and api_service.is_available():
            try:
                # Use API service (better quality)
                logger.info(f"Using API service: {api_service.provider}")
                
                # Load images
                from app.utils.image_processor import load_image, save_image
                user_img = load_image(user_path)
                cloth_img = load_image(cloth_path)
                
                # Process with API
                result_img = api_service.process_tryon(user_img, cloth_img)
                
                # Save result
                save_image(result_img, str(output_path))
                
                algorithm_used = f"api_{api_service.provider}"
                
                # Get metadata
                metadata = {
                    "algorithm": algorithm_used,
                    "person_size": f"{user_img.shape[1]}x{user_img.shape[0]}",
                    "cloth_size": f"{cloth_img.shape[1]}x{cloth_img.shape[0]}"
                }
                
            except Exception as e:
                logger.warning(f"API service failed, falling back to basic: {e}")
                # Fallback to basic
                result = tryon_service.process(user_path, cloth_path, str(output_path))
                metadata = result['metadata']
                algorithm_used = "basic_fallback"
        else:
            # Use basic algorithm
            result = tryon_service.process(user_path, cloth_path, str(output_path))
            metadata = result['metadata']
        
        # Calculate time
        time_taken = time.time() - start_time
        
        # Add algorithm info to metadata
        metadata['algorithm'] = algorithm_used
        
        # Return response
        return TryOnResponse(
            status="success",
            image_url=f"/storage/results/tryon_result_{result_id}.jpg",
            time_taken=time_taken,
            metadata=metadata
        )
        
    except (ImageProcessingError, PoseDetectionError) as e:
        logger.error(f"Processing error (422): {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Try-on processing failed (500): {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@router.post("/try-on", response_model=TryOnResponse)
async def try_on_alias(
    user_image: UploadFile = File(...),
    cloth_image: UploadFile = File(...)
):
    """
    Alias for /process endpoint (backward compatibility)
    """
    return await process_tryon(user_image, cloth_image)

