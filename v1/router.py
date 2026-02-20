"""
API v1 router
"""
from fastapi import APIRouter
from app.api.v1.endpoints import tryon


router = APIRouter()

# Include endpoint routers
router.include_router(tryon.router, prefix="/tryon", tags=["Try-On"])

