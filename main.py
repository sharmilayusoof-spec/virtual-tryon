"""
FastAPI application - Virtual Try-On Backend
"""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from datetime import datetime
from pathlib import Path

from app.core.config import settings
from app.core.middleware import setup_middleware
from app.api.v1.router import router as api_v1_router
from app.utils.file_handler import ensure_directories
from app.models.schemas import HealthResponse


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="AI-Powered Virtual Try-On Backend API",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Setup middleware
setup_middleware(app)

# Ensure directories exist
ensure_directories()

# Mount static files for serving results
Path(settings.RESULTS_DIR).mkdir(parents=True, exist_ok=True)
app.mount("/storage", StaticFiles(directory="storage"), name="storage")

# Mount frontend assets (CSS, JS, images) before API routes
app.mount("/assets", StaticFiles(directory="frontend/assets"), name="assets")

# Include API routers
app.include_router(api_v1_router, prefix="/api/v1")


@app.get("/")
async def root():
    """Serve the frontend application"""
    return FileResponse("frontend/index.html")


@app.get("/style.css")
async def serve_css():
    """Serve CSS file"""
    return FileResponse("frontend/style.css", media_type="text/css")


@app.get("/script.js")
async def serve_js():
    """Serve JavaScript file"""
    return FileResponse("frontend/script.js", media_type="application/javascript")


@app.get("/api")
async def api_info():
    """API information endpoint"""
    return {
        "name": settings.APP_NAME,
        "version": settings.VERSION,
        "status": "running",
        "docs": "/docs",
        "api": "/api/v1"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        version=settings.VERSION
    )


@app.on_event("startup")
async def startup_event():
    """Startup event handler"""
    print(f"[*] {settings.APP_NAME} v{settings.VERSION} starting...")
    print(f"[*] API Documentation: http://{settings.HOST}:{settings.PORT}/docs")
    print(f"[*] Server ready!")


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler"""
    print(f"[*] {settings.APP_NAME} shutting down...")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )

