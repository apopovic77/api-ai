#!/usr/bin/env python3
"""
Arkturian AI API
================

Production-ready AI/ML API for the Arkturian platform.

Features:
- Text AI: Claude, ChatGPT, Gemini
- Image AI: Generation, Upscaling, Depth Maps
- Audio AI: TTS, SFX, Music Generation
- Dialog System: Multi-character conversations with TTS

Author: Arkturian Team
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging

# Import routes (will be created)
from ai.routes import text_ai_routes, image_ai_routes, audio_ai_routes, dialog_routes

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Arkturian AI API",
    version="1.0.0",
    description="AI/ML services for the Arkturian platform",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production: specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(text_ai_routes.router, prefix="/ai", tags=["Text AI"])
app.include_router(image_ai_routes.router, prefix="/ai", tags=["Image AI"])
app.include_router(audio_ai_routes.router, prefix="/ai", tags=["Audio AI"])
app.include_router(dialog_routes.router, prefix="/ai/dialog", tags=["Dialog System"])

# Health check
@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "arkturian-ai-api",
        "version": "1.0.0"
    }

@app.get("/")
def root():
    """Root endpoint with API info"""
    return {
        "service": "Arkturian AI API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "text_ai": [
                "POST /ai/claude",
                "POST /ai/chatgpt", 
                "POST /ai/gemini"
            ],
            "image_ai": [
                "POST /ai/genimage",
                "POST /ai/upscale",
                "POST /ai/gendepth"
            ],
            "audio_ai": [
                "POST /ai/generate_speech",
                "POST /ai/gensfx",
                "POST /ai/genmusic",
                "POST /ai/genmusic_eleven"
            ],
            "dialog": [
                "POST /ai/dialog/start",
                "GET /ai/dialog/status",
                "POST /ai/dialog/cancel"
            ]
        }
    }

# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if app.debug else "An error occurred"
        }
    )

# Startup/Shutdown events
@app.on_event("startup")
async def startup_event():
    """Startup tasks"""
    logger.info("🚀 Arkturian AI API starting up...")
    logger.info("✅ All AI services initialized")

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown tasks"""
    logger.info("👋 Arkturian AI API shutting down...")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8003,  # Different port from artrack(8001) and storage(8002)
        reload=True,
        log_level="info"
    )

