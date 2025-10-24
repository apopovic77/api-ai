"""
AI Routes Package
=================

All API routes for the Arkturian AI service.
"""

from . import text_ai_routes
from . import image_ai_routes
from . import audio_ai_routes
from . import dialog_routes

__all__ = [
    "text_ai_routes",
    "image_ai_routes",
    "audio_ai_routes",
    "dialog_routes"
]

