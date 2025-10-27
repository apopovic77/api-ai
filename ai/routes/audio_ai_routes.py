"""
Audio AI Routes
===============

Endpoints for audio generation:
- Text-to-Speech (TTS) & Dialog Generation
- Sound Effects (SFX)
- Music Generation
"""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from typing import Optional
import logging
import re

logger = logging.getLogger(__name__)
router = APIRouter()


# Import models and services
import sys
sys.path.insert(0, '/Volumes/DatenAP/Code/api-ai/ai/services')

from ai.services.tts_models import SpeechRequest
from ai.services.speech_service import SpeechGenerator
from ai.services.audio_drama_service import AudioDramaGenerator
from ai.clients.storage_client import StorageObject


# Response Models
class AudioResponse(BaseModel):
    audio_url: str
    storage_object_id: Optional[int] = None
    duration_seconds: Optional[float] = None
    format: str = "mp3"


class SFXRequest(BaseModel):
    prompt: str
    duration: Optional[float] = 5.0
    model: Optional[str] = "audio-ldm"


class MusicRequest(BaseModel):
    prompt: str
    duration: Optional[int] = 30
    model: Optional[str] = "suno"  # suno or eleven


def get_api_key():
    return "Inetpass1"


def is_likely_json(text: str) -> bool:
    """Check if text looks like JSON to prevent expensive TTS mistakes."""
    trimmed = text.strip()

    # Check for obvious JSON structures
    if (trimmed.startswith('{') and trimmed.endswith('}')) or \
       (trimmed.startswith('[') and trimmed.endswith(']')):
        return True

    # Check for JSON patterns
    json_patterns = [
        r'"production_cues"\s*:',
        r'"cues"\s*:',
        r'"background_music"\s*:',
        r'"music"\s*:',
        r'"type"\s*:\s*"dialog"',
        r'"audio_url"\s*:',
        r'"pause_after_ms"\s*:'
    ]

    for pattern in json_patterns:
        if re.search(pattern, trimmed):
            return True

    return False


@router.post("/generate_speech")
async def generate_speech_endpoint(
    req: SpeechRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Handles both single voice TTS and complex audio drama generation.

    Accepts the full SpeechRequest model from dialog.php with:
    - id, timestamp, content (text, language, voice, speed)
    - config (dialog_mode, voice_mapping, add_sfx, add_music, etc.)
    - collection_id, save_options
    """
    try:
        # CRITICAL VALIDATION: Prevent JSON from being synthesized (costs money!)
        if req.content and req.content.text and is_likely_json(req.content.text):
            print(f"🚫 TTS BLOCKED: Text looks like JSON structure!")
            print(f"🚫 First 200 chars: {req.content.text[:200]}")
            raise HTTPException(
                status_code=400,
                detail="TTS generation blocked: Text appears to be JSON. JSON should be parsed, not spoken."
            )

        if req.config.dialog_mode:
            # Dialog/Audio Drama Generation
            generator = AudioDramaGenerator(req, api_key, image_gen_func=None)
            saved_audio_obj, production_plan, generated_image_obj = await generator.generate()

            # Analyze-only mode: return just the analysis
            if saved_audio_obj is None:
                return JSONResponse(content={
                    "analysis_result": production_plan,
                    "message": "analysis_only"
                })

            # Convert StorageObject to dict for JSON response
            response_data = {
                "id": saved_audio_obj.id,
                "file_url": saved_audio_obj.file_url,
                "audio_url": saved_audio_obj.file_url,  # Legacy compatibility
                "object_key": saved_audio_obj.object_key,
                "original_filename": saved_audio_obj.original_filename,
                "mime_type": saved_audio_obj.mime_type,
                "file_size": saved_audio_obj.file_size,
                "thumbnail_url": saved_audio_obj.thumbnail_url,
                "context": saved_audio_obj.context,
                "is_public": saved_audio_obj.is_public,
                "collection_id": saved_audio_obj.collection_id,
                "analysis_result": production_plan
            }

            if generated_image_obj:
                response_data['generated_image'] = {
                    "id": generated_image_obj.id,
                    "file_url": generated_image_obj.file_url
                }

            return JSONResponse(content=response_data)

        else:
            # Simple TTS Generation
            generator = SpeechGenerator(req, api_key, image_gen_func=None)
            saved_audio_obj, _, _ = await generator.generate()

            return {
                "id": saved_audio_obj.id,
                "file_url": saved_audio_obj.file_url,
                "audio_url": saved_audio_obj.file_url,
                "object_key": saved_audio_obj.object_key,
                "original_filename": saved_audio_obj.original_filename,
                "mime_type": saved_audio_obj.mime_type,
                "file_size": saved_audio_obj.file_size,
                "context": saved_audio_obj.context,
                "is_public": saved_audio_obj.is_public,
                "collection_id": saved_audio_obj.collection_id
            }

    except Exception as e:
        print(f"--- ERROR [TTS Endpoint]: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An error occurred during TTS generation: {str(e)}")


@router.post("/gensfx", response_model=AudioResponse)
async def generate_sfx_endpoint(
    request: SFXRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Generate sound effects from text description
    
    Examples:
    - "dog barking"
    - "thunder and rain"
    - "car engine starting"
    """
    try:
        # TODO: Implement SFX generation
        # Using AudioLDM or similar models via Replicate
        
        return AudioResponse(
            audio_url="https://placeholder.com/sfx.mp3",
            storage_object_id=None,
            duration_seconds=request.duration,
            format="mp3"
        )
    except Exception as e:
        logger.error(f"SFX generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/genmusic", response_model=AudioResponse)
async def generate_music_endpoint(
    request: MusicRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Generate music from text prompt (Suno)
    
    Examples:
    - "upbeat electronic dance music"
    - "calm piano melody"
    - "epic orchestral soundtrack"
    """
    try:
        # TODO: Implement music generation with Suno
        # This might be async/job-based like in old API
        
        return AudioResponse(
            audio_url="https://placeholder.com/music.mp3",
            storage_object_id=None,
            duration_seconds=float(request.duration),
            format="mp3"
        )
    except Exception as e:
        logger.error(f"Music generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/genmusic_eleven", response_model=AudioResponse)
async def generate_music_eleven_endpoint(
    request: MusicRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Generate music using ElevenLabs
    
    Features:
    - High quality music generation
    - Style control
    - Instrumental and vocal options
    """
    try:
        # TODO: Implement ElevenLabs music generation
        
        return AudioResponse(
            audio_url="https://placeholder.com/music_eleven.mp3",
            storage_object_id=None,
            duration_seconds=float(request.duration),
            format="mp3"
        )
    except Exception as e:
        logger.error(f"ElevenLabs music error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

