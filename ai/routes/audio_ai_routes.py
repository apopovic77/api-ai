"""
Audio AI Routes
===============

Endpoints for audio generation:
- Text-to-Speech (TTS) & Dialog Generation
- Sound Effects (SFX)
- Music Generation
"""

from fastapi import APIRouter, Depends, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from typing import Optional
import logging
import re
import io
import os

logger = logging.getLogger(__name__)
router = APIRouter()


# Import models and services
import sys
sys.path.insert(0, '/Volumes/DatenAP/Code/api-ai/ai/services')

from ai.services.tts_models import SpeechRequest
from ai.services.speech_service import SpeechGenerator
from ai.services.audio_drama_service import AudioDramaGenerator
from ai.clients.storage_client import StorageObject
from ai.routes.music_generation import generate_music_stable_audio, generate_music_elevenlabs
from openai import AsyncOpenAI


# Response Models
class AudioResponse(BaseModel):
    audio_url: str
    file_url: Optional[str] = None  # Alias for audio_url for frontend compatibility
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


class AudioGenRequest(BaseModel):
    """Request model for audio generation (SFX/Music) - compatible with legacy API"""
    prompt: str
    duration_ms: Optional[int] = None
    link_id: Optional[str] = None
    owner_user_id: Optional[int] = None


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


@router.post("/gensfx")
async def generate_sfx_endpoint(
    request: SFXRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Generates a sound effect using ElevenLabs, analyzes its volume, and saves it to storage if it's not silent.

    Examples:
    - "dog barking"
    - "thunder and rain"
    - "car engine starting"
    """
    import os
    import tempfile
    from pathlib import Path
    from ai.services import tts_service
    from ai.clients.storage_client import save_file_and_record

    try:
        from elevenlabs.client import AsyncElevenLabs

        print(f"--- SFX Gen: Generating SFX for prompt: '{request.prompt[:80]}...'")
        client = AsyncElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

        audio_stream = client.text_to_sound_effects.convert(text=request.prompt)

        audio_bytes = b""
        async for chunk in audio_stream:
            audio_bytes += chunk

        if not audio_bytes:
            raise HTTPException(status_code=500, detail="ElevenLabs SFX generation returned no data.")

        # --- Audio Analysis ---
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            temp_file.write(audio_bytes)
            temp_path = Path(temp_file.name)

        mean_volume = tts_service.analyze_audio_level(temp_path)
        print(f"--- SFX Gen: Analyzed audio level: {mean_volume} dB")

        # Discard silent or near-silent SFX
        SILENCE_THRESHOLD = -60.0
        if mean_volume < SILENCE_THRESHOLD:
            print(f"--- SFX Gen: SFX is too quiet ({mean_volume} dB), discarding.")
            temp_path.unlink()
            raise HTTPException(status_code=422, detail=f"Generated SFX was silent and has been discarded.")

        # --- Save to Storage via HTTP ---
        filename = f"sfx_{request.prompt.replace(' ', '_')[:20]}.mp3"

        saved_obj = await save_file_and_record(
            data=audio_bytes,
            original_filename=filename,
            context="sfx-generation",
            is_public=True,
            collection_id="ai-generated-sfx"
        )

        temp_path.unlink()
        print(f"--- SFX Gen: Saved SFX to storage object ID {saved_obj.id}")

        return {
            "id": saved_obj.id,
            "file_url": saved_obj.file_url,
            "audio_url": saved_obj.file_url,
            "storage_object_id": saved_obj.id,
            "format": "mp3"
        }

    except Exception as e:
        logger.error(f"SFX generation error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    model: str = "whisper-1",
    prompt: Optional[str] = None,
    api_key: str = Depends(get_api_key)
):
    """
    Transcribe an uploaded audio file using OpenAI Whisper.

    Supports typical audio MIME types (mp3, m4a, wav, webm, etc.).
    """
    try:
        if not os.getenv("OPENAI_API_KEY"):
            raise HTTPException(status_code=500, detail="OpenAI API key not configured.")

        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")

        # OpenAI SDK expects a file-like object with a name attribute
        audio_buffer = io.BytesIO(data)
        audio_buffer.name = file.filename or "audio.webm"

        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        result = await client.audio.transcriptions.create(
            model=model,
            file=audio_buffer,
            prompt=prompt
        )

        text = getattr(result, "text", None)
        if not text:
            # Fallback to serializing the full response
            return jsonable_encoder(result)

        return {"text": text, "model": model, "prompt": prompt, "filename": file.filename}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/genmusic", response_model=AudioResponse)
async def generate_music_endpoint(
    request: MusicRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Generate music from text prompt using Stable Audio (AIMLAPI)

    Examples:
    - "upbeat electronic dance music"
    - "calm piano melody"
    - "epic orchestral soundtrack"
    """
    try:
        duration_ms = request.duration * 1000  # Convert seconds to milliseconds
        result = await generate_music_stable_audio(request.prompt, duration_ms)

        return AudioResponse(
            audio_url=result["audio_url"],
            file_url=result["audio_url"],  # For frontend compatibility
            storage_object_id=result["storage_object_id"],
            duration_seconds=float(request.duration),
            format=result["format"]
        )
    except Exception as e:
        logger.error(f"Music generation error: {e}")
        import traceback
        traceback.print_exc()
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
    - Auto-retry with suggested prompt if flagged
    """
    try:
        duration_ms = request.duration * 1000  # Convert seconds to milliseconds
        result = await generate_music_elevenlabs(request.prompt, duration_ms)

        return AudioResponse(
            audio_url=result["audio_url"],
            file_url=result["audio_url"],  # For frontend compatibility
            storage_object_id=result["storage_object_id"],
            duration_seconds=float(request.duration),
            format=result["format"]
        )
    except Exception as e:
        logger.error(f"ElevenLabs music error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
