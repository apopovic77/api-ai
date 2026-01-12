"""
Video AI Routes
===============

Endpoints for video generation using Higgsfield API:
- Image-to-Video transformation
- Multiple model support (DoP, Kling, Seedance)
- Async job management with polling
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, Literal
import logging
import asyncio

from ai.clients.higgsfield_client import (
    HiggsFieldClient,
    HiggsFieldResponse,
    HiggsFieldStatus,
    HiggsFieldVideoModel,
    get_client
)
from ai.clients.storage_client import save_file_and_record

logger = logging.getLogger(__name__)
router = APIRouter()


# Request/Response Models
class VideoGenRequest(BaseModel):
    """Request model for video generation"""
    image_url: str = Field(..., description="URL of source image to animate")
    prompt: str = Field(..., description="Motion/animation description")
    duration: int = Field(default=5, ge=1, le=10, description="Video duration in seconds (1-10)")
    model: str = Field(
        default="higgsfield-ai/dop/standard",
        description="Model to use: higgsfield-ai/dop/standard, kling-video/v2.1/pro/image-to-video, bytedance/seedance/v1/pro/image-to-video"
    )
    wait_for_result: bool = Field(
        default=True,
        description="If true, wait for result (sync). If false, return request_id immediately (async)."
    )
    collection_id: Optional[str] = Field(default="ai-generated-videos", description="Storage collection")
    link_id: Optional[str] = Field(default=None, description="Link ID for related objects")


class VideoGenResponse(BaseModel):
    """Response model for video generation"""
    request_id: str
    status: str
    video_url: Optional[str] = None
    storage_object_id: Optional[int] = None
    model: str
    duration: int
    message: Optional[str] = None


class VideoStatusResponse(BaseModel):
    """Response for status check endpoint"""
    request_id: str
    status: str
    video_url: Optional[str] = None
    storage_object_id: Optional[int] = None
    error: Optional[str] = None
    progress: Optional[float] = None


def get_api_key():
    """Placeholder for API key validation"""
    return "placeholder"


async def download_and_save_video(
    video_url: str,
    request_id: str,
    collection_id: str = "ai-generated-videos",
    link_id: Optional[str] = None
) -> int:
    """
    Download video from Higgsfield and save to Storage API.

    Args:
        video_url: URL of generated video
        request_id: Higgsfield request ID (used in filename)
        collection_id: Storage collection
        link_id: Optional link ID

    Returns:
        Storage object ID
    """
    import httpx

    logger.info(f"Downloading video from Higgsfield: {video_url}")

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.get(video_url)
        response.raise_for_status()
        video_bytes = response.content

    # Generate filename
    filename = f"video_{request_id[:8]}.mp4"

    # Save to storage
    saved_obj = await save_file_and_record(
        data=video_bytes,
        original_filename=filename,
        context="video-generation",
        is_public=True,
        collection_id=collection_id,
        link_id=link_id
    )

    logger.info(f"Saved video to storage: ID={saved_obj.id}, URL={saved_obj.file_url}")
    return saved_obj.id, saved_obj.file_url


@router.post("/genvideo", response_model=VideoGenResponse)
async def generate_video_endpoint(
    request: VideoGenRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Generate video from image using Higgsfield AI.

    Image-to-Video animation with multiple model options:
    - **higgsfield-ai/dop/standard** (default): High-quality animation
    - **higgsfield-ai/dop/preview**: Fast preview quality
    - **kling-video/v2.1/pro/image-to-video**: Cinematic animations
    - **bytedance/seedance/v1/pro/image-to-video**: Professional-grade

    Set `wait_for_result=false` for async mode (returns immediately with request_id).
    Use `/ai/genvideo/status/{request_id}` to check progress.

    Example:
    ```json
    {
        "image_url": "https://example.com/photo.jpg",
        "prompt": "camera slowly zooms in, gentle wind moves the hair",
        "duration": 5,
        "model": "higgsfield-ai/dop/standard"
    }
    ```
    """
    try:
        client = get_client()

        logger.info(
            f"Video generation request: model={request.model}, "
            f"duration={request.duration}s, wait={request.wait_for_result}"
        )

        if request.wait_for_result:
            # Synchronous: wait for result
            result = await client.generate_video(
                image_url=request.image_url,
                prompt=request.prompt,
                duration=request.duration,
                model=request.model,
                poll_interval=5.0,
                max_wait=300.0  # 5 minutes max
            )

            if result.status != HiggsFieldStatus.COMPLETED:
                raise HTTPException(
                    status_code=500,
                    detail=f"Video generation failed: {result.error or result.status}"
                )

            # Download and save to storage
            storage_id, storage_url = await download_and_save_video(
                video_url=result.result_url,
                request_id=result.request_id,
                collection_id=request.collection_id,
                link_id=request.link_id
            )

            return VideoGenResponse(
                request_id=result.request_id,
                status="completed",
                video_url=storage_url,
                storage_object_id=storage_id,
                model=request.model,
                duration=request.duration,
                message="Video generation completed"
            )

        else:
            # Asynchronous: return request_id immediately
            submit_result = await client.submit_video_generation(
                image_url=request.image_url,
                prompt=request.prompt,
                duration=request.duration,
                model=request.model
            )

            return VideoGenResponse(
                request_id=submit_result.request_id,
                status="queued",
                video_url=None,
                storage_object_id=None,
                model=request.model,
                duration=request.duration,
                message="Video generation queued. Use /ai/genvideo/status/{request_id} to check progress."
            )

    except TimeoutError as e:
        logger.error(f"Video generation timeout: {e}")
        raise HTTPException(status_code=504, detail="Video generation timed out")
    except RuntimeError as e:
        logger.error(f"Video generation failed: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Video generation error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Video generation error: {str(e)}")


@router.get("/genvideo/status/{request_id}", response_model=VideoStatusResponse)
async def get_video_status(
    request_id: str,
    save_on_complete: bool = True,
    collection_id: str = "ai-generated-videos",
    link_id: Optional[str] = None,
    api_key: str = Depends(get_api_key)
):
    """
    Check status of video generation request.

    Args:
        request_id: Higgsfield request ID
        save_on_complete: Auto-save to storage when completed (default: true)
        collection_id: Storage collection for auto-save
        link_id: Link ID for auto-save

    Returns:
        Current status with video_url if completed
    """
    try:
        client = get_client()
        result = await client.get_status(request_id)

        response = VideoStatusResponse(
            request_id=request_id,
            status=result.status.value,
            error=result.error
        )

        # If completed and save requested, download and save
        if result.status == HiggsFieldStatus.COMPLETED and result.result_url:
            if save_on_complete:
                storage_id, storage_url = await download_and_save_video(
                    video_url=result.result_url,
                    request_id=request_id,
                    collection_id=collection_id,
                    link_id=link_id
                )
                response.video_url = storage_url
                response.storage_object_id = storage_id
            else:
                response.video_url = result.result_url

        return response

    except Exception as e:
        logger.error(f"Status check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/genvideo/cancel/{request_id}")
async def cancel_video_generation(
    request_id: str,
    api_key: str = Depends(get_api_key)
):
    """
    Cancel a pending video generation request.

    Only works if the request is still queued (not yet processing).

    Returns:
        Success status
    """
    try:
        client = get_client()
        cancelled = await client.cancel_request(request_id)

        if cancelled:
            return {"status": "cancelled", "request_id": request_id}
        else:
            return {
                "status": "cannot_cancel",
                "request_id": request_id,
                "message": "Request is already processing and cannot be cancelled"
            }

    except Exception as e:
        logger.error(f"Cancel error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/genvideo/models")
async def list_video_models():
    """List available video generation models"""
    return {
        "models": [
            {
                "id": "higgsfield-ai/dop/standard",
                "name": "DoP Standard",
                "provider": "Higgsfield",
                "description": "High-quality image animation",
                "max_duration": 10
            },
            {
                "id": "higgsfield-ai/dop/preview",
                "name": "DoP Preview",
                "provider": "Higgsfield",
                "description": "Fast preview quality",
                "max_duration": 5
            },
            {
                "id": "kling-video/v2.1/pro/image-to-video",
                "name": "Kling 2.1 Pro",
                "provider": "Kling Video",
                "description": "Advanced cinematic animations",
                "max_duration": 10
            },
            {
                "id": "bytedance/seedance/v1/pro/image-to-video",
                "name": "Seedance Pro",
                "provider": "ByteDance",
                "description": "Professional-grade video generation",
                "max_duration": 10
            }
        ]
    }
