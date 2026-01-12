"""
Image AI Routes
===============

Endpoints for image generation and processing:
- Image generation (Higgsfield, Gemini, DALL-E)
- Image upscaling
- Depth map generation
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


# Request/Response Models
class ImageGenRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    width: Optional[int] = 1024
    height: Optional[int] = 1024
    aspect_ratio: Optional[str] = Field(default="1:1", description="Aspect ratio (1:1, 16:9, 9:16)")
    model: Optional[str] = Field(
        default="higgsfield",
        description="Model: higgsfield (default), higgsfield-reve, nano-banana, imagen-4"
    )
    collection_id: Optional[str] = Field(default="ai-generated-images", description="Storage collection")
    link_id: Optional[str] = Field(default=None, description="Link ID for related objects")


class ImageResponse(BaseModel):
    image_url: str
    storage_object_id: Optional[int] = None
    model: str
    width: int
    height: int


class UpscaleRequest(BaseModel):
    image_url: str
    scale_factor: Optional[int] = 4
    model: Optional[str] = "real-esrgan"


class DepthRequest(BaseModel):
    image_url: str
    model: Optional[str] = "depth-anything"


def get_api_key():
    return "placeholder"


# Model mapping: user-friendly names to provider-specific IDs
MODEL_MAPPING = {
    # Higgsfield Models (NEW DEFAULT)
    "higgsfield": "higgsfield-ai/soul/standard",
    "higgsfield-soul": "higgsfield-ai/soul/standard",
    "higgsfield-reve": "reve/text-to-image",
    "reve": "reve/text-to-image",

    # Gemini Image Models (Legacy)
    "nano-banana": "gemini-2.5-flash-image",
    "nano-banana-pro": "gemini-3-pro-image-preview",
    "gemini": "gemini-2.5-flash-image",
    "gemini-2.5-flash-image": "gemini-2.5-flash-image",
    "gemini-3-pro-image-preview": "gemini-3-pro-image-preview",

    # Imagen 4
    "imagen-4": "imagen-4.0-generate-001",
    "imagen-4.0-generate-001": "imagen-4.0-generate-001",

    # Others
    "dall-e-3": "dall-e-3",
}


def is_higgsfield_model(model: str) -> bool:
    """Check if model should use Higgsfield provider"""
    higgsfield_prefixes = ["higgsfield", "reve"]
    return any(model.startswith(p) for p in higgsfield_prefixes)


def is_gemini_model(model: str) -> bool:
    """Check if model should use Gemini/Google provider"""
    gemini_prefixes = ["gemini", "imagen", "nano-banana"]
    return any(model.startswith(p) for p in gemini_prefixes)


async def generate_with_higgsfield(
    prompt: str,
    model: str,
    aspect_ratio: str,
    collection_id: str,
    link_id: Optional[str]
) -> dict:
    """Generate image using Higgsfield API"""
    import httpx
    import uuid
    from ai.clients.higgsfield_client import get_client, HiggsFieldStatus
    from ai.clients.storage_client import save_file_and_record

    client = get_client()

    # Higgsfield only accepts "720p" or "1080p"
    resolution = "1080p"

    logger.info(f"Higgsfield image gen: model={model}, aspect={aspect_ratio}")

    # Generate image (with polling)
    result = await client.generate_image(
        prompt=prompt,
        model=model,
        aspect_ratio=aspect_ratio,
        resolution=resolution,
        poll_interval=2.0,
        max_wait=120.0
    )

    if result.status != HiggsFieldStatus.COMPLETED:
        raise HTTPException(
            status_code=500,
            detail=f"Higgsfield image generation failed: {result.error or result.status}"
        )

    # Download image from Higgsfield
    logger.info(f"Downloading image from Higgsfield: {result.result_url}")
    async with httpx.AsyncClient(timeout=60.0) as http_client:
        response = await http_client.get(result.result_url)
        response.raise_for_status()
        image_bytes = response.content

    # Determine file extension from content-type
    content_type = response.headers.get("content-type", "image/png")
    ext = "png" if "png" in content_type else "jpg"
    filename = f"img_{result.request_id[:8]}.{ext}"

    # Save to storage
    saved_obj = await save_file_and_record(
        data=image_bytes,
        original_filename=filename,
        context="image-generation",
        is_public=True,
        collection_id=collection_id,
        link_id=link_id
    )

    logger.info(f"Saved Higgsfield image to storage: ID={saved_obj.id}")

    return {
        "id": saved_obj.id,
        "image_url": saved_obj.file_url,
        "file_url": saved_obj.file_url,
        "storage_object_id": saved_obj.id,
        "request_id": result.request_id
    }


async def generate_with_gemini(
    prompt: str,
    model: str,
    collection_id: str,
    link_id: Optional[str]
) -> dict:
    """Generate image using Google Gemini/Imagen API"""
    from google import genai
    from google.genai import types
    import os
    import uuid
    from ai.clients.storage_client import save_file_and_record
    from ai.services.cost_tracker import cost_tracker

    # Check budget before processing
    if cost_tracker.should_block_request():
        status = cost_tracker.get_status()
        raise HTTPException(
            status_code=429,
            detail=f"Monthly Gemini API budget exceeded: {status['total_cost_eur']:.2f}/{status['monthly_budget_eur']:.2f} EUR."
        )

    logger.info(f"Gemini image gen: model={model}")

    # Use Google GenAI SDK
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"]
        )
    )

    # Extract image from response
    image_bytes = None
    for part in response.candidates[0].content.parts:
        if part.inline_data is not None:
            image_bytes = part.inline_data.data
            break

    if not image_bytes:
        raise HTTPException(status_code=500, detail="No image found in Gemini response.")

    # Generate filename
    filename = f"img_{str(uuid.uuid4())[:8]}.png"

    # Save to storage
    saved_obj = await save_file_and_record(
        data=image_bytes,
        original_filename=filename,
        context="image-generation",
        is_public=True,
        collection_id=collection_id,
        link_id=link_id
    )

    logger.info(f"Saved Gemini image to storage: ID={saved_obj.id}")

    # Track cost
    cost_tracker.track_image_generation(model, num_images=1)

    return {
        "id": saved_obj.id,
        "image_url": saved_obj.file_url,
        "file_url": saved_obj.file_url,
        "storage_object_id": saved_obj.id
    }


@router.post("/genimage")
async def generate_image_endpoint(
    request: ImageGenRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Generate an image from text prompt using various AI models.

    **Default: Higgsfield** (high quality, fast)

    Supported models:
    - **higgsfield** (default): Higgsfield Soul - high quality text-to-image
    - **higgsfield-reve**: Reve model via Higgsfield - versatile generation
    - **nano-banana**: Gemini 2.5 Flash Image - fast, cost-effective
    - **nano-banana-pro**: Gemini 3 Pro Image - best Gemini quality
    - **imagen-4**: Google Imagen 4.0 - photorealistic

    Example:
    ```json
    {
        "prompt": "A serene mountain landscape at sunset",
        "model": "higgsfield",
        "aspect_ratio": "16:9"
    }
    ```
    """
    try:
        # Determine model
        model_name = request.model or "higgsfield"
        actual_model = MODEL_MAPPING.get(model_name, model_name)

        logger.info(f"Image gen request: model={model_name} -> {actual_model}")

        # Route to appropriate provider
        if is_higgsfield_model(actual_model):
            result = await generate_with_higgsfield(
                prompt=request.prompt,
                model=actual_model,
                aspect_ratio=request.aspect_ratio or "1:1",
                collection_id=request.collection_id or "ai-generated-images",
                link_id=request.link_id
            )

        elif is_gemini_model(model_name) or actual_model.startswith("gemini") or actual_model.startswith("imagen"):
            result = await generate_with_gemini(
                prompt=request.prompt,
                model=actual_model,
                collection_id=request.collection_id or "ai-generated-images",
                link_id=request.link_id
            )

        elif model_name == "dall-e-3":
            raise HTTPException(status_code=501, detail="DALL-E 3 support coming soon")

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported model: {model_name}")

        return {
            **result,
            "model": model_name,
            "actual_model": actual_model,
            "width": request.width,
            "height": request.height
        }

    except HTTPException:
        raise
    except TimeoutError as e:
        logger.error(f"Image generation timeout: {e}")
        raise HTTPException(status_code=504, detail="Image generation timed out")
    except RuntimeError as e:
        logger.error(f"Image generation failed: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Image generation error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Image generation error: {str(e)}")


@router.get("/genimage/models")
async def list_image_models():
    """List available image generation models"""
    return {
        "models": [
            {
                "id": "higgsfield",
                "name": "Higgsfield Soul",
                "provider": "Higgsfield",
                "description": "High quality text-to-image (default)",
                "default": True
            },
            {
                "id": "higgsfield-reve",
                "name": "Reve",
                "provider": "Higgsfield",
                "description": "Versatile text-to-image generation"
            },
            {
                "id": "nano-banana",
                "name": "Nano Banana",
                "provider": "Google Gemini",
                "description": "Fast, cost-effective image generation"
            },
            {
                "id": "nano-banana-pro",
                "name": "Nano Banana Pro",
                "provider": "Google Gemini",
                "description": "Best Gemini image quality"
            },
            {
                "id": "imagen-4",
                "name": "Imagen 4",
                "provider": "Google",
                "description": "Photorealistic image generation"
            }
        ]
    }


@router.post("/upscale", response_model=ImageResponse)
async def upscale_image_endpoint(
    request: UpscaleRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Upscale an image using AI models

    Supports:
    - Real-ESRGAN
    - GFPGAN (face enhancement)
    - Other upscaling models
    """
    try:
        # TODO: Implement actual image upscaling
        # 1. Download input image
        # 2. Call upscaling model (Replicate)
        # 3. Upload result to Storage API
        # 4. Return storage object info

        return ImageResponse(
            image_url=request.image_url,
            storage_object_id=None,
            model=request.model,
            width=1024 * request.scale_factor,
            height=1024 * request.scale_factor
        )
    except Exception as e:
        logger.error(f"Image upscale error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/gendepth", response_model=dict)
async def generate_depth_endpoint(
    request: DepthRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Generate depth map from image

    Useful for:
    - 3D reconstruction
    - AR/VR applications
    - Image analysis
    """
    try:
        # TODO: Implement depth generation
        # Could be async job-based like in old API

        return {
            "job_id": "placeholder",
            "status": "pending",
            "message": "Depth generation job started"
        }
    except Exception as e:
        logger.error(f"Depth generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/gendepth/result/{job_id}")
async def get_depth_result(
    job_id: str,
    api_key: str = Depends(get_api_key)
):
    """Get depth generation job result"""
    # TODO: Implement job result retrieval
    return {
        "job_id": job_id,
        "status": "completed",
        "depth_map_url": "https://placeholder.com/depth.png",
        "storage_object_id": None
    }
