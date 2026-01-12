"""
Higgsfield API Client
=====================

Async client for Higgsfield AI platform.
Supports image generation (text-to-image) and video generation (image-to-video).

API Documentation: https://docs.higgsfield.ai/

Architecture:
- Async request-response pattern
- Submit job → Poll status → Get result
- Auto-saves to Storage API on completion
"""

import httpx
import asyncio
import logging
import os
from typing import Optional, Dict, Any, Literal
from pydantic import BaseModel
from enum import Enum

logger = logging.getLogger(__name__)


class HiggsFieldStatus(str, Enum):
    """Higgsfield request statuses"""
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    NSFW = "nsfw"


class HiggsFieldImageModel(str, Enum):
    """Available image generation models"""
    SOUL_STANDARD = "higgsfield-ai/soul/standard"
    REVE = "reve/text-to-image"
    SEEDREAM_EDIT = "bytedance/seedream/v4/edit"


class HiggsFieldVideoModel(str, Enum):
    """Available video generation models"""
    DOP_STANDARD = "higgsfield-ai/dop/standard"
    DOP_PREVIEW = "higgsfield-ai/dop/preview"
    KLING_PRO = "kling-video/v2.1/pro/image-to-video"
    SEEDANCE_PRO = "bytedance/seedance/v1/pro/image-to-video"


class HiggsFieldResponse(BaseModel):
    """Response from Higgsfield API"""
    request_id: str
    status: HiggsFieldStatus
    result_url: Optional[str] = None
    error: Optional[str] = None
    model: str
    raw_response: Optional[Dict[str, Any]] = None


class HiggsFieldClient:
    """
    Async client for Higgsfield AI platform.

    Usage:
        client = HiggsFieldClient()

        # Image generation
        result = await client.generate_image("a sunset over mountains")

        # Video generation (with polling)
        result = await client.generate_video(
            image_url="https://example.com/image.jpg",
            prompt="camera slowly zooms in"
        )
    """

    BASE_URL = "https://platform.higgsfield.ai"

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        timeout: float = 300.0
    ):
        """
        Initialize Higgsfield client.

        Args:
            api_key: Higgsfield API key (or HIGGSFIELD_API_KEY env var)
            api_secret: Higgsfield API secret (or HIGGSFIELD_API_SECRET env var)
            timeout: Request timeout in seconds
        """
        self.api_key = api_key or os.getenv("HIGGSFIELD_API_KEY")
        self.api_secret = api_secret or os.getenv("HIGGSFIELD_API_SECRET")
        self.timeout = timeout

        if not self.api_key or not self.api_secret:
            raise ValueError(
                "Higgsfield credentials required. Set HIGGSFIELD_API_KEY and "
                "HIGGSFIELD_API_SECRET environment variables."
            )

    @property
    def _auth_header(self) -> str:
        """Generate Authorization header value"""
        return f"Key {self.api_key}:{self.api_secret}"

    @property
    def _headers(self) -> Dict[str, str]:
        """Default headers for API requests"""
        return {
            "Authorization": self._auth_header,
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

    async def _request(
        self,
        method: str,
        endpoint: str,
        json: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Make HTTP request to Higgsfield API.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (will be appended to BASE_URL)
            json: JSON body for POST requests
            **kwargs: Additional httpx request arguments

        Returns:
            JSON response as dict

        Raises:
            httpx.HTTPStatusError: On HTTP errors
        """
        url = f"{self.BASE_URL}{endpoint}"

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.request(
                method=method,
                url=url,
                headers=self._headers,
                json=json,
                **kwargs
            )

            # Log request for debugging
            logger.info(f"Higgsfield {method} {endpoint} -> {response.status_code}")

            if response.status_code >= 400:
                logger.error(f"Higgsfield error: {response.text}")
                response.raise_for_status()

            return response.json()

    async def submit_image_generation(
        self,
        prompt: str,
        model: str = HiggsFieldImageModel.SOUL_STANDARD,
        aspect_ratio: str = "1:1",
        resolution: str = "1080p"
    ) -> HiggsFieldResponse:
        """
        Submit image generation request.

        Args:
            prompt: Text description of desired image
            model: Model to use (default: higgsfield-ai/soul/standard)
            aspect_ratio: Image aspect ratio (e.g., "1:1", "16:9", "9:16")
            resolution: Image resolution ("720p" or "1080p")

        Returns:
            HiggsFieldResponse with request_id
        """
        logger.info(f"Higgsfield image gen: '{prompt[:50]}...' model={model}")

        # Higgsfield only accepts "720p" or "1080p"
        valid_resolutions = ["720p", "1080p"]
        if resolution not in valid_resolutions:
            resolution = "1080p"

        payload = {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution
        }

        result = await self._request("POST", f"/{model}", json=payload)

        return HiggsFieldResponse(
            request_id=result.get("request_id", result.get("id", "")),
            status=HiggsFieldStatus.QUEUED,
            model=model,
            raw_response=result
        )

    async def submit_video_generation(
        self,
        image_url: str,
        prompt: str,
        duration: int = 5,
        model: str = HiggsFieldVideoModel.DOP_STANDARD
    ) -> HiggsFieldResponse:
        """
        Submit video generation request (image-to-video).

        Args:
            image_url: URL of source image
            prompt: Motion/animation description
            duration: Video duration in seconds (default: 5)
            model: Model to use (default: higgsfield-ai/dop/standard)

        Returns:
            HiggsFieldResponse with request_id
        """
        logger.info(f"Higgsfield video gen: '{prompt[:50]}...' model={model}")

        payload = {
            "image_url": image_url,
            "prompt": prompt,
            "duration": duration
        }

        result = await self._request("POST", f"/{model}", json=payload)

        return HiggsFieldResponse(
            request_id=result.get("request_id", result.get("id", "")),
            status=HiggsFieldStatus.QUEUED,
            model=model,
            raw_response=result
        )

    async def get_status(self, request_id: str) -> HiggsFieldResponse:
        """
        Check status of a generation request.

        Args:
            request_id: Request ID from submit response

        Returns:
            HiggsFieldResponse with current status and result_url if completed
        """
        result = await self._request("GET", f"/requests/{request_id}/status")

        status = HiggsFieldStatus(result.get("status", "queued"))

        return HiggsFieldResponse(
            request_id=request_id,
            status=status,
            result_url=result.get("result_url") or result.get("output_url"),
            error=result.get("error"),
            model=result.get("model", "unknown"),
            raw_response=result
        )

    async def cancel_request(self, request_id: str) -> bool:
        """
        Cancel a pending request.

        Args:
            request_id: Request ID to cancel

        Returns:
            True if cancelled successfully, False otherwise
        """
        try:
            await self._request("POST", f"/requests/{request_id}/cancel")
            logger.info(f"Higgsfield request {request_id} cancelled")
            return True
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 400:
                # Request already processing, cannot cancel
                logger.warning(f"Could not cancel {request_id}: already processing")
                return False
            raise

    async def poll_until_complete(
        self,
        request_id: str,
        poll_interval: float = 2.0,
        max_wait: float = 300.0
    ) -> HiggsFieldResponse:
        """
        Poll request status until completed or failed.

        Args:
            request_id: Request ID to poll
            poll_interval: Seconds between polls (default: 2s)
            max_wait: Maximum wait time in seconds (default: 300s)

        Returns:
            Final HiggsFieldResponse

        Raises:
            TimeoutError: If max_wait exceeded
            RuntimeError: If generation failed
        """
        elapsed = 0.0

        while elapsed < max_wait:
            response = await self.get_status(request_id)

            if response.status == HiggsFieldStatus.COMPLETED:
                logger.info(f"Higgsfield {request_id} completed: {response.result_url}")
                return response

            if response.status == HiggsFieldStatus.FAILED:
                raise RuntimeError(f"Higgsfield generation failed: {response.error}")

            if response.status == HiggsFieldStatus.NSFW:
                raise RuntimeError("Higgsfield generation failed: content flagged as NSFW")

            logger.debug(f"Higgsfield {request_id} status: {response.status}")
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        raise TimeoutError(f"Higgsfield request {request_id} timed out after {max_wait}s")

    async def generate_image(
        self,
        prompt: str,
        model: str = HiggsFieldImageModel.SOUL_STANDARD,
        aspect_ratio: str = "1:1",
        resolution: str = "1080p",
        poll_interval: float = 2.0,
        max_wait: float = 120.0
    ) -> HiggsFieldResponse:
        """
        Generate image and wait for result (convenience method).

        Combines submit + poll into single call.

        Args:
            prompt: Text description of desired image
            model: Model to use
            aspect_ratio: Image aspect ratio
            resolution: Image resolution
            poll_interval: Seconds between status checks
            max_wait: Maximum wait time

        Returns:
            HiggsFieldResponse with result_url
        """
        # Submit request
        submit_response = await self.submit_image_generation(
            prompt=prompt,
            model=model,
            aspect_ratio=aspect_ratio,
            resolution=resolution
        )

        # Poll until complete
        return await self.poll_until_complete(
            request_id=submit_response.request_id,
            poll_interval=poll_interval,
            max_wait=max_wait
        )

    async def generate_video(
        self,
        image_url: str,
        prompt: str,
        duration: int = 5,
        model: str = HiggsFieldVideoModel.DOP_STANDARD,
        poll_interval: float = 5.0,
        max_wait: float = 300.0
    ) -> HiggsFieldResponse:
        """
        Generate video from image and wait for result (convenience method).

        Combines submit + poll into single call.

        Args:
            image_url: URL of source image
            prompt: Motion/animation description
            duration: Video duration in seconds
            model: Model to use
            poll_interval: Seconds between status checks
            max_wait: Maximum wait time

        Returns:
            HiggsFieldResponse with result_url
        """
        # Submit request
        submit_response = await self.submit_video_generation(
            image_url=image_url,
            prompt=prompt,
            duration=duration,
            model=model
        )

        # Poll until complete
        return await self.poll_until_complete(
            request_id=submit_response.request_id,
            poll_interval=poll_interval,
            max_wait=max_wait
        )


# Module-level convenience functions
_client: Optional[HiggsFieldClient] = None


def get_client() -> HiggsFieldClient:
    """Get or create singleton Higgsfield client"""
    global _client
    if _client is None:
        _client = HiggsFieldClient()
    return _client


async def generate_image(prompt: str, **kwargs) -> HiggsFieldResponse:
    """Generate image using default client"""
    return await get_client().generate_image(prompt, **kwargs)


async def generate_video(image_url: str, prompt: str, **kwargs) -> HiggsFieldResponse:
    """Generate video using default client"""
    return await get_client().generate_video(image_url, prompt, **kwargs)
