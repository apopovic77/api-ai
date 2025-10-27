"""
Text AI Routes
==============

Endpoints for text-based AI models:
- Claude (Anthropic)
- ChatGPT (OpenAI)
- Gemini (Google)
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Union
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


# Request/Response Models
class PromptText(BaseModel):
    """Nested text/images structure for legacy compatibility"""
    text: str
    images: Optional[List[str]] = None  # Base64 encoded images

class Prompt(BaseModel):
    prompt: Union[str, PromptText]  # Support both string and nested object
    system: Optional[str] = None
    max_tokens: Optional[int] = 1000
    temperature: Optional[float] = 0.7
    conversation_history: Optional[List[Dict[str, str]]] = None


class AIResponse(BaseModel):
    response: str
    message: Optional[str] = None  # Alias for 'response' for legacy compatibility
    model: str
    tokens_used: Optional[int] = None
    finish_reason: Optional[str] = None

    def __init__(self, **data):
        super().__init__(**data)
        # Auto-populate message field from response if not provided
        if not self.message and self.response:
            self.message = self.response


# Placeholder for API key validation
def get_api_key():
    # TODO: Implement API key validation
    return "placeholder"


@router.post("/claude", response_model=AIResponse)
async def claude_endpoint(
    prompt: Prompt,
    model: Optional[str] = None,
    api_key: str = Depends(get_api_key)
):
    """
    Claude AI endpoint (Anthropic)

    Features:
    - Large context window (200k+ tokens)
    - Excellent at reasoning and analysis
    - Strong safety guardrails
    - Vision support (can analyze images)
    """
    try:
        from anthropic import Anthropic
        import os
        import base64

        # Configure Anthropic API
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if not anthropic_key or anthropic_key == "placeholder":
            raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured")

        client = Anthropic(api_key=anthropic_key)

        # Extract prompt text and images
        prompt_text = ""
        images_data = []

        if isinstance(prompt.prompt, str):
            # Simple text prompt
            prompt_text = prompt.prompt
        else:
            # Prompt with text + images
            prompt_text = prompt.prompt.text
            if prompt.prompt.images:
                # Decode base64 images
                for img_b64 in prompt.prompt.images:
                    # Remove data:image/... prefix if present
                    if ',' in img_b64:
                        img_b64 = img_b64.split(',', 1)[1]

                    images_data.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",  # Claude auto-detects format
                            "data": img_b64
                        }
                    })

        # Build messages content
        if images_data:
            # Vision request: [text, image1, image2, ...]
            content = [{"type": "text", "text": prompt_text}] + images_data
        else:
            # Text-only request
            content = prompt_text

        # Call Claude API
        model_name = model or "claude-3-5-sonnet-20241022"  # Default to latest Claude 3.5 Sonnet
        message = client.messages.create(
            model=model_name,
            max_tokens=prompt.max_tokens or 1000,
            temperature=prompt.temperature or 0.7,
            messages=[
                {"role": "user", "content": content}
            ]
        )

        return AIResponse(
            response=message.content[0].text,
            model=message.model,
            tokens_used=message.usage.input_tokens + message.usage.output_tokens,
            finish_reason=message.stop_reason
        )
    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.error(f"Claude error: {error_msg}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)


@router.post("/chatgpt", response_model=AIResponse)
async def chatgpt_endpoint(
    prompt: Prompt,
    model: Optional[str] = None,
    api_key: str = Depends(get_api_key)
):
    """
    ChatGPT endpoint (OpenAI)

    Features:
    - GPT-4o (latest multimodal model)
    - Fast response times
    - Vision support (can analyze images)
    - Function calling support
    """
    try:
        from openai import OpenAI
        import os

        # Configure OpenAI API
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key or openai_key == "placeholder":
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")

        client = OpenAI(api_key=openai_key)

        # Extract prompt text and images
        prompt_text = ""
        images_data = []

        if isinstance(prompt.prompt, str):
            # Simple text prompt
            prompt_text = prompt.prompt
        else:
            # Prompt with text + images
            prompt_text = prompt.prompt.text
            if prompt.prompt.images:
                # OpenAI expects images in specific format
                for img_b64 in prompt.prompt.images:
                    # Keep data:image/... prefix or add it
                    if not img_b64.startswith('data:image'):
                        img_b64 = f"data:image/png;base64,{img_b64}"

                    images_data.append({
                        "type": "image_url",
                        "image_url": {
                            "url": img_b64
                        }
                    })

        # Build messages content
        if images_data:
            # Vision request: text + images
            content = [{"type": "text", "text": prompt_text}] + images_data
        else:
            # Text-only request
            content = prompt_text

        # Call OpenAI API
        model_name = model or "gpt-4o"  # Default to GPT-4o with vision support
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": content}
            ],
            max_tokens=prompt.max_tokens or 1000,
            temperature=prompt.temperature or 0.7
        )

        return AIResponse(
            response=response.choices[0].message.content,
            model=response.model,
            tokens_used=response.usage.total_tokens,
            finish_reason=response.choices[0].finish_reason
        )
    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.error(f"ChatGPT error: {error_msg}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)


@router.post("/gemini", response_model=AIResponse)
async def gemini_endpoint(
    prompt: Prompt,
    model: Optional[str] = None,
    api_key: str = Depends(get_api_key)
):
    """
    Gemini endpoint (Google)

    Features:
    - Gemini Pro or Ultra
    - Multimodal capabilities (vision support)
    - Fast and cost-effective

    Accepts:
    - Simple string prompt: {"prompt": "your text"}
    - Vision prompt: {"prompt": {"text": "describe this", "images": ["base64..."]}}
    """
    try:
        import google.generativeai as genai
        import os
        import base64
        from PIL import Image
        import io

        # Configure Gemini API
        gemini_key = os.getenv("GOOGLE_API_KEY")
        if not gemini_key or gemini_key == "placeholder":
            raise HTTPException(status_code=500, detail="GOOGLE_API_KEY not configured")

        genai.configure(api_key=gemini_key)

        # Extract prompt text and images
        prompt_text = ""
        images_data = []

        # Select model
        model_name = model or 'gemini-2.5-flash'  # Default to Gemini 2.5 Flash
        gemini_model = genai.GenerativeModel(model_name)

        if isinstance(prompt.prompt, str):
            # Simple text prompt
            prompt_text = prompt.prompt
        else:
            # Vision prompt with text + images
            prompt_text = prompt.prompt.text
            if prompt.prompt.images:
                # Decode base64 images
                for img_b64 in prompt.prompt.images:
                    # Remove data:image/... prefix if present
                    if ',' in img_b64:
                        img_b64 = img_b64.split(',', 1)[1]

                    img_bytes = base64.b64decode(img_b64)
                    img = Image.open(io.BytesIO(img_bytes))
                    images_data.append(img)

        # Build content parts
        if images_data:
            # Vision request: [text, image1, image2, ...]
            content_parts = [prompt_text] + images_data
        else:
            # Text-only request
            content_parts = [prompt_text]

        # Generate response
        response = gemini_model.generate_content(content_parts)

        return AIResponse(
            response=response.text,
            model=f"{model_name}-vision" if images_data else model_name,
            tokens_used=None,  # Gemini doesn't provide token count directly
            finish_reason="stop"
        )
    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.error(f"Gemini error: {error_msg}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)


@router.get("/models")
async def list_text_models():
    """List available text AI models"""
    return {
        "models": [
            {
                "id": "claude-3-opus",
                "provider": "anthropic",
                "context_window": 200000,
                "endpoint": "/ai/claude"
            },
            {
                "id": "gpt-4-turbo",
                "provider": "openai",
                "context_window": 128000,
                "endpoint": "/ai/chatgpt"
            },
            {
                "id": "gemini-pro",
                "provider": "google",
                "context_window": 32000,
                "endpoint": "/ai/gemini"
            }
        ]
    }

