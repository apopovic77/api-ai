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
    api_key: str = Depends(get_api_key)
):
    """
    Claude AI endpoint (Anthropic)
    
    Features:
    - Large context window (100k+ tokens)
    - Excellent at reasoning and analysis
    - Strong safety guardrails
    """
    try:
        # TODO: Implement actual Claude API call
        # from anthropic import Anthropic
        # client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        # ...
        
        return AIResponse(
            response="Claude response - TO BE IMPLEMENTED",
            model="claude-3-sonnet",
            tokens_used=0,
            finish_reason="stop"
        )
    except Exception as e:
        logger.error(f"Claude error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chatgpt", response_model=AIResponse)
async def chatgpt_endpoint(
    prompt: Prompt,
    api_key: str = Depends(get_api_key)
):
    """
    ChatGPT endpoint (OpenAI)
    
    Features:
    - GPT-4 or GPT-3.5-turbo
    - Fast response times
    - Function calling support
    """
    try:
        # TODO: Implement actual OpenAI API call
        # from openai import OpenAI
        # client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # ...
        
        return AIResponse(
            response="ChatGPT response - TO BE IMPLEMENTED",
            model="gpt-4",
            tokens_used=0,
            finish_reason="stop"
        )
    except Exception as e:
        logger.error(f"ChatGPT error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/gemini", response_model=AIResponse)
async def gemini_endpoint(
    prompt: Prompt,
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

        if isinstance(prompt.prompt, str):
            # Simple text prompt
            prompt_text = prompt.prompt
            model = genai.GenerativeModel('gemini-pro')
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

                # Use vision model
                model = genai.GenerativeModel('gemini-pro-vision')
            else:
                model = genai.GenerativeModel('gemini-pro')

        # Build content parts
        if images_data:
            # Vision request: [text, image1, image2, ...]
            content_parts = [prompt_text] + images_data
        else:
            # Text-only request
            content_parts = [prompt_text]

        # Generate response
        response = model.generate_content(content_parts)

        return AIResponse(
            response=response.text,
            model="gemini-pro-vision" if images_data else "gemini-pro",
            tokens_used=None,  # Gemini doesn't provide token count directly
            finish_reason="stop"
        )
    except Exception as e:
        logger.error(f"Gemini error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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

