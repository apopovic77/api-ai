"""
Text AI Routes
==============

Endpoints for text-based AI models:
- Claude (Anthropic)
- ChatGPT (OpenAI)
- Gemini (Google)
"""

from fastapi import APIRouter, Depends, HTTPException, Request
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
    Claude AI endpoint via Claude Code CLI (claude -p)

    Features:
    - Uses logged-in Claude account (no API costs on your bill!)
    - Claude Code in print mode (-p) for non-interactive use
    - JSON output for token tracking
    - Supports --model parameter (sonnet, opus, haiku)
    - System prompt support via --system-prompt
    - Vision support via --image flag (URLs or local files)
    - Cost tracking at /ai/claude/cost-status

    Note: Costs shown are informational - uses your Claude subscription.
    """
    import subprocess
    import asyncio
    import json as json_module
    import tempfile
    import os
    import httpx
    import base64
    from ..services.claude_cost_tracker import claude_cost_tracker

    temp_files = []  # Track temp files for cleanup

    try:
        # Extract prompt text and images
        prompt_text = ""
        image_paths = []

        if isinstance(prompt.prompt, str):
            prompt_text = prompt.prompt
        else:
            prompt_text = prompt.prompt.text
            if prompt.prompt.images:
                logger.info(f"Processing {len(prompt.prompt.images)} images for Claude vision")
                for idx, img_source in enumerate(prompt.prompt.images):
                    try:
                        # Check if it's a URL or base64
                        if img_source.startswith('http://') or img_source.startswith('https://'):
                            # Download image from URL
                            logger.info(f"Downloading image {idx+1} from URL: {img_source[:100]}...")
                            async with httpx.AsyncClient(timeout=30.0) as client:
                                resp = await client.get(img_source)
                                resp.raise_for_status()
                                img_data = resp.content

                                # Determine extension from content-type
                                content_type = resp.headers.get('content-type', 'image/png')
                                ext = '.png'
                                if 'jpeg' in content_type or 'jpg' in content_type:
                                    ext = '.jpg'
                                elif 'webp' in content_type:
                                    ext = '.webp'
                                elif 'gif' in content_type:
                                    ext = '.gif'

                                # Save to temp file
                                temp_file = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
                                temp_file.write(img_data)
                                temp_file.close()
                                temp_files.append(temp_file.name)
                                image_paths.append(temp_file.name)
                                logger.info(f"Downloaded image to {temp_file.name} ({len(img_data)} bytes)")

                        elif img_source.startswith('data:image'):
                            # Base64 data URL
                            header, b64_data = img_source.split(',', 1)
                            img_data = base64.b64decode(b64_data)

                            # Determine extension
                            ext = '.png'
                            if 'jpeg' in header or 'jpg' in header:
                                ext = '.jpg'
                            elif 'webp' in header:
                                ext = '.webp'

                            temp_file = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
                            temp_file.write(img_data)
                            temp_file.close()
                            temp_files.append(temp_file.name)
                            image_paths.append(temp_file.name)
                            logger.info(f"Decoded base64 image to {temp_file.name}")

                        else:
                            # Assume raw base64 (no data: prefix)
                            img_data = base64.b64decode(img_source)
                            temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                            temp_file.write(img_data)
                            temp_file.close()
                            temp_files.append(temp_file.name)
                            image_paths.append(temp_file.name)
                            logger.info(f"Decoded raw base64 image to {temp_file.name}")

                    except Exception as img_err:
                        logger.warning(f"Failed to process image {idx+1}: {img_err}")

        logger.info(f"Calling claude -p with prompt length: {len(prompt_text)} chars, {len(image_paths)} images")

        # Build CLI command
        cmd = ["claude", "-p", prompt_text, "--output-format", "json"]

        # Add images using --image flag
        for img_path in image_paths:
            cmd.extend(["--image", img_path])

        # Add model if specified
        if model:
            cmd.extend(["--model", model])

        # Add system prompt if provided
        if prompt.system:
            cmd.extend(["--system-prompt", prompt.system])

        # Call claude -p in subprocess
        def run_claude_cli():
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout for longer prompts
                env={**subprocess.os.environ, "NO_COLOR": "1"}
            )
            return result

        result = await asyncio.to_thread(run_claude_cli)

        if result.returncode != 0:
            error_msg = result.stderr or "Unknown error from claude CLI"
            logger.error(f"Claude CLI error: {error_msg}")
            raise HTTPException(status_code=500, detail=f"Claude CLI error: {error_msg}")

        raw_output = result.stdout.strip()

        if not raw_output:
            raise HTTPException(status_code=500, detail="Claude CLI returned empty response")

        # Parse JSON response
        try:
            cli_response = json_module.loads(raw_output)
        except json_module.JSONDecodeError as e:
            logger.error(f"Failed to parse Claude CLI JSON: {e}")
            # Fallback: treat as plain text
            return AIResponse(
                response=raw_output,
                model="claude-cli",
                tokens_used=None,
                finish_reason="stop"
            )

        # Check for error
        if cli_response.get("is_error"):
            error_msg = cli_response.get("result", "Unknown error")
            raise HTTPException(status_code=500, detail=f"Claude error: {error_msg}")

        # Track usage
        claude_cost_tracker.track_usage(cli_response)

        # Extract response text
        response_text = cli_response.get("result", "")

        # Calculate total tokens
        usage = cli_response.get("usage", {})
        tokens_used = (
            usage.get("input_tokens", 0) +
            usage.get("output_tokens", 0) +
            usage.get("cache_read_input_tokens", 0)
        )

        # Determine model used
        model_usage = cli_response.get("modelUsage", {})
        models_used = list(model_usage.keys())
        model_name = models_used[0] if models_used else "claude-cli"

        logger.info(
            f"Claude CLI response: {len(response_text)} chars, "
            f"${cli_response.get('total_cost_usd', 0):.4f}, "
            f"{tokens_used} tokens"
        )

        return AIResponse(
            response=response_text,
            model=model_name,
            tokens_used=tokens_used,
            finish_reason="stop"
        )

    except subprocess.TimeoutExpired:
        logger.error("Claude CLI timeout after 300 seconds")
        raise HTTPException(status_code=504, detail="Claude CLI timeout - prompt may be too long")
    except FileNotFoundError:
        logger.error("Claude CLI not found - is 'claude' installed?")
        raise HTTPException(status_code=500, detail="Claude CLI not installed on server")
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.error(f"Claude error: {error_msg}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)
    finally:
        # Cleanup temp files
        for temp_path in temp_files:
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    logger.debug(f"Cleaned up temp file: {temp_path}")
            except Exception as cleanup_err:
                logger.warning(f"Failed to cleanup temp file {temp_path}: {cleanup_err}")


@router.get("/claude/cost-status")
async def claude_cost_status():
    """
    Get current Claude CLI cost tracking status.

    Returns:
    - Current month's usage and costs
    - Token breakdown by model
    - Request statistics

    Note: Costs are informational - Claude CLI uses your subscription.
    """
    from ..services.claude_cost_tracker import claude_cost_tracker
    return claude_cost_tracker.get_status()


@router.post("/chatgpt", response_model=AIResponse)
async def chatgpt_endpoint(
    prompt: Prompt,
    model: Optional[str] = None,
    api_key: str = Depends(get_api_key)
):
    """
    ChatGPT endpoint (OpenAI)

    Features:
    - GPT-4.1 (default) - Latest 2025 model, best coding & instruction following
    - Context window: up to 1M tokens
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
        model_name = model or "gpt-4.1"  # Default to GPT-4.1 (latest 2025, best performance)
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
    - Gemini 2.5 Flash (default) - Stable GA model, fast and cost-effective
    - Alternative models: gemini-2.5-pro, gemini-2.5-flash-lite
    - Multimodal capabilities (vision support)
    - Note: All Gemini 1.x models are retired
    - Cost tracking with monthly budget limit (default: 30 EUR)

    Accepts:
    - Simple string prompt: {"prompt": "your text"}
    - Vision prompt: {"prompt": {"text": "describe this", "images": ["base64..."]}}

    Returns 429 if monthly budget is exceeded.
    """
    try:
        import google.generativeai as genai
        import os
        import base64
        from PIL import Image
        import io
        from ..services.cost_tracker import cost_tracker

        # Check budget before processing
        if cost_tracker.should_block_request():
            status = cost_tracker.get_status()
            raise HTTPException(
                status_code=429,
                detail=f"Monthly Gemini API budget exceeded: {status['total_cost_eur']:.2f}/{status['monthly_budget_eur']:.2f} EUR. Requests blocked until next month."
            )

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

        # Extract token usage from usage_metadata
        tokens_used = None
        input_tokens = 0
        output_tokens = 0

        # Debug log to see what we get from Gemini
        logger.info(f"Gemini response type: {type(response)}")
        logger.info(f"Has usage_metadata: {hasattr(response, 'usage_metadata')}")
        if hasattr(response, 'usage_metadata'):
            logger.info(f"usage_metadata: {response.usage_metadata}")

        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            input_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0) or 0
            output_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0) or 0
            tokens_used = input_tokens + output_tokens
            logger.info(f"Extracted tokens: input={input_tokens}, output={output_tokens}")

        # Track usage for cost monitoring (even if 0, to count requests)
        if input_tokens > 0 or output_tokens > 0:
            cost_tracker.track_usage(
                model=model_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens
            )
            logger.info(f"Tracked Gemini usage: {input_tokens}in/{output_tokens}out")
        else:
            logger.warning(f"No token info from Gemini - cannot track usage")

        return AIResponse(
            response=response.text,
            model=f"{model_name}-vision" if images_data else model_name,
            tokens_used=tokens_used,
            finish_reason="stop"
        )
    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.error(f"Gemini error: {error_msg}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)


@router.get("/gemini/cost-status")
async def gemini_cost_status():
    """
    Get current Gemini API cost tracking status.

    Returns:
    - Current month's usage and costs
    - Budget information
    - Whether requests are blocked
    """
    from ..services.cost_tracker import cost_tracker
    return cost_tracker.get_status()


@router.post("/gemini/send-report")
async def gemini_send_report():
    """
    Manually trigger a daily report via Telegram.
    """
    from ..services.cost_tracker import cost_tracker
    cost_tracker.send_daily_report()
    return {"status": "Report sent"}


@router.post("/gemini/gcp-budget-webhook")
async def gcp_budget_webhook(request: Request):
    """
    Webhook endpoint for GCP Budget Pub/Sub push notifications.

    Configure in GCP:
    1. Create Push Subscription on gemini-budget-alerts Topic
    2. Endpoint URL: https://api-ai.arkturian.com/ai/gemini/gcp-budget-webhook

    This is a BACKUP alert system with ~24h delay.
    Primary alerting is done via real-time token tracking.
    """
    import base64
    import json
    from ..services.cost_tracker import cost_tracker

    try:
        body = await request.json()

        # Decode Pub/Sub message
        message_data = body.get("message", {}).get("data", "")
        if message_data:
            decoded = base64.b64decode(message_data).decode()
            budget_data = json.loads(decoded)

            budget_name = budget_data.get("budgetDisplayName", "Unknown")
            cost_amount = budget_data.get("costAmount", 0)
            budget_amount = budget_data.get("budgetAmount", 0)
            threshold = budget_data.get("alertThresholdExceeded", 0) * 100

            # Determine emoji and status
            if threshold >= 100:
                emoji = "🚨"
                status = "GCP: BUDGET EXCEEDED"
            elif threshold >= 80:
                emoji = "⚠️"
                status = "GCP: WARNING"
            else:
                emoji = "📊"
                status = "GCP: INFO"

            message = f"""
{emoji} <b>{status}</b>

<b>Budget:</b> {budget_name}
<b>Threshold:</b> {threshold:.0f}%
<b>GCP Cost:</b> ${cost_amount:.2f} / ${budget_amount:.2f}

<i>Note: GCP data has ~24h delay</i>
<i>Real-time status: /ai/gemini/cost-status</i>
"""
            cost_tracker._send_telegram_message(message.strip())
            logger.info(f"GCP Budget webhook: {threshold}% threshold for {budget_name}")

        return {"status": "ok"}

    except Exception as e:
        logger.error(f"GCP Budget webhook error: {e}")
        return {"status": "error", "message": str(e)}


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

