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

        # Build CLI command with default model sonnet (cost-effective)
        cmd = ["claude", "-p", prompt_text, "--output-format", "json"]

        # Add images using --image flag (vision support)
        for img_path in image_paths:
            cmd.extend(["--image", img_path])

        # Add model - default to sonnet (günstig), user can override with opus/haiku
        selected_model = model or "sonnet"
        cmd.extend(["--model", selected_model])

        # Add system prompt if provided
        if prompt.system:
            cmd.extend(["--system-prompt", prompt.system])

        # Call claude -p in subprocess
        def run_claude_cli():
            import os
            import pwd
            # Create env with HOME for Claude credentials
            # Use CLI_HOME env var, or detect from current user (gsgbot on aiserver, root on arkturian)
            env = os.environ.copy()
            env["NO_COLOR"] = "1"
            cli_home = os.getenv("CLI_HOME") or pwd.getpwuid(os.getuid()).pw_dir
            env["HOME"] = cli_home

            # IMPORTANT: Remove ANTHROPIC_API_KEY so Claude CLI uses OAuth credentials
            # If ANTHROPIC_API_KEY is set (even to "placeholder"), Claude CLI will try
            # to use it instead of the logged-in OAuth credentials
            env.pop("ANTHROPIC_API_KEY", None)

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout for longer prompts
                env=env
            )
            return result

        result = await asyncio.to_thread(run_claude_cli)

        # Log the result for debugging
        logger.info(f"Claude CLI returncode: {result.returncode}")
        logger.info(f"Claude CLI stdout length: {len(result.stdout) if result.stdout else 0}")
        if result.stdout:
            logger.info(f"Claude CLI stdout preview: {result.stdout[:500]}")
        if result.stderr:
            logger.info(f"Claude CLI stderr: {result.stderr[:200]}")

        raw_output = result.stdout.strip() if result.stdout else ""

        # Claude CLI returns errors in JSON stdout with is_error: true
        # So we need to parse JSON FIRST before checking returncode
        if raw_output:
            try:
                cli_response = json_module.loads(raw_output)

                # Check for error in JSON response (Claude returns is_error: true)
                if cli_response.get("is_error"):
                    error_msg = cli_response.get("result", "Unknown error")
                    logger.error(f"Claude CLI error (from JSON): {error_msg}")
                    raise HTTPException(status_code=500, detail=f"Claude error: {error_msg}")

            except json_module.JSONDecodeError as e:
                # Not valid JSON - check if it's a plain error
                if result.returncode != 0:
                    error_msg = result.stderr or raw_output or "Unknown error from claude CLI"
                    logger.error(f"Claude CLI error (non-JSON): {error_msg}")
                    raise HTTPException(status_code=500, detail=f"Claude CLI error: {error_msg}")

                # Otherwise treat as plain text response (unusual but possible)
                logger.warning(f"Claude CLI returned non-JSON output: {raw_output[:200]}")
                return AIResponse(
                    response=raw_output,
                    model="claude-cli",
                    tokens_used=None,
                    finish_reason="stop"
                )
        else:
            # No stdout - check stderr for errors
            if result.returncode != 0:
                error_msg = result.stderr or "Unknown error from claude CLI (empty response)"
                logger.error(f"Claude CLI error (empty stdout): {error_msg}")
                raise HTTPException(status_code=500, detail=f"Claude CLI error: {error_msg}")

            raise HTTPException(status_code=500, detail="Claude CLI returned empty response")

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
    ChatGPT endpoint via OpenAI Codex CLI

    Features:
    - Uses logged-in ChatGPT account (no API costs on your bill!)
    - Codex CLI in exec mode for non-interactive use
    - JSONL output for response extraction
    - Default model: o4-mini (fast and cost-effective)
    - System prompt support
    - Cost tracking at /ai/chatgpt/cost-status

    Note: Uses your ChatGPT Plus/Pro subscription - no API billing.
    """
    import subprocess
    import asyncio
    import json as json_module
    from ..services.codex_cost_tracker import codex_cost_tracker

    try:
        # Extract prompt text
        prompt_text = ""
        if isinstance(prompt.prompt, str):
            prompt_text = prompt.prompt
        else:
            prompt_text = prompt.prompt.text
            if prompt.prompt.images:
                logger.warning("Codex CLI mode does not support images via this endpoint - ignoring images")

        # Codex CLI has no --system-prompt, so prepend system prompt to user prompt
        if prompt.system:
            prompt_text = f"{prompt.system}\n\n{prompt_text}"
            logger.info(f"Prepended system prompt ({len(prompt.system)} chars) to user prompt")

        logger.info(f"Calling codex exec with prompt length: {len(prompt_text)} chars")

        # Build CLI command
        # codex exec --json --skip-git-repo-check "prompt"
        cmd = ["codex", "exec", "--json", "--skip-git-repo-check"]

        # Only add model if explicitly specified by user
        # (ChatGPT subscription has limited model access, let Codex use its default)
        selected_model = model  # None if not specified
        if selected_model:
            cmd.extend(["--model", selected_model])

        # Add prompt as argument
        cmd.append(prompt_text)

        # Call codex exec in subprocess
        def run_codex_cli():
            import os
            import pwd
            env = os.environ.copy()
            env["NO_COLOR"] = "1"
            cli_home = os.getenv("CLI_HOME") or pwd.getpwuid(os.getuid()).pw_dir
            env["HOME"] = cli_home

            # Remove OPENAI_API_KEY so Codex CLI uses OAuth credentials
            env.pop("OPENAI_API_KEY", None)

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                env=env
            )
            return result

        result = await asyncio.to_thread(run_codex_cli)

        # Log the result for debugging
        logger.info(f"Codex CLI returncode: {result.returncode}")
        logger.info(f"Codex CLI stdout length: {len(result.stdout) if result.stdout else 0}")
        if result.stderr:
            logger.info(f"Codex CLI stderr: {result.stderr[:200]}")

        raw_output = result.stdout.strip() if result.stdout else ""

        if not raw_output:
            if result.returncode != 0:
                error_msg = result.stderr or "Unknown error from codex CLI (empty response)"
                logger.error(f"Codex CLI error (empty stdout): {error_msg}")
                raise HTTPException(status_code=500, detail=f"Codex CLI error: {error_msg}")
            raise HTTPException(status_code=500, detail="Codex CLI returned empty response")

        # Parse JSONL output - each line is a separate JSON object
        # We need to find the agent_message item and usage info
        response_text = ""
        input_tokens = 0
        output_tokens = 0
        model_name = selected_model or "codex-default"  # Track as codex-default if no model specified

        for line in raw_output.split('\n'):
            if not line.strip():
                continue
            try:
                event = json_module.loads(line)

                # Extract agent message
                if event.get("type") == "item.completed":
                    item = event.get("item", {})
                    if item.get("type") == "agent_message":
                        response_text = item.get("text", "")

                # Extract usage info
                if event.get("type") == "turn.completed":
                    usage = event.get("usage", {})
                    input_tokens = usage.get("input_tokens", 0)
                    output_tokens = usage.get("output_tokens", 0)

            except json_module.JSONDecodeError:
                continue

        if not response_text:
            logger.error(f"No agent_message found in Codex output: {raw_output[:500]}")
            raise HTTPException(status_code=500, detail="No response from Codex CLI")

        # Track usage
        codex_cost_tracker.track_usage({
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "model": model_name
        })

        tokens_used = input_tokens + output_tokens

        logger.info(
            f"Codex CLI response: {len(response_text)} chars, "
            f"{tokens_used} tokens ({input_tokens}in/{output_tokens}out)"
        )

        return AIResponse(
            response=response_text,
            model=model_name,
            tokens_used=tokens_used,
            finish_reason="stop"
        )

    except subprocess.TimeoutExpired:
        logger.error("Codex CLI timeout after 300 seconds")
        raise HTTPException(status_code=504, detail="Codex CLI timeout - prompt may be too long")
    except FileNotFoundError:
        logger.error("Codex CLI not found - is 'codex' installed?")
        raise HTTPException(status_code=500, detail="Codex CLI not installed on server")
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.error(f"Codex error: {error_msg}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)


@router.get("/chatgpt/cost-status")
async def chatgpt_cost_status():
    """
    Get current Codex CLI cost tracking status.

    Returns:
    - Current month's usage and costs
    - Token breakdown by model
    - Request statistics

    Note: Costs are informational - Codex CLI uses your subscription.
    """
    from ..services.codex_cost_tracker import codex_cost_tracker
    return codex_cost_tracker.get_status()


@router.post("/gemini", response_model=AIResponse)
async def gemini_endpoint(
    prompt: Prompt,
    model: Optional[str] = None,
    api_key: str = Depends(get_api_key)
):
    """
    Gemini endpoint via Google Gemini CLI

    Features:
    - Uses free tier (60 req/min, 1000 req/day - NO API costs!)
    - Gemini 2.5 Flash-Lite with 1M token context
    - JSON output for token tracking
    - Usage tracking at /ai/gemini/cost-status

    Note: Uses Google's free tier CLI - no API billing.
    """
    import subprocess
    import asyncio
    import json as json_module
    from ..services.gemini_cli_cost_tracker import gemini_cli_cost_tracker

    try:
        # Extract prompt text
        prompt_text = ""
        if isinstance(prompt.prompt, str):
            prompt_text = prompt.prompt
        else:
            prompt_text = prompt.prompt.text
            if prompt.prompt.images:
                logger.warning("Gemini CLI mode does not support images via this endpoint - ignoring images")

        # Gemini CLI has no --system-prompt, so prepend system prompt to user prompt
        if prompt.system:
            prompt_text = f"{prompt.system}\n\n{prompt_text}"
            logger.info(f"Prepended system prompt ({len(prompt.system)} chars) to user prompt")

        logger.info(f"Calling gemini CLI with prompt length: {len(prompt_text)} chars")

        # Build CLI command
        # gemini --output-format json "prompt"
        cmd = ["gemini", "--output-format", "json", prompt_text]

        # Call gemini in subprocess
        def run_gemini_cli():
            import os
            import pwd
            env = os.environ.copy()
            env["NO_COLOR"] = "1"
            cli_home = os.getenv("CLI_HOME") or pwd.getpwuid(os.getuid()).pw_dir
            env["HOME"] = cli_home

            # Gemini CLI expects GEMINI_API_KEY, but we use GOOGLE_API_KEY
            google_key = os.getenv("GOOGLE_API_KEY", "")
            if google_key:
                env["GEMINI_API_KEY"] = google_key

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                env=env
            )
            return result

        result = await asyncio.to_thread(run_gemini_cli)

        # Log the result for debugging
        logger.info(f"Gemini CLI returncode: {result.returncode}")
        logger.info(f"Gemini CLI stdout length: {len(result.stdout) if result.stdout else 0}")
        if result.stderr:
            logger.info(f"Gemini CLI stderr: {result.stderr[:200]}")

        raw_output = result.stdout.strip() if result.stdout else ""

        if not raw_output:
            if result.returncode != 0:
                error_msg = result.stderr or "Unknown error from gemini CLI (empty response)"
                logger.error(f"Gemini CLI error (empty stdout): {error_msg}")
                raise HTTPException(status_code=500, detail=f"Gemini CLI error: {error_msg}")
            raise HTTPException(status_code=500, detail="Gemini CLI returned empty response")

        # Parse JSON output
        # Format: {"response": "...", "stats": {"models": {"gemini-2.5-flash-lite": {"tokens": {...}}}}}
        try:
            cli_response = json_module.loads(raw_output)
        except json_module.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini CLI JSON: {e}")
            logger.error(f"Raw output: {raw_output[:500]}")
            # Return raw output as response if JSON parsing fails
            return AIResponse(
                response=raw_output,
                model="gemini-cli",
                tokens_used=None,
                finish_reason="stop"
            )

        # Extract response text
        response_text = cli_response.get("response", "")

        # Extract token info from stats
        input_tokens = 0
        output_tokens = 0
        model_name = "gemini-cli"

        stats = cli_response.get("stats", {})
        models = stats.get("models", {})

        # Get the first model from stats (usually gemini-2.5-flash-lite)
        for m_name, m_data in models.items():
            model_name = m_name
            tokens = m_data.get("tokens", {})
            input_tokens = tokens.get("prompt", 0)
            output_tokens = tokens.get("candidates", 0)
            break

        # Track usage
        gemini_cli_cost_tracker.track_usage({
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "model": model_name
        })

        tokens_used = input_tokens + output_tokens

        logger.info(
            f"Gemini CLI response: {len(response_text)} chars, "
            f"{tokens_used} tokens ({input_tokens}in/{output_tokens}out)"
        )

        return AIResponse(
            response=response_text,
            model=model_name,
            tokens_used=tokens_used,
            finish_reason="stop"
        )

    except subprocess.TimeoutExpired:
        logger.error("Gemini CLI timeout after 300 seconds")
        raise HTTPException(status_code=504, detail="Gemini CLI timeout - prompt may be too long")
    except FileNotFoundError:
        logger.error("Gemini CLI not found - is 'gemini' installed?")
        raise HTTPException(status_code=500, detail="Gemini CLI not installed on server")
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.error(f"Gemini CLI error: {error_msg}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)


@router.get("/gemini/cost-status")
async def gemini_cost_status():
    """
    Get current Gemini CLI usage tracking status.

    Returns:
    - Current month's usage and estimated costs
    - Token breakdown by model
    - Request statistics

    Note: Costs are informational only - Gemini CLI uses free tier (no actual billing).
    """
    from ..services.gemini_cli_cost_tracker import gemini_cli_cost_tracker
    return gemini_cli_cost_tracker.get_status()


@router.post("/gemini/vision", response_model=AIResponse)
async def gemini_vision_endpoint(
    prompt: Prompt,
    model: Optional[str] = None,
    api_key: str = Depends(get_api_key)
):
    """
    Gemini Vision endpoint via Google Generative AI API.

    Features:
    - Supports multimodal input (text + images)
    - Uses Gemini 2.0 Flash for vision tasks
    - Base64 encoded images in prompt.images array
    - Perfect for image analysis, safety checks, content moderation

    Note: Uses GOOGLE_API_KEY from environment.
    """
    import os
    import base64
    import google.generativeai as genai
    from ..services.gemini_cli_cost_tracker import gemini_cli_cost_tracker

    try:
        # Get API key from environment
        google_api_key = os.getenv("GOOGLE_API_KEY", "")
        if not google_api_key:
            raise HTTPException(status_code=500, detail="GOOGLE_API_KEY not configured")

        # Configure Gemini
        genai.configure(api_key=google_api_key)

        # Extract prompt text and images
        prompt_text = ""
        images_b64 = []

        if isinstance(prompt.prompt, str):
            prompt_text = prompt.prompt
        else:
            prompt_text = prompt.prompt.text
            if prompt.prompt.images:
                images_b64 = prompt.prompt.images

        logger.info(f"Calling Gemini Vision API with prompt length: {len(prompt_text)} chars, images: {len(images_b64)}")

        # Select model - use gemini-2.0-flash for vision (fast and capable)
        model_name = model or "gemini-2.0-flash"
        gemini_model = genai.GenerativeModel(model_name)

        # Build content parts
        content_parts = []

        # Add images first (better for vision tasks)
        for i, img_b64 in enumerate(images_b64):
            try:
                # Handle data URL format: "data:image/jpeg;base64,..."
                if img_b64.startswith("data:"):
                    # Extract mime type and base64 data
                    header, b64_data = img_b64.split(",", 1)
                    mime_type = header.split(":")[1].split(";")[0]
                else:
                    # Assume raw base64 JPEG
                    b64_data = img_b64
                    mime_type = "image/jpeg"

                # Decode and create image part
                image_data = base64.b64decode(b64_data)
                content_parts.append({
                    "mime_type": mime_type,
                    "data": image_data
                })
                logger.info(f"Added image {i+1}: {mime_type}, {len(image_data)} bytes")
            except Exception as e:
                logger.warning(f"Failed to process image {i+1}: {e}")
                continue

        # Add text prompt
        content_parts.append(prompt_text)

        # Generate response
        response = gemini_model.generate_content(content_parts)

        # Extract response text
        response_text = response.text if response.text else ""

        # Extract token counts if available
        input_tokens = 0
        output_tokens = 0
        if hasattr(response, 'usage_metadata'):
            input_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0)
            output_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0)

        # Track usage
        gemini_cli_cost_tracker.track_usage({
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "model": model_name
        })

        tokens_used = input_tokens + output_tokens

        logger.info(
            f"Gemini Vision response: {len(response_text)} chars, "
            f"{tokens_used} tokens ({input_tokens}in/{output_tokens}out)"
        )

        return AIResponse(
            response=response_text,
            model=model_name,
            tokens_used=tokens_used,
            finish_reason="stop"
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.error(f"Gemini Vision error: {error_msg}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)


@router.post("/gemini/send-report")
async def gemini_send_report():
    """
    Manually trigger a usage report via Telegram.

    Note: Shows CLI usage statistics (no actual costs - free tier).
    """
    from ..services.gemini_cli_cost_tracker import gemini_cli_cost_tracker

    # Get status and send via Telegram
    status = gemini_cli_cost_tracker.get_status()

    import httpx
    import os
    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat = os.getenv("TELEGRAM_ADMIN_CHAT_ID", "")

    if telegram_token and telegram_chat:
        message = f"""
📊 <b>Gemini CLI Usage Report</b>

<b>Month:</b> {status.get('month', 'N/A')}
<b>Requests:</b> {status.get('request_count', 0):,}
<b>Input Tokens:</b> {status.get('total_input_tokens', 0):,}
<b>Output Tokens:</b> {status.get('total_output_tokens', 0):,}

<b>Estimated Value:</b> ${status.get('total_cost_usd', 0):.4f} (if API)

<i>Note: Gemini CLI uses free tier - no actual costs!</i>
"""
        try:
            url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
            with httpx.Client(timeout=10.0) as client:
                client.post(url, json={"chat_id": telegram_chat, "text": message.strip(), "parse_mode": "HTML"})
        except Exception:
            pass

    return {"status": "Report sent", "usage": status}


@router.post("/gemini/gcp-budget-webhook")
async def gcp_budget_webhook(request: Request):
    """
    Webhook endpoint for GCP Budget Pub/Sub push notifications.

    Configure in GCP:
    1. Create Push Subscription on gemini-budget-alerts Topic
    2. Endpoint URL: https://api-ai.arkturian.com/ai/gemini/gcp-budget-webhook

    This is a BACKUP alert system with ~24h delay.
    Primary alerting is done via real-time token tracking.

    NOTE: Uses NotificationManager to limit alerts to max 1 per day.
    """
    import base64
    import json
    from ..services.cost_tracker import cost_tracker
    from ..services.notification_manager import notification_manager

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

            logger.info(f"GCP Budget webhook received: {threshold:.0f}% threshold for {budget_name}")

            # Check if we can send notification (cooldown: 24h)
            if not notification_manager.can_send("gcp_budget_alert"):
                next_allowed = notification_manager.get_next_allowed("gcp_budget_alert")
                logger.info(
                    f"GCP Budget alert suppressed (cooldown). "
                    f"Next allowed: {next_allowed.isoformat() if next_allowed else 'now'}"
                )
                return {"status": "ok", "notification": "suppressed_cooldown"}

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
            notification_manager.mark_sent("gcp_budget_alert")
            logger.info(f"GCP Budget webhook: sent alert for {threshold:.0f}% threshold")

        return {"status": "ok", "notification": "sent"}

    except Exception as e:
        logger.error(f"GCP Budget webhook error: {e}")
        return {"status": "error", "message": str(e)}


@router.get("/notifications/status")
async def get_notification_status():
    """
    Get notification manager status.

    Shows cooldown state for all notification types.
    """
    from ..services.notification_manager import notification_manager
    return notification_manager.get_status()


@router.post("/notifications/reset/{notification_type}")
async def reset_notification_cooldown(notification_type: str):
    """
    Reset cooldown for a specific notification type.

    Args:
        notification_type: Type to reset (e.g., "gcp_budget_alert")
    """
    from ..services.notification_manager import notification_manager
    notification_manager.reset(notification_type)
    return {"status": "reset", "notification_type": notification_type}


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

