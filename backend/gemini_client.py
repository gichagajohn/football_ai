"""
Shared Gemini client for Football Pulse AI.

All LLM calls go through this module so both the Scout and pipeline agents
use the same model selection, retry, rate-limit, and truncation behaviour.
The client is created lazily because GitHub Actions injects secrets at runtime.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

# Keep the model configurable. The second model is only used when the primary
# model is unavailable or rate-limited; both are Gemini API model IDs.
_DEFAULT_MODELS = ("gemini-2.5-flash", "gemini-2.5-flash-lite")
_model_names = tuple(
    name.strip()
    for name in os.environ.get("GEMINI_MODELS", "").split(",")
    if name.strip()
) or _DEFAULT_MODELS

_current_model_index = 0
_client: genai.Client | None = None
_last_finish_reason: str | None = None
_last_model: str | None = None
_last_request_at = 0.0

MAX_RETRIES = max(1, int(os.environ.get("GEMINI_MAX_RETRIES", "3")))
MAX_SINGLE_WAIT_SECONDS = float(os.environ.get("GEMINI_MAX_SINGLE_WAIT_SECONDS", "30"))
REQUEST_INTERVAL_SECONDS = float(os.environ.get("GEMINI_REQUEST_INTERVAL_SECONDS", "0.25"))
TRUNCATION_RETRY_CEILING = int(os.environ.get("GEMINI_TRUNCATION_RETRY_CEILING", "12000"))


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Add your Gemini API key to the "
                "GEMINI_API_KEY environment variable or GitHub Secret."
            )
        _client = genai.Client(api_key=api_key)
    return _client


def get_last_finish_reason() -> str | None:
    """Return the finish reason from the most recent Gemini response."""
    return _last_finish_reason


def get_current_model() -> str:
    return _model_names[_current_model_index]


def _switch_to_next_model() -> bool:
    global _current_model_index
    if _current_model_index >= len(_model_names) - 1:
        return False
    _current_model_index += 1
    logger.warning("[GEMINI] Switching to fallback model %s", get_current_model())
    return True


def _error_code(error: Exception) -> int | None:
    code = getattr(error, "code", None)
    if code is None:
        response = getattr(error, "response", None)
        code = getattr(response, "status_code", None)
    try:
        return int(code) if code is not None else None
    except (TypeError, ValueError):
        return None


def _is_retryable(error: Exception) -> bool:
    code = _error_code(error)
    message = str(error).lower()
    return (
        code in {408, 409, 429, 500, 502, 503, 504}
        or "rate limit" in message
        or "resource exhausted" in message
        or "quota" in message
        or "temporarily unavailable" in message
    )


def _retry_after_seconds(error: Exception) -> float | None:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if headers:
        for key in ("retry-after", "Retry-After"):
            value = headers.get(key)
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    pass
    return None


def _pace() -> None:
    """Avoid bursts that commonly trigger Gemini free-tier RPM limits."""
    global _last_request_at
    wait = REQUEST_INTERVAL_SECONDS - (time.monotonic() - _last_request_at)
    if wait > 0:
        time.sleep(wait)
    _last_request_at = time.monotonic()


def _build_contents(messages: list[dict[str, Any]]) -> tuple[str, list[types.Content]]:
    system_parts: list[str] = []
    contents: list[types.Content] = []

    for message in messages:
        text = str(message.get("content", ""))
        role = message.get("role", "user")
        if role == "system":
            system_parts.append(text)
            continue
        gemini_role = "model" if role == "assistant" else "user"
        contents.append(
            types.Content(
                role=gemini_role,
                parts=[types.Part.from_text(text=text)],
            )
        )

    return "\n\n".join(system_parts), contents


def _wants_json(messages: list[dict[str, Any]]) -> bool:
    """Use Gemini's JSON response mode for the existing structured agents."""
    text = "\n".join(str(message.get("content", "")) for message in messages).lower()
    if "plain text (not json)" in text:
        return False
    return any(
        phrase in text
        for phrase in (
            "respond with only the json",
            "return json",
            "output json",
            "valid json",
        )
    )


def _extract_text(response: Any) -> str:
    try:
        return response.text or ""
    except (AttributeError, ValueError):
        # Safety/blocked responses may not expose response.text.
        try:
            candidate = response.candidates[0]
            parts = candidate.content.parts or []
            return "".join(getattr(part, "text", "") or "" for part in parts)
        except (AttributeError, IndexError, TypeError):
            return ""


def _finish_reason(response: Any) -> str | None:
    try:
        reason = response.candidates[0].finish_reason
    except (AttributeError, IndexError, TypeError):
        return None
    name = getattr(reason, "name", None)
    value = str(name or reason).lower() if reason is not None else None
    # Keep the diagnostic value used by the existing agents provider-neutral.
    return "length" if value and "max_token" in value else value


def gemini_chat(
    *,
    max_tokens: int,
    messages: list[dict[str, Any]],
    truncation_ceiling: int = TRUNCATION_RETRY_CEILING,
) -> str:
    """Generate one response using Gemini.

    ``messages`` intentionally keeps the existing role/content shape so the
    agent prompts do not need to be rewritten. These are bounded structured
    output tasks, so the Gemini client uses JSON response mode when the prompt
    requests JSON and plain text mode for the publisher.
    """
    global _last_finish_reason, _last_model

    system_instruction, contents = _build_contents(messages)
    if not contents:
        contents = [types.Content(role="user", parts=[types.Part.from_text(text="")])]

    config_kwargs: dict[str, Any] = {"max_output_tokens": max_tokens}
    if system_instruction:
        config_kwargs["system_instruction"] = system_instruction
    if _wants_json(messages):
        config_kwargs["response_mime_type"] = "application/json"

    attempt = 0
    while True:
        model = get_current_model()
        _pace()
        try:
            response = _get_client().models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(**config_kwargs),
            )
        except Exception as error:
            code = _error_code(error)
            if code == 404 and _switch_to_next_model():
                attempt = 0
                continue
            if not _is_retryable(error):
                raise

            retry_after = _retry_after_seconds(error)
            wait = retry_after if retry_after is not None else min(2 ** attempt, 10)
            if wait > MAX_SINGLE_WAIT_SECONDS:
                if _switch_to_next_model():
                    attempt = 0
                    continue
                raise RuntimeError(
                    f"Gemini API: {model} is unavailable for approximately "
                    f"{wait:.0f}s and no fallback model remains."
                ) from error

            attempt += 1
            if attempt >= MAX_RETRIES:
                if _switch_to_next_model():
                    attempt = 0
                    continue
                raise RuntimeError(
                    f"Gemini API: {model} remained unavailable after "
                    f"{MAX_RETRIES} retries."
                ) from error

            logger.warning(
                "[GEMINI] %s error on %s (attempt %d/%d) — retrying in %.1fs",
                code or "API",
                model,
                attempt,
                MAX_RETRIES,
                wait,
            )
            time.sleep(wait)
            continue

        _last_model = model
        _last_finish_reason = _finish_reason(response)
        content = _extract_text(response)

        # Gemini uses MAX_TOKENS for an output truncation. Retry with a larger
        # output budget before the caller sees an incomplete JSON document.
        if (
            _last_finish_reason
            and ("max_token" in _last_finish_reason or "length" in _last_finish_reason)
            and max_tokens < truncation_ceiling
        ):
            bumped = min(max_tokens * 2, truncation_ceiling)
            logger.warning(
                "[GEMINI] %s response reached %d output tokens — retrying with %d",
                model,
                max_tokens,
                bumped,
            )
            max_tokens = bumped
            config_kwargs["max_output_tokens"] = max_tokens
            continue

        return content
