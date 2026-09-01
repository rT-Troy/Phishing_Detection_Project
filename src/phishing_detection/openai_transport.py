"""Small HTTPS adapter for the frozen OpenAI chat request."""

from __future__ import annotations

import json
import socket
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .llm import RetryableTransportError, TransportResponse

OPENAI_CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"


class OpenAIRequestError(RuntimeError):
    """A non-retryable provider rejection with content-free diagnostics."""


def _safe(value: object) -> str:
    if not isinstance(value, str):
        return "unknown"
    cleaned = "".join(
        character
        for character in value[:80]
        if character.isalnum() or character in "._-"
    )
    return cleaned or "unknown"


class OpenAIHttpTransport:
    def __init__(
        self,
        *,
        api_key: str,
        opener: Callable[..., Any] = urlopen,
        timeout_seconds: int = 60,
    ) -> None:
        if not api_key.strip():
            raise ValueError("OpenAI API key is empty")
        self._api_key, self._opener, self._timeout = api_key, opener, timeout_seconds

    def complete_chat(
        self, request: dict[str, object], *, request_key: str
    ) -> TransportResponse:
        http_request = Request(
            OPENAI_CHAT_COMPLETIONS_URL,
            data=json.dumps(request, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Idempotency-Key": request_key,
            },
            method="POST",
        )
        try:
            with self._opener(http_request, timeout=self._timeout) as response:
                body = response.read()
        except HTTPError as error:
            try:
                detail = json.loads(error.read(64 * 1024))["error"]
                error_type, error_code = _safe(detail.get("type")), _safe(
                    detail.get("code")
                )
            except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
                error_type = error_code = "unknown"
            if error.code == 429 or error.code >= 500:
                raise RetryableTransportError(
                    f"OpenAI transient HTTP {error.code} (type={error_type}, code={error_code})"
                ) from error
            raise OpenAIRequestError(
                f"OpenAI HTTP {error.code} (type={error_type}, code={error_code})"
            ) from error
        except (URLError, TimeoutError, socket.timeout) as error:
            raise RetryableTransportError("OpenAI connection failed") from error
        try:
            payload = json.loads(body)
            usage = payload["usage"]
            content = payload["choices"][0]["message"]["content"]
            cached = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
            if not isinstance(content, str):
                raise TypeError
            return TransportResponse(
                provider_request_id=str(payload["id"]),
                resolved_model=str(payload["model"]),
                content=content,
                input_tokens=int(usage["prompt_tokens"]),
                cached_input_tokens=int(cached),
                output_tokens=int(usage["completion_tokens"]),
            )
        except (
            json.JSONDecodeError,
            KeyError,
            IndexError,
            TypeError,
            ValueError,
        ) as error:
            raise RuntimeError(
                "OpenAI returned an invalid response envelope"
            ) from error
