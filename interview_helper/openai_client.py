"""Opt-in OpenAI answers using the same streaming contract as local Qwen."""

from __future__ import annotations

import os
import re
import threading
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import httpx

from .qwen import LocalQwenClient, QwenCancelledError, QwenError

OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENAI_MODEL = "gpt-5.6-luna"
# Models verified on 2026-10-02 to accept this client's streaming, reasoning-off
# request. Others (e.g. gpt-6-astra, gpt-6.1-sol, gpt-4.1) reject it with HTTP 400.
OPENAI_MODELS: dict[str, str] = {
    "gpt-5.6-luna": "GPT-5.6 Luna",
    "gpt-6-luna": "GPT-6 Luna",
    "gpt-6-sol": "GPT-6 Sol",
    "gpt-5.6-terra": "GPT-5.6 Terra",
    "gpt-5.6-sol": "GPT-5.6 Sol",
}


class OpenAIError(QwenError):
    """Safe, actionable hosted-provider error without request or response bodies."""


def _load_key(path: Path | None) -> str:
    # An explicit file takes precedence over the environment.
    if path is not None:
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            raise OpenAIError("Cannot read the OpenAI API key file; check its path and permissions.") from None
        # Plain text and RTF both contain literal keys. Never render or log the file.
        keys: set[str] = set(re.findall(r"(?<![A-Za-z0-9_-])sk-[A-Za-z0-9_-]{20,}(?![A-Za-z0-9_-])", source))
        if len(keys) != 1:
            raise OpenAIError("OpenAI API key file must contain exactly one distinct API key.")
        return keys.pop()
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not re.fullmatch(r"sk-[A-Za-z0-9_-]{20,}", key):
        raise OpenAIError("Set OPENAI_API_KEY or select an OpenAI API key file.")
    return key


def _safe_error(error: QwenError) -> OpenAIError:
    cause = error.__cause__
    if isinstance(cause, httpx.HTTPStatusError):
        status = cause.response.status_code
        if status == 401:
            return OpenAIError("OpenAI authentication failed; check the API key.")
        if status == 403:
            return OpenAIError("OpenAI denied access; check the key's project and model permissions.")
        if status == 429:
            return OpenAIError("OpenAI quota or rate limit reached; check API billing and limits before trying again.")
        if status == 404:
            return OpenAIError("The selected OpenAI model is unavailable to this project.")
        return OpenAIError(f"OpenAI request failed (HTTP {status}); check model access or service status.")
    if isinstance(cause, httpx.HTTPError):
        return OpenAIError("Cannot reach OpenAI; check the internet connection and service status.")
    # The shared parser uses fixed messages; do not propagate arbitrary exception text.
    if "token limit" in str(error):
        return OpenAIError("Answer reached its token limit before completion; increase the answer token limit.")
    return OpenAIError("OpenAI returned an incomplete or invalid answer; try the question again.")


class OpenAIAnswerClient(LocalQwenClient):
    def __init__(
        self,
        *,
        api_key_file: Path | None = None,
        model: str = OPENAI_MODEL,
        timeout: float = 45.0,
        health_timeout: float = 5.0,
        check_health: bool = True,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._api_key = _load_key(api_key_file)
        super().__init__(
            base_url=OPENAI_BASE_URL, model=model, timeout=timeout,
            health_timeout=health_timeout, check_health=check_health,
            http_client=http_client,
        )

    def _request_options(self) -> dict[str, Any]:
        # Do not set shared client headers or follow provider redirects with credentials.
        return {"headers": {"Authorization": f"Bearer {self._api_key}"}, "follow_redirects": False}

    def _payload(
        self, messages: Sequence[Mapping[str, str]], *, max_tokens: int, stream: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [dict(message) for message in messages],
            "stream": stream,
            "max_completion_tokens": max_tokens,
            "reasoning_effort": "none",
            "service_tier": "default",
            "store": False,
        }
        if stream:
            payload["stream_options"] = {"include_usage": True}
        return payload

    def health_check(self) -> None:
        # Check authentication without generating or forwarding resume context.
        # Model aliases need not appear in /models; generation checks model access.
        try:
            response = self._http.get(
                f"{OPENAI_BASE_URL}/models", timeout=self.health_timeout,
                **self._request_options(),
            )
            response.raise_for_status()
        except httpx.HTTPError as error:
            wrapped = QwenError()
            wrapped.__cause__ = error
            raise _safe_error(wrapped) from None

    def complete(
        self, messages: Sequence[Mapping[str, str]], *, max_tokens: int = 120,
    ) -> str:
        try:
            return super().complete(messages, max_tokens=max_tokens)
        except QwenError as error:
            raise _safe_error(error) from None

    def complete_stream_cancellable(
        self, messages: Sequence[Mapping[str, str]], *,
        on_update: Callable[[str], None], cancel_event: threading.Event,
        max_tokens: int = 120,
    ) -> str:
        try:
            return super().complete_stream_cancellable(
                messages, on_update=on_update, cancel_event=cancel_event,
                max_tokens=max_tokens,
            )
        except QwenCancelledError:
            raise
        except QwenError as error:
            raise _safe_error(error) from None
