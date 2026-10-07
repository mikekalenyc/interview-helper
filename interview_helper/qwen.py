"""Serialized client for the workstation's local OpenAI-compatible Qwen server."""

from __future__ import annotations

import json
import socket
import threading
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import httpx


DEFAULT_BASE_URL = "http://127.0.0.1:8082/v1"
DEFAULT_MODEL = "qwen3.8-27b-uncensored"
_QWEN_LOCK = threading.Lock()


class QwenError(RuntimeError):
    """Base error for local Qwen communication."""


class QwenUnavailableError(QwenError):
    """The configured local server is unavailable."""


class QwenProtocolError(QwenError):
    """The server returned an unusable response."""


class QwenCancelledError(QwenError):
    """An intentionally cancelled completion."""


class LocalQwenClient:
    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout: float = 45.0,
        health_timeout: float = 2.0,
        check_health: bool = True,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.health_timeout = health_timeout
        self._owns_client = http_client is None
        self._http = http_client or httpx.Client(timeout=timeout)
        try:
            if check_health:
                self.health_check()
        except Exception:
            self.close()
            raise

    def _request_options(self) -> dict[str, Any]:
        return {}

    def _payload(
        self, messages: Sequence[Mapping[str, str]], *, max_tokens: int, stream: bool,
    ) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": [dict(message) for message in messages],
            "stream": stream,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }

    def health_check(self) -> None:
        try:
            response = self._http.get(
                f"{self.base_url}/models", timeout=self.health_timeout, **self._request_options()
            )
            response.raise_for_status()
            body = response.json()
            models = {
                item.get("id")
                for item in body.get("data", [])
                if isinstance(item, dict)
            }
        except (httpx.HTTPError, json.JSONDecodeError, TypeError, AttributeError) as error:
            raise QwenUnavailableError(
                f"Local Qwen endpoint is unavailable at {self.base_url}: {error}"
            ) from error
        if self.model not in models:
            raise QwenUnavailableError(
                f"Model {self.model!r} is not advertised by {self.base_url}; "
                f"available: {sorted(str(item) for item in models)}"
            )

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        max_tokens: int = 120,
    ) -> str:
        payload = self._payload(messages, max_tokens=max_tokens, stream=False)
        with _QWEN_LOCK:
            try:
                response = self._http.post(
                    f"{self.base_url}/chat/completions", json=payload, **self._request_options()
                )
                response.raise_for_status()
                body = response.json()
            except (httpx.HTTPError, json.JSONDecodeError) as error:
                raise QwenUnavailableError(
                    f"Local Qwen completion failed: {error}"
                ) from error
        try:
            message = body["choices"][0]["message"]
            content = message.get("content")
            finish_reason = body["choices"][0].get("finish_reason")
        except (KeyError, IndexError, TypeError, AttributeError) as error:
            raise QwenProtocolError("Qwen response has no assistant message") from error
        if finish_reason == "length":
            raise QwenProtocolError("Answer reached its token limit before completion; increase the answer token limit")
        if not isinstance(content, str) or not content.strip():
            raise QwenProtocolError("Qwen response contains no answer text")
        return content.strip()

    def complete_stream(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        on_update: Callable[[str], None],
        max_tokens: int = 120,
    ) -> str:
        """Deliver cumulative visible text; only return a complete, stopped answer.

        Updates are provisional until this method returns. Failures never retry a
        generation, and the process-wide server lock covers the entire stream.
        """
        return self.complete_stream_cancellable(
            messages, on_update=on_update, cancel_event=threading.Event(),
            max_tokens=max_tokens,
        )

    def complete_stream_cancellable(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        on_update: Callable[[str], None],
        cancel_event: threading.Event,
        max_tokens: int = 120,
    ) -> str:
        """Close the active HTTP stream on cancellation, releasing the server slot."""
        def check_cancelled() -> None:
            if cancel_event.is_set():
                raise QwenCancelledError("Answer cancelled")

        check_cancelled()
        payload = self._payload(messages, max_tokens=max_tokens, stream=True)
        content = ""
        stopped = False
        done = False
        with _QWEN_LOCK:
            check_cancelled()
            watcher_done = threading.Event()
            watcher: threading.Thread | None = None
            try:
                with self._http.stream(
                    "POST", f"{self.base_url}/chat/completions", json=payload, **self._request_options()
                ) as response:
                    def disconnect_on_cancel() -> None:
                        while not watcher_done.wait(0.01):
                            if cancel_event.is_set():
                                # shutdown interrupts a blocking read on Linux.
                                network = response.extensions.get("network_stream")
                                if network is not None:
                                    connection = network.get_extra_info("socket")
                                    if connection is not None:
                                        try:
                                            connection.shutdown(socket.SHUT_RDWR)
                                        except OSError:
                                            pass
                                response.close()
                                return

                    watcher = threading.Thread(
                        target=disconnect_on_cancel, name="qwen-stream-cancel", daemon=True,
                    )
                    watcher.start()
                    check_cancelled()
                    response.raise_for_status()
                    for line in response.iter_lines():
                        check_cancelled()
                        if not line or line.startswith(":"):
                            continue
                        if not line.startswith("data:"):
                            # SSE metadata does not carry generated content.
                            if line.startswith(("event:", "id:", "retry:")):
                                continue
                            raise QwenProtocolError("Malformed Qwen streaming event")
                        data = line[5:].strip()
                        if data == "[DONE]":
                            done = True
                            break
                        try:
                            body = json.loads(data)
                        except json.JSONDecodeError as error:
                            raise QwenProtocolError("Malformed Qwen streaming JSON") from error
                        if not isinstance(body, dict) or "error" in body:
                            raise QwenProtocolError("Qwen stream returned an error payload")
                        choices = body.get("choices")
                        if not isinstance(choices, list):
                            raise QwenProtocolError("Qwen stream has no choices")
                        if not choices:
                            # Optional final usage report.
                            continue
                        if len(choices) != 1 or not isinstance(choices[0], dict):
                            raise QwenProtocolError("Malformed Qwen streaming choice")
                        choice = choices[0]
                        if stopped:
                            raise QwenProtocolError("Qwen sent answer data after completion")
                        delta = choice.get("delta")
                        if not isinstance(delta, dict):
                            raise QwenProtocolError("Qwen stream has no assistant delta")
                        fragment = delta.get("content")
                        if fragment is not None and not isinstance(fragment, str):
                            raise QwenProtocolError("Qwen stream contains invalid answer text")
                        finish_reason = choice.get("finish_reason")
                        if finish_reason == "length":
                            raise QwenProtocolError(
                                "Answer reached its token limit before completion; "
                                "increase the answer token limit"
                            )
                        if finish_reason not in (None, "stop"):
                            raise QwenProtocolError("Qwen stream ended without a complete answer")
                        if fragment:
                            content += fragment
                            on_update(content)
                        stopped = finish_reason == "stop"
            except httpx.HTTPError as error:
                check_cancelled()
                raise QwenUnavailableError(
                    f"Local Qwen streaming completion failed: {error}"
                ) from error
            finally:
                watcher_done.set()
                if watcher is not None:
                    watcher.join()
        check_cancelled()
        if not done or not stopped:
            raise QwenProtocolError("Qwen stream disconnected before terminal completion")
        if not content.strip():
            raise QwenProtocolError("Qwen response contains no answer text")
        return content.strip()

    def close(self) -> None:
        if self._owns_client:
            self._http.close()
