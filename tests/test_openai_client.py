import json
import threading
import traceback
from pathlib import Path

import httpx
import pytest

from interview_helper.openai_client import OPENAI_MODEL, OpenAIAnswerClient, OpenAIError
from interview_helper.qwen import QwenCancelledError

KEY = "sk-" + "test-secret-" * 4


@pytest.fixture(autouse=True)
def api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", KEY)


def test_auth_health_and_complete_payload() -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.url.host == "api.openai.com"
        assert request.headers["authorization"] == f"Bearer {KEY}"
        return httpx.Response(200, json={"choices": [{"message": {"content": "Answer"}, "finish_reason": "stop"}]})

    http = httpx.Client(transport=httpx.MockTransport(handle))
    client = OpenAIAnswerClient(http_client=http)
    assert client.complete([{"role": "user", "content": "question"}], max_tokens=224) == "Answer"
    assert requests[0].url.path == "/v1/models"
    assert json.loads(requests[1].content) == {
        "model": OPENAI_MODEL, "messages": [{"role": "user", "content": "question"}],
        "stream": False, "max_completion_tokens": 224, "reasoning_effort": "none",
        "service_tier": "default", "store": False,
    }
    assert "authorization" not in http.headers
    client.close()
    assert not http.is_closed


@pytest.mark.parametrize("rtf", [False, True])
def test_key_file_overrides_environment(tmp_path: Path, rtf: bool) -> None:
    key = "sk-" + "file-key-" * 5
    path = tmp_path / "key"
    path.write_text(r"{\rtf1\ansi " + key + "}" if rtf else key + "\n")
    def handle(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == f"Bearer {key}"
        return httpx.Response(200)
    OpenAIAnswerClient(api_key_file=path, http_client=httpx.Client(transport=httpx.MockTransport(handle)))


@pytest.mark.parametrize("content", ["", "not a key", KEY + " sk-" + "different" * 5])
def test_invalid_or_ambiguous_key_file(tmp_path: Path, content: str) -> None:
    path = tmp_path / "key"
    path.write_text(content)
    with pytest.raises(OpenAIError, match="exactly one") as caught:
        OpenAIAnswerClient(api_key_file=path, check_health=False)
    assert KEY not in str(caught.value)


def test_missing_key_and_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(OpenAIError, match="OPENAI_API_KEY"):
        OpenAIAnswerClient(check_health=False)
    with pytest.raises(OpenAIError, match="Cannot read"):
        OpenAIAnswerClient(api_key_file=tmp_path / "missing", check_health=False)


@pytest.mark.parametrize("status,word", [(401, "authentication"), (403, "permissions"), (429, "quota"), (404, "unavailable"), (500, "HTTP 500")])
@pytest.mark.parametrize("operation", ["health", "complete", "stream"])
def test_http_errors_safe_and_no_retries(status: int, word: str, operation: str) -> None:
    calls = 0
    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(status, text=KEY + " private context")
    http = httpx.Client(transport=httpx.MockTransport(handle))
    with pytest.raises(OpenAIError, match=word) as caught:
        client = OpenAIAnswerClient(http_client=http, check_health=operation == "health")
        if operation == "complete":
            client.complete([])
        elif operation == "stream":
            client.complete_stream([], on_update=lambda text: None)
    formatted = "".join(traceback.format_exception(caught.value))
    assert KEY not in formatted
    assert "private context" not in formatted
    assert calls == 1


def test_network_error_sanitized_and_owned_client_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(KEY)
    http = httpx.Client(transport=httpx.MockTransport(handle))
    monkeypatch.setattr("interview_helper.qwen.httpx.Client", lambda **kwargs: http)
    with pytest.raises(OpenAIError, match="internet") as caught:
        OpenAIAnswerClient()
    assert http.is_closed
    assert KEY not in "".join(traceback.format_exception(caught.value))


def test_redirect_not_followed_even_with_injected_client() -> None:
    calls: list[httpx.Request] = []
    def handle(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(307, headers={"Location": "https://other.example/v1/models"})
    with pytest.raises(OpenAIError):
        OpenAIAnswerClient(http_client=httpx.Client(transport=httpx.MockTransport(handle), follow_redirects=True))
    assert len(calls) == 1


def sse(finish: str = "stop", done: bool = True) -> bytes:
    events = [
        {"choices": [{"delta": {"content": "Hello"}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": finish}]},
        {"choices": [], "usage": {"completion_tokens": 1}},
    ]
    return ("".join("data: " + json.dumps(event) + "\n\n" for event in events) + ("data: [DONE]\n\n" if done else "")).encode()


def test_stream_payload_and_cumulative_updates() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["stream"] is True
        assert payload["stream_options"] == {"include_usage": True}
        assert payload["max_completion_tokens"] == 224
        assert "chat_template_kwargs" not in payload
        return httpx.Response(200, content=sse())
    client = OpenAIAnswerClient(http_client=httpx.Client(transport=httpx.MockTransport(handle)), check_health=False)
    updates: list[str] = []
    assert client.complete_stream([], on_update=updates.append, max_tokens=224) == "Hello"
    assert updates == ["Hello"]


@pytest.mark.parametrize("content,match", [(sse("length"), "token limit"), (sse(done=False), "incomplete"), (b'data: {"error":"secret"}\n\n', "invalid")])
def test_stream_failures(content: bytes, match: str) -> None:
    client = OpenAIAnswerClient(http_client=httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, content=content))), check_health=False)
    with pytest.raises(OpenAIError, match=match):
        client.complete_stream([], on_update=lambda text: None)


def test_cancellation_after_update_preserves_cancel_type() -> None:
    client = OpenAIAnswerClient(http_client=httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, content=sse()))), check_health=False)
    cancel = threading.Event()
    with pytest.raises(QwenCancelledError):
        client.complete_stream_cancellable([], on_update=lambda text: cancel.set(), cancel_event=cancel)
