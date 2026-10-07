import json
from collections.abc import Iterator

import httpx
import pytest

from interview_helper.qwen import (
    LocalQwenClient,
    QwenProtocolError,
    QwenUnavailableError,
    _QWEN_LOCK,
)


def test_health_and_completion_payload_disable_thinking() -> None:
    payloads: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(
                200, json={"data": [{"id": "qwen3.8-27b-uncensored"}]}
            )
        payloads.append(__import__("json").loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": "Grounded answer."},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    http = httpx.Client(transport=httpx.MockTransport(handler))
    client = LocalQwenClient(http_client=http)

    answer = client.complete([{"role": "user", "content": "question"}])

    assert answer == "Grounded answer."
    assert payloads == [
        {
            "model": "qwen3.8-27b-uncensored",
            "messages": [{"role": "user", "content": "question"}],
            "stream": False,
            "max_tokens": 120,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    ]


def test_truncated_answer_is_not_presented_as_complete() -> None:
    import pytest
    from interview_helper.qwen import QwenProtocolError
    http = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(
        200, json={"choices": [{"message": {"content": "A cut off answer"}, "finish_reason": "length"}]},
    )))
    client = LocalQwenClient(http_client=http, check_health=False)
    with pytest.raises(QwenProtocolError, match="token limit"):
        client.complete([{"role": "user", "content": "question"}])


def event(delta: dict[str, object], finish: str | None = None) -> bytes:
    return ('data: ' + json.dumps({"choices": [{"delta": delta, "finish_reason": finish}]}, ensure_ascii=False) + '\n\n').encode()


class ByteStream(httpx.SyncByteStream):
    def __init__(self, chunks: list[bytes | Exception]) -> None:
        self.chunks = chunks
        self.closed = False

    def __iter__(self) -> Iterator[bytes]:
        for chunk in self.chunks:
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk

    def close(self) -> None:
        self.closed = True


def test_stream_delivers_before_exhaustion_and_keeps_lock() -> None:
    updates: list[str] = []
    requests: list[dict[str, object]] = []

    class CheckedStream(ByteStream):
        def __iter__(self) -> Iterator[bytes]:
            yield event({"role": "assistant", "reasoning_content": "hidden"})
            yield event({"content": "I led "})
            assert updates == ["I led "]
            assert _QWEN_LOCK.locked()
            # Split every byte, including the multi-byte accented character.
            for value in event({"content": "the café migration."}):
                yield bytes([value])
            assert updates == ["I led ", "I led the café migration."]
            yield event({}, "stop")
            yield b'data: {"choices": [], "usage": {"completion_tokens": 8}}\n\n'
            yield b'data: [DONE]\n\n'

    stream = CheckedStream([])

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, stream=stream)

    client = LocalQwenClient(http_client=httpx.Client(transport=httpx.MockTransport(handler)), check_health=False)
    assert client.complete_stream([{"role": "user", "content": "question"}], on_update=updates.append, max_tokens=224) == "I led the café migration."
    assert requests == [{"model": "qwen3.8-27b-uncensored", "messages": [{"role": "user", "content": "question"}], "stream": True, "max_tokens": 224, "chat_template_kwargs": {"enable_thinking": False}}]
    assert stream.closed
    assert not _QWEN_LOCK.locked()


@pytest.mark.parametrize("tail,expected", [
    ([b'data: broken\n\n'], QwenProtocolError),
    ([b'data: {"error": "failure"}\n\n'], QwenProtocolError),
    ([b'data: {"choices": "invalid"}\n\n'], QwenProtocolError),
    ([event({"content": 1})], QwenProtocolError),
    ([event({}, "length"), b'data: [DONE]\n\n'], QwenProtocolError),
    ([event({}, "content_filter"), b'data: [DONE]\n\n'], QwenProtocolError),
    ([event({}, "stop")], QwenProtocolError),
    ([b'data: [DONE]\n\n'], QwenProtocolError),
    ([], QwenProtocolError),
    ([httpx.ReadError("disconnected")], QwenUnavailableError),
])
def test_bad_stream_never_retries_and_releases_lock(tail: list[bytes | Exception], expected: type[Exception]) -> None:
    stream = ByteStream([event({"content": "Provisional"}), *tail])
    count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal count
        count += 1
        return httpx.Response(200, stream=stream)

    client = LocalQwenClient(http_client=httpx.Client(transport=httpx.MockTransport(handler)), check_health=False)
    updates: list[str] = []
    with pytest.raises(expected):
        client.complete_stream([], on_update=updates.append)
    assert updates == ["Provisional"]
    assert count == 1
    assert stream.closed
    assert not _QWEN_LOCK.locked()


def test_reasoning_only_stream_is_not_an_answer() -> None:
    stream = ByteStream([event({"role": "assistant", "content": None, "reasoning_content": "hidden"}), event({}, "stop"), b'data: [DONE]\n\n'])
    client = LocalQwenClient(http_client=httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=stream))), check_health=False)
    updates: list[str] = []
    with pytest.raises(QwenProtocolError, match="no answer text"):
        client.complete_stream([], on_update=updates.append)
    assert updates == []


def test_callback_exception_closes_stream_and_releases_lock() -> None:
    stream = ByteStream([event({"content": "Hello"})])
    client = LocalQwenClient(http_client=httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=stream))), check_health=False)

    def fail(text: str) -> None:
        raise ValueError("consumer failed")

    with pytest.raises(ValueError, match="consumer failed"):
        client.complete_stream([], on_update=fail)
    assert stream.closed
    assert not _QWEN_LOCK.locked()


def test_cancel_before_dispatch_and_while_waiting_for_lock_sends_no_request() -> None:
    import threading
    from interview_helper.qwen import QwenCancelledError

    requests: list[httpx.Request] = []
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(500)

    client = LocalQwenClient(http_client=httpx.Client(transport=httpx.MockTransport(handler)), check_health=False)
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(QwenCancelledError):
        client.complete_stream_cancellable([], on_update=lambda text: None, cancel_event=cancel)
    cancel.clear()
    errors: list[BaseException] = []
    entered = threading.Event()
    def generate() -> None:
        entered.set()
        try:
            client.complete_stream_cancellable([], on_update=lambda text: None, cancel_event=cancel)
        except BaseException as error:
            errors.append(error)
    with _QWEN_LOCK:
        thread = threading.Thread(target=generate)
        thread.start()
        assert entered.wait(2)
        cancel.set()
    thread.join(2)
    assert not thread.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], QwenCancelledError)
    assert not requests and not _QWEN_LOCK.locked()


def test_cancel_interrupts_real_http_read_disconnects_and_releases_lock() -> None:
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from interview_helper.qwen import QwenCancelledError

    disconnected = threading.Event()
    updated = threading.Event()
    cancel = threading.Event()
    errors: list[BaseException] = []
    updates: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'
        def do_POST(self) -> None:
            self.rfile.read(int(self.headers['Content-Length']))
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Transfer-Encoding', 'chunked')
            self.end_headers()
            chunk = event({'content': 'Draft'})
            self.wfile.write(f'{len(chunk):x}\r\n'.encode() + chunk + b'\r\n')
            self.wfile.flush()
            self.connection.settimeout(3)
            if self.connection.recv(1) == b'':
                disconnected.set()
            self.close_connection = True
        def log_message(self, format: str, *args: object) -> None:
            pass

    server = HTTPServer(('127.0.0.1', 0), Handler)
    serving = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': 0.01}, daemon=True)
    serving.start()
    client = LocalQwenClient(base_url=f'http://127.0.0.1:{server.server_port}/v1', check_health=False, timeout=10)
    def update(text: str) -> None:
        updates.append(text)
        updated.set()
    def generate() -> None:
        try:
            client.complete_stream_cancellable([], on_update=update, cancel_event=cancel)
        except BaseException as error:
            errors.append(error)
    worker = threading.Thread(target=generate, daemon=True)
    try:
        worker.start()
        assert updated.wait(2)
        cancel.set()
        assert disconnected.wait(2)
        worker.join(2)
        assert not worker.is_alive()
        assert updates == ['Draft']
        assert len(errors) == 1 and isinstance(errors[0], QwenCancelledError)
        assert not _QWEN_LOCK.locked()
        assert not any(t.name == 'qwen-stream-cancel' for t in threading.enumerate())
    finally:
        cancel.set()
        worker.join(3)
        client.close()
        server.shutdown()
        server.server_close()
        serving.join(2)
