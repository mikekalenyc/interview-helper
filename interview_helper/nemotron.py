"""Local NeMo-Speech.cpp v0.1 C ABI streaming adapter.

Each stream has one worker and bounded PCM ingress. Native calls cannot be
interrupted: close discards queued audio immediately and defers native release
until the active call returns. Callbacks already entered cannot be recalled.
"""
from __future__ import annotations

import ctypes as C
import os
import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from interview_helper.core import Transcript


class NemotronError(RuntimeError):
    """Native model setup or streaming recognition failed."""


class _BackendConfig(C.Structure):
    _fields_ = [("size", C.c_size_t), ("gpu", C.c_int32)]


class _ModelConfig(C.Structure):
    _fields_ = [("size", C.c_size_t), ("path", C.c_char_p), ("name", C.c_char_p)]


class _EndpointConfig(C.Structure):
    _fields_ = [("size", C.c_size_t), ("enable", C.c_bool), ("vad_based", C.c_bool),
                ("stop_history_eou_ms", C.c_int32)]


class _Config(C.Structure):
    _fields_ = [("size", C.c_size_t), ("backend", C.POINTER(_BackendConfig)),
                ("model", C.POINTER(_ModelConfig)), ("streaming", C.c_void_p),
                ("decoder", C.c_void_p), ("vad", C.c_void_p),
                ("endpointing", C.POINTER(_EndpointConfig)), ("postproc", C.c_void_p),
                ("diar", C.c_void_p), ("batching", C.c_void_p)]


class _Context(C.Structure):
    _fields_ = [("size", C.c_size_t), ("phrases", C.POINTER(C.c_char_p)),
                ("phrase_count", C.c_size_t), ("boost", C.c_float)]


class _Options(C.Structure):
    _fields_ = [("size", C.c_size_t), ("request_id", C.c_char_p), ("language_code", C.c_char_p),
                ("interim_results", C.c_bool), ("enable_word_time_offsets", C.c_bool),
                ("enable_automatic_punctuation", C.c_bool), ("verbatim_transcripts", C.c_bool),
                ("profanity_filter", C.c_bool), ("stop_history_eou_ms", C.c_int32),
                ("speech_contexts", C.POINTER(_Context)), ("speech_context_count", C.c_size_t),
                ("max_alternatives", C.c_int32), ("enable_speaker_diarization", C.c_bool),
                ("max_speaker_count", C.c_int32)]


class NativeBackend(Protocol):
    def create(self, path: Path, device: str, index: int) -> Any: ...
    def destroy(self, recognizer: Any) -> None: ...
    def open(self, recognizer: Any, keyterms: tuple[str, ...]) -> Any: ...
    def push(self, stream: Any, pcm: bytes) -> None: ...
    def finish(self, stream: Any) -> None: ...
    def next(self, stream: Any) -> tuple[str, bool] | None: ...
    def close(self, stream: Any) -> None: ...


class _Native:
    def __init__(self, path: Path) -> None:
        try:
            self.lib = C.CDLL(str(path.resolve()))
            self._bind("last_error", [], C.c_char_p)
            self._bind("create", [C.POINTER(_Config), C.POINTER(C.c_void_p)], C.c_int)
            self._bind("destroy", [C.c_void_p], None)
            self._bind("streaming_recognize", [C.c_void_p, C.POINTER(_Options), C.POINTER(C.c_void_p)], C.c_int)
            self._bind("stream_push_f32", [C.c_void_p, C.POINTER(C.c_float), C.c_size_t, C.c_int32], C.c_int)
            for name in ("stream_finish",):
                self._bind(name, [C.c_void_p], C.c_int)
            self._bind("stream_next", [C.c_void_p, C.POINTER(C.c_void_p)], C.c_int)
            self._bind("stream_close", [C.c_void_p], None)
            self._bind("result_is_final", [C.c_void_p], C.c_bool)
            self._bind("result_transcript", [C.c_void_p, C.c_size_t], C.c_char_p)
            self._bind("result_destroy", [C.c_void_p], None)
            self._bind("recognition_options_default", [], _Options)
        except (OSError, AttributeError) as error:
            raise NemotronError(f"Cannot load NeMo-Speech runtime {path}: {error}") from error
        self.path = path

    def _bind(self, name: str, args: list[Any], result: Any) -> None:
        function = getattr(self.lib, "nemo_speech_asr_" + name)
        function.argtypes, function.restype = args, result

    def _check(self, status: int, operation: str) -> None:
        if status:
            message = self.lib.nemo_speech_asr_last_error()
            detail = message.decode("utf-8", errors="replace") if message else f"status {status}"
            raise NemotronError(f"Nemotron {operation}: {detail}")

    def create(self, path: Path, device: str, index: int) -> Any:
        if device == "cuda":
            try:
                cuda = C.CDLL(str(self.path.parent / "libggml-cuda.so"))
                cuda.ggml_backend_cuda_get_device_count.argtypes = []
                cuda.ggml_backend_cuda_get_device_count.restype = C.c_int
                if index >= cuda.ggml_backend_cuda_get_device_count():
                    raise NemotronError(f"CUDA device {index} is unavailable")
            except (OSError, AttributeError) as error:
                raise NemotronError(f"CUDA runtime is unavailable: {error}") from error
        backend = _BackendConfig(C.sizeof(_BackendConfig), index if device == "cuda" else -1)
        model = _ModelConfig(C.sizeof(_ModelConfig), os.fsencode(path.resolve()), None)
        endpoint = _EndpointConfig(C.sizeof(_EndpointConfig), False, False, 0)
        config = _Config()
        config.size = C.sizeof(config)
        config.backend, config.model, config.endpointing = C.pointer(backend), C.pointer(model), C.pointer(endpoint)
        handle = C.c_void_p()
        self._check(self.lib.nemo_speech_asr_create(C.byref(config), C.byref(handle)), "model initialization")
        if not handle.value:
            raise NemotronError("Native runtime returned an empty recognizer")
        return handle

    def destroy(self, recognizer: Any) -> None:
        self.lib.nemo_speech_asr_destroy(recognizer)

    def open(self, recognizer: Any, keyterms: tuple[str, ...]) -> Any:
        options = self.lib.nemo_speech_asr_recognition_options_default()
        options.interim_results = True
        options.language_code = b"en-US"
        if keyterms:
            phrases = (C.c_char_p * len(keyterms))(*(term.encode("utf-8") for term in keyterms))
            context = _Context(C.sizeof(_Context), phrases, len(keyterms), 1.0)
            options.speech_contexts = C.pointer(context)
            options.speech_context_count = 1
        handle = C.c_void_p()
        self._check(self.lib.nemo_speech_asr_streaming_recognize(recognizer, C.byref(options), C.byref(handle)), "open stream")
        if not handle.value:
            raise NemotronError("Native runtime returned an empty stream")
        return handle

    def push(self, stream: Any, pcm: bytes) -> None:
        samples = np.frombuffer(pcm, dtype="<f4")
        self._check(self.lib.nemo_speech_asr_stream_push_f32(
            stream, samples.ctypes.data_as(C.POINTER(C.c_float)), len(samples), 16000), "push audio")

    def finish(self, stream: Any) -> None:
        self._check(self.lib.nemo_speech_asr_stream_finish(stream), "finish stream")

    def next(self, stream: Any) -> tuple[str, bool] | None:
        result = C.c_void_p()
        self._check(self.lib.nemo_speech_asr_stream_next(stream, C.byref(result)), "decode audio")
        if not result.value:
            return None
        try:
            text = self.lib.nemo_speech_asr_result_transcript(result, 0)
            return (text.decode("utf-8", errors="replace") if text else "",
                    bool(self.lib.nemo_speech_asr_result_is_final(result)))
        finally:
            self.lib.nemo_speech_asr_result_destroy(result)

    def close(self, stream: Any) -> None:
        self.lib.nemo_speech_asr_stream_close(stream)


_runtime_lock = threading.Lock()
_runtime: _Native | None = None


def _prepare_shared_cuda(path: Path) -> None:
    # The native archive carries a reduced cuBLAS library with the same SONAME
    # as full cuBLAS. Load the GPU detector profile's full libraries first so
    # both runtimes can coexist, including after changing detection settings.
    if not (path.parent / "libggml-cuda.so").is_file():
        return
    try:
        import onnxruntime as ort
    except ImportError:
        return
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        return
    preload = getattr(ort, "preload_dlls", None)
    if preload is None:
        raise NemotronError("Combined GPU support requires the pinned gpu-desktop installation")
    preload(directory="")


def _get_runtime(path: Path, device: str) -> _Native:
    """Shared SONAMEs make switching from a loaded CPU build unsafe in-process."""
    global _runtime
    with _runtime_lock:
        if _runtime is not None:
            if device == "cuda" and not (_runtime.path.parent / "libggml-cuda.so").is_file():
                raise NemotronError(
                    "The CPU-only NeMo-Speech runtime is already loaded. Restart the app "
                    "before selecting CUDA. A CUDA runtime can subsequently serve CPU or CUDA."
                )
            return _runtime
        _prepare_shared_cuda(path)
        runtime = _Native(path)
        _runtime = runtime
        return runtime


class NemotronTranscriber:
    def __init__(self, model_path: Path, *, library_path: Path, device: str = "cpu",
                 device_index: int = 0, keyterms: tuple[str, ...] = (),
                 backend: NativeBackend | None = None) -> None:
        if device not in {"cpu", "cuda"} or type(device_index) is not int or device_index < 0:
            raise NemotronError("Nemotron needs cpu or cuda and a nonnegative device index")
        if not model_path.is_file() or not model_path.stat().st_size:
            raise NemotronError(f"Local Nemotron model is missing or empty: {model_path}")
        if not library_path.is_file():
            raise NemotronError(f"Local NeMo-Speech runtime is missing: {library_path}")
        self._native = backend if backend is not None else _get_runtime(library_path, device)
        self._lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._closed = False
        self._sessions: set[NemotronSession] = set()
        self._keyterms = keyterms
        self._model = self._native.create(model_path, device, device_index)

    def open_stream(self, *, partial_callback: Callable[[str], None],
                    complete_callback: Callable[[Transcript], None]) -> NemotronSession:
        with self._state_lock:
            if self._closed:
                raise NemotronError("Nemotron transcriber is closed")
            session = NemotronSession(self, partial_callback, complete_callback)
            self._sessions.add(session)
            session._worker.start()
            return session

    def _release(self, session: NemotronSession) -> None:
        with self._state_lock:
            self._sessions.discard(session)
            self._destroy_if_idle()

    def _destroy_if_idle(self) -> None:
        if self._closed and not self._sessions and self._model is not None:
            self._native.destroy(self._model)
            self._model = None

    def close(self) -> None:
        with self._state_lock:
            self._closed = True
            sessions = tuple(self._sessions)
            self._destroy_if_idle()
        for session in sessions:
            session.close()


class NemotronSession:
    _limit = 185 * 16000 * 4
    _queue_limit = 4 * 16000 * 4
    _finish_timeout = 60.0

    def __init__(self, owner: NemotronTranscriber, partial: Callable[[str], None],
                 complete: Callable[[Transcript], None]) -> None:
        self._owner, self._partial, self._complete = owner, partial, complete
        self._condition = threading.Condition()
        self._queue: deque[bytes] = deque()
        self._queued = self._total = 0
        self._cancelled = self._finishing = False
        self._nonzero = False
        self._finish_started = 0.0
        self._done = threading.Event()
        self._error: Exception | None = None
        self._worker = threading.Thread(target=self._run, name="nemotron-stream", daemon=True)

    def _raise_error(self) -> None:
        if self._error:
            raise NemotronError(f"Nemotron streaming failed: {self._error}") from self._error

    def add_pcm(self, pcm: bytes, sample_rate: int) -> None:
        with self._condition:
            self._raise_error()
            if self._cancelled or self._finishing or self._done.is_set():
                return
            if sample_rate != 16000 or not pcm or len(pcm) % 4:
                raise NemotronError("Expected nonempty mono float32 PCM at 16000 Hz")
            samples = np.frombuffer(pcm, dtype="<f4")
            if not np.isfinite(samples).all():
                raise NemotronError("PCM contains non-finite samples")
            if self._total + len(pcm) > self._limit or self._queued + len(pcm) > self._queue_limit:
                raise NemotronError("Nemotron audio buffer limit exceeded; discard this utterance")
            self._nonzero = self._nonzero or bool(np.any(np.abs(samples) > 1e-7))
            self._queue.append(pcm)
            self._queued += len(pcm)
            self._total += len(pcm)
            self._condition.notify()

    def finish(self) -> None:
        if threading.current_thread() is self._worker:
            raise NemotronError("Cannot finish a stream from its own partial callback")
        with self._condition:
            self._raise_error()
            if self._cancelled or self._done.is_set():
                return
            if not self._finishing:
                self._finish_started = time.monotonic()
            self._finishing = True
            self._condition.notify()
        if not self._done.wait(self._finish_timeout):
            self.close()
            raise NemotronError("Timed out finishing Nemotron; utterance discarded")
        self._raise_error()

    def close(self) -> None:
        with self._condition:
            self._cancelled = True
            self._queue.clear()
            self._queued = 0
            self._condition.notify_all()
            self._done.set()

    def _run(self) -> None:
        native, owner = self._owner._native, self._owner
        stream: Any = None
        segments: list[str] = []
        current = ""
        try:
            with owner._lock:
                with self._condition:
                    if self._cancelled:
                        return
                stream = native.open(owner._model, owner._keyterms)
            while True:
                with self._condition:
                    self._condition.wait_for(lambda: self._cancelled or self._queue or self._finishing)
                    if self._cancelled:
                        return
                    chunk = self._queue.popleft() if self._queue else None
                    if chunk is not None:
                        self._queued -= len(chunk)
                    final = chunk is None and self._finishing
                with owner._lock:
                    with self._condition:
                        if self._cancelled:
                            return
                    if final:
                        native.finish(stream)
                    elif chunk is not None:
                        native.push(stream, chunk)
                    updates: list[str] = []
                    while True:
                        with self._condition:
                            if self._cancelled:
                                return
                        result = native.next(stream)
                        if result is None:
                            break
                        text, segment_final = result
                        current = text.strip()
                        if segment_final:
                            if current:
                                segments.append(current)
                            current = ""
                        updates.append(" ".join(segments + ([current] if current else [])))
                for text in updates:
                    with self._condition:
                        deliver = not self._cancelled and not self._finishing and self._nonzero
                    if deliver:
                        self._partial(text)
                if final:
                    with self._condition:
                        deliver = not self._cancelled
                        text = " ".join(segments + ([current] if current else [])) if self._nonzero else ""
                    if deliver:
                        self._complete(Transcript(text, time.monotonic() - self._finish_started))
                    return
        except Exception as error:
            with self._condition:
                if not self._cancelled:
                    self._error = error
        finally:
            try:
                if stream is not None:
                    with owner._lock:
                        native.close(stream)
            finally:
                with self._condition:
                    self._queue.clear()
                    self._queued = 0
                    self._done.set()
                owner._release(self)
