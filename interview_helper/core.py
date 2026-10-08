"""Hardware-independent hold-to-transcribe state machine."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class RuntimeState(str, Enum):
    IDLE = "idle"
    RECORDING = "recording"
    TRANSCRIBING = "transcribing"
    ERROR = "error"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class Transcript:
    text: str
    transcription_seconds: float


class TranscriptionSession(Protocol):
    def add_pcm(self, pcm: bytes, sample_rate: int) -> None: ...

    def finish(self) -> None: ...

    def close(self) -> None: ...


class StreamingTranscriber(Protocol):
    def open_stream(
        self,
        *,
        partial_callback: Callable[[str], None],
        complete_callback: Callable[[Transcript], None],
    ) -> TranscriptionSession: ...


class HoldToTranscribe:
    """Serialize held controls into one streaming transcription at a time."""

    def __init__(
        self,
        transcriber: StreamingTranscriber,
        *,
        sample_rate: int,
        transcript_callback: Callable[[Transcript], None],
        partial_callback: Callable[[str], None] | None = None,
        state_callback: Callable[[RuntimeState], None] | None = None,
        error_callback: Callable[[BaseException], None] | None = None,
    ) -> None:
        self.transcriber = transcriber
        self.sample_rate = sample_rate
        self.transcript_callback = transcript_callback
        self.partial_callback = partial_callback or (lambda _text: None)
        self.state_callback = state_callback or (lambda _state: None)
        self.error_callback = error_callback or (lambda _error: None)
        self._lock = threading.Lock()
        self._state = RuntimeState.IDLE
        self._session: TranscriptionSession | None = None
        self._finishing_session: TranscriptionSession | None = None
        self._generation = 0
        self._received_pcm = False
        self._finish_thread: threading.Thread | None = None

    @property
    def state(self) -> RuntimeState:
        with self._lock:
            return self._state

    def press(self) -> bool:
        """Begin one recording; return false for repeats or busy states."""
        with self._lock:
            if self._state is not RuntimeState.IDLE:
                return False
            self._state = RuntimeState.RECORDING
            self._received_pcm = False
            self._generation += 1
            generation = self._generation
            try:
                self._session = self.transcriber.open_stream(
                    partial_callback=lambda text: self._partial(text, generation),
                    complete_callback=lambda result: self._complete(result, generation),
                )
            except BaseException as error:
                self._session = None
                self._state = RuntimeState.ERROR
                self.error_callback(error)
                self._state = RuntimeState.IDLE
                self.state_callback(RuntimeState.IDLE)
                return False
        self.state_callback(RuntimeState.RECORDING)
        return True

    def add_pcm(self, pcm: bytes) -> None:
        if not pcm:
            return
        with self._lock:
            if self._state is not RuntimeState.RECORDING or self._session is None:
                return
            session = self._session
            generation = self._generation
            self._received_pcm = True
        try:
            session.add_pcm(pcm, self.sample_rate)
        except BaseException as error:
            self.abort(error, generation=generation)

    def release(self) -> bool:
        """Finish asynchronously; return false for a stray release."""
        with self._lock:
            if self._state is not RuntimeState.RECORDING or self._session is None:
                return False
            session = self._session
            self._session = None
            self._finishing_session = session
            generation = self._generation
            received_pcm = self._received_pcm
            self._state = RuntimeState.TRANSCRIBING
        self.state_callback(RuntimeState.TRANSCRIBING)

        def finish() -> None:
            try:
                if received_pcm:
                    session.finish()
                    self._return_idle(generation)
                else:
                    session.close()
                    self._return_idle(generation)
            except BaseException as error:
                try:
                    session.close()
                finally:
                    self._fail(error, generation)
            finally:
                with self._lock:
                    if self._finishing_session is session:
                        self._finishing_session = None

        thread = threading.Thread(target=finish, name="transcription-finish", daemon=True)
        self._finish_thread = thread
        thread.start()
        return True

    def abort(self, error: BaseException, *, generation: int | None = None) -> None:
        with self._lock:
            if generation is not None and generation != self._generation:
                return
            session = self._session
            finishing = self._finishing_session
            self._session = None
            self._finishing_session = None
            self._generation += 1
            generation = self._generation
        for active in (session, finishing):
            if active is None:
                continue
            try:
                active.close()
            except BaseException:
                pass
        self._fail(error, generation)

    def close(self) -> None:
        with self._lock:
            session = self._session
            finishing = self._finishing_session
            self._session = None
            self._finishing_session = None
            self._generation += 1
            self._state = RuntimeState.CLOSED
        if session is not None:
            session.close()
        if finishing is not None:
            finishing.close()
        self.state_callback(RuntimeState.CLOSED)

    def wait_until_idle(self, timeout: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.state is RuntimeState.IDLE:
                return True
            time.sleep(0.01)
        return self.state is RuntimeState.IDLE

    def _partial(self, text: str, generation: int) -> None:
        with self._lock:
            if generation != self._generation or self._state is RuntimeState.CLOSED:
                return
        self.partial_callback(text)

    def _complete(self, transcript: Transcript, generation: int) -> None:
        with self._lock:
            if generation != self._generation or self._state is RuntimeState.CLOSED:
                return
        text = transcript.text.strip()
        if text:
            self.transcript_callback(
                Transcript(text=text, transcription_seconds=transcript.transcription_seconds)
            )
        # The finishing thread returns to idle after finish() itself returns.

    def _return_idle(self, generation: int | None = None) -> None:
        with self._lock:
            if generation is not None and generation != self._generation:
                return
            if self._state in {RuntimeState.CLOSED, RuntimeState.IDLE}:
                return
            self._state = RuntimeState.IDLE
        self.state_callback(RuntimeState.IDLE)

    def _fail(self, error: BaseException, generation: int | None = None) -> None:
        with self._lock:
            if generation is not None and generation != self._generation:
                return
            if self._state in {RuntimeState.CLOSED, RuntimeState.IDLE}:
                return
            self._state = RuntimeState.ERROR
        self.state_callback(RuntimeState.ERROR)
        self.error_callback(error)
        self._return_idle(generation)
