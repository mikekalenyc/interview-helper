"""Composition root for capture, held input, and transcription."""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Callable
from typing import Protocol

from interview_helper.capture import CaptureError, MonitorSource
from interview_helper.core import RuntimeState, Transcript


class CaptureAdapter(Protocol):
    def start(
        self,
        pcm_callback: Callable[[bytes], None],
        error_callback: Callable[[BaseException], None],
    ) -> MonitorSource: ...

    def default_sink_changed(self) -> bool: ...

    def stop(self) -> None: ...


class HoldControl(Protocol):
    def run(
        self,
        press_callback: Callable[[], object],
        release_callback: Callable[[], object],
    ) -> None: ...

    def close(self) -> None: ...


class TranscriptionEngine(Protocol):
    @property
    def state(self) -> RuntimeState: ...

    def press(self) -> bool: ...

    def release(self) -> bool: ...

    def add_pcm(self, pcm: bytes) -> None: ...

    def abort(self, error: BaseException) -> None: ...

    def close(self) -> None: ...


class InterviewDaemon:
    def __init__(
        self,
        *,
        capture: CaptureAdapter,
        control: HoldControl,
        engine: TranscriptionEngine,
        status: Callable[[str], None] | None = None,
        capture_ready_callback: Callable[[bool], None] | None = None,
        reconnect_interval_seconds: float = 1.0,
        release_drain_seconds: float = 0.3,
        abort_on_capture_loss: bool = False,
        control_description: str = "waiting for held control",
    ) -> None:
        if reconnect_interval_seconds <= 0:
            raise ValueError("Capture reconnect interval must be positive")
        if not 0 <= release_drain_seconds <= 2:
            raise ValueError("Release drain must be between 0 and 2 seconds")
        self.capture = capture
        self.abort_on_capture_loss = abort_on_capture_loss
        self.control_description = control_description
        self.control = control
        self.engine = engine
        self.status = status or (lambda message: print(message, file=sys.stderr, flush=True))
        self.capture_ready_callback = capture_ready_callback or (lambda _ready: None)
        self.reconnect_interval_seconds = reconnect_interval_seconds
        self.release_drain_seconds = release_drain_seconds
        self._closed = threading.Event()
        self._capture_ready = threading.Event()
        self._capture_wakeup = threading.Event()
        self._capture_lifecycle_lock = threading.Lock()
        self._capture_error_lock = threading.Lock()
        self._capture_error: BaseException | None = None
        self._supervisor: threading.Thread | None = None
        self._release_lock = threading.Lock()
        self._release_pending = False
        self._release_thread: threading.Thread | None = None

    def run(self) -> None:
        monitor = self._connect_capture()
        assert monitor is not None
        self._capture_ready.set()
        self.capture_ready_callback(True)
        self.status(f"ready: capturing {monitor.source}; {self.control_description}")
        self._supervisor = threading.Thread(
            target=self._supervise_capture,
            name="capture-supervisor",
            daemon=True,
        )
        self._supervisor.start()
        try:
            self.control.run(self._press, self._release)
        except KeyboardInterrupt:
            self.status("stopping")
        finally:
            self.close()

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        self._capture_ready.clear()
        self.capture_ready_callback(False)
        self._capture_wakeup.set()
        self.control.close()
        release_thread = self._release_thread
        if release_thread is not None and release_thread is not threading.current_thread():
            release_thread.join(timeout=2)
        supervisor = self._supervisor
        if supervisor is not None and supervisor is not threading.current_thread():
            supervisor.join(timeout=2)
        self._disconnect_capture()
        self.engine.close()

    def _capture_failed(self, error: BaseException) -> None:
        with self._capture_error_lock:
            if self._capture_error is None:
                self._capture_error = error
        self._capture_wakeup.set()

    def _press(self) -> bool:
        if not self._capture_ready.is_set():
            self.status("capture unavailable: held control ignored")
            return False
        with self._release_lock:
            if self._release_pending:
                return False
        return self.engine.press()

    def _release(self) -> bool:
        """Drain queued capture frames before finalizing the utterance."""
        with self._release_lock:
            if self._release_pending or self.engine.state is not RuntimeState.RECORDING:
                return False
            self._release_pending = True
        self.status("finishing question: draining final audio")

        def finish_release() -> None:
            try:
                if not self._closed.wait(self.release_drain_seconds):
                    self.engine.release()
            finally:
                with self._release_lock:
                    self._release_pending = False

        thread = threading.Thread(
            target=finish_release,
            name="capture-release-drain",
            daemon=True,
        )
        self._release_thread = thread
        thread.start()
        return True

    def _connect_capture(self) -> MonitorSource | None:
        with self._capture_lifecycle_lock:
            if self._closed.is_set():
                return None
            return self.capture.start(self.engine.add_pcm, self._capture_failed)

    def _disconnect_capture(self) -> None:
        with self._capture_lifecycle_lock:
            self.capture.stop()

    def _take_capture_error(self) -> BaseException | None:
        with self._capture_error_lock:
            error = self._capture_error
            self._capture_error = None
        return error

    def _supervise_capture(self) -> None:
        while not self._closed.is_set():
            signaled = self._capture_wakeup.wait(self.reconnect_interval_seconds)
            self._capture_wakeup.clear()
            if self._closed.is_set():
                return

            error = self._take_capture_error() if signaled else None
            if error is None:
                try:
                    if not self.capture.default_sink_changed():
                        continue
                    error = CaptureError("Default audio source changed")
                except BaseException as resolver_error:
                    error = resolver_error

            self._capture_ready.clear()
            self.capture_ready_callback(False)
            if self.engine.state is RuntimeState.RECORDING or self.abort_on_capture_loss:
                self.engine.abort(error)
                self.status(f"capture error: {error}; recording aborted")
            else:
                self.status(f"capture error: {error}")
            self._reconnect_capture()

    def _reconnect_capture(self) -> None:
        self._disconnect_capture()
        self._take_capture_error()
        self._capture_wakeup.clear()
        while not self._closed.is_set():
            try:
                monitor = self._connect_capture()
            except BaseException as error:
                self._disconnect_capture()
                self.status(f"capture reconnect failed: {error}; retrying")
                if self._closed.wait(self.reconnect_interval_seconds):
                    return
                continue
            if monitor is None:
                return
            self._capture_ready.set()
            self.capture_ready_callback(True)
            self.status(f"ready: reconnected to {monitor.source}")
            return


def stderr_state(state: RuntimeState) -> None:
    print(f"state: {state.value}", file=sys.stderr, flush=True)


def stderr_error(error: BaseException) -> None:
    print(f"error: {error}", file=sys.stderr, flush=True)


def stdout_transcript(transcript: Transcript) -> None:
    print(f"transcript: {transcript.text}", flush=True)


def stdout_answer(answer: str) -> None:
    print(f"answer: {answer}", flush=True)


def stderr_partial(text: str) -> None:
    timestamp = time.strftime("%H:%M:%S")
    print(f"partial {timestamp}: {text}", file=sys.stderr, flush=True)
