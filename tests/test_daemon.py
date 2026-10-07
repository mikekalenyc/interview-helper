from __future__ import annotations

import threading
import time
from collections.abc import Callable

from interview_helper.capture import CaptureError, MonitorSource
from interview_helper.core import RuntimeState
from interview_helper.daemon import InterviewDaemon


class FakeCapture:
    def __init__(self) -> None:
        self.starts = 0
        self.stops = 0
        self.changed = False
        self.error_callback: Callable[[BaseException], None] | None = None
        self.pcm_callback: Callable[[bytes], None] | None = None

    def start(
        self,
        pcm_callback: Callable[[bytes], None],
        error_callback: Callable[[BaseException], None],
    ) -> MonitorSource:
        self.starts += 1
        self.pcm_callback = pcm_callback
        self.error_callback = error_callback
        return MonitorSource(f"sink-{self.starts}", f"sink-{self.starts}.monitor")

    def default_sink_changed(self) -> bool:
        changed = self.changed
        self.changed = False
        return changed

    def stop(self) -> None:
        self.stops += 1

    def fail(self, error: BaseException) -> None:
        assert self.error_callback is not None
        self.error_callback(error)

    def emit(self, pcm: bytes) -> None:
        assert self.pcm_callback is not None
        self.pcm_callback(pcm)


class FakeControl:
    def __init__(self) -> None:
        self.closed = threading.Event()
        self.press_callback: Callable[[], object] | None = None
        self.release_callback: Callable[[], object] | None = None

    def run(
        self,
        press_callback: Callable[[], object],
        release_callback: Callable[[], object],
    ) -> None:
        self.press_callback = press_callback
        self.release_callback = release_callback
        self.closed.wait(timeout=2)

    def close(self) -> None:
        self.closed.set()


class FakeEngine:
    def __init__(self) -> None:
        self.state = RuntimeState.IDLE
        self.presses = 0
        self.releases = 0
        self.pcm: list[bytes] = []
        self.aborted: list[BaseException] = []
        self.closed = False

    def press(self) -> bool:
        self.presses += 1
        self.state = RuntimeState.RECORDING
        return True

    def release(self) -> bool:
        self.releases += 1
        self.state = RuntimeState.IDLE
        return True

    def add_pcm(self, pcm: bytes) -> None:
        if self.state is RuntimeState.RECORDING:
            self.pcm.append(pcm)

    def abort(self, error: BaseException) -> None:
        self.aborted.append(error)
        self.state = RuntimeState.IDLE

    def close(self) -> None:
        self.closed = True
        self.state = RuntimeState.CLOSED


def wait_for(predicate: Callable[[], bool]) -> None:
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("Condition was not reached")


def start_daemon(
    *, engine_state: RuntimeState = RuntimeState.IDLE
) -> tuple[InterviewDaemon, FakeCapture, FakeControl, FakeEngine, list[str], threading.Thread]:
    capture = FakeCapture()
    control = FakeControl()
    engine = FakeEngine()
    engine.state = engine_state
    statuses: list[str] = []
    daemon = InterviewDaemon(
        capture=capture,
        control=control,
        engine=engine,
        status=statuses.append,
        reconnect_interval_seconds=0.01,
    )
    thread = threading.Thread(target=daemon.run)
    thread.start()
    wait_for(lambda: control.press_callback is not None)
    return daemon, capture, control, engine, statuses, thread


def stop_daemon(daemon: InterviewDaemon, thread: threading.Thread) -> None:
    daemon.close()
    thread.join(timeout=1)
    assert not thread.is_alive()


def test_capture_process_failure_reconnects_while_idle() -> None:
    daemon, capture, _control, engine, statuses, thread = start_daemon()

    capture.fail(CaptureError("parec exited"))
    wait_for(lambda: capture.starts == 2)

    assert engine.aborted == []
    assert statuses[-1] == "ready: reconnected to sink-2.monitor"
    stop_daemon(daemon, thread)


def test_default_sink_change_aborts_recording_before_reconnect() -> None:
    daemon, capture, _control, engine, statuses, thread = start_daemon(
        engine_state=RuntimeState.RECORDING
    )

    capture.changed = True
    wait_for(lambda: capture.starts == 2)

    assert len(engine.aborted) == 1
    assert str(engine.aborted[0]) == "Default audio source changed"
    assert "recording aborted" in statuses[-2]
    stop_daemon(daemon, thread)


def test_holds_are_ignored_until_capture_reconnects() -> None:
    class RetryCapture(FakeCapture):
        allow_reconnect = threading.Event()

        def start(
            self,
            pcm_callback: Callable[[bytes], None],
            error_callback: Callable[[BaseException], None],
        ) -> MonitorSource:
            if self.starts and not self.allow_reconnect.is_set():
                self.starts += 1
                raise CaptureError("monitor unavailable")
            return super().start(pcm_callback, error_callback)

    capture = RetryCapture()
    control = FakeControl()
    engine = FakeEngine()
    statuses: list[str] = []
    daemon = InterviewDaemon(
        capture=capture,
        control=control,
        engine=engine,
        status=statuses.append,
        reconnect_interval_seconds=0.01,
    )
    thread = threading.Thread(target=daemon.run)
    thread.start()
    wait_for(lambda: control.press_callback is not None)

    capture.fail(CaptureError("parec exited"))
    wait_for(lambda: any("reconnect failed" in message for message in statuses))
    assert control.press_callback is not None
    assert control.press_callback() is False
    assert engine.presses == 0

    capture.allow_reconnect.set()
    wait_for(lambda: capture.starts >= 3 and statuses[-1].startswith("ready: reconnected"))
    assert control.press_callback() is True
    assert engine.presses == 1
    stop_daemon(daemon, thread)


def test_release_drains_queued_capture_audio_before_finalizing() -> None:
    capture = FakeCapture()
    control = FakeControl()
    engine = FakeEngine()
    statuses: list[str] = []
    daemon = InterviewDaemon(
        capture=capture,
        control=control,
        engine=engine,
        status=statuses.append,
        reconnect_interval_seconds=1,
        release_drain_seconds=0.03,
    )
    thread = threading.Thread(target=daemon.run)
    thread.start()
    wait_for(
        lambda: control.press_callback is not None
        and control.release_callback is not None
    )
    assert control.press_callback is not None
    assert control.release_callback is not None

    assert control.press_callback() is True
    capture.emit(b"held")
    assert control.release_callback() is True
    capture.emit(b"queued tail")

    assert engine.releases == 0
    assert control.press_callback() is False
    assert "draining final audio" in statuses[-1]
    wait_for(lambda: engine.releases == 1)
    assert engine.pcm == [b"held", b"queued tail"]
    stop_daemon(daemon, thread)
