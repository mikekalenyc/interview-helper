"""PipeWire/PulseAudio monitor discovery and persistent parec capture."""

from __future__ import annotations

import json
import subprocess
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import IO, Any, Protocol, cast

from interview_helper.config import AudioConfig


class CaptureError(RuntimeError):
    """The configured local audio source could not be resolved or captured."""


class CommandRunner(Protocol):
    def __call__(self, command: Sequence[str]) -> str: ...


def run_text(command: Sequence[str]) -> str:
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise CaptureError(f"Command failed: {' '.join(command)}: {error}") from error
    return result.stdout


@dataclass(frozen=True, slots=True)
class MonitorSource:
    sink: str
    source: str


@dataclass(frozen=True, slots=True)
class MicrophoneSource:
    name: str
    description: str
    is_default: bool = False


def list_microphone_sources(
    runner: CommandRunner = run_text,
) -> tuple[MicrophoneSource, ...]:
    """Return the non-monitor sources shown by the Linux audio server."""
    default = runner(("pactl", "get-default-source")).strip()
    try:
        payload: Any = json.loads(
            runner(("pactl", "--format=json", "list", "sources"))
        )
    except (json.JSONDecodeError, TypeError) as error:
        raise CaptureError(f"Could not read Linux microphone devices: {error}") from error
    if not isinstance(payload, list):
        raise CaptureError("Linux audio server returned an invalid source list")
    sources: list[MicrophoneSource] = []
    for value in payload:
        if not isinstance(value, dict):
            continue
        name = value.get("name")
        description = value.get("description")
        properties = value.get("properties")
        device_class = (
            properties.get("device.class") if isinstance(properties, dict) else None
        )
        if not isinstance(name, str) or name.endswith(".monitor"):
            continue
        if device_class == "monitor":
            continue
        sources.append(
            MicrophoneSource(
                name=name,
                description=(
                    description
                    if isinstance(description, str) and description.strip()
                    else name
                ),
                is_default=name == default,
            )
        )
    return tuple(sources)


class PulseMonitorResolver:
    def __init__(self, runner: CommandRunner = run_text) -> None:
        self.runner = runner

    def resolve_default(self) -> MonitorSource:
        sink = self.runner(("pactl", "get-default-sink")).strip()
        if not sink:
            raise CaptureError("PipeWire did not report a default output sink")
        source = f"{sink}.monitor"
        rows = self.runner(("pactl", "list", "short", "sources")).splitlines()
        available = {
            fields[1]
            for row in rows
            if len(fields := row.split("\t")) >= 2
        }
        if source not in available:
            raise CaptureError(
                f"Default sink monitor {source!r} is unavailable; "
                "select an active headphone output and retry"
            )
        return MonitorSource(sink=sink, source=source)


class PulseMicrophoneResolver:
    """Resolve and validate the current default non-monitor input source."""

    def __init__(
        self,
        runner: CommandRunner = run_text,
        *,
        source: str | None = None,
    ) -> None:
        self.source = source
        self.runner = runner

    def resolve_default(self) -> MonitorSource:
        source = self.source or self.runner(("pactl", "get-default-source")).strip()
        if not source:
            raise CaptureError("PipeWire did not report a default microphone")
        if source.endswith(".monitor"):
            raise CaptureError(
                "The default input is an output monitor, not a microphone; "
                "select a microphone in system sound settings and retry"
            )
        rows = self.runner(("pactl", "list", "short", "sources")).splitlines()
        available = {
            fields[1]
            for row in rows
            if len(fields := row.split("\t")) >= 2
        }
        if source not in available:
            raise CaptureError(
                f"Default microphone {source!r} is unavailable; "
                "select an active input and retry"
            )
        return MonitorSource(sink=source, source=source)


class PulseSourceResolver(Protocol):
    def resolve_default(self) -> MonitorSource: ...


class Process(Protocol):
    stdout: IO[bytes] | None

    def poll(self) -> int | None: ...

    def terminate(self) -> None: ...

    def wait(self, timeout: float | None = None) -> int: ...

    def kill(self) -> None: ...


ProcessFactory = Callable[[Sequence[str]], Process]


def start_process(command: Sequence[str]) -> Process:
    try:
        return cast(Process, subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL))
    except OSError as error:
        raise CaptureError(f"Could not start parec: {error}") from error


class ParecCapture:
    """Continuously read normalized PCM from one resolved Pulse audio source."""

    def __init__(
        self,
        config: AudioConfig,
        *,
        resolver: PulseSourceResolver | None = None,
        process_factory: ProcessFactory = start_process,
    ) -> None:
        self.config = config
        self.resolver: PulseSourceResolver = resolver or PulseMonitorResolver()
        self.process_factory = process_factory
        self.monitor: MonitorSource | None = None
        self._process: Process | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def command_for(self, monitor: MonitorSource) -> tuple[str, ...]:
        return (
            "parec",
            f"--device={monitor.source}",
            "--raw",
            f"--format={self.config.sample_format}",
            f"--rate={self.config.sample_rate}",
            f"--channels={self.config.channels}",
            f"--latency-msec={self.config.chunk_milliseconds}",
        )

    def start(
        self,
        pcm_callback: Callable[[bytes], None],
        error_callback: Callable[[BaseException], None],
    ) -> MonitorSource:
        if self._thread is not None and self._thread.is_alive():
            raise CaptureError("Audio capture is already running")
        self.monitor = self.resolver.resolve_default()
        self._process = self.process_factory(self.command_for(self.monitor))
        if self._process.stdout is None:
            raise CaptureError("parec did not expose an audio stream")
        self._stop.clear()

        def read() -> None:
            assert self._process is not None
            assert self._process.stdout is not None
            try:
                while not self._stop.is_set():
                    chunk = self._process.stdout.read(self.config.chunk_bytes)
                    if not chunk:
                        if not self._stop.is_set():
                            code = self._process.poll()
                            raise CaptureError(f"parec stopped unexpectedly (exit {code})")
                        return
                    if len(chunk) % 4:
                        raise CaptureError("parec returned a truncated float32 PCM frame")
                    pcm_callback(chunk)
            except BaseException as error:
                if not self._stop.is_set():
                    error_callback(error)

        self._thread = threading.Thread(target=read, name="parec-capture", daemon=True)
        self._thread.start()
        return self.monitor

    def default_sink_changed(self) -> bool:
        if self.monitor is None:
            return False
        return self.resolver.resolve_default().sink != self.monitor.sink

    def stop(self) -> None:
        self._stop.set()
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._thread = None
        self._process = None


ParecMonitorCapture = ParecCapture
