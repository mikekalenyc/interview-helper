"""Toolkit-neutral presentation state for the optional desktop application."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import TypeAlias, assert_never

from interview_helper.core import RuntimeState


class PresentationPhase(str, Enum):
    """User-visible phases of the desktop application."""

    STOPPED = "stopped"
    STOPPING = "stopping"
    STARTING = "starting"
    READY = "ready"
    RECORDING = "recording"
    TRANSCRIBING = "transcribing"
    GENERATING = "generating"
    RECONNECTING = "reconnecting"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class PresentationSnapshot:
    """Immutable application state that can be rendered by any GUI toolkit.

    ``run_id`` identifies the current controller run.  Callbacks carry that id,
    allowing the reducer to ignore work that completes after a stop or restart.
    """

    run_id: int = 0
    running: bool = False
    stopping: bool = False
    startup_complete: bool = False
    capture_ready: bool = False
    runtime_state: RuntimeState = RuntimeState.CLOSED
    answer_busy: bool = False
    transcript: str = ""
    answer: str = ""
    answer_draft: str = ""
    error_message: str | None = None

    @property
    def phase(self) -> PresentationPhase:
        """Derive the visible phase while preserving independent busy states."""
        if self.error_message is not None or self.runtime_state is RuntimeState.ERROR:
            return PresentationPhase.ERROR
        if not self.running or self.runtime_state is RuntimeState.CLOSED:
            return PresentationPhase.STOPPED
        if self.stopping:
            return PresentationPhase.STOPPING
        if self.runtime_state is RuntimeState.RECORDING:
            return PresentationPhase.RECORDING
        if self.runtime_state is RuntimeState.TRANSCRIBING:
            return PresentationPhase.TRANSCRIBING
        if not self.startup_complete:
            return PresentationPhase.STARTING
        if self.answer_busy:
            return PresentationPhase.GENERATING
        if not self.capture_ready:
            return PresentationPhase.RECONNECTING
        return PresentationPhase.READY


@dataclass(frozen=True, slots=True)
class StartRequested:
    """Begin a new run and clear output left by the previous one."""


@dataclass(frozen=True, slots=True)
class StopRequested:
    """Stop the active run and invalidate its outstanding callbacks."""


@dataclass(frozen=True, slots=True)
class RunFinished:
    """The background lifecycle ended, retaining any visible failure."""

    run_id: int


@dataclass(frozen=True, slots=True)
class CaptureStatusChanged:
    run_id: int
    ready: bool


@dataclass(frozen=True, slots=True)
class RuntimeStateChanged:
    run_id: int
    state: RuntimeState


@dataclass(frozen=True, slots=True)
class TranscriptFinalized:
    run_id: int
    text: str


@dataclass(frozen=True, slots=True)
class AnswerStarted:
    run_id: int


@dataclass(frozen=True, slots=True)
class AnswerCancelled:
    run_id: int


@dataclass(frozen=True, slots=True)
class AnswerProgress:
    run_id: int
    text: str


@dataclass(frozen=True, slots=True)
class AnswerCompleted:
    run_id: int
    text: str


@dataclass(frozen=True, slots=True)
class ErrorReported:
    run_id: int
    message: str


PresentationEvent: TypeAlias = (
    StartRequested
    | StopRequested
    | RunFinished
    | CaptureStatusChanged
    | RuntimeStateChanged
    | TranscriptFinalized
    | AnswerStarted
    | AnswerCompleted
    | AnswerProgress
    | AnswerCancelled
    | ErrorReported
)


def reduce_presentation(
    snapshot: PresentationSnapshot, event: PresentationEvent
) -> PresentationSnapshot:
    """Apply one controller event without mutating the previous snapshot."""
    if isinstance(event, StartRequested):
        if snapshot.running:
            return snapshot
        return PresentationSnapshot(
            run_id=snapshot.run_id + 1,
            running=True,
            runtime_state=RuntimeState.IDLE,
        )

    if isinstance(event, StopRequested):
        if not snapshot.running or snapshot.stopping:
            return snapshot
        return replace(snapshot, stopping=True, answer_busy=False, answer_draft="")

    if isinstance(event, RunFinished):
        if event.run_id != snapshot.run_id or not snapshot.running:
            return snapshot
        return PresentationSnapshot(
            run_id=snapshot.run_id + 1,
            error_message=snapshot.error_message,
        )

    if event.run_id != snapshot.run_id or not snapshot.running:
        return snapshot

    if snapshot.stopping:
        return snapshot

    if isinstance(event, CaptureStatusChanged):
        return replace(
            snapshot,
            startup_complete=True,
            capture_ready=event.ready,
            error_message=None if event.ready else snapshot.error_message,
        )
    if isinstance(event, RuntimeStateChanged):
        if event.state is RuntimeState.CLOSED:
            return replace(
                snapshot,
                capture_ready=False,
                runtime_state=RuntimeState.CLOSED,
                answer_busy=False,
                answer_draft="",
            )
        return replace(snapshot, runtime_state=event.state)
    if isinstance(event, TranscriptFinalized):
        return replace(
            snapshot, transcript=event.text.strip(), answer="", error_message=None
        )
    if isinstance(event, AnswerStarted):
        return replace(snapshot, answer_busy=True, answer="", answer_draft="", error_message=None)
    if isinstance(event, AnswerCancelled):
        return replace(snapshot, answer_busy=False, answer_draft="", answer="", transcript="")
    if isinstance(event, AnswerProgress):
        if not snapshot.answer_busy:
            return snapshot
        return replace(snapshot, answer_draft=event.text)
    if isinstance(event, AnswerCompleted):
        return replace(
            snapshot,
            answer_busy=False,
            answer=event.text.strip(),
            answer_draft="",
            error_message=None,
        )
    if isinstance(event, ErrorReported):
        message = event.message.strip() or "An unknown error occurred."
        return replace(snapshot, answer_busy=False, answer_draft="", error_message=message)

    assert_never(event)
