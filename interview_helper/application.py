"""Reusable lifecycle for the daemon and desktop presentation layer."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from interview_helper.answer import (
    ANSWER_TOKEN_LIMIT,
    AnswerConversation,
    AnswerTurn,
    GroundedAnswerPipeline,
)
from interview_helper.automatic import AutomaticTranscriber
from interview_helper.turn_detection import SileroSpeechDetector, SmartTurnDetector
from interview_helper.capture import (
    ParecMonitorCapture,
    PulseMicrophoneResolver,
)
from interview_helper.config import AudioConfig, MoonshineConfig
from interview_helper.context import CandidateProfile, ContextError, TechnicalAnswers
from interview_helper.core import HoldToTranscribe, RuntimeState, Transcript
from interview_helper.daemon import HoldControl, InterviewDaemon, TranscriptionEngine
from interview_helper.input import EvdevHoldControl, ManualHoldControl
from interview_helper.moonshine import MoonshineTranscriber
from interview_helper.qwen import DEFAULT_BASE_URL, DEFAULT_MODEL, LocalQwenClient
from interview_helper.openai_client import OPENAI_MODEL, OPENAI_MODELS, OpenAIAnswerClient
from interview_helper.library import TechnicalLibrary
from interview_helper.spoken import SpokenConversation, SpokenTurn
from interview_helper.duplex import DuplexDaemon


def _ignore(_value: object) -> None:
    return


class CaptureMode(str, Enum):
    HEADPHONE_MONITOR = "headphone_monitor"
    DEFAULT_MICROPHONE = "default_microphone"


class ControlMode(str, Enum):
    EVDEV = "evdev"
    MANUAL = "manual"


class AnswerProvider(str, Enum):
    LOCAL_QWEN = "local"
    OPENAI = "openai"


@dataclass(frozen=True, slots=True)
class ApplicationConfig:
    input_device: Path | None
    event_code: str
    model_path: Path
    capture_mode: CaptureMode = CaptureMode.HEADPHONE_MONITOR
    control_mode: ControlMode = ControlMode.EVDEV
    microphone_source: str | None = None
    keyterms: tuple[str, ...] = ()
    resume: Path | None = None
    context: tuple[Path, ...] = ()
    qwen_base_url: str = DEFAULT_BASE_URL
    qwen_model: str = DEFAULT_MODEL
    answer_max_tokens: int = ANSWER_TOKEN_LIMIT
    technical_library: Path | None = None
    show_partials: bool = True
    automatic_listening: bool = False
    listen_to_candidate: bool = False
    answer_provider: AnswerProvider = AnswerProvider.LOCAL_QWEN
    openai_api_key_file: Path | None = None
    openai_model: str = OPENAI_MODEL
    technical_answers: Path | None = None

    def __post_init__(self) -> None:
        if self.listen_to_candidate and (
            not self.automatic_listening or self.capture_mode is not CaptureMode.HEADPHONE_MONITOR
        ):
            raise ValueError("Spoken-answer experiment requires hands-free headphone capture")
        if self.control_mode is ControlMode.EVDEV and not self.automatic_listening:
            if self.input_device is None:
                raise ValueError("Input device is required for keyboard control")
            if not self.event_code.strip():
                raise ValueError("Input event code cannot be empty")
        if self.context and self.resume is None:
            raise ContextError("Context files require a resume")
        if self.technical_answers is not None and self.resume is None:
            raise ContextError("Technical mode requires a resume")
        if self.answer_max_tokens <= 0:
            raise ValueError("Answer token limit must be positive")
        if not isinstance(self.answer_provider, AnswerProvider):
            raise ValueError("Unknown answer provider")
        if self.openai_model not in OPENAI_MODELS:
            raise ValueError(f"Unsupported OpenAI model: {self.openai_model}")
        if self.answer_provider is AnswerProvider.LOCAL_QWEN and (
            not self.qwen_base_url.strip() or not self.qwen_model.strip()
        ):
            raise ValueError("Qwen URL and model cannot be empty")


@dataclass(slots=True)
class ApplicationCallbacks:
    candidate_partial: Callable[[str], None] = field(default=_ignore)
    candidate_state: Callable[[RuntimeState], None] = field(default=_ignore)
    candidate_transcript: Callable[[SpokenTurn], None] = field(default=_ignore)
    state: Callable[[RuntimeState], None] = field(default=_ignore)
    capture_ready: Callable[[bool], None] = field(default=_ignore)
    status: Callable[[str], None] = field(default=_ignore)
    partial: Callable[[str], None] = field(default=_ignore)
    transcript: Callable[[Transcript], None] = field(default=_ignore)
    answer: Callable[[str], None] = field(default=_ignore)
    answer_progress: Callable[[str], None] | None = None
    answer_cancelled: Callable[[], None] = field(default=lambda: None)
    evidence: Callable[[str], None] = field(default=_ignore)
    exchange: Callable[[AnswerTurn], None] = field(default=_ignore)
    answer_busy: Callable[[bool], None] = field(default=_ignore)
    error: Callable[[BaseException], None] = field(default=_ignore)


class InterviewApplication:
    """Own one configured daemon run and all of its dependent resources."""

    def __init__(
        self,
        config: ApplicationConfig,
        callbacks: ApplicationCallbacks | None = None,
        conversation: AnswerConversation | None = None,
        spoken_conversation: SpokenConversation | None = None,
    ) -> None:
        self.config = config
        self.callbacks = callbacks or ApplicationCallbacks()
        self.conversation = conversation or AnswerConversation()
        self.spoken_conversation = spoken_conversation if spoken_conversation is not None else SpokenConversation()
        self._candidate_engine: AutomaticTranscriber | None = None
        self._question_lock = threading.Lock()
        self._current_question = ""
        self._candidate_spoke = False
        self._source_ready = {"interviewer": False, "candidate": False}
        self._lock = threading.Lock()
        self._running = False
        self._stop_requested = threading.Event()
        self._daemon: InterviewDaemon | DuplexDaemon | None = None
        self._answer_pipeline: GroundedAnswerPipeline | None = None
        self._qwen: LocalQwenClient | OpenAIAnswerClient | None = None
        self._manual_control: ManualHoldControl | None = None
        self._automatic_engine: AutomaticTranscriber | None = None
        self._library: TechnicalLibrary | None = None

    @property
    def running(self) -> bool:
        with self._lock:
            return self._running

    def run(self) -> None:
        """Build and run the application; call this from a worker thread."""
        with self._lock:
            if self._running:
                raise RuntimeError("Interview Helper is already running")
            self._running = True
            self._stop_requested.clear()
        try:
            daemon = self._build_daemon()
            with self._lock:
                self._daemon = daemon
            if self._stop_requested.is_set():
                daemon.close()
                return
            daemon.run()
        finally:
            with self._lock:
                active_daemon = self._daemon
            if active_daemon is not None:
                active_daemon.close()
            self._close_resources()
            with self._lock:
                self._daemon = None
                self._running = False

    def stop(self) -> None:
        """Request shutdown without waiting for model or HTTP cleanup."""
        self._stop_requested.set()
        pipeline = self._answer_pipeline
        if pipeline is not None:
            pipeline.cancel_pending()
        with self._lock:
            daemon = self._daemon
        if daemon is not None:
            daemon.close()

    def answer_now(self) -> bool:
        engine = self._automatic_engine
        return engine.press() if engine is not None else False

    def cancel_answer(self) -> bool:
        engine = self._automatic_engine
        if engine is not None:
            engine.cancel()
        pipeline = self._answer_pipeline
        return pipeline.cancel_pending() if pipeline is not None else False

    def _interrupt_answer(self) -> bool:
        pipeline = self._answer_pipeline
        return pipeline.cancel_pending() if pipeline is not None else False

    def _interrupt_continuation(self) -> bool:
        if self.config.listen_to_candidate:
            with self._question_lock:
                spoke = self._candidate_spoke
            if spoke or (self._candidate_engine is not None and
                         self._candidate_engine.state in (RuntimeState.RECORDING, RuntimeState.TRANSCRIBING)):
                return False
        return self._interrupt_answer()

    def manual_press(self) -> bool:
        control = self._manual_control
        return control.press() if control is not None else False

    def manual_release(self) -> bool:
        control = self._manual_control
        return control.release() if control is not None else False

    def _build_daemon(self) -> InterviewDaemon | DuplexDaemon:
        audio = AudioConfig()

        def on_transcript(transcript: Transcript) -> None:
            if self._stop_requested.is_set():
                return
            candidate = self._candidate_engine
            if candidate is not None:
                with self._question_lock:
                    ready = all(self._source_ready.values())
                if not ready:
                    self.callbacks.status("Both audio inputs must be ready before answering a follow-up")
                    return
                # The interviewer worker waits for preceding microphone frames,
                # including final ASR words, before the answer takes its snapshot.
                if not candidate.finish_pending(timeout=5.0):
                    self.callbacks.error(RuntimeError(
                        "Microphone transcription could not finish; follow-up was not submitted."
                    ))
                    return
                if self._stop_requested.is_set():
                    return
                self._interrupt_answer()
                with self._question_lock:
                    self._current_question = transcript.text
                    self._candidate_spoke = False
            self.callbacks.transcript(transcript)
            pipeline = self._answer_pipeline
            if pipeline is not None:
                pipeline.submit(transcript)

        if self.config.resume is not None:
            profile = CandidateProfile.load(self.config.resume, self.config.context)
            technical_answers = (
                TechnicalAnswers.load(self.config.technical_answers)
                if self.config.technical_answers is not None else None
            )
            if technical_answers is not None:
                self.callbacks.status(
                    f"Technical mode: {technical_answers.source} and your last five "
                    "answers are sent with each question."
                )
            if self.config.answer_provider is AnswerProvider.OPENAI:
                self.callbacks.status(
                    f"Connecting to OpenAI {OPENAI_MODELS[self.config.openai_model]}; "
                    "selected context and transcripts are sent to OpenAI…"
                )
                self._qwen = OpenAIAnswerClient(
                    api_key_file=self.config.openai_api_key_file, model=self.config.openai_model,
                )
            else:
                self._qwen = LocalQwenClient(
                    base_url=self.config.qwen_base_url,
                    model=self.config.qwen_model,
                )
            if self.config.technical_library is not None:
                self.callbacks.status("Preparing local documentation index…")
                self._library = TechnicalLibrary.open(self.config.technical_library)
            self._answer_pipeline = GroundedAnswerPipeline(
                profile,
                self._qwen,
                answer_callback=self.callbacks.answer,
                progress_callback=self.callbacks.answer_progress,
                cancelled_callback=self.callbacks.answer_cancelled,
                error_callback=self.callbacks.error,
                busy_callback=self.callbacks.answer_busy,
                exchange_callback=self.callbacks.exchange,
                conversation=self.conversation,
                max_tokens=self.config.answer_max_tokens,
                technical_library=self._library,
                status_callback=self.callbacks.status,
                evidence_callback=self.callbacks.evidence,
                spoken_conversation=self.spoken_conversation if self.config.listen_to_candidate else None,
                technical_answers=technical_answers,
            )
        transcriber = MoonshineTranscriber(
            MoonshineConfig(
                model_path=self.config.model_path,
                keyterms=self.config.keyterms,
            )
        )
        engine: TranscriptionEngine
        if self.config.automatic_listening:
            self.callbacks.status("Loading local hands-free detectors…")
            self._automatic_engine = AutomaticTranscriber(
                transcriber,
                speech_detector=SileroSpeechDetector(),
                turn_detector=SmartTurnDetector(),
                transcript_callback=on_transcript,
                partial_callback=self.callbacks.partial,
                state_callback=self.callbacks.state,
                error_callback=self.callbacks.error,
                status_callback=self.callbacks.status,
                interrupt_callback=self._interrupt_continuation,
            )
            engine = self._automatic_engine
        else:
            engine = HoldToTranscribe(
                transcriber,
                sample_rate=audio.sample_rate,
                transcript_callback=on_transcript,
                partial_callback=self.callbacks.partial if self.config.show_partials else None,
                state_callback=self.callbacks.state,
                error_callback=self.callbacks.error,
            )
        capture = (
            ParecMonitorCapture(
                audio,
                resolver=PulseMicrophoneResolver(
                    source=self.config.microphone_source
                ),
            )
            if self.config.capture_mode is CaptureMode.DEFAULT_MICROPHONE
            else ParecMonitorCapture(audio)
        )
        control: HoldControl
        if self.config.control_mode is ControlMode.MANUAL or self.config.automatic_listening:
            manual_control = ManualHoldControl()
            self._manual_control = manual_control
            control = manual_control
        else:
            assert self.config.input_device is not None
            control = EvdevHoldControl(self.config.input_device, self.config.event_code)
        interviewer = InterviewDaemon(
            capture=capture,
            control=control,
            engine=engine,
            status=self.callbacks.status,
            capture_ready_callback=(lambda ready: self._source_capture_ready("interviewer", ready))
            if self.config.listen_to_candidate else self.callbacks.capture_ready,
            release_drain_seconds=audio.release_drain_milliseconds / 1_000,
            abort_on_capture_loss=self.config.automatic_listening,
            control_description=("hands-free listening" if self.config.automatic_listening else "waiting for held control"),
        )
        if not self.config.listen_to_candidate:
            return interviewer
        candidate_transcriber = MoonshineTranscriber(
            MoonshineConfig(model_path=self.config.model_path, keyterms=self.config.keyterms)
        )
        self._candidate_engine = AutomaticTranscriber(
            candidate_transcriber,
            speech_detector=SileroSpeechDetector(),
            turn_detector=SmartTurnDetector(),
            transcript_callback=self._on_candidate_transcript,
            partial_callback=self.callbacks.candidate_partial,
            state_callback=self._on_candidate_state,
            error_callback=self.callbacks.error,
            status_callback=lambda text: self.callbacks.status(f"Microphone: {text}"),
            question_only=False,
            max_utterance_seconds=180.0,
        )
        candidate_daemon = InterviewDaemon(
            capture=ParecMonitorCapture(audio, resolver=PulseMicrophoneResolver(
                source=self.config.microphone_source,
            )),
            control=ManualHoldControl(),
            engine=self._candidate_engine,
            status=lambda text: self.callbacks.status(f"Microphone: {text}"),
            capture_ready_callback=lambda ready: self._source_capture_ready("candidate", ready),
            abort_on_capture_loss=True,
            control_description="listening to your spoken answers",
        )
        return DuplexDaemon(interviewer=interviewer, candidate=candidate_daemon)

    def _source_capture_ready(self, source: str, ready: bool) -> None:
        with self._question_lock:
            self._source_ready[source] = ready
            all_ready = all(self._source_ready.values())
        self.callbacks.capture_ready(all_ready)
        if source == "candidate" and not self._stop_requested.is_set():
            engine = self._candidate_engine
            self.callbacks.candidate_state(
                engine.state if ready and engine is not None else RuntimeState.ERROR
            )
        if not ready:
            self._interrupt_answer()

    def _on_candidate_transcript(self, transcript: Transcript) -> None:
        if self._stop_requested.is_set() or not transcript.text.strip():
            return
        with self._question_lock:
            question = self._current_question
            self._candidate_spoke = True
        turn = self.spoken_conversation.append(transcript.text, question=question)
        self.callbacks.candidate_transcript(turn)

    def _on_candidate_state(self, state: RuntimeState) -> None:
        if self._stop_requested.is_set():
            return
        if state in (RuntimeState.RECORDING, RuntimeState.TRANSCRIBING):
            with self._question_lock:
                self._candidate_spoke = True
        self.callbacks.candidate_state(state)

    def correct_candidate_speech(self, turn_id: int, text: str) -> bool:
        # A draft may already contain the old words. Do not leave it presented
        # as a current answer after a correction; no automatic extra model call.
        changed = self.spoken_conversation.replace(turn_id, text)
        if changed:
            self._interrupt_answer()
        return changed

    def _close_resources(self) -> None:
        self._manual_control = None
        candidate_engine, self._candidate_engine = self._candidate_engine, None
        if candidate_engine is not None:
            candidate_engine.close()
        automatic_engine = self._automatic_engine
        self._automatic_engine = None
        if automatic_engine is not None:
            automatic_engine.close()
        pipeline = self._answer_pipeline
        self._answer_pipeline = None
        if pipeline is not None:
            pipeline.close()
        library = self._library
        self._library = None
        if library is not None:
            library.close()
        qwen = self._qwen
        self._qwen = None
        if qwen is not None:
            qwen.close()
