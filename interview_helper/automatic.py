"""Bounded, opt-in automatic interview turns on a dedicated audio worker."""

from __future__ import annotations

import re
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from interview_helper.core import RuntimeState, StreamingTranscriber, Transcript, TranscriptionSession


class SpeechDetector(Protocol):
    def is_speech(self, pcm: bytes) -> bool: ...

    def reset(self) -> None: ...


class TurnDetector(Protocol):
    def is_complete(self, pcm: bytes) -> bool: ...


_ACK = re.compile(
    r"^(?:(?:okay|ok|yes|yeah|yep|right|sure|great|good|cool|perfect|thanks|thank you|"
    r"that makes sense|i see|understood|got it|all right|hello|hi|bye|goodbye|"
    r"nice to meet you|sounds good|you're welcome|you are welcome|very good|"
    r"thank you very much|thanks for that|thanks for explaining that|"
    r"thank you for explaining that|that was helpful|well|"
    r"how are you|how are you doing|how is your day|can you hear me|"
    r"can you see my screen|good morning|good afternoon)[\s,.!?]*)+$", re.IGNORECASE,
)
_REQUEST = re.compile(
    r"\b(?:what|why|how|when|where|which|who|whose|describe|explain|compare|"
    r"contrast|demonstrate|outline|summarize|elaborate|clarify|"
    r"tell me|walk (?:me|us) through|give (?:me|us)|talk (?:me|us) through|"
    r"take (?:me|us) through|pick a|can you|could you|would you|have you|"
    r"did you|do you|are you|is there|any (?:examples|risks|caveats|limitations))\b",
    re.IGNORECASE,
)


def is_interview_prompt(text: str) -> bool:
    """A deliberately modest intent gate, not a claim of question understanding."""
    text = text.strip()
    if not text or _ACK.fullmatch(text):
        return False
    if _REQUEST.search(text):
        return True
    if re.match(r"^(?:and|but|specifically|also|more about|for example)\b", text, re.I):
        return len(text.split()) >= 2
    return text.endswith("?") and len(text.split()) >= 2


@dataclass
class _Command:
    kind: str
    error: BaseException | None = None
    barrier: threading.Event | None = None
    generation: int = 0
    succeeded: bool = False


class AutomaticTranscriber:
    """Analyze 16kHz float32 PCM without blocking the capture reader.

    No audio is saved. Retained audio is limited to four queued seconds, a
    200ms preroll, and the detector's last eight seconds. The ASR stream keeps
    its complete text across acoustic line boundaries until a turn ends.
    """

    sample_rate = 16_000
    _frame_bytes = 512 * 4
    _bytes_per_second = sample_rate * 4
    _queue_limit = 4 * _bytes_per_second

    def __init__(
        self,
        transcriber: StreamingTranscriber,
        speech_detector: SpeechDetector,
        turn_detector: TurnDetector,
        *,
        transcript_callback: Callable[[Transcript], None],
        partial_callback: Callable[[str], None] | None = None,
        state_callback: Callable[[RuntimeState], None] | None = None,
        error_callback: Callable[[BaseException], None] | None = None,
        status_callback: Callable[[str], None] | None = None,
        interrupt_callback: Callable[[], bool] | None = None,
        question_only: bool = True,
        minimum_silence_seconds: float = 0.65,
        max_silence_seconds: float = 3.0,
        max_utterance_seconds: float = 90.0,
    ) -> None:
        if not 0 < minimum_silence_seconds <= max_silence_seconds:
            raise ValueError("Silence thresholds must be positive and ordered")
        if max_utterance_seconds <= max_silence_seconds:
            raise ValueError("Maximum utterance must exceed maximum silence")
        self.question_only = question_only
        self._failure_generation = 0
        self.transcriber = transcriber
        self.speech_detector = speech_detector
        self.turn_detector = turn_detector
        self.transcript_callback = transcript_callback
        self.partial_callback = partial_callback or (lambda _text: None)
        self.state_callback = state_callback or (lambda _state: None)
        self.error_callback = error_callback or (lambda _error: None)
        self.status_callback = status_callback or (lambda _text: None)
        self.interrupt_callback = interrupt_callback or (lambda: False)
        self.minimum_silence_seconds = minimum_silence_seconds
        self.max_silence_seconds = max_silence_seconds
        self.max_utterance_seconds = max_utterance_seconds
        self._condition = threading.Condition()
        self._queue: deque[bytes | _Command] = deque()
        self._queued_bytes = 0
        self._closed = threading.Event()
        self._reset_pending = threading.Event()
        self._state = RuntimeState.IDLE
        self._session: TranscriptionSession | None = None
        self._pending = bytearray()
        self._preroll = bytearray()
        self._window = bytearray()
        self._clock = 0.0
        self._started = 0.0
        self._silence = 0.0
        self._next_check = minimum_silence_seconds
        self._text = ""
        self._result: Transcript | None = None
        self._previous = ""
        self._previous_end = -100.0
        self._continuation = False
        self._interrupt_checked = False
        self._prefix = ""
        self._lead_in = ""
        self._lead_in_end = -100.0
        self._active_lead_in = ""
        self._worker = threading.Thread(target=self._run, name="automatic-transcription", daemon=True)
        self._worker.start()

    @property
    def state(self) -> RuntimeState:
        with self._condition:
            return self._state

    def press(self) -> bool:
        """Explicit answer-now bypasses the automatic intent gate."""
        return self._command(_Command("finish"))

    def release(self) -> bool:
        return False

    def cancel(self) -> bool:
        return self._command(_Command("cancel"), discard_audio=True)

    def abort(self, error: BaseException) -> None:
        self._command(_Command("abort", error=error), discard_audio=True)

    def add_pcm(self, pcm: bytes) -> None:
        if not pcm or self._closed.is_set():
            return
        if len(pcm) % 4:
            self.abort(ValueError("Invalid float32 PCM chunk"))
            return
        with self._condition:
            if self._closed.is_set():
                return
            if self._queued_bytes + len(pcm) > self._queue_limit:
                self._discard_queued_audio()
                self._reset_pending.set()
                if not any(isinstance(item, _Command) and item.kind == "abort" for item in self._queue):
                    self._queue.appendleft(_Command(
                        "abort", error=RuntimeError("Automatic audio queue overflow; question discarded" if self.question_only
                                             else "Automatic audio queue overflow; candidate speech discarded"),
                    ))
            else:
                self._queue.append(pcm)
                self._queued_bytes += len(pcm)
            self._condition.notify()

    def wait_until_drained(self, timeout: float = 5.0) -> bool:
        """Wait for submitted audio/controls, without waiting for an endpoint."""
        barrier = threading.Event()
        return self._command(_Command("barrier", barrier=barrier)) and barrier.wait(timeout)

    def finish_pending(self, timeout: float = 5.0) -> bool:
        """Finalize queued audio and wait for transcript delivery from another worker."""
        if threading.current_thread() is self._worker:
            raise RuntimeError("Cannot wait for transcription on its own worker")
        command = _Command("finish", barrier=threading.Event())
        with self._condition:
            command.generation = self._failure_generation
            if not self._command(command):
                return False
        assert command.barrier is not None
        if not command.barrier.wait(timeout):
            self.error_callback(TimeoutError("Timed out finishing pending speech"))
            return False
        return command.succeeded

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        with self._condition:
            self._condition.notify_all()
        if threading.current_thread() is not self._worker:
            self._worker.join(timeout=5)

    def _command(self, command: _Command, *, discard_audio: bool = False) -> bool:
        with self._condition:
            if self._closed.is_set():
                return False
            if discard_audio:
                self._discard_queued_audio()
                self._reset_pending.set()
                self._queue.appendleft(command)
            else:
                self._queue.append(command)
            self._condition.notify()
        return True

    def _discard_queued_audio(self) -> None:
        self._failure_generation += 1
        for item in self._queue:
            if isinstance(item, _Command) and item.kind == "finish" and item.barrier is not None:
                item.barrier.set()
        self._queue = deque(item for item in self._queue if isinstance(item, _Command) and item.kind in {"barrier", "abort", "cancel"})
        self._queued_bytes = 0

    def _set_state(self, state: RuntimeState) -> None:
        with self._condition:
            self._state = state
        self.state_callback(state)

    def _run(self) -> None:
        try:
            while not self._closed.is_set():
                with self._condition:
                    self._condition.wait_for(lambda: bool(self._queue) or self._closed.is_set())
                    if self._closed.is_set():
                        break
                    item = self._queue.popleft()
                    if isinstance(item, bytes):
                        self._queued_bytes -= len(item)
                try:
                    if isinstance(item, bytes):
                        self._pending.extend(item)
                        while (len(self._pending) >= self._frame_bytes
                               and not self._closed.is_set() and not self._reset_pending.is_set()):
                            frame = bytes(self._pending[:self._frame_bytes])
                            del self._pending[:self._frame_bytes]
                            self._frame(frame)
                    elif item.kind == "barrier":
                        assert item.barrier is not None
                        item.barrier.set()
                    elif item.kind == "finish":
                        self._finish(manual=True)
                        item.succeeded = (item.generation == self._failure_generation
                                          and not self._closed.is_set()
                                          and not self._reset_pending.is_set())
                    else:
                        self._discard(clear_previous=True)
                        self._reset_pending.clear()
                        if item.kind == "cancel" and self.question_only:
                            self.interrupt_callback()
                        if item.error is not None:
                            self.error_callback(item.error)
                        self.status_callback("Listening for the next question" if self.question_only else "Listening to you")
                except Exception as error:
                    self._failure_generation += 1
                    self._discard(clear_previous=True)
                    self.error_callback(error)
                finally:
                    if isinstance(item, _Command) and item.kind == "finish" and item.barrier is not None:
                        item.barrier.set()
        finally:
            self._discard(clear_previous=True)
            with self._condition:
                for command in self._queue:
                    if isinstance(command, _Command) and command.barrier is not None:
                        command.barrier.set()
                self._queue.clear()
                self._queued_bytes = 0
            self._set_state(RuntimeState.CLOSED)

    def _frame(self, pcm: bytes) -> None:
        self._clock += len(pcm) / self._bytes_per_second
        speech = self.speech_detector.is_speech(pcm)
        if self._closed.is_set() or self._reset_pending.is_set():
            return
        if self._session is None:
            if self._clock - self._lead_in_end > 3.0:
                self._lead_in = ""
            if not speech:
                self._preroll.extend(pcm)
                del self._preroll[:-int(0.2 * self._bytes_per_second)]
                return
            self._started = self._clock
            self._active_lead_in = self._lead_in
            self._continuation = self.question_only and bool(self._previous) and self._clock - self._previous_end <= 2.0
            self._interrupt_checked = False
            self._session = self.transcriber.open_stream(
                partial_callback=self._partial, complete_callback=self._complete,
            )
            self._set_state(RuntimeState.RECORDING)
            self.status_callback("Listening to the question" if self.question_only else "Listening to your answer")
            if self._preroll:
                self._session.add_pcm(bytes(self._preroll), self.sample_rate)
                self._window.extend(self._preroll)
                self._preroll.clear()
        self._window.extend(pcm)
        del self._window[:-8 * self._bytes_per_second]
        self._session.add_pcm(pcm, self.sample_rate)
        if self._clock - self._started >= self.max_utterance_seconds:
            raise RuntimeError("Automatic question exceeded time limit; question discarded" if self.question_only
                               else "Automatic candidate speech exceeded time limit; speech discarded")
        if speech:
            self._silence = 0.0
            self._next_check = self.minimum_silence_seconds
            return
        self._silence += len(pcm) / self._bytes_per_second
        if self._silence >= self.max_silence_seconds:
            self._finish()
        elif self._silence >= self._next_check:
            self._next_check = self._silence + self.minimum_silence_seconds
            if self.turn_detector.is_complete(bytes(self._window)):
                self._finish()
            else:
                self.status_callback("Waiting for the rest of the question" if self.question_only else "Waiting for the rest of your answer")

    def _partial(self, text: str) -> None:
        if self._closed.is_set() or self._reset_pending.is_set():
            return
        self._text = text.strip()
        self._check_continuation(self._text)
        self.partial_callback(" ".join(filter(None, (self._prefix, self._active_lead_in, self._text))))

    def _check_continuation(self, text: str) -> None:
        if self._closed.is_set() or self._reset_pending.is_set():
            return
        if (self.question_only and self._continuation and not self._interrupt_checked
                and len(text.split()) >= 2 and not _ACK.fullmatch(text.strip())):
            self._interrupt_checked = True
            if self.interrupt_callback():
                self._prefix = self._previous
                self.status_callback("Question continued; updating the answer")

    def _complete(self, transcript: Transcript) -> None:
        self._result = transcript

    def _finish(self, *, manual: bool = False) -> None:
        session = self._session
        if session is None:
            return
        self._set_state(RuntimeState.TRANSCRIBING)
        if manual and self._pending:
            session.add_pcm(bytes(self._pending), self.sample_rate)
            self._pending.clear()
        session.finish()
        result = self._result
        text = (result.text if result is not None else self._text).strip()
        self._check_continuation(text)
        # Pending setup provides context, never permission to answer a statement.
        accepted = bool(text) and (not self.question_only or manual or bool(self._prefix) or is_interview_prompt(text))
        if accepted:
            text = " ".join(filter(None, (self._prefix, self._active_lead_in, text)))
            self._lead_in = ""
        elif text and not _ACK.fullmatch(text):
            lead_in = " ".join(filter(None, (self._active_lead_in, text)))
            if len(lead_in) > 4_000:
                raise RuntimeError("Automatic question setup exceeded 4000 characters; question discarded")
            self._lead_in = lead_in
            self._lead_in_end = self._clock
        latency = result.transcription_seconds if result is not None else 0.0
        self._discard()
        if accepted and not self._closed.is_set() and not self._reset_pending.is_set():
            self._previous = text if self.question_only else ""
            self._previous_end = self._clock
            self.transcript_callback(Transcript(text, latency))
            self.status_callback("Listening for the next question" if self.question_only else "Listening to you")
        elif not self._closed.is_set():
            self.status_callback("No interview question detected; listening" if self.question_only else "No speech detected; listening to you")

    def _discard(self, *, clear_previous: bool = False) -> None:
        session, self._session = self._session, None
        if session is not None:
            session.close()
        self._window.clear()
        self._preroll.clear()
        self._text = ""
        self._result = None
        self._silence = 0.0
        self._prefix = ""
        self._active_lead_in = ""
        self._continuation = False
        self._interrupt_checked = False
        self.speech_detector.reset()
        if clear_previous:
            self._pending.clear()
            self._previous = ""
            self._previous_end = -100.0
            self._lead_in = ""
            self._lead_in_end = -100.0
        if not self._closed.is_set():
            self.partial_callback("")
            self._set_state(RuntimeState.IDLE)
