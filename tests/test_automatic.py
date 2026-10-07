"""Automatic turns tested without devices, models, or retained recordings."""

from __future__ import annotations

import struct
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

import pytest

from interview_helper.automatic import AutomaticTranscriber, is_interview_prompt
from interview_helper.core import RuntimeState, Transcript

SPEECH = struct.pack('<f', 0.2) * 512
SILENCE = bytes(512 * 4)


class Speech:
    resets = 0

    def is_speech(self, pcm: bytes) -> bool:
        return pcm != bytes(len(pcm))

    def reset(self) -> None:
        self.resets += 1


class Turn:
    def __init__(self, decisions: list[bool] | None = None) -> None:
        self.decisions = decisions or [True]
        self.windows: list[bytes] = []

    def is_complete(self, pcm: bytes) -> bool:
        self.windows.append(pcm)
        return self.decisions.pop(0) if len(self.decisions) > 1 else self.decisions[0]


class Session:
    def __init__(self, text: str, partial: Callable[[str], None], complete: Callable[[Transcript], None]) -> None:
        self.text = text
        self.partial = partial
        self.complete = complete
        self.closed = False
        self.finished = False
        self.audio: list[bytes] = []

    def add_pcm(self, pcm: bytes, sample_rate: int) -> None:
        assert sample_rate == 16_000
        self.audio.append(pcm)
        if any(pcm):
            self.partial(self.text)

    def finish(self) -> None:
        self.finished = True
        self.complete(Transcript(self.text, 0.012))

    def close(self) -> None:
        self.closed = True


class Transcriber:
    def __init__(self, texts: list[str]) -> None:
        self.texts = iter(texts)
        self.sessions: list[Session] = []

    def open_stream(self, *, partial_callback: Callable[[str], None], complete_callback: Callable[[Transcript], None]) -> Session:
        session = Session(next(self.texts), partial_callback, complete_callback)
        self.sessions.append(session)
        return session


@dataclass
class Harness:
    engine: AutomaticTranscriber
    transcriber: Transcriber
    turn: Turn
    transcripts: list[Transcript]
    partials: list[str]
    errors: list[BaseException]
    states: list[RuntimeState] = field(default_factory=list)

    def feed(self, frame: bytes, count: int = 1) -> None:
        # Replay faster than real time, but drain each short batch rather than
        # deliberately overflowing the bounded live-capture queue.
        for offset in range(0, count, 20):
            self.engine.add_pcm(frame * min(20, count - offset))
            assert self.engine.wait_until_drained()


@pytest.fixture
def make() -> Iterator[Callable[..., Harness]]:
    engines: list[AutomaticTranscriber] = []

    def factory(texts: list[str], *, decisions: list[bool] | None = None,
                interrupt: Callable[[], bool] | None = None,
                max_utterance: float = 90.0, question_only: bool = True) -> Harness:
        transcripts: list[Transcript] = []
        partials: list[str] = []
        errors: list[BaseException] = []
        states: list[RuntimeState] = []
        transcriber = Transcriber(texts)
        turn = Turn(decisions)
        engine = AutomaticTranscriber(
            transcriber, Speech(), turn, transcript_callback=transcripts.append,
            partial_callback=partials.append, error_callback=errors.append,
            state_callback=states.append, interrupt_callback=interrupt,
            max_utterance_seconds=max_utterance, question_only=question_only,
        )
        engines.append(engine)
        return Harness(engine, transcriber, turn, transcripts, partials, errors, states)

    yield factory
    for engine in engines:
        engine.close()


def test_silence_does_not_open_asr_stream(make: Callable[..., Harness]) -> None:
    h = make([])
    h.feed(SILENCE, 300)
    assert h.transcriber.sessions == []
    assert h.transcripts == []
    assert len(h.engine._preroll) == int(0.2 * 64_000)


def test_endpoint_waits_for_semantics_and_keeps_same_stream(make: Callable[..., Harness]) -> None:
    h = make(['Describe your design and the failure modes.'], decisions=[False, True])
    h.feed(SPEECH, 10)
    h.feed(SILENCE, 21)
    assert h.transcripts == []
    assert h.engine.state is RuntimeState.RECORDING
    h.transcriber.sessions[0].partial('Describe your design. Also explain the failure modes.')
    h.feed(SPEECH, 10)
    h.feed(SILENCE, 21)
    assert len(h.transcriber.sessions) == 1
    assert len(h.transcripts) == 1
    assert h.transcripts[0].transcription_seconds == 0.012
    assert len(h.turn.windows) == 2
    assert RuntimeState.TRANSCRIBING in h.states
    assert h.engine.state is RuntimeState.IDLE


def test_max_silence_finishes_incomplete_detector(make: Callable[..., Harness]) -> None:
    h = make(['How would you isolate the fault?'], decisions=[False])
    h.feed(SPEECH)
    h.feed(SILENCE, 93)
    assert not h.transcripts
    h.feed(SILENCE)
    assert len(h.transcripts) == 1


def test_audio_window_is_bounded_but_full_question_text_survives(make: Callable[..., Harness]) -> None:
    text = 'Describe the initial design. ' + 'Then explain the next constraint. ' * 40
    h = make([text])
    h.feed(SPEECH, 400)
    h.feed(SILENCE, 21)
    assert len(h.turn.windows[0]) == 8 * 64_000
    assert h.transcripts[0].text == text.strip()


def test_ack_filtered_but_explicit_answer_now_bypasses_filter(make: Callable[..., Harness]) -> None:
    h = make(['Okay, thank you.', 'The backup route.'])
    h.feed(SPEECH)
    h.feed(SILENCE, 21)
    assert not h.transcripts
    h.feed(SPEECH)
    assert h.engine.press()
    assert h.engine.wait_until_drained()
    assert [t.text for t in h.transcripts] == ['The backup route.']
    assert h.engine.release() is False


def test_short_continuation_cancels_once_and_combines(make: Callable[..., Harness]) -> None:
    calls: list[bool] = []

    def interrupt() -> bool:
        calls.append(True)
        return True

    h = make(['Describe the design.', 'Specifically the routing.'], interrupt=interrupt)
    h.feed(SPEECH)
    h.feed(SILENCE, 21)
    h.feed(SPEECH, 10)
    assert calls == [True]
    assert h.partials[-1] == 'Describe the design. Specifically the routing.'
    h.feed(SILENCE, 21)
    assert h.transcripts[-1].text == 'Describe the design. Specifically the routing.'
    assert calls == [True]


def test_ack_does_not_cancel_pending_answer(make: Callable[..., Harness]) -> None:
    calls: list[bool] = []
    h = make(['Explain failover.', 'Thank you.'], interrupt=lambda: bool(calls.append(True)))
    h.feed(SPEECH)
    h.feed(SILENCE, 21)
    h.feed(SPEECH, 10)
    h.feed(SILENCE, 21)
    assert calls == []
    assert len(h.transcripts) == 1


def test_finished_answer_and_late_followup_do_not_merge(make: Callable[..., Harness]) -> None:
    calls: list[bool] = []
    h = make(['Explain failover.', 'What about DNS?', 'And recovery?'],
             interrupt=lambda: bool(calls.append(True)))
    for index in range(3):
        if index == 2:
            h.feed(SILENCE, 64)
        h.feed(SPEECH)
        h.feed(SILENCE, 21)
    assert calls == [True]
    assert [t.text for t in h.transcripts] == ['Explain failover.', 'What about DNS?', 'And recovery?']


def test_cancel_discards_and_interrupts(make: Callable[..., Harness]) -> None:
    calls: list[bool] = []
    h = make(['Explain the topology.'], interrupt=lambda: bool(calls.append(True)))
    h.feed(SPEECH)
    assert h.engine.cancel()
    assert h.engine.wait_until_drained()
    h.feed(SILENCE, 20)
    assert not h.transcripts
    assert calls == [True]
    assert h.transcriber.sessions[0].closed
    assert not h.transcriber.sessions[0].finished


def test_close_does_not_finalize(make: Callable[..., Harness]) -> None:
    h = make(['Explain the topology.'])
    h.feed(SPEECH)
    h.engine.close()
    assert not h.transcripts
    assert h.engine.state is RuntimeState.CLOSED
    assert h.transcriber.sessions[0].closed
    assert not h.transcriber.sessions[0].finished
    assert not h.engine.press()


def test_abort_discards_and_recovers(make: Callable[..., Harness]) -> None:
    h = make(['Explain the old source.', 'Explain the new source.'])
    h.feed(SPEECH)
    error = RuntimeError('source changed')
    h.engine.abort(error)
    assert h.engine.wait_until_drained()
    h.feed(SPEECH)
    h.feed(SILENCE, 21)
    assert h.errors == [error]
    assert [t.text for t in h.transcripts] == ['Explain the new source.']


def test_queue_overflow_is_explicit_and_recovers(make: Callable[..., Harness]) -> None:
    h = make(['Explain recovery.'])
    h.engine.add_pcm(SPEECH * 126)
    assert h.engine.wait_until_drained()
    assert len(h.errors) == 1
    assert 'overflow' in str(h.errors[0])
    assert not h.transcriber.sessions
    h.feed(SPEECH)
    h.feed(SILENCE, 21)
    assert len(h.transcripts) == 1


def test_duration_limit_discards_instead_of_answering_cut_off_question(make: Callable[..., Harness]) -> None:
    h = make(['Explain a very long question.'], max_utterance=3.1)
    h.feed(SPEECH, 98)
    assert not h.transcripts
    assert len(h.errors) == 1
    assert 'time limit' in str(h.errors[0])
    assert h.engine.state is RuntimeState.IDLE


def test_capture_reader_does_not_wait_for_detector_and_abort_suppresses_result() -> None:
    entered, unblock = threading.Event(), threading.Event()

    class SlowSpeech(Speech):
        def is_speech(self, pcm: bytes) -> bool:
            entered.set()
            assert unblock.wait(3)
            return super().is_speech(pcm)

    transcripts: list[Transcript] = []
    engine = AutomaticTranscriber(Transcriber(['Explain routing.']), SlowSpeech(), Turn(),
                                  transcript_callback=transcripts.append)
    try:
        engine.add_pcm(SPEECH + SILENCE * 21)
        assert entered.wait(2)
        engine.abort(RuntimeError('source changed'))
        # This returns while the worker is still blocked in detector inference.
        engine.add_pcm(SILENCE)
        unblock.set()
        assert engine.wait_until_drained()
        assert transcripts == []
        assert engine.state is RuntimeState.IDLE
    finally:
        unblock.set()
        engine.close()


@pytest.mark.parametrize('text', [
    'Describe your last project.', 'Walk me through a packet drop.',
    'You mentioned an outage. What happened next?', 'And the rollback?',
    'Specifically routing.', 'Any caveats?', 'Why?',
    'Explain association versus propagation.', 'How would you check the route?',
])
def test_interview_intent_accepts_requests_and_followups(text: str) -> None:
    assert is_interview_prompt(text)


@pytest.mark.parametrize('text', [
    '', 'Okay.', 'Thank you.', 'Okay, thanks.', 'That makes sense.',
    'Nice to meet you.', "You're welcome!", 'The weather is nice today.',
    'How are you?', 'Can you hear me?', 'Thank you for explaining that.',
])
def test_interview_intent_filters_acknowledgments_and_small_talk(text: str) -> None:
    assert not is_interview_prompt(text)


def test_asr_line_completion_does_not_end_interview_turn(make: Callable[..., Harness]) -> None:
    h = make(['Describe the route. Then explain the return path.'], decisions=[False, True])
    h.feed(SPEECH)
    session = h.transcriber.sessions[0]
    session.complete(Transcript('Describe the route.', 0.01))
    h.feed(SILENCE, 21)
    assert h.transcripts == []
    assert not session.closed
    h.feed(SPEECH)
    h.feed(SILENCE, 21)
    assert [t.text for t in h.transcripts] == ['Describe the route. Then explain the return path.']


def test_default_threshold_keeps_half_second_multipart_pause(make: Callable[..., Harness]) -> None:
    h = make(['Describe your design. Explain the rollback.'])
    assert h.engine.minimum_silence_seconds == 0.65
    h.feed(SPEECH)
    h.feed(SILENCE, 17)  # 544ms inside one multipart question.
    assert not h.turn.windows
    assert not h.transcripts
    h.feed(SPEECH)
    h.feed(SILENCE, 21)
    assert len(h.transcriber.sessions) == 1
    assert [t.text for t in h.transcripts] == ['Describe your design. Explain the rollback.']


def test_setup_is_kept_for_adjacent_question(make: Callable[..., Harness]) -> None:
    h = make(['The two offices share an address range.', 'Describe a safe transition.'])
    h.feed(SPEECH)
    h.feed(SILENCE, 21)
    assert not h.transcripts
    h.feed(SPEECH)
    assert h.partials[-1] == 'The two offices share an address range. Describe a safe transition.'
    h.feed(SILENCE, 21)
    assert [t.text for t in h.transcripts] == [
        'The two offices share an address range. Describe a safe transition.',
    ]


def test_multiple_setup_statements_do_not_bypass_question_gate(make: Callable[..., Harness]) -> None:
    h = make(['The first link is down.', 'The backup route is missing.', 'Explain your next steps.'])
    for _ in range(2):
        h.feed(SPEECH)
        h.feed(SILENCE, 21)
        assert not h.transcripts
    h.feed(SPEECH)
    h.feed(SILENCE, 21)
    assert h.transcripts[0].text == (
        'The first link is down. The backup route is missing. Explain your next steps.'
    )


def test_setup_expiry_is_based_on_next_speech_start(make: Callable[..., Harness]) -> None:
    h = make(['The firewall is down.', 'Explain the recovery process.'])
    h.feed(SPEECH)
    h.feed(SILENCE, 21)
    h.feed(SILENCE, 80)  # 2.56s after endpoint: setup still eligible.
    h.feed(SPEECH, 120)  # The next question lasts beyond the setup expiry.
    h.feed(SILENCE, 21)
    assert h.transcripts[0].text == 'The firewall is down. Explain the recovery process.'


def test_expired_setup_is_not_attached_to_later_question(make: Callable[..., Harness]) -> None:
    h = make(['The firewall is down.', 'Explain DNS caching.'])
    h.feed(SPEECH)
    h.feed(SILENCE, 21)
    h.feed(SILENCE, 94)
    h.feed(SPEECH)
    h.feed(SILENCE, 21)
    assert h.transcripts[0].text == 'Explain DNS caching.'


def test_acknowledgment_not_attached_and_does_not_extend_setup_expiry(make: Callable[..., Harness]) -> None:
    h = make(['The firewall is down.', 'Okay, thanks.', 'Explain DNS caching.'])
    h.feed(SPEECH)
    h.feed(SILENCE, 21)
    h.feed(SILENCE, 70)
    h.feed(SPEECH)
    h.feed(SILENCE, 21)
    assert not h.transcripts
    h.feed(SILENCE, 4)  # More than3s since setup, less than3s since acknowledgment.
    h.feed(SPEECH)
    h.feed(SILENCE, 21)
    assert h.transcripts[0].text == 'Explain DNS caching.'


def test_short_ack_preserves_setup_without_becoming_part_of_question(make: Callable[..., Harness]) -> None:
    h = make(['The firewall is down.', 'Okay.', 'Explain recovery.'])
    for _ in range(3):
        h.feed(SPEECH)
        h.feed(SILENCE, 21)
    assert [t.text for t in h.transcripts] == ['The firewall is down. Explain recovery.']


@pytest.mark.parametrize('reset', ['abort', 'cancel', 'close'])
def test_resets_clear_pending_setup(make: Callable[..., Harness], reset: str) -> None:
    h = make(['The firewall is down.', 'Explain DNS caching.'])
    h.feed(SPEECH)
    h.feed(SILENCE, 21)
    assert h.engine._lead_in
    if reset == 'abort':
        h.engine.abort(RuntimeError('capture reconnected'))
    elif reset == 'cancel':
        h.engine.cancel()
    else:
        h.engine.close()
    if reset != 'close':
        assert h.engine.wait_until_drained()
    assert h.engine._lead_in == ''
    assert h.engine._active_lead_in == ''
    if reset != 'close':
        h.feed(SPEECH)
        h.feed(SILENCE, 21)
        assert h.transcripts[0].text == 'Explain DNS caching.'


def test_setup_character_limit_resets_without_truncating_facts(make: Callable[..., Harness]) -> None:
    h = make(['The first site is down. ' * 100, 'The second site is down. ' * 100,
              'Explain recovery.'])
    for _ in range(2):
        h.feed(SPEECH)
        h.feed(SILENCE, 21)
    assert not h.transcripts
    assert len(h.errors) == 1
    assert '4000 characters' in str(h.errors[0])
    assert h.engine._lead_in == ''
    h.feed(SPEECH)
    h.feed(SILENCE, 21)
    assert h.transcripts[0].text == 'Explain recovery.'


def test_candidate_speech_accepts_statements_ack_and_questions_without_merging(make: Callable[..., Harness]) -> None:
    calls: list[bool] = []
    texts = ['We retained the ASA.', 'Yes.', 'No.', 'Why would we change that?']
    h = make(texts, question_only=False, interrupt=lambda: bool(calls.append(True)))
    for _ in texts:
        h.feed(SPEECH)
        h.feed(SILENCE, 21)
    assert [t.text for t in h.transcripts] == texts
    assert calls == []
    assert h.engine._lead_in == ''
    h.engine.cancel()
    assert h.engine.wait_until_drained()
    assert calls == []


def test_finish_pending_delivers_queued_tail_before_return(make: Callable[..., Harness]) -> None:
    h = make(['We retained the ASA.'], question_only=False)
    tail = SPEECH[:40]
    h.engine.add_pcm(SPEECH + tail)
    assert h.engine.finish_pending()
    assert [t.text for t in h.transcripts] == ['We retained the ASA.']
    assert b''.join(h.transcriber.sessions[0].audio) == SPEECH + tail


def test_finish_pending_does_not_report_asr_failure_as_success(make: Callable[..., Harness]) -> None:
    h = make(['My answer.'], question_only=False)
    h.feed(SPEECH)

    def fail() -> None:
        raise RuntimeError('ASR failed')

    h.transcriber.sessions[0].finish = fail  # type: ignore[method-assign]
    assert not h.engine.finish_pending()
    assert not h.transcripts
    assert str(h.errors[0]) == 'ASR failed'


def test_finish_pending_times_out_and_closed_engine_refuses(make: Callable[..., Harness]) -> None:
    h = make(['My answer.'], question_only=False)
    h.feed(SPEECH)
    unblock = threading.Event()
    original = h.transcriber.sessions[0].finish

    def slow() -> None:
        assert unblock.wait(3)
        original()

    h.transcriber.sessions[0].finish = slow  # type: ignore[method-assign]
    try:
        assert not h.engine.finish_pending(timeout=0.01)
        assert isinstance(h.errors[0], TimeoutError)
    finally:
        unblock.set()
        h.engine.close()
    assert not h.engine.finish_pending()


def test_pending_finish_reports_audio_error_before_finish_command() -> None:
    entered, unblock = threading.Event(), threading.Event()
    errors: list[BaseException] = []

    class FailedSpeech(Speech):
        def is_speech(self, pcm: bytes) -> bool:
            entered.set()
            assert unblock.wait(3)
            raise RuntimeError('detector failed')

    engine = AutomaticTranscriber(Transcriber([]), FailedSpeech(), Turn(),
                                  transcript_callback=lambda _t: None,
                                  error_callback=errors.append, question_only=False)
    result: list[bool] = []
    try:
        engine.add_pcm(SPEECH)
        assert entered.wait(2)
        waiter = threading.Thread(target=lambda: result.append(engine.finish_pending()))
        waiter.start()
        # Wait for the finish command to enter the queue before failing audio.
        with engine._condition:
            assert engine._condition.wait_for(lambda: bool(engine._queue), timeout=2)
        unblock.set()
        waiter.join(2)
        assert result == [False]
        assert str(errors[0]) == 'detector failed'
    finally:
        unblock.set()
        engine.close()


def test_candidate_statuses_never_describe_question_intent(make: Callable[..., Harness]) -> None:
    h = make(['Yes.'], question_only=False)
    statuses: list[str] = []
    h.engine.status_callback = statuses.append
    h.feed(SPEECH)
    h.feed(SILENCE, 21)
    assert statuses
    assert all('question' not in status.lower() for status in statuses)
