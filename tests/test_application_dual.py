"""Dual source application routing without devices or models."""
from pathlib import Path
from typing import Any

import pytest

import interview_helper.application as module
from interview_helper.application import ApplicationCallbacks, ApplicationConfig, CaptureMode, InterviewApplication
from interview_helper.core import RuntimeState, Transcript
from interview_helper.duplex import DuplexDaemon
from interview_helper.spoken import SpokenConversation


def test_dual_config_requires_automatic_headphones() -> None:
    for automatic, mode in [(False, CaptureMode.HEADPHONE_MONITOR),
                            (True, CaptureMode.DEFAULT_MICROPHONE)]:
        with pytest.raises(ValueError, match="requires hands-free headphone"):
            ApplicationConfig(input_device=None, event_code="", model_path=Path('/model'),
                              listen_to_candidate=True, automatic_listening=automatic,
                              capture_mode=mode)


def test_source_isolation_flush_order_correction_and_cleanup(monkeypatch: pytest.MonkeyPatch,
                                                           tmp_path: Path) -> None:
    engines: list[Any] = []
    pipeline_calls: list[tuple[str, tuple[str, ...]]] = []
    speech = SpokenConversation()
    errors: list[BaseException] = []
    received: list[str] = []
    ready: list[bool] = []
    candidate_states: list[RuntimeState] = []

    class Engine:
        state = RuntimeState.IDLE
        flush_ok = True
        pending = ""
        closed = False
        def __init__(self, transcriber: object, **kwargs: Any) -> None:
            self.transcriber = transcriber
            self.options = kwargs
            engines.append(self)
        def finish_pending(self, timeout: float) -> bool:
            assert timeout == 5.0
            if self.pending:
                self.emit(self.pending)
                self.pending = ""
            return self.flush_ok
        def emit(self, text: str) -> None:
            self.options['transcript_callback'](Transcript(text, 0))
        def close(self) -> None:
            self.closed = True

    class Pipeline:
        cancels = 0
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            assert kwargs['spoken_conversation'] is speech
        def submit(self, transcript: Transcript) -> None:
            pipeline_calls.append((transcript.text, tuple(t.text for t in speech.snapshot())))
        def cancel_pending(self) -> bool:
            self.cancels += 1
            return True
        def close(self) -> None:
            pass

    class Client:
        def close(self) -> None:
            pass

    monkeypatch.setattr(module, 'MoonshineTranscriber', lambda *_: object())
    monkeypatch.setattr(module, 'SileroSpeechDetector', object)
    monkeypatch.setattr(module, 'SmartTurnDetector', object)
    monkeypatch.setattr(module, 'AutomaticTranscriber', Engine)
    monkeypatch.setattr(module, 'GroundedAnswerPipeline', Pipeline)
    monkeypatch.setattr(module, 'LocalQwenClient', lambda **_: Client())
    resume = tmp_path / 'resume.md'
    resume.write_text('Network engineer')
    config = ApplicationConfig(input_device=None, event_code='', model_path=Path('/model'),
                               resume=resume, automatic_listening=True, listen_to_candidate=True)
    app = InterviewApplication(config, ApplicationCallbacks(error=errors.append,
        candidate_transcript=lambda turn: received.append(turn.text), capture_ready=ready.append,
        candidate_state=candidate_states.append),
        spoken_conversation=speech)
    pair = app._build_daemon()
    assert isinstance(pair, DuplexDaemon)
    assert len(engines) == 2 and engines[0].transcriber is not engines[1].transcriber
    assert engines[1].options['question_only'] is False
    assert pair.candidate.capture.resolver.source is None  # type: ignore[attr-defined]
    app._source_capture_ready('interviewer', True)
    assert not ready[-1]
    app._source_capture_ready('candidate', True)
    assert ready[-1]
    assert candidate_states[-1] is RuntimeState.IDLE
    engines[0].emit('What did you do?')
    engines[1].state = RuntimeState.TRANSCRIBING
    assert not app._interrupt_continuation()
    engines[1].state = RuntimeState.IDLE
    assert app._answer_pipeline is not None
    cancels = app._answer_pipeline.cancels  # type: ignore[attr-defined]
    assert not app.correct_candidate_speech(123, 'stale')
    assert app._answer_pipeline.cancels == cancels  # type: ignore[attr-defined]
    engines[1].emit('I kept the old firewall.')
    assert len(pipeline_calls) == 1
    assert received == ['I kept the old firewall.']
    assert speech.snapshot()[0].question == 'What did you do?'
    assert not app._interrupt_continuation()
    engines[1].pending = 'Two owners were unknown; I did not remove their tunnels.'
    engines[0].emit('Why did you choose that?')
    assert pipeline_calls[-1][1][-1] == 'Two owners were unknown; I did not remove their tunnels.'
    assert len(pipeline_calls) == 2
    assert speech.snapshot()[-1].question == 'What did you do?'
    turn = speech.snapshot()[-1]
    assert app.correct_candidate_speech(turn.id, 'Three owners were unknown.')
    assert speech.snapshot()[-1].text == 'Three owners were unknown.'
    assert len(pipeline_calls) == 2
    engines[1].flush_ok = False
    engines[0].emit('How did you check?')
    assert errors and len(pipeline_calls) == 2
    app._source_capture_ready('candidate', False)
    assert candidate_states[-1] is RuntimeState.ERROR
    engines[0].emit('What next?')
    assert not ready[-1] and len(pipeline_calls) == 2
    app.stop()
    engines[1].emit('Must not enter stopped session')
    assert speech.snapshot()[-1].text == 'Three owners were unknown.'
    app._close_resources()
    assert all(engine.closed for engine in engines)
