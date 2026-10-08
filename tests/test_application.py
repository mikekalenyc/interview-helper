from __future__ import annotations

import threading
from dataclasses import replace
from pathlib import Path

import pytest

import interview_helper.application as application_module
from interview_helper.application import (
    ApplicationConfig,
    CaptureMode,
    ControlMode,
    InterviewApplication,
)
from interview_helper.answer import ANSWER_TOKEN_LIMIT, AnswerConversation
from interview_helper.capture import PulseMicrophoneResolver
from interview_helper.context import ContextError
from interview_helper.input import ManualHoldControl


def config() -> ApplicationConfig:
    return ApplicationConfig(
        input_device=Path("/dev/input/example"),
        event_code="KEY_M",
        model_path=Path("/models/moonshine"),
    )


class BlockingDaemon:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.closed = threading.Event()
        self.close_calls = 0

    def run(self) -> None:
        self.started.set()
        self.closed.wait(timeout=2)

    def close(self) -> None:
        self.close_calls += 1
        self.closed.set()


class FakeApplication(InterviewApplication):
    def __init__(self, daemon: BlockingDaemon) -> None:
        super().__init__(config())
        self.test_daemon = daemon

    def _build_daemon(self) -> BlockingDaemon:  # type: ignore[override]
        return self.test_daemon


def test_config_rejects_context_without_resume() -> None:
    with pytest.raises(ContextError, match="require a resume"):
        ApplicationConfig(
            input_device=Path("/dev/input/example"),
            event_code="KEY_M",
            model_path=Path("/models/moonshine"),
            context=(Path("projects.md"),),
        )


def test_manual_control_does_not_require_evdev_configuration() -> None:
    configured = ApplicationConfig(
        input_device=None,
        event_code="",
        model_path=Path("/models/moonshine"),
        control_mode=ControlMode.MANUAL,
    )

    assert configured.input_device is None


def test_application_uses_window_owned_conversation() -> None:
    conversation = AnswerConversation()

    application = InterviewApplication(config(), conversation=conversation)

    assert application.conversation is conversation


def test_microphone_test_composes_microphone_capture_and_evdev_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        application_module, "MoonshineTranscriber", lambda _config: object()
    )
    control = object()
    monkeypatch.setattr(
        application_module,
        "EvdevHoldControl",
        lambda _device, _event: control,
    )
    configured = ApplicationConfig(
        input_device=Path("/dev/input/by-id/test-event-kbd"),
        event_code="KEY_M",
        model_path=Path("/models/moonshine"),
        capture_mode=CaptureMode.DEFAULT_MICROPHONE,
        control_mode=ControlMode.EVDEV,
        microphone_source="alsa_input.dji",
    )

    daemon = InterviewApplication(configured)._build_daemon()

    assert isinstance(daemon.capture.resolver, PulseMicrophoneResolver)  # type: ignore[attr-defined]
    assert daemon.capture.resolver.source == "alsa_input.dji"  # type: ignore[attr-defined]
    assert daemon.control is control


def test_stop_unblocks_background_run_and_cleanup_is_idempotent() -> None:
    daemon = BlockingDaemon()
    application = FakeApplication(daemon)
    worker = threading.Thread(target=application.run)
    worker.start()
    assert daemon.started.wait(timeout=1)
    assert application.running

    application.stop()
    worker.join(timeout=1)

    assert not worker.is_alive()
    assert not application.running
    assert daemon.close_calls >= 1


def test_failed_daemon_start_is_still_closed() -> None:
    class FailingDaemon(BlockingDaemon):
        def run(self) -> None:
            raise RuntimeError("capture failed")

    daemon = FailingDaemon()
    application = FakeApplication(daemon)

    with pytest.raises(RuntimeError, match="capture failed"):
        application.run()

    assert daemon.closed.is_set()
    assert not application.running


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("token_override", [None, 240])
@pytest.mark.parametrize("cloud", [False, True])
def test_library_wiring_and_owned_cleanup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, enabled: bool,
    token_override: int | None, cloud: bool,
) -> None:
    from dataclasses import replace
    from interview_helper.application import ApplicationCallbacks, AnswerProvider

    resume = tmp_path / "resume.md"
    resume.write_text("Network engineer")
    closed: list[str] = []
    captured: dict[str, object] = {}

    class Resource:
        def __init__(self, name: str) -> None:
            self.name = name

        def close(self) -> None:
            closed.append(self.name)

    library = Resource("library")

    class LibraryFactory:
        @staticmethod
        def open(root: Path) -> Resource:
            assert root == tmp_path
            return library
    monkeypatch.setattr(application_module, "TechnicalLibrary", LibraryFactory)
    def local_client(**_kwargs: object) -> Resource:
        assert not cloud
        return Resource("qwen")

    def cloud_client(**kwargs: object) -> Resource:
        assert cloud
        assert kwargs == {"api_key_file": tmp_path / "key.rtf", "model": "gpt-6-luna"}
        return Resource("openai")

    monkeypatch.setattr(application_module, "LocalQwenClient", local_client)
    monkeypatch.setattr(application_module, "OpenAIAnswerClient", cloud_client)
    monkeypatch.setattr(application_module, "MoonshineTranscriber", lambda _config: object())
    monkeypatch.setattr(application_module, "EvdevHoldControl", lambda *_args: object())

    def pipeline(*_args: object, **kwargs: object) -> Resource:
        captured.update(kwargs)
        return Resource("pipeline")

    monkeypatch.setattr(application_module, "GroundedAnswerPipeline", pipeline)
    evidence: list[str] = []
    statuses: list[str] = []
    callbacks = ApplicationCallbacks(
        evidence=evidence.append, status=statuses.append, answer_progress=statuses.append,
    )
    configured = replace(config(), resume=resume, technical_library=tmp_path if enabled else None)
    configured = replace(
        configured, answer_provider=AnswerProvider.OPENAI if cloud else AnswerProvider.LOCAL_QWEN,
        openai_api_key_file=tmp_path / "key.rtf", openai_model="gpt-6-luna",
    )
    if token_override is not None:
        configured = replace(configured, answer_max_tokens=token_override)
    app = InterviewApplication(configured, callbacks)
    app._build_daemon()
    assert captured["max_tokens"] == (
        ANSWER_TOKEN_LIMIT if token_override is None else token_override
    )
    assert captured["technical_library"] is (library if enabled else None)
    assert captured["evidence_callback"] == callbacks.evidence
    assert captured["status_callback"] == callbacks.status
    assert captured["progress_callback"] == callbacks.answer_progress
    app._close_resources()
    app._close_resources()
    assert closed == ["pipeline", *(["library"] if enabled else []), "openai" if cloud else "qwen"]


def test_automatic_mode_needs_no_event_device_and_composes_detectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(application_module, "MoonshineTranscriber", lambda _config: object())
    monkeypatch.setattr(application_module, "SileroSpeechDetector", lambda: "vad")
    monkeypatch.setattr(application_module, "SmartTurnDetector", lambda: "turn")
    monkeypatch.setattr(application_module, "EvdevHoldControl", lambda *_: pytest.fail("No evdev in automatic mode"))
    captured = {}

    class Automatic:
        def __init__(self, transcriber, **kwargs):
            captured.update(kwargs)
        def close(self):
            captured["closed"] = True
        def press(self):
            captured["answer_now"] = True
            return True
        def cancel(self):
            captured["cancelled"] = True
            return True

    monkeypatch.setattr(application_module, "AutomaticTranscriber", Automatic)
    configured = ApplicationConfig(input_device=None,event_code="",model_path=Path('/model'),automatic_listening=True)
    app = InterviewApplication(configured)
    daemon = app._build_daemon()
    assert captured['speech_detector'] == 'vad'
    assert captured['turn_detector'] == 'turn'
    assert daemon.abort_on_capture_loss
    assert 'hands-free' in daemon.control_description
    assert app.answer_now()
    app.cancel_answer()
    assert captured['answer_now'] and captured['cancelled']
    app._close_resources()
    assert captured['closed']


def test_unverified_openai_model_is_rejected() -> None:
    from dataclasses import replace
    with pytest.raises(ValueError, match="Unsupported OpenAI model"):
        replace(config(), openai_model="gpt-6-astra")


@pytest.mark.parametrize("changes", [
    {"transcription_backend": "unknown"},
    {"transcription_device": "auto"},
    {"detection_device": "metal"},
    {"transcription_device": "cuda"},
    {"gpu_device_index": -1},
    {"gpu_device_index": True},
    {"gpu_device_index": 0.5},
    {"moonshine_architecture": "large"},
])
def test_unsupported_compute_choices_fail_before_runtime(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        replace(config(), **changes)


def test_moonshine_medium_and_gpu_detection_are_independent(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = []
    monkeypatch.setattr(application_module, "MoonshineTranscriber",
                        lambda configuration: captured.append(configuration) or object())
    detector_calls = []
    monkeypatch.setattr(application_module, "SileroSpeechDetector",
                        lambda **kwargs: detector_calls.append(kwargs) or "vad")
    monkeypatch.setattr(application_module, "SmartTurnDetector",
                        lambda **kwargs: detector_calls.append(kwargs) or "turn")
    app = InterviewApplication(replace(config(), moonshine_architecture="medium",
                                       detection_device="cuda", gpu_device_index=2))
    first = app._create_transcriber()
    second = app._create_transcriber()
    assert first is not second
    assert captured[0].architecture == "medium"
    assert captured[0].model_path == config().model_path
    assert app._create_detectors() == ("vad", "turn")
    assert detector_calls == [{"device": "cuda", "device_index": 2}] * 2


def test_nemotron_shares_model_and_closes_after_stream_engines(monkeypatch: pytest.MonkeyPatch) -> None:
    import interview_helper.nemotron as native
    calls = []
    closed = []

    class Native:
        def __init__(self, path: Path, **kwargs: object) -> None:
            calls.append((path, kwargs))

        def close(self) -> None:
            closed.append("model")

    class Engine:
        def close(self) -> None:
            closed.append("engine")

    monkeypatch.setattr(native, "NemotronTranscriber", Native)
    configured = replace(config(), transcription_backend="nemotron", transcription_device="cuda",
                         gpu_device_index=1, nemotron_library=Path("/runtime/lib.so"))
    app = InterviewApplication(configured)
    assert app._create_transcriber() is app._create_transcriber()
    assert calls == [(config().model_path, {"library_path": Path("/runtime/lib.so"),
                                          "device": "cuda", "device_index": 1, "keyterms": ()})]
    app._automatic_engine = Engine()
    app._candidate_engine = Engine()
    app._close_resources()
    app._close_resources()
    assert closed == ["engine", "engine", "model"]


def test_transcriber_is_released_when_detector_startup_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    closed = []

    class Model:
        def close(self) -> None:
            closed.append("model")

    def fail() -> None:
        raise RuntimeError("detector missing")

    monkeypatch.setattr(application_module, "MoonshineTranscriber", lambda _: Model())
    monkeypatch.setattr(application_module, "SileroSpeechDetector", fail)
    app = InterviewApplication(replace(config(), automatic_listening=True))
    with pytest.raises(RuntimeError, match="detector missing"):
        app.run()
    assert closed == ["model"]
    assert not app.running
