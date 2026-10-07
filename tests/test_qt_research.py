"""Offscreen source display lifecycle; no hardware, HTTP, or models."""
from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication

import interview_helper.qt_gui as gui
from interview_helper.answer import ANSWER_TOKEN_LIMIT
from interview_helper.history import InterviewHistoryStore
from interview_helper.presentation import RunFinished, StartRequested, StopRequested, reduce_presentation


@pytest.fixture(scope="module")
def qt_app() -> Iterator[QApplication]:
    app = QApplication.instance() or QApplication([])
    yield app
    app.processEvents()


@pytest.fixture(autouse=True)
def dispose_test_windows(qt_app: QApplication) -> Iterator[None]:
    yield
    # close() hides widgets; explicitly destroy them while QApplication is alive.
    for widget in QApplication.topLevelWidgets():
        widget.close()
        widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    qt_app.processEvents()


def test_sources_reject_stale_callbacks_and_clear_on_new_start(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, qt_app: QApplication,
) -> None:
    monkeypatch.setattr(gui, "default_hotkey_device", lambda: Path("/dev/input/test"))
    monkeypatch.setattr(gui, "list_microphone_sources", lambda: ())
    monkeypatch.setattr(gui, "InterviewHistoryStore", lambda: InterviewHistoryStore(tmp_path))
    window = gui.MainWindow()
    window.technical_library.setText(str(tmp_path))
    assert window._config().technical_library == tmp_path
    window.technical_library.clear()
    assert window._config().technical_library is None
    window._snapshot = reduce_presentation(window._snapshot, StartRequested())
    run_id = window._snapshot.run_id
    window._bridge.evidence.emit(run_id, "Guide section\nlocal-library/aws.md")
    assert window.sources.toPlainText().startswith("Guide section")
    window._bridge.evidence.emit(run_id - 1, "stale")
    assert "stale" not in window.sources.toPlainText()
    window._snapshot = reduce_presentation(window._snapshot, StopRequested())
    window._bridge.evidence.emit(run_id, "late")
    assert "late" not in window.sources.toPlainText()
    window._snapshot = reduce_presentation(window._snapshot, RunFinished(run_id))

    class IdleApplication:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def run(self) -> None:
            pass

    monkeypatch.setattr(gui, "InterviewApplication", IdleApplication)
    window._start()
    assert window.sources.toPlainText() == ""
    assert window._worker is not None
    window._worker.join(timeout=1)
    qt_app.processEvents()
    window.close()


def test_startup_contexts_reach_runtime_configuration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, qt_app: QApplication,
) -> None:
    monkeypatch.setattr(gui, 'default_hotkey_device', lambda: Path('/dev/input/test'))
    monkeypatch.setattr(gui, 'list_microphone_sources', lambda: ())
    monkeypatch.setattr(gui, 'InterviewHistoryStore', lambda: InterviewHistoryStore(tmp_path))
    resume = tmp_path / 'resume.txt'
    background = tmp_path / 'resume background.txt'
    window = gui.MainWindow(resume_path=resume, context_paths=(background, background))
    assert window.context_paths.count() == 1
    assert window._config().resume == resume
    assert window._config().context == (background,)
    window.close()


def test_openai_startup_selection_and_local_switch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, qt_app: QApplication,
) -> None:
    from interview_helper.application import AnswerProvider
    monkeypatch.setattr(gui, "default_hotkey_device", lambda: Path("/dev/input/test"))
    monkeypatch.setattr(gui, "list_microphone_sources", lambda: ())
    monkeypatch.setattr(gui, "InterviewHistoryStore", lambda: InterviewHistoryStore(tmp_path))
    key_path = tmp_path / "key.rtf"
    window = gui.MainWindow(answer_provider="openai", openai_api_key_file=key_path)
    cfg = window._config()
    assert cfg.answer_provider is AnswerProvider.OPENAI
    assert cfg.openai_api_key_file == key_path
    assert not window.qwen_url.isEnabled()
    assert window.openai_key_file.isEnabled()
    assert "OpenAI GPT-5.6 Luna" in window.provider_status.text()
    assert "sent to OpenAI" in window.provider_status.text()
    window.answer_provider.setCurrentIndex(0)
    assert window._config().answer_provider is AnswerProvider.LOCAL_QWEN
    assert window.qwen_url.isEnabled()
    assert not window.openai_key_file.isEnabled()
    assert "Local Qwen" in window.provider_status.text()
    window.close()
    qt_app.processEvents()


def test_dual_audio_controls_spoken_corrections_history_and_stale_events(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, qt_app: QApplication,
) -> None:
    from interview_helper.application import InterviewApplication
    monkeypatch.setattr(gui, 'default_hotkey_device', lambda: Path('/dev/input/test'))
    monkeypatch.setattr(gui, 'list_microphone_sources', lambda: ())
    monkeypatch.setattr(gui, 'InterviewHistoryStore', lambda: InterviewHistoryStore(tmp_path))
    window = gui.MainWindow()
    assert "always enabled" in window.spoken_mode_label.text()
    assert not hasattr(window, 'listen_to_candidate')
    assert not hasattr(window, 'hands_free')
    assert not hasattr(window, 'capture_mode')
    assert not window.microphone_selector.isHidden()
    config = window._config()
    assert config.listen_to_candidate and config.microphone_source is None
    assert config.automatic_listening and config.input_device is None
    assert config.capture_mode is gui.CaptureMode.HEADPHONE_MONITOR
    window.show()
    qt_app.processEvents()
    assert window.candidate_state_label.isVisible()
    assert "Start Interview" in window.partial_label.text()
    window.tabs.setCurrentIndex(1)
    qt_app.processEvents()
    assert window.microphone_selector.isVisible()
    assert not window.input_device.isVisible()
    assert not window.event_code.isVisible()
    assert not window.learn_hotkey_button.isVisible()
    window.tabs.setCurrentIndex(0)
    window._snapshot = reduce_presentation(window._snapshot, StartRequested())
    run_id = window._snapshot.run_id
    app = InterviewApplication(config, spoken_conversation=window._spoken_conversation)
    window._application = app
    turn = window._spoken_conversation.append('I kept the firewall.', 'What did you do?')
    window._bridge.spoken_changed.emit(run_id)
    assert window.spoken_detail.toPlainText() == 'I kept the firewall.'
    assert window._history_store.records()[0].spoken[0].answer == turn.text
    window._bridge.candidate_partial.emit(run_id - 1, 'stale')
    assert not window.spoken_partial.text()
    window._bridge.candidate_partial.emit(run_id, 'Current speech')
    assert 'Current speech' in window.spoken_partial.text()
    window._handle_event(RunFinished(run_id - 1))
    assert window._application is app
    monkeypatch.setattr(gui.QInputDialog, 'getMultiLineText', lambda *_: ('I only observed.', True))
    window._correct_spoken()
    assert window._spoken_conversation.snapshot()[0].text == 'I only observed.'
    assert window._history_store.records()[0].spoken[0].answer == 'I only observed.'

    def finish_during_modal(*_args: object) -> tuple[str, bool]:
        window._end_session_when_finished = True
        window._handle_event(RunFinished(run_id))
        return 'Must not enter a new session', True

    monkeypatch.setattr(gui.QInputDialog, 'getMultiLineText', finish_during_modal)
    window._correct_spoken()
    assert window._spoken_conversation.snapshot() == ()
    assert window.spoken_detail.toPlainText() == ''
    assert window._history_store.records()[0].spoken[0].answer == 'I only observed.'
    window.close()


def test_transparent_window_keeps_subtitle_answers_and_controls_available(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, qt_app: QApplication,
) -> None:
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtWidgets import QPushButton
    monkeypatch.setattr(gui, 'default_hotkey_device', lambda: Path('/dev/input/test'))
    monkeypatch.setattr(gui, 'list_microphone_sources', lambda: ())
    monkeypatch.setattr(gui, 'InterviewHistoryStore', lambda: InterviewHistoryStore(tmp_path))
    window = gui.MainWindow()
    for i in range(8):
        window._conversation.append(f'Question {i}', 'A long answer. ' * 40)
    window._render()
    window.show()
    qt_app.processEvents()
    assert window.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
    assert window.windowFlags() & Qt.WindowType.FramelessWindowHint
    assert window.windowFlags() & Qt.WindowType.WindowStaysOnTopHint
    assert window.windowOpacity() == 1.0
    assert window.phase_label.isVisible() and window.start_button.isVisible()
    assert window.show_questions.isChecked() and window.transcript.isVisible()
    window.show_questions.setChecked(False)
    qt_app.processEvents()
    assert not window.transcript.isVisible()
    frame = window.grab().toImage()
    assert frame.pixelColor(window.width() // 2, 300).alpha() == 0
    bar = window.findChild(gui.QGroupBox, 'answerBar')
    assert bar is not None
    point = bar.mapTo(window, QPoint(5, bar.height() // 2))
    assert frame.pixelColor(point).alpha() == 0
    cursor = gui.QTextCursor(window.answer.document())
    cursor.movePosition(gui.QTextCursor.MoveOperation.NextCharacter)
    assert cursor.charFormat().background().color().name() == '#000000'
    assert cursor.charFormat().foreground().color().name() == '#ffffff'
    assert window.answer.verticalScrollBar().value() == window.answer.verticalScrollBar().maximum()
    window.show_questions.setChecked(True)
    qt_app.processEvents()
    assert window.transcript.isVisible()
    assert window.resize_grip.isVisible()
    buttons = {b.accessibleName(): b for b in window.title_bar.findChildren(QPushButton)}
    buttons['Maximize / restore'].click()
    assert window.isMaximized()
    buttons['Maximize / restore'].click()
    assert not window.isMaximized()
    buttons['Close'].click()
    qt_app.processEvents()
    assert not window.isVisible()


def test_startup_error_is_visible_in_main_banner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, qt_app: QApplication,
) -> None:
    from interview_helper.presentation import ErrorReported

    monkeypatch.setattr(gui, "default_hotkey_device", lambda: Path("/dev/input/test"))
    monkeypatch.setattr(gui, "list_microphone_sources", lambda: ())
    monkeypatch.setattr(gui, "InterviewHistoryStore", lambda: InterviewHistoryStore(tmp_path))
    window = gui.MainWindow()
    window._snapshot = reduce_presentation(window._snapshot, StartRequested())
    run_id = window._snapshot.run_id
    message = "Moonshine model is missing: /missing/model"
    window._snapshot = reduce_presentation(window._snapshot, ErrorReported(run_id, message))
    window._snapshot = reduce_presentation(window._snapshot, RunFinished(run_id))
    window._render()
    assert message in window.phase_label.text()
    assert window.phase_label.wordWrap()
    window.close()
    qt_app.processEvents()


def test_answer_token_default_and_edit_reach_runtime_configuration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, qt_app: QApplication,
) -> None:
    monkeypatch.setattr(gui, "default_hotkey_device", lambda: Path("/dev/input/test"))
    monkeypatch.setattr(gui, "list_microphone_sources", lambda: ())
    monkeypatch.setattr(gui, "InterviewHistoryStore", lambda: InterviewHistoryStore(tmp_path))
    window = gui.MainWindow()
    assert window._config().answer_max_tokens == ANSWER_TOKEN_LIMIT
    window.answer_tokens.setValue(240)
    assert window._config().answer_max_tokens == 240
    window.close()
    qt_app.processEvents()


def test_streamed_answer_renders_before_final_without_saving_draft(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, qt_app: QApplication,
) -> None:
    from interview_helper.presentation import AnswerStarted, AnswerProgress, AnswerCompleted, ErrorReported

    monkeypatch.setattr(gui, "default_hotkey_device", lambda: Path("/dev/input/test"))
    monkeypatch.setattr(gui, "list_microphone_sources", lambda: ())
    monkeypatch.setattr(gui, "InterviewHistoryStore", lambda: InterviewHistoryStore(tmp_path))
    window = gui.MainWindow()
    window._handle_event(StartRequested())
    run_id = window._snapshot.run_id
    window._handle_event(AnswerStarted(run_id))
    window._handle_event(AnswerProgress(run_id, "I checked <routes>"))
    assert "I checked <routes>" in window.answer.toPlainText()
    assert "generating" in window.answer.toPlainText()
    assert window._conversation.snapshot() == ()
    assert not window.copy_button.isEnabled()
    window._conversation.append("What did you check?", "I checked <routes>.")
    window._handle_event(AnswerCompleted(run_id, "I checked <routes>."))
    assert window.answer.toPlainText().count("I checked <routes>.") == 1
    assert "generating" not in window.answer.toPlainText()
    window._handle_event(AnswerStarted(run_id))
    window._handle_event(AnswerProgress(run_id, "unfinished text"))
    window._handle_event(ErrorReported(run_id, "stream interrupted"))
    assert "unfinished text" not in window.answer.toPlainText()
    assert "I checked <routes>." in window.answer.toPlainText()
    window._handle_event(StopRequested())
    window._handle_event(AnswerProgress(run_id, "late text"))
    assert "late text" not in window.answer.toPlainText()
    window.close()


def test_live_question_is_visible_during_recording_and_resets_for_next_question(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, qt_app: QApplication,
) -> None:
    from interview_helper.core import RuntimeState
    from interview_helper.presentation import RuntimeStateChanged, TranscriptFinalized

    monkeypatch.setattr(gui, "default_hotkey_device", lambda: Path("/dev/input/test"))
    monkeypatch.setattr(gui, "list_microphone_sources", lambda: ())
    monkeypatch.setattr(gui, "InterviewHistoryStore", lambda: InterviewHistoryStore(tmp_path))
    window = gui.MainWindow()
    window.show()
    window._handle_event(StartRequested())
    run_id = window._snapshot.run_id
    window._handle_event(RuntimeStateChanged(run_id, RuntimeState.RECORDING))
    window._bridge.partial.emit(run_id, "How did you")
    qt_app.processEvents()
    assert window.partial_label.isVisible()
    assert window.partial_label.text() == "Live question: How did you"
    assert window._snapshot.transcript == ""
    window._bridge.partial.emit(run_id, "How did you check the routes?")
    assert window.partial_label.text().endswith("check the routes?")
    window._handle_event(TranscriptFinalized(run_id, "How did you check the routes?"))
    assert "How did you check the routes?" in window.transcript.toPlainText()
    assert window.partial_label.text() == "Question received."
    window._handle_event(RuntimeStateChanged(run_id, RuntimeState.IDLE))
    window._handle_event(RuntimeStateChanged(run_id, RuntimeState.RECORDING))
    assert window.partial_label.text() == "Listening…"
    window._handle_event(StopRequested())
    window._bridge.partial.emit(run_id, "stale partial")
    assert "stale partial" not in window.partial_label.text()
    window.close()


def test_answer_height_can_change_with_questions_shown_or_hidden(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, qt_app: QApplication,
) -> None:
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    monkeypatch.setattr(gui, "default_hotkey_device", lambda: Path("/dev/input/test"))
    monkeypatch.setattr(gui, "list_microphone_sources", lambda: ())
    monkeypatch.setattr(gui, "InterviewHistoryStore", lambda: InterviewHistoryStore(tmp_path))
    window = gui.MainWindow()
    window.show()
    qt_app.processEvents()
    bar = window.findChild(gui.QGroupBox, "answerBar")
    assert bar is not None
    for show in (True, False):
        window.show_questions.setChecked(show)
        window.answer_splitter.setSizes([350, 230])
        qt_app.processEvents()
        before = bar.height()
        handle = window.answer_splitter.handle(1)
        assert handle.isVisible()
        center = handle.rect().center()
        QTest.mousePress(handle, Qt.MouseButton.LeftButton, pos=center)
        QTest.mouseMove(handle, center - gui.QPoint(0, 90))
        QTest.mouseRelease(handle, Qt.MouseButton.LeftButton, pos=center - gui.QPoint(0, 90))
        qt_app.processEvents()
        assert bar.height() > before
        assert bar.height() > 240
        assert window.start_button.isVisible()
    window.close()


def test_hands_free_controls_configure_without_hotkey_and_allow_stop_during_speech(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, qt_app: QApplication,
) -> None:
    from interview_helper.core import RuntimeState
    from interview_helper.presentation import RuntimeStateChanged, CaptureStatusChanged, AnswerStarted, AnswerProgress, AnswerCancelled

    monkeypatch.setattr(gui, "default_hotkey_device", lambda: None)
    monkeypatch.setattr(gui, "list_microphone_sources", lambda: ())
    monkeypatch.setattr(gui, "InterviewHistoryStore", lambda: InterviewHistoryStore(tmp_path))
    window = gui.MainWindow()
    assert window._config().listen_to_candidate
    assert window._config().capture_mode is gui.CaptureMode.HEADPHONE_MONITOR
    assert window._config().automatic_listening
    assert window._config().input_device is None
    window._automatic_mode = True
    window._handle_event(StartRequested())
    run_id = window._snapshot.run_id
    window._handle_event(CaptureStatusChanged(run_id, True))
    assert 'Listening automatically' in window.phase_label.text()
    assert window.answer_now_button.isEnabled()
    window._handle_event(AnswerStarted(run_id))
    window._handle_event(AnswerProgress(run_id, 'stale answer'))
    window._handle_event(AnswerCancelled(run_id))
    assert 'stale answer' not in window.answer.toPlainText()
    window._handle_event(RuntimeStateChanged(run_id, RuntimeState.RECORDING))
    monkeypatch.setattr(gui.QMessageBox, 'critical', lambda *_: pytest.fail('Auto mode must be stoppable during speech'))
    window._interview_done()
    assert window._snapshot.stopping
    assert not window.answer_now_button.isEnabled()
    window._handle_event(RunFinished(run_id))
    window.close()


def test_technical_mode_startup_skips_default_library_and_shows_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, qt_app: QApplication,
) -> None:
    monkeypatch.setattr(gui, "default_hotkey_device", lambda: Path("/dev/input/test"))
    monkeypatch.setattr(gui, "list_microphone_sources", lambda: ())
    monkeypatch.setattr(gui, "InterviewHistoryStore", lambda: InterviewHistoryStore(tmp_path))
    prepared = tmp_path / "firewall.md"
    window = gui.MainWindow(resume_path=tmp_path / "resume.txt", technical_answers_path=prepared)
    config = window._config()
    assert config.technical_answers == prepared
    assert config.technical_library is None
    assert window.enable_technical_mode.isChecked()
    assert "Technical mode" in window.provider_status.text()
    window.enable_technical_mode.setChecked(False)
    assert window._config().technical_answers is None
    assert window.technical_answers.text() == str(prepared)
    assert "Technical mode" not in window.provider_status.text()
    window.close()


def test_technical_checkbox_round_trip_and_session_lock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, qt_app: QApplication,
) -> None:
    monkeypatch.setattr(gui, "list_microphone_sources", lambda: ())
    monkeypatch.setattr(gui, "InterviewHistoryStore", lambda: InterviewHistoryStore(tmp_path))
    window = gui.MainWindow(resume_path=tmp_path / "resume.txt")
    window.technical_library.setText(str(tmp_path / "library"))
    original = window._config()
    assert not window.enable_technical_mode.isChecked()
    assert original.technical_answers is None
    assert not window.technical_answers.parentWidget().isEnabled()
    # A fresh installation has no owner's prepared answers. Supply a fixture
    # rather than requiring a private file from the developer's checkout.
    prepared = tmp_path / "technical-answers.md"
    prepared.write_text("# Questions\n\n## What is DNS?\n\nIt resolves names.\n")
    window.technical_answers.setText(str(prepared))
    window.enable_technical_mode.setChecked(True)
    technical = window._config()
    assert technical.technical_answers == prepared
    assert technical.technical_library is None
    assert technical.resume == original.resume
    assert technical.answer_provider == original.answer_provider
    assert technical.openai_model == original.openai_model
    assert window.technical_answers.parentWidget().isEnabled()
    assert not window.technical_library.parentWidget().isEnabled()
    assert "Technical mode" in window.provider_status.text()
    window._handle_event(StartRequested())
    run_id = window._snapshot.run_id
    assert not window.enable_technical_mode.isEnabled()
    window._handle_event(StopRequested())
    assert not window.enable_technical_mode.isEnabled()
    window._handle_event(RunFinished(run_id))
    assert window.enable_technical_mode.isEnabled()
    window.enable_technical_mode.setChecked(False)
    assert window._config() == original
    assert window.technical_answers.text() == str(prepared)
    assert window.technical_library.parentWidget().isEnabled()
    assert "Technical mode" not in window.provider_status.text()
    window.close()


def test_enabled_technical_mode_requires_answers_before_start(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, qt_app: QApplication,
) -> None:
    monkeypatch.setattr(gui, "list_microphone_sources", lambda: ())
    monkeypatch.setattr(gui, "InterviewHistoryStore", lambda: InterviewHistoryStore(tmp_path))
    window = gui.MainWindow(resume_path=tmp_path / "resume.txt")
    window.enable_technical_mode.setChecked(True)
    window.technical_answers.clear()
    warnings: list[str] = []
    monkeypatch.setattr(gui.QMessageBox, "warning", lambda _parent, _title, text: warnings.append(text))
    window._start()
    assert warnings == ["Select a technical answers file or uncheck Enable technical mode."]
    assert window._worker is None
    assert not window._snapshot.running
    window.technical_answers.setText(str(tmp_path / "custom.md"))
    assert window._config().technical_answers == tmp_path / "custom.md"
    window.resume.clear()
    with pytest.raises(ValueError, match="Technical mode requires a resume"):
        window._config()
    window.enable_technical_mode.setChecked(False)
    assert window._config().technical_answers is None
    window.close()


def test_openai_model_picker_reaches_runtime_configuration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, qt_app: QApplication,
) -> None:
    from interview_helper.openai_client import OPENAI_MODEL, OPENAI_MODELS
    monkeypatch.setattr(gui, "default_hotkey_device", lambda: Path("/dev/input/test"))
    monkeypatch.setattr(gui, "list_microphone_sources", lambda: ())
    monkeypatch.setattr(gui, "InterviewHistoryStore", lambda: InterviewHistoryStore(tmp_path))
    window = gui.MainWindow(answer_provider="openai")
    assert window._config().openai_model == OPENAI_MODEL
    assert [window.openai_model.itemData(i) for i in range(window.openai_model.count())] == list(OPENAI_MODELS)
    window.openai_model.setCurrentIndex(window.openai_model.findData("gpt-6-sol"))
    assert window._config().openai_model == "gpt-6-sol"
    assert "OpenAI GPT-6 Sol" in window.provider_status.text()
    window.answer_provider.setCurrentIndex(0)
    assert not window.openai_model.isEnabled()
    window.close()
    preselected = gui.MainWindow(answer_provider="openai", openai_model="gpt-6-luna")
    assert preselected._config().openai_model == "gpt-6-luna"
    preselected.close()
    with pytest.raises(ValueError, match="Unsupported OpenAI model"):
        gui.MainWindow(answer_provider="openai", openai_model="gpt-6-astra")
