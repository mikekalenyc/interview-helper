"""Model Setup interactions with no downloads, capture, or inference."""
from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication

import interview_helper.qt_gui as gui
from interview_helper.history import InterviewHistoryStore
from interview_helper.core import RuntimeState
from interview_helper.presentation import (
    RunFinished, StartRequested, RuntimeStateChanged, reduce_presentation,
)


@pytest.fixture
def window(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[gui.MainWindow]:
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(gui, "list_microphone_sources", lambda: ())
    monkeypatch.setattr(gui, "default_hotkey_device", lambda: None)
    monkeypatch.setattr(gui, "InterviewHistoryStore", lambda: InterviewHistoryStore(tmp_path / "history"))
    monkeypatch.setattr(gui, "is_installed", lambda *_a, **_k: False)
    monkeypatch.setattr(gui, "catalog_model_path", lambda spec: tmp_path / spec.id)
    def unexpected(*_args: object, **_kwargs: object) -> Path:
        raise AssertionError("Opening Setup must not download")
    monkeypatch.setattr(gui, "install_model", unexpected)
    result = gui.MainWindow()
    yield result
    result._snapshot = reduce_presentation(result._snapshot, RunFinished(result._snapshot.run_id))
    result.close()
    if result._download_worker:
        result._download_worker.join(timeout=2)
    result.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()


def pump() -> None:
    app = QApplication.instance()
    assert app is not None
    app.processEvents()


def finish_download(window: gui.MainWindow) -> None:
    assert window._download_worker is not None
    window._download_worker.join(timeout=2)
    assert not window._download_worker.is_alive()
    pump()


def test_choices_and_independent_devices(window: gui.MainWindow) -> None:
    assert window._download_worker is None
    assert window._application is None
    config = window._config()
    assert config.transcription_backend == "moonshine"
    assert config.transcription_device == config.detection_device == "cpu"
    assert window.transcription_device.findData("cuda") == -1
    window.transcription_model.setCurrentIndex(window.transcription_model.findData("nemotron-en"))
    window.transcription_device.setCurrentIndex(window.transcription_device.findData("cuda"))
    window.gpu_device_index.setValue(1)
    config = window._config()
    assert config.transcription_backend == "nemotron"
    assert config.transcription_device == "cuda"
    assert config.detection_device == "cpu"
    assert config.gpu_device_index == 1
    assert config.nemotron_library is not None
    window.detection_device.setCurrentIndex(window.detection_device.findData("cuda"))
    window.transcription_model.setCurrentIndex(window.transcription_model.findData("moonshine-medium"))
    config = window._config()
    assert config.transcription_device == "cpu"
    assert config.detection_device == "cuda"
    assert config.moonshine_architecture == "medium"
    assert window._download_worker is None


def test_explicit_download_progress_completion_and_locking(
    window: gui.MainWindow, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    started, release = threading.Event(), threading.Event()
    output = tmp_path / "installed"
    def install(model_id: str, **kwargs: Any) -> Path:
        assert model_id == "moonshine-small"
        kwargs["progress"](5, 10, "weights")
        started.set()
        assert release.wait(2)
        return output
    monkeypatch.setattr(gui, "install_model", install)
    window.download_model_button.click()
    assert started.wait(1)
    pump()
    assert not window.setup_group.isEnabled()
    assert not window.transcription_model.isEnabled()
    assert not window.detection_device.isEnabled()
    assert not window.resume.isEnabled()
    assert not window.start_button.isEnabled()
    assert window.cancel_download_button.isEnabled()
    assert window.download_progress.value() == 50
    window._bridge.model_progress.emit(window._download_generation, 3_000_000_000, 4_000_000_000, "large archive")
    assert window.download_progress.value() == 75
    window._start()
    assert window._application is None
    release.set()
    finish_download(window)
    assert window.setup_group.isEnabled()
    assert window.start_button.isEnabled()
    assert window.model_path.text() == str(output)
    assert "complete" in window.download_status.text()
    # Once an interview starts, selection and downloading are locked.
    window._snapshot = reduce_presentation(window._snapshot, StartRequested())
    window._render()
    assert not window.setup_group.isEnabled()
    assert not window.transcription_device.isEnabled()
    window._snapshot = reduce_presentation(
        window._snapshot, RuntimeStateChanged(window._snapshot.run_id, RuntimeState.RECORDING),
    )
    window._render()
    assert not window.setup_group.isEnabled()
    window._download_model()
    assert not window._download_active


def test_reuses_installed_model_without_network(window: gui.MainWindow, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gui, "is_installed", lambda *_a, **_k: True)
    window._download_model()
    assert window._download_worker is None
    assert window.model_path.text() == str(gui.catalog_model_path(window._selected_model()))
    assert "already installed" in window.download_status.text()


def test_cancel_and_error_recover_setup(window: gui.MainWindow, monkeypatch: pytest.MonkeyPatch) -> None:
    started = threading.Event()
    original_path = window.model_path.text()
    def install(_model: str, **kwargs: Any) -> Path:
        started.set()
        assert kwargs["cancel_event"].wait(2)
        raise RuntimeError("network read interrupted")
    monkeypatch.setattr(gui, "install_model", install)
    window._download_model()
    assert started.wait(1)
    window._cancel_model_download()
    finish_download(window)
    assert window.model_path.text() == original_path
    assert "cancelled" in window.download_status.text()
    assert window.start_button.isEnabled()
    def fail(*_args: object, **_kwargs: object) -> Path:
        raise OSError("disk full")
    monkeypatch.setattr(gui, "install_model", fail)
    window._download_model()
    finish_download(window)
    assert "disk full" in window.download_status.text()
    assert window.setup_group.isEnabled()
    cancelled_generation = window._download_generation - 1
    monkeypatch.setattr(gui, "install_model", lambda *_a, **_k: Path("retry-success"))
    window._download_model()
    window._model_download_finished(cancelled_generation, "stale-cancelled", "")
    assert window.model_path.text() == original_path
    finish_download(window)
    assert window.model_path.text() == "retry-success"
    assert "complete" in window.download_status.text()


def test_close_and_stale_completion_do_not_modify_selection(window: gui.MainWindow, monkeypatch: pytest.MonkeyPatch) -> None:
    started = threading.Event()
    original_path = window.model_path.text()
    def install(_model: str, **kwargs: Any) -> Path:
        started.set()
        assert kwargs["cancel_event"].wait(2)
        return Path("late")
    monkeypatch.setattr(gui, "install_model", install)
    window._download_model()
    assert started.wait(1)
    window._model_download_finished(window._download_generation - 1, "stale", "")
    assert window.model_path.text() == original_path
    window.close()
    assert window._download_cancel.is_set()
    finish_download(window)
    assert window.model_path.text() == original_path
