# mypy: disable-error-code="misc,untyped-decorator"
"""PySide6 desktop window for the local Interview Helper pipeline."""

from __future__ import annotations

import threading
import time
from datetime import datetime
from collections.abc import Callable
from pathlib import Path
from typing import cast

from PySide6.QtCore import QObject, QPoint, Qt, Signal, Slot
from PySide6.QtGui import QColor, QCloseEvent, QMouseEvent, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QInputDialog,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QProgressBar,
    QScrollArea,
    QSplitter,
    QSizeGrip,
    QStatusBar,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from interview_helper.openai_client import OPENAI_MODELS
from interview_helper.model_catalog import (
    MODEL_CATALOG, ModelSpec, install_model, is_installed,
    model_path as catalog_model_path, runtime_library_path,
)
from interview_helper.application import (
    ApplicationCallbacks,
    ApplicationConfig,
    AnswerProvider,
    CaptureMode,
    ControlMode,
    InterviewApplication,
)
from interview_helper.answer import ANSWER_TOKEN_LIMIT, AnswerConversation, AnswerTurn
from interview_helper.spoken import SpokenConversation, SpokenTurn
from interview_helper.history import HistoryEntry
from interview_helper.cli import DEFAULT_MODEL
from interview_helper.capture import CaptureError, list_microphone_sources
from interview_helper.core import RuntimeState, Transcript
from interview_helper.history import (
    InterviewHistoryStore,
    InterviewRecord,
)
from interview_helper.input import (
    HotkeyLearningCancelled,
    LearnedHotkey,
    default_hotkey_device,
    learning_event_paths,
    learn_hotkey,
)
from interview_helper.presentation import (
    AnswerCompleted,
    AnswerCancelled,
    AnswerProgress,
    AnswerStarted,
    CaptureStatusChanged,
    ErrorReported,
    PresentationEvent,
    PresentationPhase,
    PresentationSnapshot,
    RunFinished,
    RuntimeStateChanged,
    StartRequested,
    StopRequested,
    TranscriptFinalized,
    reduce_presentation,
)
from interview_helper.qwen import DEFAULT_BASE_URL, DEFAULT_MODEL as DEFAULT_QWEN_MODEL


class WindowTitleBar(QWidget):
    """Window controls and a drag surface for the transparent window."""

    def __init__(self, window: QMainWindow) -> None:
        super().__init__(window)
        self._window = window
        self._drag_offset: QPoint | None = None
        self.setObjectName("windowTitleBar")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 4, 4, 4)
        title = QLabel("Interview Helper")
        title.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        layout.addWidget(title)
        layout.addStretch()
        for label, tooltip, action in (
            ("−", "Minimize", window.showMinimized),
            ("□", "Maximize / restore", self._toggle_maximized),
            ("×", "Close", window.close),
        ):
            button = QPushButton(label)
            button.setToolTip(tooltip)
            button.setAccessibleName(tooltip)
            button.setFixedSize(32, 28)
            button.clicked.connect(action)
            layout.addWidget(button)

    def _toggle_maximized(self) -> None:
        if self._window.isMaximized():
            self._window.showNormal()
        else:
            self._window.showMaximized()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            handle = self._window.windowHandle()
            if handle is None or not handle.startSystemMove():
                self._drag_offset = event.globalPosition().toPoint() - self._window.pos()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_offset is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self._window.move(event.globalPosition().toPoint() - self._drag_offset)
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._drag_offset = None
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._toggle_maximized()
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)


class CallbackBridge(QObject):
    spoken_changed = Signal(int)
    candidate_partial = Signal(int, str)
    candidate_state = Signal(int, str)
    presentation_event = Signal(object)
    status = Signal(int, str)
    partial = Signal(int, str)
    evidence = Signal(int, str)
    exchange = Signal(object)
    hotkey_learned = Signal(object)
    hotkey_failed = Signal(str)
    hotkey_cancelled = Signal()
    model_progress = Signal(int, object, object, str)
    model_finished = Signal(int, str, str)


class MainWindow(QMainWindow):
    def __init__(
        self,
        *,
        resume_path: Path | None = None,
        context_paths: tuple[Path, ...] = (),
        answer_provider: str = "local",
        openai_api_key_file: Path | None = None,
        openai_model: str | None = None,
        technical_answers_path: Path | None = None,
    ) -> None:
        super().__init__()
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint)
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setWindowTitle("Interview Helper")
        self.resize(1040, 760)
        self.setMinimumSize(720, 480)
        self.setStyleSheet("""
            QWidget { color: #ffffff; }
            QMainWindow, QWidget#transparentRoot, QWidget#mainPage,
            QTabWidget::pane { background: transparent; border: none; }
            QWidget#windowTitleBar, QStatusBar { background: #111111; }
            QWidget#setupPage, QWidget#historyPage, QWidget#spokenPage, QDialog { background: #181818; }
            QGroupBox { border: 1px solid #555555; border-radius: 6px;
                        margin-top: 12px; padding: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 12px; }
            QPushButton, QTabBar::tab { background: #242424; color: white;
                border: 1px solid #555555; border-radius: 4px; padding: 6px 12px; }
            QPushButton:hover, QTabBar::tab:selected { background: #414141; }
            QPushButton:disabled { color: #929292; background: #1b1b1b; }
            QLineEdit, QComboBox, QSpinBox, QListWidget, QTextEdit {
                background: #181818; color: white; border: 1px solid #555555;
                selection-background-color: #376ea6; selection-color: white; }
            QComboBox QAbstractItemView { background: #181818; color: white; }
            QGroupBox#answerBar { background: transparent; border: none; }
            QTextEdit#answerText { background: transparent; color: #ffffff;
                border: none; font-size: 16px; padding: 8px; }
            QGroupBox#questionPanel { background: transparent; border: none; }
            QTextEdit#questionText { background: transparent; color: #ffffff;
                border: none; font-size: 18px; }
            QCheckBox { background: #181818; padding: 6px; border-radius: 4px; }
        """)
        self._snapshot = PresentationSnapshot()
        self._conversation = AnswerConversation()
        self._spoken_conversation = SpokenConversation()
        self._history_store = InterviewHistoryStore()
        self._history_session = self._history_store.new_session()
        self._history_records: tuple[InterviewRecord, ...] = ()
        self._application: InterviewApplication | None = None
        self._worker: threading.Thread | None = None
        self._microphone_mode = False
        self._automatic_mode = True
        self._close_when_finished = False
        self._end_session_when_finished = False
        self._hotkey_learning = False
        self._hotkey_cancel = threading.Event()
        self._hotkey_worker: threading.Thread | None = None
        self._download_active = False
        self._download_generation = 0
        self._download_cancel = threading.Event()
        self._download_worker: threading.Thread | None = None
        self._window_closed = False
        self._bridge = CallbackBridge()
        self._bridge.model_progress.connect(self._model_download_progress)
        self._bridge.model_finished.connect(self._model_download_finished)
        self._bridge.spoken_changed.connect(self._show_run_spoken)
        self._bridge.candidate_partial.connect(self._show_candidate_partial)
        self._bridge.candidate_state.connect(self._show_candidate_state)
        self._bridge.presentation_event.connect(self._handle_event)
        self._bridge.status.connect(self._show_run_status)
        self._bridge.partial.connect(self._show_run_partial)
        self._bridge.evidence.connect(self._show_run_evidence)
        self._bridge.exchange.connect(self._handle_exchange)
        self._bridge.hotkey_learned.connect(self._hotkey_was_learned)
        self._bridge.hotkey_failed.connect(self._hotkey_learning_failed)
        self._bridge.hotkey_cancelled.connect(self._hotkey_learning_cancelled)
        self._build_ui()
        provider = AnswerProvider(answer_provider)
        self.answer_provider.setCurrentIndex(self.answer_provider.findData(provider.value))
        if openai_api_key_file is not None:
            self.openai_key_file.setText(str(openai_api_key_file))
        if openai_model is not None:
            if openai_model not in OPENAI_MODELS:
                raise ValueError(
                    f"Unsupported OpenAI model {openai_model!r}; choose one of: "
                    + ", ".join(OPENAI_MODELS)
                )
            self.openai_model.setCurrentIndex(self.openai_model.findData(openai_model))
        if resume_path is not None:
            self.resume.setText(str(resume_path))
        self.context_paths.addItems([str(path) for path in dict.fromkeys(context_paths)])
        if technical_answers_path is not None:
            self.technical_answers.setText(str(technical_answers_path))
            self.enable_technical_mode.setChecked(True)
        self._refresh_history()
        self._render()

    def _build_ui(self) -> None:
        central = QWidget()
        central.setObjectName("transparentRoot")
        root = QVBoxLayout(central)
        self.title_bar = WindowTitleBar(self)
        root.addWidget(self.title_bar)
        self.tabs = QTabWidget()
        root.addWidget(self.tabs)

        main_tab = QWidget()
        main_tab.setObjectName("mainPage")
        main_layout = QVBoxLayout(main_tab)
        self.tabs.addTab(main_tab, "Main")

        setup_page = QWidget()
        setup_page.setObjectName("setupPage")
        setup_outer = QVBoxLayout(setup_page)
        setup_outer.setContentsMargins(0, 0, 0, 0)
        setup_tab = QWidget()
        setup_tab.setObjectName("setupPage")
        setup_layout = QVBoxLayout(setup_tab)
        setup_scroll = QScrollArea()
        setup_scroll.setWidgetResizable(True)
        setup_scroll.setWidget(setup_tab)
        setup_outer.addWidget(setup_scroll, 1)
        self.tabs.addTab(setup_page, "Setup")

        history_tab = QWidget()
        history_tab.setObjectName("historyPage")
        history_layout = QHBoxLayout(history_tab)
        self.history_list = QListWidget()
        self.history_detail = QTextEdit()
        self.history_detail.setReadOnly(True)
        self.history_list.currentRowChanged.connect(self._show_history)
        history_layout.addWidget(self.history_list, 1)
        history_layout.addWidget(self.history_detail, 2)
        self.tabs.addTab(history_tab, "History")

        self.sources = QTextEdit()
        self.sources.setReadOnly(True)
        self.sources.setAcceptRichText(False)
        self.tabs.addTab(self.sources, "Retrieved sources")

        self.phase_label = QLabel()
        self.phase_label.setWordWrap(True)
        self.phase_label.setObjectName("phaseBanner")
        self.phase_label.setStyleSheet(
            "QLabel { font-size: 18px; font-weight: 700; padding: 10px; "
            "border-radius: 6px; background: #273142; color: white; }"
        )
        main_layout.addWidget(self.phase_label)

        setup = QGroupBox("Setup")
        form = QFormLayout(setup)
        form.addRow("Audio workflow", QLabel("Interviewer audio + your spoken answers"))
        microphone_selector = QWidget()
        microphone_layout = QHBoxLayout(microphone_selector)
        microphone_layout.setContentsMargins(0, 0, 0, 0)
        self.microphone_source = QComboBox()
        self.refresh_microphones_button = QPushButton("Refresh")
        self.refresh_microphones_button.clicked.connect(self._refresh_microphones)
        microphone_layout.addWidget(self.microphone_source, 1)
        microphone_layout.addWidget(self.refresh_microphones_button)
        form.addRow("Microphone", microphone_selector)
        self.microphone_selector = microphone_selector
        self.microphone_label = form.labelForField(microphone_selector)
        self.input_device = self._path_row(form, "Hotkey device", self._browse_input)
        self.event_code = QLineEdit("KEY_M")
        form.addRow("Held key / button", self.event_code)
        hotkey_learning = QWidget()
        hotkey_learning_layout = QVBoxLayout(hotkey_learning)
        hotkey_learning_layout.setContentsMargins(0, 0, 0, 0)
        self.learn_hotkey_button = QPushButton("Learn hotkey…")
        self.learn_hotkey_button.clicked.connect(self._toggle_hotkey_learning)
        self.learn_hotkey_help = QLabel(
            "Press Learn, then press the desired key once. Input is observed, "
            "never grabbed. Learn scans all accessible stable devices and also "
            "includes the manually selected device."
        )
        self.learn_hotkey_help.setWordWrap(True)
        hotkey_learning_layout.addWidget(self.learn_hotkey_button)
        hotkey_learning_layout.addWidget(self.learn_hotkey_help)
        form.addRow("Hotkey setup", hotkey_learning)
        # Legacy hotkey helpers are retained for now but are not part of the desktop workflow.
        hotkey_row = self.input_device.parentWidget()
        assert hotkey_row is not None
        form.setRowVisible(hotkey_row, False)
        form.setRowVisible(self.event_code, False)
        form.setRowVisible(hotkey_learning, False)
        self.transcription_model = QComboBox()
        for spec in MODEL_CATALOG:
            self.transcription_model.addItem(spec.label, spec.id)
        self.transcription_model.setCurrentIndex(self.transcription_model.findData("moonshine-small"))
        form.addRow("Transcription model", self.transcription_model)
        self.transcription_device = QComboBox()
        form.addRow("Transcription device", self.transcription_device)
        self.detection_device = QComboBox()
        self.detection_device.addItem("CPU (recommended)", "cpu")
        self.detection_device.addItem("NVIDIA GPU (CUDA; GPU installation required)", "cuda")
        form.addRow("Speech / turn detection", self.detection_device)
        self.gpu_device_index = QSpinBox()
        self.gpu_device_index.setRange(0, 31)
        self.gpu_device_index.setToolTip("Zero selects the first NVIDIA GPU.")
        form.addRow("GPU index", self.gpu_device_index)
        form.addRow("Document lookup", QLabel("CPU (recommended)"))
        self.model_path = self._path_row(form, "Installed model path", self._browse_model)
        self.model_path.setText(str(DEFAULT_MODEL))
        self.model_description = QLabel()
        self.model_description.setWordWrap(True)
        form.addRow(self.model_description)
        self.model_status = QLabel()
        self.model_status.setWordWrap(True)
        form.addRow(self.model_status)
        self.download_model_button = QPushButton("Download selected model")
        self.download_model_button.clicked.connect(self._download_model)
        form.addRow(self.download_model_button)
        self.transcription_model.currentIndexChanged.connect(self._model_changed)
        self.transcription_device.currentIndexChanged.connect(self._model_device_changed)
        self.model_path.textChanged.connect(self._refresh_model_status)
        self._model_changed(preserve_path=True)
        self.resume = self._path_row(form, "Resume (optional)", self._browse_resume)

        context_box = QWidget()
        context_layout = QHBoxLayout(context_box)
        context_layout.setContentsMargins(0, 0, 0, 0)
        self.context_paths = QListWidget()
        context_buttons = QVBoxLayout()
        add_context = QPushButton("Add…")
        add_context.clicked.connect(self._add_context)
        remove_context = QPushButton("Remove")
        remove_context.clicked.connect(self._remove_context)
        context_buttons.addWidget(add_context)
        context_buttons.addWidget(remove_context)
        context_buttons.addStretch()
        context_layout.addWidget(self.context_paths)
        context_layout.addLayout(context_buttons)
        form.addRow("Interview guides and background", context_box)
        self.technical_library = self._path_row(
            form, "Technical library (optional)", self._browse_technical_library
        )
        default_library = Path("/mnt/raid10/interview assistant/technical-library")
        if default_library.is_dir():
            self.technical_library.setText(str(default_library))
        library_help = QLabel(
            "Local reference documents; no online lookup. Clear the folder to disable. "
            "Technical references cannot prove personal experience. "
            "Retrieved sources shows the passages supplied to the model."
        )
        library_help.setWordWrap(True)
        form.addRow(library_help)
        self.enable_technical_mode = QCheckBox("Enable technical mode")
        form.addRow(self.enable_technical_mode)
        self.technical_answers = self._path_row(
            form, "Technical answers", self._browse_technical_answers
        )
        default_answers = (
            Path(__file__).resolve().parent.parent / "technical mode"
            / "cisco_firewall_engineer_50_questions_original_with_concise_answers.md"
        )
        if default_answers.is_file():
            self.technical_answers.setText(str(default_answers))
        technical_help = QLabel(
            "Enable for short technical answers using this prepared Q&A and the model's "
            "technical knowledge, with your last five questions and spoken answers. "
            "The general technical library is paused in this mode. "
            "Uncheck to return to general interviews. Change modes between interviews."
        )
        technical_help.setWordWrap(True)
        form.addRow(technical_help)

        self.keyterms = QLineEdit()
        self.keyterms.setPlaceholderText("BGP, EVPN, Kubernetes")
        form.addRow("Vocabulary", self.keyterms)
        self.answer_provider = QComboBox()
        self.answer_provider.addItem("Local Qwen", AnswerProvider.LOCAL_QWEN.value)
        self.answer_provider.addItem("OpenAI", AnswerProvider.OPENAI.value)
        form.addRow("Answer provider", self.answer_provider)
        self.openai_model = QComboBox()
        for model, label in OPENAI_MODELS.items():
            self.openai_model.addItem(label, model)
        form.addRow("OpenAI model", self.openai_model)
        self.openai_key_file = self._path_row(
            form, "OpenAI key file", self._browse_openai_key
        )
        self.openai_key_file.setPlaceholderText("Or use the OPENAI_API_KEY environment variable")
        self.qwen_url = QLineEdit(DEFAULT_BASE_URL)
        form.addRow("Qwen URL", self.qwen_url)
        self.qwen_model = QLineEdit(DEFAULT_QWEN_MODEL)
        form.addRow("Qwen model", self.qwen_model)
        self.answer_tokens = QSpinBox()
        self.answer_tokens.setRange(1, 4096)
        self.answer_tokens.setValue(ANSWER_TOKEN_LIMIT)
        form.addRow("Answer token limit", self.answer_tokens)
        self.provider_help = QLabel()
        self.provider_help.setWordWrap(True)
        form.addRow(self.provider_help)
        self.answer_provider.currentIndexChanged.connect(self._provider_changed)
        self.openai_model.currentIndexChanged.connect(self._provider_changed)
        self.enable_technical_mode.toggled.connect(self._provider_changed)
        setup_layout.addWidget(setup)
        self.download_status = QLabel()
        self.download_status.setWordWrap(True)
        self.download_progress = QProgressBar()
        self.download_progress.setRange(0, 100)
        self.download_progress.hide()
        self.cancel_download_button = QPushButton("Cancel download")
        self.cancel_download_button.clicked.connect(self._cancel_model_download)
        self.cancel_download_button.hide()
        # Keep progress and cancellation visible even when the long form scrolls.
        setup_outer.addWidget(self.download_status)
        setup_outer.addWidget(self.download_progress)
        setup_outer.addWidget(self.cancel_download_button)
        setup_layout.addStretch()
        self.setup_group = setup

        controls = QHBoxLayout()
        self.start_button = QPushButton("Start Interview")
        self.start_button.clicked.connect(self._start)
        self.interview_done_button = QPushButton("Interview Done")
        self.interview_done_button.clicked.connect(self._interview_done)
        controls.addWidget(self.start_button)
        controls.addWidget(self.interview_done_button)
        self.show_questions = QCheckBox("Show questions")
        controls.addWidget(self.show_questions)
        controls.addStretch()
        main_layout.addLayout(controls)
        self.provider_status = QLabel()
        self.provider_status.setWordWrap(True)
        main_layout.addWidget(self.provider_status)
        self._provider_changed()
        automatic_controls = QHBoxLayout()
        self.answer_now_button = QPushButton("Answer now")
        self.answer_now_button.clicked.connect(self._answer_now)
        self.cancel_answer_button = QPushButton("Cancel question / answer")
        self.cancel_answer_button.clicked.connect(self._cancel_answer)
        automatic_controls.addWidget(self.answer_now_button)
        automatic_controls.addWidget(self.cancel_answer_button)
        automatic_controls.addStretch()
        main_layout.addLayout(automatic_controls)
        self.spoken_mode_label = QLabel("Use my spoken answers · always enabled")
        self.spoken_mode_label.setWordWrap(True)
        self.spoken_mode_label.setToolTip(
            "Start Interview listens to interviewer audio and your microphone. "
            "Only interviewer questions generate suggestions."
        )
        main_layout.addWidget(self.spoken_mode_label)
        self.candidate_state_label = QLabel("Microphone stopped")
        self.candidate_state_label.setStyleSheet("background: #181818; padding: 4px;")
        self.candidate_state_label.setWordWrap(True)
        main_layout.addWidget(self.candidate_state_label)

        spoken_page = QWidget()
        spoken_page.setObjectName("spokenPage")
        spoken_layout = QVBoxLayout(spoken_page)
        spoken_help = QLabel(
            "Your finalized speech informs follow-up answers in this session. "
            "Correct or discard transcription errors here. Suggestions are kept separately. "
            "Completed text is saved in History; audio is not saved."
        )
        spoken_help.setWordWrap(True)
        spoken_layout.addWidget(spoken_help)
        self.spoken_partial = QLabel("")
        self.spoken_partial.setWordWrap(True)
        spoken_layout.addWidget(self.spoken_partial)
        self.spoken_list = QListWidget()
        self.spoken_list.currentRowChanged.connect(self._show_spoken_detail)
        spoken_layout.addWidget(self.spoken_list)
        self.spoken_detail = QTextEdit()
        self.spoken_detail.setReadOnly(True)
        spoken_layout.addWidget(self.spoken_detail)
        spoken_buttons = QHBoxLayout()
        self.correct_spoken_button = QPushButton("Correct selected text…")
        self.correct_spoken_button.clicked.connect(self._correct_spoken)
        self.discard_spoken_button = QPushButton("Discard selected text")
        self.discard_spoken_button.clicked.connect(self._discard_spoken)
        spoken_buttons.addWidget(self.correct_spoken_button)
        spoken_buttons.addWidget(self.discard_spoken_button)
        spoken_layout.addLayout(spoken_buttons)
        self.tabs.addTab(spoken_page, "You said")

        question_group = QGroupBox("Session questions")
        question_group.setObjectName("questionPanel")
        question_layout = QVBoxLayout(question_group)
        self.partial_label = QLabel("Start Interview to listen to questions and your spoken answers.")
        self.partial_label.setWordWrap(True)
        self.partial_label.setTextFormat(Qt.TextFormat.PlainText)
        self.partial_label.setStyleSheet("color: white; font-size: 18px;")
        self.transcript = QTextEdit()
        self.transcript.setObjectName("questionText")
        self.transcript.setReadOnly(True)
        self.transcript.viewport().setAutoFillBackground(False)
        question_layout.addWidget(self.partial_label)
        question_layout.addWidget(self.transcript)
        self.show_questions.toggled.connect(question_group.setVisible)
        self.show_questions.setChecked(True)
        # Keep an upper pane when questions are hidden so the answer divider
        # remains draggable in the transparent, answers-only view.
        question_area = QWidget()
        question_area.setMinimumHeight(24)
        question_area_layout = QVBoxLayout(question_area)
        question_area_layout.setContentsMargins(0, 0, 0, 0)
        question_area_layout.addWidget(question_group)
        answer_group = QGroupBox()
        answer_group.setObjectName("answerBar")
        answer_layout = QVBoxLayout(answer_group)
        answer_header = QHBoxLayout()
        answer_header.addWidget(QLabel("Session answers"))
        answer_header.addStretch()
        self._answer_text_pixels = 16
        self.answer_smaller_button = QPushButton("A−")
        self.answer_smaller_button.setAccessibleName("Make answer text smaller")
        self.answer_smaller_button.setToolTip("Make answer text smaller")
        self.answer_smaller_button.clicked.connect(
            lambda: self._set_answer_text_size(self._answer_text_pixels - 2)
        )
        self.answer_size_label = QLabel("16 px")
        self.answer_larger_button = QPushButton("A+")
        self.answer_larger_button.setAccessibleName("Make answer text larger")
        self.answer_larger_button.setToolTip("Make answer text larger")
        self.answer_larger_button.clicked.connect(
            lambda: self._set_answer_text_size(self._answer_text_pixels + 2)
        )
        answer_header.addWidget(self.answer_smaller_button)
        answer_header.addWidget(self.answer_size_label)
        answer_header.addWidget(self.answer_larger_button)
        answer_layout.addLayout(answer_header)
        self.answer = QTextEdit()
        self.answer.setObjectName("answerText")
        self.answer.setReadOnly(True)
        self.answer.viewport().setAutoFillBackground(False)
        self.answer.setMinimumHeight(140)
        self.copy_button = QPushButton("Copy answer")
        self.copy_button.clicked.connect(self._copy_answer)
        answer_layout.addWidget(self.answer)
        answer_layout.addWidget(self.copy_button)
        self.answer_splitter = QSplitter(Qt.Orientation.Vertical)
        self.answer_splitter.setObjectName("answerSplitter")
        self.answer_splitter.setChildrenCollapsible(False)
        self.answer_splitter.setHandleWidth(8)
        self.answer_splitter.setStyleSheet(
            "QSplitter#answerSplitter { background: transparent; }"
            "QSplitter#answerSplitter::handle { background: #444444; }"
            "QSplitter#answerSplitter::handle:hover { background: #888888; }"
        )
        self.answer_splitter.addWidget(question_area)
        self.answer_splitter.addWidget(answer_group)
        self.answer_splitter.setStretchFactor(0, 1)
        self.answer_splitter.setStretchFactor(1, 0)
        self.answer_splitter.setSizes([340, 240])
        self.answer_splitter.handle(1).setToolTip("Drag to resize session answers")
        main_layout.addWidget(self.answer_splitter, 1)

        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar())
        self.statusBar().setSizeGripEnabled(False)
        self.resize_grip = QSizeGrip(self)
        self.statusBar().addPermanentWidget(self.resize_grip)
        self._refresh_microphones(show_error=False)

    def _path_row(
        self, form: QFormLayout, label: str, browse: Callable[[], None]
    ) -> QLineEdit:
        container = QWidget()
        layout = QGridLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        field = QLineEdit()
        button = QPushButton("Browse…")
        button.clicked.connect(browse)
        layout.addWidget(field, 0, 0)
        layout.addWidget(button, 0, 1)
        form.addRow(label, container)
        return field

    @Slot()
    def _start(self) -> None:
        if self._download_active or self._window_closed or self._snapshot.running:
            return
        try:
            config = self._config()
        except ValueError as error:
            QMessageBox.warning(self, "Invalid setup", str(error))
            return
        self._snapshot = reduce_presentation(self._snapshot, StartRequested())
        self._microphone_mode = (
            config.capture_mode is CaptureMode.DEFAULT_MICROPHONE
        )
        self._automatic_mode = config.automatic_listening
        if config.listen_to_candidate:
            self.candidate_state_label.setText("Microphone — starting…")
        run_id = self._snapshot.run_id
        self.sources.clear()
        callbacks = ApplicationCallbacks(
            candidate_partial=lambda text: self._bridge.candidate_partial.emit(run_id, text),
            candidate_state=lambda state: self._bridge.candidate_state.emit(run_id, state.value),
            candidate_transcript=lambda _turn: self._bridge.spoken_changed.emit(run_id),
            state=lambda state: self._emit(RuntimeStateChanged(run_id, state)),
            capture_ready=lambda ready: self._emit(CaptureStatusChanged(run_id, ready)),
            status=lambda text: self._bridge.status.emit(run_id, text),
            partial=lambda text: self._bridge.partial.emit(run_id, text),
            evidence=lambda text: self._bridge.evidence.emit(run_id, text),
            transcript=lambda result: self._on_transcript(run_id, result),
            answer=lambda text: self._emit(AnswerCompleted(run_id, text)),
            answer_progress=lambda text: self._emit(AnswerProgress(run_id, text)),
            answer_cancelled=lambda: self._emit(AnswerCancelled(run_id)),
            exchange=lambda turn: self._record_exchange(run_id, turn),
            answer_busy=lambda busy: self._answer_busy(run_id, busy),
            error=lambda error: self._emit(ErrorReported(run_id, str(error))),
        )
        application = InterviewApplication(
            config,
            callbacks,
            conversation=self._conversation,
            spoken_conversation=self._spoken_conversation,
        )
        self._application = application

        def run() -> None:
            try:
                application.run()
            except BaseException as error:
                self._emit(ErrorReported(run_id, str(error)))
            finally:
                self._emit(RunFinished(run_id))

        self._worker = threading.Thread(target=run, name="interview-app", daemon=True)
        self._worker.start()
        self._render()

    @Slot()
    def _answer_now(self) -> None:
        if self._application is not None:
            self._application.answer_now()

    @Slot()
    def _cancel_answer(self) -> None:
        if self._application is not None:
            self._application.cancel_answer()

    @Slot()
    def _stop(self) -> None:
        application = self._application
        if application is not None:
            self._show_status("Stopping…")
        self._snapshot = reduce_presentation(self._snapshot, StopRequested())
        self._render()
        if application is not None:
            threading.Thread(
                target=application.stop,
                name="interview-app-stop",
                daemon=True,
            ).start()

    @Slot()
    def _interview_done(self) -> None:
        if not self._snapshot.running:
            return
        if self._snapshot.runtime_state is RuntimeState.RECORDING and not self._automatic_mode:
            message = (
                "Release the configured hotkey before ending the interview. "
                "The current recording is still active."
            )
            QMessageBox.critical(self, "Cannot end interview while recording", message)
            self._show_status(message)
            return
        self._end_session_when_finished = True
        self._stop()

    def _config(self) -> ApplicationConfig:
        model = self.model_path.text().strip()
        if not model:
            raise ValueError("A transcription model path is required; download a model or browse to one.")
        technical_mode = self.enable_technical_mode.isChecked()
        if technical_mode and not self.technical_answers.text().strip():
            raise ValueError("Select a technical answers file or uncheck Enable technical mode.")
        resume_text = self.resume.text().strip()
        contexts = tuple(
            Path(self.context_paths.item(row).text())
            for row in range(self.context_paths.count())
        )
        keyterms = tuple(
            term.strip() for term in self.keyterms.text().split(",") if term.strip()
        )
        return ApplicationConfig(
            input_device=None,
            event_code="",
            model_path=Path(model),
            transcription_backend=self._selected_model().backend,
            transcription_device=str(self.transcription_device.currentData()),
            detection_device=str(self.detection_device.currentData()),
            gpu_device_index=self.gpu_device_index.value(),
            moonshine_architecture=self._selected_model().architecture,
            nemotron_library=(runtime_library_path(str(self.transcription_device.currentData()))
                              if self._selected_model().backend == "nemotron" else None),
            capture_mode=CaptureMode.HEADPHONE_MONITOR,
            control_mode=ControlMode.MANUAL,
            automatic_listening=True,
            listen_to_candidate=True,
            microphone_source=(
                str(self.microphone_source.currentData())
                if self.microphone_source.currentData()
                else None
            ),
            keyterms=keyterms,
            resume=Path(resume_text) if resume_text else None,
            context=contexts,
            qwen_base_url=self.qwen_url.text().strip(),
            qwen_model=self.qwen_model.text().strip(),
            answer_provider=AnswerProvider(self.answer_provider.currentData()),
            openai_model=str(self.openai_model.currentData()),
            openai_api_key_file=(
                Path(self.openai_key_file.text().strip())
                if self.openai_key_file.text().strip() else None
            ),
            answer_max_tokens=self.answer_tokens.value(),
            technical_library=(
                Path(self.technical_library.text().strip())
                if not technical_mode and self.technical_library.text().strip() else None
            ),
            technical_answers=(
                Path(self.technical_answers.text().strip())
                if technical_mode else None
            ),
        )

    def _emit(self, event: PresentationEvent) -> None:
        self._bridge.presentation_event.emit(event)

    def _on_transcript(self, run_id: int, transcript: Transcript) -> None:
        self._emit(TranscriptFinalized(run_id, transcript.text))

    def _answer_busy(self, run_id: int, busy: bool) -> None:
        if busy:
            self._emit(AnswerStarted(run_id))

    def _record_exchange(self, run_id: int, turn: AnswerTurn) -> None:
        try:
            self._history_session.append(turn)
        except BaseException as error:
            self._emit(ErrorReported(run_id, f"Could not save interview history: {error}"))
        self._bridge.exchange.emit(turn)

    @Slot()
    def _refresh_microphones(self, *, show_error: bool = True) -> None:
        selected = self.microphone_source.currentData()
        try:
            sources = list_microphone_sources()
        except CaptureError as error:
            self.microphone_source.clear()
            self.microphone_source.addItem("System default (unavailable)", "")
            if show_error:
                QMessageBox.warning(self, "Could not list microphones", str(error))
            return
        default = next((source for source in sources if source.is_default), None)
        default_description = default.description if default is not None else "not set"
        self.microphone_source.clear()
        self.microphone_source.addItem(
            f"System default — {default_description}",
            "",
        )
        for source in sources:
            self.microphone_source.addItem(source.description, source.name)
        if selected:
            index = self.microphone_source.findData(selected)
            if index >= 0:
                self.microphone_source.setCurrentIndex(index)

    @Slot()
    def _toggle_hotkey_learning(self) -> None:
        if self._hotkey_learning:
            self._hotkey_cancel.set()
            self.learn_hotkey_button.setText("Cancelling…")
            self.learn_hotkey_button.setEnabled(False)
            return
        selected = self.input_device.text().strip()
        paths = learning_event_paths(Path(selected) if selected else None)
        self._hotkey_learning = True
        self._hotkey_cancel = threading.Event()
        self.learn_hotkey_button.setText("Cancel learning")
        self._show_status(
            "Learning hotkey: press the desired physical key once (20 second timeout)"
        )
        self._render()

        def learn() -> None:
            try:
                learned = learn_hotkey(
                    paths,
                    cancel=self._hotkey_cancel,
                    timeout=20.0,
                )
            except HotkeyLearningCancelled:
                self._bridge.hotkey_cancelled.emit()
            except BaseException as error:
                self._bridge.hotkey_failed.emit(str(error))
            else:
                self._bridge.hotkey_learned.emit(learned)

        self._hotkey_worker = threading.Thread(
            target=learn,
            name="hotkey-learning",
            daemon=True,
        )
        self._hotkey_worker.start()

    @Slot(object)
    def _hotkey_was_learned(self, value: object) -> None:
        learned = cast(LearnedHotkey, value)
        self.input_device.setText(str(learned.device))
        self.event_code.setText(learned.event_code)
        self._finish_hotkey_learning()
        self._show_status(
            f"Learned {learned.event_code} from {learned.device_name}"
        )

    @Slot(str)
    def _hotkey_learning_failed(self, message: str) -> None:
        self._finish_hotkey_learning()
        QMessageBox.warning(self, "Could not learn hotkey", message)
        self._show_status(message)

    @Slot()
    def _hotkey_learning_cancelled(self) -> None:
        self._finish_hotkey_learning()
        self._show_status("Hotkey learning cancelled")

    def _finish_hotkey_learning(self) -> None:
        self._hotkey_learning = False
        self._hotkey_worker = None
        self.learn_hotkey_button.setText("Learn hotkey…")
        self._render()

    @Slot(object)
    def _handle_event(self, event: object) -> None:
        previous = self._snapshot
        self._snapshot = reduce_presentation(
            self._snapshot, cast(PresentationEvent, event)
        )
        if self._snapshot != previous:
            if isinstance(event, RuntimeStateChanged) and event.state is RuntimeState.RECORDING:
                self.partial_label.setText("Listening…")
            elif isinstance(event, TranscriptFinalized):
                self.partial_label.setText("Question received.")
        if isinstance(event, RunFinished) and event.run_id == previous.run_id and previous.running:
            self._application = None
            if self._end_session_when_finished:
                self._end_session_when_finished = False
                self._conversation.clear()
                self._spoken_conversation.clear()
                self._refresh_spoken()
                self._history_session = self._history_store.new_session()
            if self._close_when_finished:
                self.close()
        self._render()

    @Slot(object)
    def _handle_exchange(self, _turn: object) -> None:
        self._refresh_history()
        self._render()

    def _refresh_history(self) -> None:
        selected_session = None
        current_row = self.history_list.currentRow()
        if 0 <= current_row < len(self._history_records):
            selected_session = self._history_records[current_row].session_id
        self._history_records = self._history_store.records()
        self.history_list.clear()
        selected_row = -1
        for row, record in enumerate(self._history_records):
            self.history_list.addItem(
                f"{record.started_at}\n{record.title}"
            )
            if record.session_id == selected_session:
                selected_row = row
        if self._history_records:
            self.history_list.setCurrentRow(selected_row if selected_row >= 0 else 0)
        else:
            self.history_detail.clear()

    @Slot(int)
    def _show_history(self, row: int) -> None:
        if not 0 <= row < len(self._history_records):
            self.history_detail.clear()
            return
        record = self._history_records[row]
        lines = [f"Interview started: {record.started_at}"]
        for index, entry in enumerate(record.entries, start=1):
            lines.extend(
                (
                    "",
                    f"Question {index}",
                    entry.question,
                    "",
                    f"Answer {index}",
                    entry.answer,
                )
            )
        self.history_detail.setPlainText("\n".join(lines))

        if record.spoken:
            lines.extend(("", "Your actual spoken responses"))
            for entry in record.spoken:
                lines.extend(("", f"Preceding question: {entry.question or '(not yet finalized)'}",
                              entry.answer))
            self.history_detail.setPlainText("\n".join(lines))

    @Slot(int, str)
    def _show_candidate_partial(self, run_id: int, text: str) -> None:
        if run_id == self._snapshot.run_id and self._snapshot.running and not self._snapshot.stopping:
            self.spoken_partial.setText(f"You are saying: {text}" if text else "")

    @Slot(int, str)
    def _show_candidate_state(self, run_id: int, state: str) -> None:
        if run_id == self._snapshot.run_id and self._snapshot.running and not self._snapshot.stopping:
            descriptions = {"recording": "Microphone — listening to your answer",
                            "transcribing": "Microphone — finishing your transcript",
                            "idle": "Microphone — listening for your answer",
                            "error": "Microphone unavailable — reconnecting"}
            self.candidate_state_label.setText(descriptions.get(state, f"Microphone — {state}"))

    @Slot(int)
    def _show_run_spoken(self, run_id: int) -> None:
        if run_id != self._snapshot.run_id or not self._snapshot.running or self._snapshot.stopping:
            return
        self._save_spoken()
        self._refresh_spoken()

    def _save_spoken(self) -> None:
        entries = tuple(HistoryEntry(
            turn.question, turn.text,
            datetime.fromtimestamp(time.time() - time.monotonic() + turn.timestamp)
            .astimezone().isoformat(timespec="seconds"),
        ) for turn in self._spoken_conversation.snapshot())
        try:
            self._history_session.set_spoken(entries)
        except BaseException as error:
            self._show_status(f"Could not save spoken history: {error}")
        self._refresh_history()

    def _refresh_spoken(self) -> None:
        selected = self.spoken_list.currentItem()
        selected_id = selected.data(Qt.ItemDataRole.UserRole) if selected else None
        turns = self._spoken_conversation.snapshot()
        self.spoken_list.clear()
        selected_row = -1
        for row, turn in enumerate(turns):
            self.spoken_list.addItem(f"You said: {turn.text[:100]}")
            self.spoken_list.item(row).setData(Qt.ItemDataRole.UserRole, turn.id)
            if turn.id == selected_id:
                selected_row = row
        self.spoken_list.setCurrentRow(selected_row if selected_row >= 0 else len(turns) - 1)
        self._show_spoken_detail(self.spoken_list.currentRow())

    def _selected_spoken(self) -> SpokenTurn | None:
        item = self.spoken_list.currentItem()
        if item is None:
            return None
        return next((turn for turn in self._spoken_conversation.snapshot()
                     if turn.id == item.data(Qt.ItemDataRole.UserRole)), None)

    @Slot(int)
    def _show_spoken_detail(self, _row: int) -> None:
        turn = self._selected_spoken()
        if hasattr(self, "spoken_detail"):
            self.spoken_detail.setPlainText(turn.text if turn else "")
            self.correct_spoken_button.setEnabled(turn is not None)
            self.discard_spoken_button.setEnabled(turn is not None)

    @Slot()
    def _correct_spoken(self) -> None:
        turn = self._selected_spoken()
        if turn is None:
            return
        run_id = self._snapshot.run_id
        session = self._history_session
        text, accepted = QInputDialog.getMultiLineText(
            self, "Correct your spoken response", "Your words:", turn.text,
        )
        if accepted and run_id == self._snapshot.run_id and session is self._history_session:
            self._replace_spoken(turn.id, text)

    @Slot()
    def _discard_spoken(self) -> None:
        turn = self._selected_spoken()
        if turn is not None:
            self._replace_spoken(turn.id, "")

    def _replace_spoken(self, turn_id: int, text: str) -> None:
        try:
            if self._application is not None:
                changed = self._application.correct_candidate_speech(turn_id, text)
            else:
                changed = self._spoken_conversation.replace(turn_id, text)
        except ValueError as error:
            QMessageBox.warning(self, "Cannot update transcript", str(error))
            return
        if changed:
            self._save_spoken()
            self._refresh_spoken()
            self._show_status("Spoken context updated for the next question.")

    @Slot(int, str)
    def _show_run_evidence(self, run_id: int, text: str) -> None:
        if run_id == self._snapshot.run_id and not self._snapshot.stopping:
            self.sources.setPlainText(text)

    @Slot(int, str)
    def _show_run_status(self, run_id: int, message: str) -> None:
        if run_id == self._snapshot.run_id and not self._snapshot.stopping:
            self._show_status(message)

    def _show_status(self, message: str) -> None:
        self.statusBar().showMessage(message)

    @Slot(int, str)
    def _show_run_partial(self, run_id: int, text: str) -> None:
        if run_id == self._snapshot.run_id and self._snapshot.running and not self._snapshot.stopping:
            self.partial_label.setText(f"Live question: {text}" if text else "Listening…")

    def _render(self) -> None:
        phase = self._snapshot.phase
        labels = {
            PresentationPhase.STOPPED: "Stopped — no audio is being retained",
            PresentationPhase.STOPPING: "Stopping capture…",
            PresentationPhase.STARTING: "Starting local models and capture…",
            PresentationPhase.READY: "Ready — hold the configured control to record",
            PresentationPhase.RECORDING: "Recording",
            PresentationPhase.TRANSCRIBING: "Transcribing locally…",
            PresentationPhase.GENERATING: "Drafting a grounded answer locally…",
            PresentationPhase.RECONNECTING: "Audio output changed — reconnecting…",
            PresentationPhase.ERROR: f"Error — {self._snapshot.error_message or 'Unknown error'}",
        }
        if phase is PresentationPhase.READY and self._microphone_mode:
            labels[phase] = "Ready — hold the configured control while you speak"
        if self._automatic_mode:
            if phase is PresentationPhase.READY:
                labels[phase] = "Listening automatically for the next question"
            elif phase is PresentationPhase.RECORDING:
                labels[phase] = "Listening — interviewer speaking"
            elif phase is PresentationPhase.GENERATING:
                labels[phase] = "Answering — still listening for a continuation"
        self.phase_label.setText(labels[phase])
        active_auto = self._automatic_mode and self._snapshot.running and not self._snapshot.stopping
        self.answer_now_button.setEnabled(active_auto)
        self.cancel_answer_button.setEnabled(active_auto)
        self.setup_group.setEnabled(not self._snapshot.running and not self._download_active)
        self.start_button.setEnabled(
            not self._snapshot.running and not self._hotkey_learning and not self._download_active
        )
        self.interview_done_button.setEnabled(self._snapshot.running)
        if self._snapshot.stopping:
            self.interview_done_button.setEnabled(False)
        self.learn_hotkey_button.setEnabled(
            not self._snapshot.running and not self._hotkey_learning
        )
        turns = self._conversation.snapshot()
        questions = [
            f"Question {index}\n{turn.question}"
            for index, turn in enumerate(turns, start=1)
        ]
        if self._snapshot.transcript and (
            not turns or turns[-1].question != self._snapshot.transcript
        ):
            questions.append(
                f"Question {len(turns) + 1} — awaiting answer\n"
                f"{self._snapshot.transcript}"
            )
        answers = [
            f"Answer {index}\n{turn.answer}"
            for index, turn in enumerate(turns, start=1)
        ]
        if self._snapshot.answer_draft and self._snapshot.answer_busy:
            answers.append(f"Answer — generating…\n{self._snapshot.answer_draft}")
        self.transcript.setPlainText("\n\n".join(questions))
        # Character backgrounds follow each wrapped line, leaving unused space clear.
        question_cursor = QTextCursor(self.transcript.document())
        question_cursor.select(QTextCursor.SelectionType.Document)
        subtitle_format = QTextCharFormat()
        subtitle_format.setForeground(QColor("white"))
        subtitle_format.setBackground(QColor("black"))
        question_cursor.mergeCharFormat(subtitle_format)
        self.answer.setPlainText("\n\n".join(answers))
        answer_cursor = QTextCursor(self.answer.document())
        answer_cursor.select(QTextCursor.SelectionType.Document)
        answer_cursor.mergeCharFormat(subtitle_format)
        self.answer.moveCursor(QTextCursor.MoveOperation.End)
        self.copy_button.setEnabled(bool(turns))
        if not self._snapshot.running or self._snapshot.stopping:
            self.candidate_state_label.setText("Microphone stopped" if not self._snapshot.running else "Stopping microphone…")
            self.spoken_partial.clear()
            self.partial_label.setText("Start Interview to listen to questions and your spoken answers.")
        if self._snapshot.error_message:
            self.statusBar().showMessage(self._snapshot.error_message)

    @Slot()
    def _set_answer_text_size(self, size: int) -> None:
        size = max(12, min(48, size))
        self._answer_text_pixels = size
        self.answer_size_label.setText(f"{size} px")
        self.answer_smaller_button.setEnabled(size > 12)
        self.answer_larger_button.setEnabled(size < 48)
        self.answer.setStyleSheet(f"QTextEdit#answerText {{ font-size: {size}px; }}")

    def _copy_answer(self) -> None:
        turns = self._conversation.snapshot()
        if not turns:
            return
        QApplication.clipboard().setText(turns[-1].answer)
        self._show_status("Answer copied to clipboard")

    def _browse_input(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select input event device", "/dev/input/by-id")
        if path:
            self.input_device.setText(path)

    def _selected_model(self) -> ModelSpec:
        return next(spec for spec in MODEL_CATALOG
                    if spec.id == self.transcription_model.currentData())

    def _model_changed(self, _index: int = 0, *, preserve_path: bool = False) -> None:
        spec = self._selected_model()
        previous = self.transcription_device.currentData()
        self.transcription_device.blockSignals(True)
        self.transcription_device.clear()
        for device in spec.devices:
            self.transcription_device.addItem(
                "CPU" if device == "cpu" else "NVIDIA GPU (CUDA)", device,
            )
        if previous in spec.devices:
            self.transcription_device.setCurrentIndex(self.transcription_device.findData(previous))
        self.transcription_device.blockSignals(False)
        if not preserve_path:
            self.model_path.setText(str(catalog_model_path(spec)))
        self.model_description.setText(
            f"{spec.description}\nDownload: approximately {spec.size_bytes / 1_000_000:.0f} MB. "
            "Moonshine Small on CPU is the current baseline. GPU speed depends on hardware; "
            "Nemotron's GPU runtime downloads with the model. GPU speech detection needs "
            "the separate GPU installation. Selection alone never downloads."
        )
        self._refresh_model_status()

    def _model_device_changed(self, _index: int = 0) -> None:
        self._refresh_model_status()

    def _refresh_model_status(self, _text: str = "") -> None:
        spec = self._selected_model()
        device = str(self.transcription_device.currentData())
        installed = is_installed(spec, device=device)
        selected = Path(self.model_path.text().strip())
        managed = selected == catalog_model_path(spec)
        if not managed:
            status = ("Custom model path exists; compatibility checked at Start Interview."
                      if selected.exists() else "Custom model path is missing. Browse or download a model.")
        else:
            status = ("Installed locally — ready to select." if installed else
                      "Model or runtime missing — download before starting.")
        self.model_status.setText(status)
        self.download_model_button.setText("Use installed model" if installed else "Download selected model")

    @Slot()
    def _download_model(self) -> None:
        if self._download_active or self._snapshot.running or self._window_closed:
            return
        spec = self._selected_model()
        device = str(self.transcription_device.currentData())
        if is_installed(spec, device=device):
            self.model_path.setText(str(catalog_model_path(spec)))
            self.download_status.setText("Using the model already installed on this computer.")
            return
        self._download_active = True
        self._download_generation += 1
        generation = self._download_generation
        cancel = threading.Event()
        self._download_cancel = cancel
        bridge = self._bridge
        self.download_status.setText(f"Downloading {spec.label}…")
        self.download_progress.setValue(0)
        self.download_progress.show()
        self.cancel_download_button.setEnabled(True)
        self.cancel_download_button.show()
        self._render()

        def progress(done: int, total: int, label: str) -> None:
            if not cancel.is_set():
                try:
                    bridge.model_progress.emit(generation, done, total, label)
                except RuntimeError:  # The window may have been destroyed.
                    cancel.set()

        def run() -> None:
            path, error = "", ""
            try:
                path = str(install_model(spec.id, device=device, progress=progress, cancel_event=cancel))
            except Exception as failure:
                error = str(failure)
            try:
                bridge.model_finished.emit(generation, path, error)
            except RuntimeError:
                pass

        self._download_worker = threading.Thread(target=run, name="model-download", daemon=True)
        self._download_worker.start()

    @Slot()
    def _cancel_model_download(self) -> None:
        self._download_cancel.set()
        self.download_status.setText("Cancelling download…")
        self.cancel_download_button.setEnabled(False)

    @Slot(int, object, object, str)
    def _model_download_progress(self, generation: int, done: int, total: int, label: str) -> None:
        if (self._window_closed or generation != self._download_generation
                or not self._download_active or self._download_cancel.is_set()):
            return
        self.download_progress.setRange(0, 100 if total > 0 else 0)
        if total > 0:
            self.download_progress.setValue(min(100, max(0, int(done * 100 / total))))
        self.download_status.setText(f"Downloading {label}: {done / 1_000_000:.1f} MB")

    @Slot(int, str, str)
    def _model_download_finished(self, generation: int, path: str, error: str) -> None:
        if self._window_closed or generation != self._download_generation or not self._download_active:
            return
        self._download_active = False
        self.download_progress.hide()
        self.cancel_download_button.hide()
        if self._download_cancel.is_set():
            self.download_status.setText("Download cancelled. You can retry when ready.")
        elif error or not path:
            self.download_status.setText(f"Download failed: {error or 'No model was installed.'}")
        else:
            self.model_path.setText(path)
            self.download_status.setText("Download complete. Model stored locally for future use.")
        self._refresh_model_status()
        self._render()

    def _browse_model(self) -> None:
        if self._selected_model().backend == "moonshine":
            path = QFileDialog.getExistingDirectory(self, "Select Moonshine model directory")
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Select Nemotron model", "", "GGUF (*.gguf)")
        if path:
            self.model_path.setText(path)

    def _browse_resume(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select resume", "", "Text (*.txt *.md)")
        if path:
            self.resume.setText(path)

    def _provider_changed(self) -> None:
        technical_mode = self.enable_technical_mode.isChecked()
        # Disable the complete path rows, including their Browse buttons.
        answers_row = self.technical_answers.parentWidget()
        library_row = self.technical_library.parentWidget()
        assert answers_row is not None and library_row is not None
        answers_row.setEnabled(technical_mode)
        library_row.setEnabled(not technical_mode)
        cloud = self.answer_provider.currentData() == AnswerProvider.OPENAI.value
        self.qwen_url.setEnabled(not cloud)
        self.qwen_model.setEnabled(not cloud)
        self.openai_key_file.setEnabled(cloud)
        self.openai_model.setEnabled(cloud)
        self.provider_help.setText(
            "OpenAI receives selected resume/background passages, questions and conversation "
            "context. Audio transcription and document search stay local. "
            "Uses standard API billing with reasoning disabled. Luna models are the fast, "
            "low-cost tier; Terra and Sol cost roughly 10–20 times more per answer. "
            "A model change applies at the next Start Interview."
            if cloud else "Answers use your local Qwen server. No OpenAI API requests."
        )
        self.provider_status.setText(
            (f"Answers: OpenAI {self.openai_model.currentText()} · selected context and "
             "transcripts sent to OpenAI"
             if cloud else "Answers: Local Qwen")
            + (" · Technical mode" if technical_mode else "")
        )

    def _browse_technical_answers(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select prepared technical answers", "", "Text (*.txt *.md)"
        )
        if path:
            self.technical_answers.setText(path)

    def _browse_openai_key(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select OpenAI key file", "", "Key files (*.txt *.rtf);;All files (*)"
        )
        if path:
            self.openai_key_file.setText(path)

    def _browse_technical_library(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select local technical library")
        if path:
            self.technical_library.setText(path)

    def _add_context(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Add interview guides and background", "", "Text (*.txt *.md)")
        self.context_paths.addItems(paths)

    def _remove_context(self) -> None:
        for item in self.context_paths.selectedItems():
            self.context_paths.takeItem(self.context_paths.row(item))

    def closeEvent(self, event: QCloseEvent) -> None:
        self._window_closed = True
        self._download_cancel.set()
        if self._hotkey_learning:
            self._hotkey_cancel.set()
        application = self._application
        if application is None or not self._snapshot.running:
            event.accept()
            return
        self._close_when_finished = True
        self._stop()
        event.ignore()


def run_gui(
    *, resume_path: Path | None = None,
    context_paths: tuple[Path, ...] = (),
    answer_provider: str = "local",
    openai_api_key_file: Path | None = None,
    openai_model: str | None = None,
    technical_answers_path: Path | None = None,
) -> int:
    application = QApplication.instance() or QApplication([])
    application.setApplicationName("Interview Helper")
    window = MainWindow(
        resume_path=resume_path,
        context_paths=context_paths,
        answer_provider=answer_provider,
        openai_api_key_file=openai_api_key_file,
        openai_model=openai_model,
        technical_answers_path=technical_answers_path,
    )
    window.show()
    return int(application.exec())
