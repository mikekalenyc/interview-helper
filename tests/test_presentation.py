from interview_helper.core import RuntimeState
from interview_helper.presentation import (
    AnswerCompleted,
    AnswerStarted,
    CaptureStatusChanged,
    ErrorReported,
    PresentationPhase,
    PresentationSnapshot,
    RunFinished,
    RuntimeStateChanged,
    StartRequested,
    StopRequested,
    TranscriptFinalized,
    reduce_presentation,
)


def dispatch(
    snapshot: PresentationSnapshot, *events: object
) -> PresentationSnapshot:
    for event in events:
        assert isinstance(
            event,
            (
                StartRequested,
                StopRequested,
                RunFinished,
                CaptureStatusChanged,
                RuntimeStateChanged,
                TranscriptFinalized,
                AnswerStarted,
                AnswerCompleted,
                ErrorReported,
            ),
        )
        snapshot = reduce_presentation(snapshot, event)
    return snapshot


def test_startup_becomes_ready_only_after_capture_reports_ready() -> None:
    initial = PresentationSnapshot()

    starting = reduce_presentation(initial, StartRequested())
    ready = reduce_presentation(
        starting, CaptureStatusChanged(run_id=starting.run_id, ready=True)
    )

    assert initial.phase is PresentationPhase.STOPPED
    assert starting.phase is PresentationPhase.STARTING
    assert starting.capture_ready is False
    assert ready.phase is PresentationPhase.READY
    assert ready.capture_ready is True


def test_asr_activity_takes_precedence_over_answer_generation() -> None:
    snapshot = dispatch(
        PresentationSnapshot(),
        StartRequested(),
        CaptureStatusChanged(run_id=1, ready=True),
        AnswerStarted(run_id=1),
    )
    assert snapshot.answer_busy is True
    assert snapshot.phase is PresentationPhase.GENERATING

    recording = reduce_presentation(
        snapshot, RuntimeStateChanged(run_id=1, state=RuntimeState.RECORDING)
    )
    transcribing = reduce_presentation(
        recording, RuntimeStateChanged(run_id=1, state=RuntimeState.TRANSCRIBING)
    )
    generating = reduce_presentation(
        transcribing, RuntimeStateChanged(run_id=1, state=RuntimeState.IDLE)
    )

    assert recording.phase is PresentationPhase.RECORDING
    assert transcribing.phase is PresentationPhase.TRANSCRIBING
    assert generating.phase is PresentationPhase.GENERATING


def test_capture_loss_and_recovery_are_visible_without_changing_other_state() -> None:
    ready = dispatch(
        PresentationSnapshot(),
        StartRequested(),
        CaptureStatusChanged(run_id=1, ready=True),
        TranscriptFinalized(run_id=1, text="  Tell me about automation.  "),
    )

    reconnecting = reduce_presentation(
        ready, CaptureStatusChanged(run_id=1, ready=False)
    )
    recovered = reduce_presentation(
        reconnecting, CaptureStatusChanged(run_id=1, ready=True)
    )

    assert reconnecting.phase is PresentationPhase.RECONNECTING
    assert reconnecting.transcript == "Tell me about automation."
    assert recovered.phase is PresentationPhase.READY


def test_error_is_sticky_and_has_highest_precedence() -> None:
    running = dispatch(
        PresentationSnapshot(),
        StartRequested(),
        CaptureStatusChanged(run_id=1, ready=True),
        AnswerStarted(run_id=1),
        RuntimeStateChanged(run_id=1, state=RuntimeState.RECORDING),
    )

    failed = reduce_presentation(
        running, ErrorReported(run_id=1, message="  input device disconnected  ")
    )
    later_idle = reduce_presentation(
        failed, RuntimeStateChanged(run_id=1, state=RuntimeState.IDLE)
    )

    assert failed.phase is PresentationPhase.ERROR
    assert failed.error_message == "input device disconnected"
    assert failed.answer_busy is False
    assert later_idle.phase is PresentationPhase.ERROR


def test_runtime_error_is_visible_before_error_details_arrive() -> None:
    starting = reduce_presentation(PresentationSnapshot(), StartRequested())

    failed = reduce_presentation(
        starting, RuntimeStateChanged(run_id=starting.run_id, state=RuntimeState.ERROR)
    )

    assert failed.phase is PresentationPhase.ERROR
    assert failed.error_message is None


def test_new_transcript_clears_answer_and_answer_completion_ends_busy_state() -> None:
    snapshot = dispatch(
        PresentationSnapshot(),
        StartRequested(),
        CaptureStatusChanged(run_id=1, ready=True),
        TranscriptFinalized(run_id=1, text="First question"),
        AnswerStarted(run_id=1),
        AnswerCompleted(run_id=1, text="  First answer  "),
        TranscriptFinalized(run_id=1, text="  Second question  "),
    )

    assert snapshot.transcript == "Second question"
    assert snapshot.answer == ""
    assert snapshot.answer_busy is False
    assert snapshot.phase is PresentationPhase.READY


def test_stop_resets_content_and_invalidates_callbacks_from_old_run() -> None:
    active = dispatch(
        PresentationSnapshot(),
        StartRequested(),
        CaptureStatusChanged(run_id=1, ready=True),
        TranscriptFinalized(run_id=1, text="Private question"),
        AnswerStarted(run_id=1),
    )

    stopped = reduce_presentation(active, StopRequested())
    stale_result = reduce_presentation(
        stopped, AnswerCompleted(run_id=1, text="Stale private answer")
    )

    assert stopped.phase is PresentationPhase.STOPPING
    assert stopped.stopping is True
    assert stopped.answer_busy is False
    assert stale_result is stopped

    finished = reduce_presentation(stopped, RunFinished(run_id=1))
    assert finished.phase is PresentationPhase.STOPPED
    assert finished.capture_ready is False
    assert finished.transcript == ""
    assert finished.answer == ""


def test_previous_run_callback_is_ignored_after_restart() -> None:
    first_run = reduce_presentation(PresentationSnapshot(), StartRequested())
    stopping = reduce_presentation(first_run, StopRequested())
    stopped = reduce_presentation(stopping, RunFinished(first_run.run_id))
    second_run = reduce_presentation(stopped, StartRequested())

    stale = reduce_presentation(
        second_run, TranscriptFinalized(run_id=first_run.run_id, text="stale")
    )

    assert second_run.run_id != first_run.run_id
    assert stale is second_run
    assert stale.transcript == ""


def test_duplicate_start_and_stop_requests_are_idempotent() -> None:
    initial = PresentationSnapshot()
    started = reduce_presentation(initial, StartRequested())

    assert reduce_presentation(started, StartRequested()) is started

    stopped = reduce_presentation(started, StopRequested())
    assert reduce_presentation(stopped, StopRequested()) is stopped


def test_finished_run_retains_error_and_allows_restart() -> None:
    snapshot = reduce_presentation(PresentationSnapshot(), StartRequested())
    run_id = snapshot.run_id
    snapshot = reduce_presentation(snapshot, ErrorReported(run_id, "capture failed"))

    finished = reduce_presentation(snapshot, RunFinished(run_id))

    assert not finished.running
    assert finished.phase is PresentationPhase.ERROR
    assert finished.error_message == "capture failed"
    restarted = reduce_presentation(finished, StartRequested())
    assert restarted.running
    assert restarted.error_message is None


def test_runtime_closed_does_not_clear_error_before_run_finishes() -> None:
    snapshot = reduce_presentation(PresentationSnapshot(), StartRequested())
    run_id = snapshot.run_id
    snapshot = reduce_presentation(snapshot, ErrorReported(run_id, "input failed"))

    closed = reduce_presentation(
        snapshot, RuntimeStateChanged(run_id, RuntimeState.CLOSED)
    )
    finished = reduce_presentation(closed, RunFinished(run_id))

    assert closed.error_message == "input failed"
    assert finished.error_message == "input failed"
    assert finished.phase is PresentationPhase.ERROR


def test_answer_draft_is_temporary_and_late_updates_are_ignored() -> None:
    from interview_helper.presentation import AnswerProgress

    snapshot = reduce_presentation(PresentationSnapshot(), StartRequested())
    run_id = snapshot.run_id
    snapshot = reduce_presentation(snapshot, AnswerStarted(run_id))
    snapshot = reduce_presentation(snapshot, AnswerProgress(run_id, "I checked"))
    assert snapshot.answer_draft == "I checked" and snapshot.answer == ""
    assert snapshot.answer_busy
    done = reduce_presentation(snapshot, AnswerCompleted(run_id, "I checked the routes."))
    assert done.answer == "I checked the routes." and done.answer_draft == ""
    assert reduce_presentation(done, AnswerProgress(run_id, "late")) == done
    failed = reduce_presentation(snapshot, ErrorReported(run_id, "stream failed"))
    assert failed.answer_draft == "" and failed.answer == ""
    stopped = reduce_presentation(snapshot, StopRequested())
    assert stopped.answer_draft == ""
    assert reduce_presentation(stopped, AnswerProgress(run_id, "late")) == stopped
    restarted = reduce_presentation(reduce_presentation(stopped, RunFinished(run_id)), StartRequested())
    assert reduce_presentation(restarted, AnswerProgress(run_id, "old run")) == restarted
