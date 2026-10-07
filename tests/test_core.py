import threading

from interview_helper.core import HoldToTranscribe, RuntimeState, Transcript


class Session:
    def __init__(self, complete_callback: object) -> None:
        self.complete_callback = complete_callback
        self.chunks: list[tuple[bytes, int]] = []
        self.finished = 0
        self.closed = 0

    def add_pcm(self, pcm: bytes, sample_rate: int) -> None:
        self.chunks.append((pcm, sample_rate))

    def finish(self) -> None:
        self.finished += 1
        self.complete_callback(Transcript("  hello there  ", 0.12))

    def close(self) -> None:
        self.closed += 1


class Transcriber:
    def __init__(self) -> None:
        self.sessions: list[Session] = []

    def open_stream(self, **callbacks: object) -> Session:
        session = Session(callbacks["complete_callback"])
        self.sessions.append(session)
        return session


def test_hold_records_only_between_press_and_release() -> None:
    transcriber = Transcriber()
    transcripts: list[Transcript] = []
    states: list[RuntimeState] = []
    engine = HoldToTranscribe(
        transcriber,
        sample_rate=16_000,
        transcript_callback=transcripts.append,
        state_callback=states.append,
    )

    engine.add_pcm(b"idle")
    assert engine.press()
    engine.add_pcm(b"held")
    assert engine.release()
    assert engine.wait_until_idle()
    engine.add_pcm(b"idle again")

    assert transcriber.sessions[0].chunks == [(b"held", 16_000)]
    assert transcriber.sessions[0].finished == 1
    assert transcripts == [Transcript("hello there", 0.12)]
    assert states == [
        RuntimeState.RECORDING,
        RuntimeState.TRANSCRIBING,
        RuntimeState.IDLE,
    ]


def test_repeat_press_and_stray_release_do_not_duplicate_turns() -> None:
    transcriber = Transcriber()
    engine = HoldToTranscribe(
        transcriber,
        sample_rate=16_000,
        transcript_callback=lambda _result: None,
    )

    assert not engine.release()
    assert engine.press()
    assert not engine.press()
    engine.add_pcm(b"pcm0")
    transcriber.sessions[0].complete_callback = lambda _result: None
    assert engine.release()
    assert not engine.release()
    assert engine.wait_until_idle()

    assert len(transcriber.sessions) == 1
    assert transcriber.sessions[0].finished == 1


def test_release_without_audio_closes_without_transcribing() -> None:
    transcriber = Transcriber()
    transcripts: list[Transcript] = []
    engine = HoldToTranscribe(
        transcriber,
        sample_rate=16_000,
        transcript_callback=transcripts.append,
    )

    assert engine.press()
    assert engine.release()
    assert engine.wait_until_idle()

    assert transcriber.sessions[0].finished == 0
    assert transcriber.sessions[0].closed == 1
    assert transcripts == []


def test_transcription_finishes_off_the_input_thread() -> None:
    started = threading.Event()
    allow_finish = threading.Event()

    class BlockingSession(Session):
        def finish(self) -> None:
            started.set()
            assert allow_finish.wait(timeout=2)
            super().finish()

    class BlockingTranscriber(Transcriber):
        def open_stream(self, **callbacks: object) -> Session:
            session = BlockingSession(callbacks["complete_callback"])
            self.sessions.append(session)
            return session

    transcriber = BlockingTranscriber()
    engine = HoldToTranscribe(
        transcriber,
        sample_rate=16_000,
        transcript_callback=lambda _result: None,
    )
    engine.press()
    engine.add_pcm(b"pcm0")
    transcriber.sessions[0].complete_callback = lambda _result: None

    assert engine.release()
    assert started.wait(timeout=1)
    assert engine.state is RuntimeState.TRANSCRIBING
    allow_finish.set()
    assert engine.wait_until_idle()
