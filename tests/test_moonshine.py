from types import SimpleNamespace

import numpy as np
import pytest

from interview_helper.config import MoonshineConfig
from interview_helper.moonshine import MoonshineTranscriber


class LineTextChanged:
    def __init__(self, text: str) -> None:
        self.line = SimpleNamespace(text=text, last_transcription_latency_ms=80)


class LineCompleted:
    def __init__(self, text: str) -> None:
        self.line = SimpleNamespace(text=text, last_transcription_latency_ms=175)


class Stream:
    def __init__(self) -> None:
        self.listener = None
        self.started = False
        self.stopped = False
        self.closed = False
        self.audio: list[tuple[list[float], int]] = []

    def add_listener(self, listener: object) -> None:
        self.listener = listener

    def start(self) -> None:
        self.started = True

    def add_audio(self, samples: list[float], sample_rate: int) -> None:
        self.audio.append((samples, sample_rate))
        self.listener(LineTextChanged("project"))

    def stop(self) -> None:
        self.stopped = True
        self.listener(LineCompleted("project deadline is Friday"))

    def close(self) -> None:
        self.closed = True


class Model:
    def __init__(self) -> None:
        self.stream = Stream()
        self.keyterms = None
        self.interval = None

    def set_keyterms(self, keyterms: object) -> None:
        self.keyterms = keyterms

    def create_stream(self, *, update_interval: float) -> Stream:
        self.interval = update_interval
        return self.stream


def test_configurable_vocabulary_and_explicit_release_finalization(tmp_path: object) -> None:
    model = Model()
    transcriber = MoonshineTranscriber(
        MoonshineConfig(
            model_path=tmp_path,
            keyterms=("OpenAI", "CACI"),
        ),
        transcriber=model,
    )
    partials: list[str] = []
    completed: list[object] = []
    session = transcriber.open_stream(
        partial_callback=partials.append,
        complete_callback=completed.append,
    )
    pcm = np.array([0.1, -0.2], dtype="<f4").tobytes()

    session.add_pcm(pcm, 16_000)
    session.finish()

    assert model.keyterms == ("OpenAI", "CACI")
    assert model.interval == 0.2
    assert model.stream.started and model.stream.stopped and model.stream.closed
    assert partials == ["project", "project deadline is Friday"]
    assert completed[0].text == "project deadline is Friday"
    assert completed[0].transcription_seconds == 0.175
    assert model.stream.audio == [([pytest.approx(0.1), pytest.approx(-0.2)], 16_000)]
