from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from interview_helper.turn_detection import SileroSpeechDetector, SmartTurnDetector


class Session:
    def __init__(self, probabilities: list[float]) -> None:
        self.probabilities = iter(probabilities)
        self.inputs: list[dict[str, Any]] = []

    def run(self, output_names: None, input_feed: dict[str, Any]) -> list[Any]:
        self.inputs.append({k: v.copy() for k, v in input_feed.items()})
        result = [np.array([[next(self.probabilities)]], dtype=np.float32)]
        if "state" in input_feed:
            result.append(input_feed["state"] + 1)
        return result


class Extractor:
    def __init__(self) -> None:
        self.audio: Any = None
        self.kwargs: Any = None

    def __call__(self, audio: Any, **kwargs: Any) -> Any:
        self.audio, self.kwargs = audio.copy(), kwargs
        return SimpleNamespace(input_features=np.ones((1, 80, 800), dtype=np.float32))


def test_silero_remainder_context_state_and_any_speech() -> None:
    session = Session([0.8, 0.1, 0.2])
    detector = SileroSpeechDetector(session=session)
    audio = np.arange(1536, dtype=np.float32)
    assert detector.is_speech(audio[:1280].tobytes())  # 80 ms; two frames plus remainder
    assert len(session.inputs) == 2  # no early return on the first speech frame
    np.testing.assert_array_equal(session.inputs[0]["input"][0, :64], np.zeros(64))
    np.testing.assert_array_equal(session.inputs[1]["input"][0, :64], audio[448:512])
    assert np.all(session.inputs[1]["state"] == 1)
    assert not detector.is_speech(audio[1280:].tobytes())
    np.testing.assert_array_equal(session.inputs[2]["input"][0, 64:], audio[1024:])
    assert session.inputs[0]["sr"].item() == 16000
    detector.reset()
    assert not detector.is_speech(b"")
    assert detector._remainder.size == 0
    assert np.all(detector._state == 0)
    assert np.all(detector._context == 0)


def test_subframe_calls_retain_previous_speech_decision() -> None:
    detector = SileroSpeechDetector(session=Session([0.5]))
    assert not detector.is_speech(np.zeros(256, dtype=np.float32).tobytes())
    assert detector.is_speech(np.zeros(256, dtype=np.float32).tobytes())
    assert detector.is_speech(np.zeros(16, dtype=np.float32).tobytes())


def test_smart_turn_left_padding_normalization_and_threshold() -> None:
    session, extractor = Session([0.5, 0.51]), Extractor()
    detector = SmartTurnDetector(session=session, feature_extractor=extractor)
    assert not detector.is_complete(b"")
    assert not detector.is_complete(np.ones(16000, dtype=np.float32).tobytes())
    assert len(extractor.audio) == 128000
    assert np.all(extractor.audio[:112000] == 0)
    assert np.all(extractor.audio[112000:] == 1)
    assert extractor.kwargs["do_normalize"] is True
    assert extractor.kwargs["max_length"] == 128000
    assert session.inputs[0]["input_features"].shape == (1, 80, 800)
    long = np.arange(160000, dtype=np.float32)
    assert detector.is_complete(long.tobytes())
    np.testing.assert_array_equal(extractor.audio, long[-128000:])


@pytest.mark.parametrize("detector_type", [SileroSpeechDetector, SmartTurnDetector])
def test_missing_models_are_actionable(tmp_path: Path, detector_type: Any) -> None:
    with pytest.raises(RuntimeError, match="interview-helper-install-turn-models"):
        detector_type(tmp_path / "missing.onnx")


@pytest.mark.parametrize("pcm", [b"x", np.array([np.nan], dtype=np.float32).tobytes()])
def test_invalid_pcm_fails_before_inference(pcm: bytes) -> None:
    detector = SileroSpeechDetector(session=Session([]))
    with pytest.raises(ValueError):
        detector.is_speech(pcm)
    smart = SmartTurnDetector(session=Session([]), feature_extractor=Extractor())
    with pytest.raises(ValueError):
        smart.is_complete(pcm)
