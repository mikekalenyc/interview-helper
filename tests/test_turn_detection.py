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


@pytest.fixture
def fake_ort(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    import json
    import sys

    state = SimpleNamespace(available=["CUDAExecutionProvider", "CPUExecutionProvider"],
                            actual=["CUDAExecutionProvider", "CPUExecutionProvider"],
                            placement=["CUDAExecutionProvider", "CPUExecutionProvider"],
                            operation="MatMul",
                            calls=[], profiles=[], fallback_disabled=False)

    class Runtime:
        def __init__(self, path: str, *, sess_options: Any, providers: Any) -> None:
            state.calls.append((path, sess_options, providers))
            self.options = sess_options

        def get_providers(self) -> list[str]:
            return state.actual

        def disable_fallback(self) -> None:
            state.fallback_disabled = True

        def run(self, outputs: Any, inputs: Any) -> list[Any]:
            state.probe = inputs
            return [np.array([[0.7]]), np.zeros((2, 1, 128))]

        def end_profiling(self) -> str:
            path = Path(self.options.profile_file_prefix + '.json')
            path.write_text(json.dumps([
                {"cat": "Node", "args": {"provider": provider, "op_name": state.operation}}
                for provider in state.placement
            ]))
            state.profiles.append(path)
            return str(path)

    module = SimpleNamespace(InferenceSession=Runtime, SessionOptions=SimpleNamespace,
                             ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL=1),
                             get_available_providers=lambda: state.available)
    monkeypatch.setitem(sys.modules, "onnxruntime", module)
    state.path = tmp_path / "model.onnx"
    state.path.write_bytes(b"fake")
    return state


@pytest.mark.parametrize("detector_type", [SileroSpeechDetector, SmartTurnDetector])
def test_cuda_probes_real_placement_and_reports_mixed_provider(fake_ort: Any, detector_type: Any) -> None:
    kwargs = {"feature_extractor": Extractor()} if detector_type is SmartTurnDetector else {}
    detector = detector_type(fake_ort.path, device="cuda", device_index=2, **kwargs)
    assert fake_ort.fallback_disabled
    assert fake_ort.calls[0][2] == [("CUDAExecutionProvider", {"device_id": 2}), "CPUExecutionProvider"]
    assert detector.provider_node_counts == {"CUDAExecutionProvider": 1, "CPUExecutionProvider": 1}
    assert detector.execution_providers == ("CUDAExecutionProvider", "CPUExecutionProvider")
    assert all(not path.exists() for path in fake_ort.profiles)
    if detector_type is SileroSpeechDetector:
        assert fake_ort.probe["input"].shape == (1, 576)
        assert np.all(detector._state == 0)  # synthetic probe never changes live state
    else:
        assert fake_ort.probe["input_features"].shape == (1, 80, 800)


def test_cpu_default_does_not_enable_profiling(fake_ort: Any) -> None:
    SileroSpeechDetector(fake_ort.path)
    assert fake_ort.calls[0][2] == ["CPUExecutionProvider"]
    assert not hasattr(fake_ort.calls[0][1], "enable_profiling")
    assert not fake_ort.fallback_disabled


@pytest.mark.parametrize("failure", ["unavailable", "fallback", "no_cuda_nodes", "copy_only"])
def test_cuda_never_silently_runs_cpu_only(fake_ort: Any, failure: str) -> None:
    if failure == "unavailable":
        fake_ort.available = ["CPUExecutionProvider"]
    elif failure == "fallback":
        fake_ort.actual = ["CPUExecutionProvider"]
    elif failure == "no_cuda_nodes":
        fake_ort.placement = ["CPUExecutionProvider"]
    else:
        fake_ort.operation = "MemcpyFromHost"
    with pytest.raises(RuntimeError, match="unavailable|fell back|no CUDA compute"):
        SileroSpeechDetector(fake_ort.path, device="cuda")
    assert all(not path.exists() for path in fake_ort.profiles)


@pytest.mark.parametrize("detector_type", [SileroSpeechDetector, SmartTurnDetector])
@pytest.mark.parametrize("options", [{"device": "auto"}, {"device_index": -1}])
def test_invalid_device_rejected_even_for_injected_sessions(detector_type: Any, options: Any) -> None:
    with pytest.raises(ValueError, match="device"):
        detector_type(session=Session([]), **options)


def test_cuda_missing_model_message_includes_installer_option(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="install-turn-models --device cuda"):
        SmartTurnDetector(tmp_path / "missing.onnx", device="cuda")
