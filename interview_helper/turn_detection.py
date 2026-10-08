"""Optional, local CPU/CUDA speech and end-of-turn inference; never downloads at runtime.

Smart Turn preprocessing follows pipecat-ai/smart-turn inference.py/audio_utils.py
(BSD-2-Clause, Copyright (c) 2024–2025 Daily). Silero recurrent/context handling
follows snakers4/silero-vad v6.2 utils_vad.py (MIT, Silero Team). The explicit
installer includes the upstream notices alongside the model files.

Upstream notices retained for the adapted preprocessing/state handling:

BSD 2-Clause License

Copyright (c) 2024–2025, Daily

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

MIT License

Copyright (c) 2020-present Silero Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""
from __future__ import annotations

from collections import Counter
from collections.abc import Callable
import json
import tempfile
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray

DEFAULT_MODEL_DIRECTORY = Path.home() / ".local/share/interview-helper/turn-models"
DEFAULT_SILERO_MODEL = DEFAULT_MODEL_DIRECTORY / "silero-v6.2.onnx"
DEFAULT_SMART_TURN_MODEL = DEFAULT_MODEL_DIRECTORY / "smart-turn-v3.2-cpu.onnx"
DEFAULT_SMART_TURN_CUDA_MODEL = DEFAULT_MODEL_DIRECTORY / "smart-turn-v3.2-gpu.onnx"


class InferenceSession(Protocol):
    def run(self, output_names: None, input_feed: dict[str, Any]) -> list[Any]: ...


class FeatureExtractor(Protocol):
    def __call__(self, audio: NDArray[np.float32], **kwargs: Any) -> Any: ...


def _validate_device(device: str, device_index: int) -> None:
    if (device not in {"cpu", "cuda"} or isinstance(device_index, bool)
            or not isinstance(device_index, int) or device_index < 0):
        raise ValueError("Detector device must be cpu or cuda with a nonnegative device index")


class _RuntimeSession:
    """Expose measured placement without claiming all operations execute on CUDA."""

    def __init__(self, session: Any, counts: dict[str, int]) -> None:
        self._session = session
        self.provider_node_counts = counts
        self.execution_providers = tuple(session.get_providers())

    def run(self, output_names: None, input_feed: dict[str, Any]) -> list[Any]:
        return cast(list[Any], self._session.run(output_names, input_feed))


def _session(path: Path, device: str, device_index: int,
             probe: dict[str, Any]) -> InferenceSession:
    _validate_device(device, device_index)
    if not path.is_file():
        command = "interview-helper-install-turn-models" + (" --device cuda" if device == "cuda" else "")
        raise RuntimeError(f"Automatic listening model is missing: {path}. Run {command}.")
    try:
        import onnxruntime as ort
    except ImportError as error:
        package = "onnxruntime-gpu" if device == "cuda" else "onnxruntime"
        raise RuntimeError(
            f"Automatic listening needs {package}. Install the matching automatic dependencies "
            "in the app environment; CPU and GPU ONNX Runtime packages must not coexist."
        ) from error
    if device == "cuda" and "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("CUDAExecutionProvider is unavailable; use a compatible onnxruntime-gpu environment")
    if device == "cuda":
        preload = getattr(ort, "preload_dlls", None)
        if preload is not None:
            preload(directory="")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    if device == "cpu":
        session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        return _RuntimeSession(session, {})
    # Profile exactly one synthetic initialization call, never recorded audio.
    # CPU shape/control operators are allowed and reported; zero CUDA compute
    # kernels is an error even when the CUDA provider was successfully registered.
    with tempfile.TemporaryDirectory(prefix="interview-helper-ort-") as directory:
        options.enable_profiling = True
        options.profile_file_prefix = str(Path(directory) / "placement")
        try:
            session = ort.InferenceSession(
                str(path), sess_options=options,
                providers=[("CUDAExecutionProvider", {"device_id": device_index}), "CPUExecutionProvider"],
            )
            session.disable_fallback()
            if "CUDAExecutionProvider" not in session.get_providers():
                raise RuntimeError("ONNX Runtime fell back to CPU while initializing CUDA")
            try:
                session.run(None, probe)
            finally:
                profile_path = session.end_profiling()
            events = json.loads(Path(profile_path).read_text(encoding="utf-8"))
            counts: Counter[str] = Counter()
            cuda_compute = 0
            for event in events:
                args = event.get("args", {})
                provider = args.get("provider")
                if event.get("cat") == "Node" and provider:
                    counts[provider] += 1
                    if provider == "CUDAExecutionProvider" and not str(args.get("op_name", "")).startswith("Memcpy"):
                        cuda_compute += 1
            if not cuda_compute:
                raise RuntimeError("CUDA requested but initialization profiling found no CUDA compute operators")
            return _RuntimeSession(session, dict(counts))
        except Exception as error:
            raise RuntimeError(f"Cannot initialize detector on CUDA device {device_index}: {error}") from error


def _samples(pcm: bytes) -> NDArray[np.float32]:
    if len(pcm) % 4:
        raise ValueError("Expected mono float32 little-endian PCM at 16 kHz")
    samples = np.frombuffer(pcm, dtype="<f4")
    if not np.isfinite(samples).all():
        raise ValueError("PCM contains non-finite samples")
    return samples


class SileroSpeechDetector:
    """Stateful 512-sample Silero frames, including its 64-sample context.

    Each call consumes all complete frames and returns whether ANY is speech.
    Remainders are retained; calls too short for a frame retain the last decision.
    One instance belongs to one capture stream; reset at stream discontinuities.
    """

    def __init__(self, model_path: Path | None = None, *,
                 session: InferenceSession | None = None,
                 device: str = "cpu", device_index: int = 0) -> None:
        _validate_device(device, device_index)
        self._session = session if session is not None else _session(
            model_path or DEFAULT_SILERO_MODEL, device, device_index,
            {"input": np.zeros((1, 576), dtype=np.float32),
             "state": np.zeros((2, 1, 128), dtype=np.float32), "sr": np.array(16000, dtype=np.int64)},
        )
        self.provider_node_counts = dict(getattr(self._session, "provider_node_counts", {}))
        self.execution_providers = tuple(getattr(self._session, "execution_providers", ()))
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, 64), dtype=np.float32)
        self._remainder = np.empty(0, dtype=np.float32)
        self._last_speech = False

    def is_speech(self, pcm: bytes) -> bool:
        samples = np.concatenate((self._remainder, _samples(pcm)))
        count = len(samples) // 512
        speech = False
        for index in range(count):
            frame = samples[index * 512:(index + 1) * 512].reshape(1, 512)
            inputs = np.concatenate((self._context, frame), axis=1)
            output, self._state = self._session.run(None, {
                "input": inputs, "state": self._state,
                "sr": np.array(16000, dtype=np.int64),
            })
            self._context = frame[:, -64:].copy()
            self._last_speech = float(np.asarray(output).flat[0]) >= 0.5
            speech = speech or self._last_speech
        self._remainder = samples[count * 512:].copy()
        return speech if count else self._last_speech


class SmartTurnDetector:
    """Estimate completion from the last eight seconds; no intent classification."""

    def __init__(self, model_path: Path | None = None, *,
                 session: InferenceSession | None = None,
                 feature_extractor: FeatureExtractor | None = None,
                 device: str = "cpu", device_index: int = 0) -> None:
        _validate_device(device, device_index)
        default_model = DEFAULT_SMART_TURN_CUDA_MODEL if device == "cuda" else DEFAULT_SMART_TURN_MODEL
        self._session = session if session is not None else _session(
            model_path or default_model, device, device_index,
            {"input_features": np.zeros((1, 80, 800), dtype=np.float32)},
        )
        self.provider_node_counts = dict(getattr(self._session, "provider_node_counts", {}))
        self.execution_providers = tuple(getattr(self._session, "execution_providers", ()))
        if feature_extractor is None:
            try:
                from transformers import WhisperFeatureExtractor
            except ImportError as error:
                raise RuntimeError(
                    "Automatic listening needs Whisper audio features. Install with "
                    "python -m pip install 'transformers>=4.45,<5' in the app environment."
                ) from error
            extractor_factory = cast(Callable[..., FeatureExtractor], WhisperFeatureExtractor)
            feature_extractor = extractor_factory(chunk_length=8)
        assert feature_extractor is not None
        self._extractor = feature_extractor

    def is_complete(self, pcm: bytes) -> bool:
        audio = _samples(pcm)[-128000:]
        if len(audio) == 0:
            return False
        audio = np.pad(audio, (128000 - len(audio), 0))
        inputs = self._extractor(
            audio, sampling_rate=16000, return_tensors="np", padding="max_length",
            max_length=128000, truncation=True, do_normalize=True,
        )
        features = np.asarray(inputs.input_features, dtype=np.float32)
        outputs = self._session.run(None, {"input_features": features})
        return float(np.asarray(outputs[0]).flat[0]) > 0.5
