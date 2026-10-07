"""Optional, local CPU speech and end-of-turn inference; never downloads at runtime.

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

from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray

DEFAULT_MODEL_DIRECTORY = Path.home() / ".local/share/interview-helper/turn-models"
DEFAULT_SILERO_MODEL = DEFAULT_MODEL_DIRECTORY / "silero-v6.2.onnx"
DEFAULT_SMART_TURN_MODEL = DEFAULT_MODEL_DIRECTORY / "smart-turn-v3.2-cpu.onnx"


class InferenceSession(Protocol):
    def run(self, output_names: None, input_feed: dict[str, Any]) -> list[Any]: ...


class FeatureExtractor(Protocol):
    def __call__(self, audio: NDArray[np.float32], **kwargs: Any) -> Any: ...


def _session(path: Path) -> InferenceSession:
    if not path.is_file():
        raise RuntimeError(
            f"Automatic listening model is missing: {path}. Run "
            "interview-helper-install-turn-models."
        )
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError(
            "Automatic listening needs ONNX Runtime. Install with "
            "python -m pip install 'onnxruntime>=1.20,<2' 'transformers>=4.45,<5' in the app environment."
        ) from error
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(  # type: ignore[no-any-return]
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )


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
                 session: InferenceSession | None = None) -> None:
        self._session = session if session is not None else _session(model_path or DEFAULT_SILERO_MODEL)
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
                 feature_extractor: FeatureExtractor | None = None) -> None:
        self._session = session if session is not None else _session(model_path or DEFAULT_SMART_TURN_MODEL)
        if feature_extractor is None:
            try:
                from transformers import WhisperFeatureExtractor
            except ImportError as error:
                raise RuntimeError(
                    "Automatic listening needs Whisper audio features. Install with "
                    "python -m pip install 'onnxruntime>=1.20,<2' 'transformers>=4.45,<5' in the app environment."
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
