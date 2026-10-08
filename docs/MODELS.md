# Transcription models and devices

These choices turn local audio into text. They are separate from the OpenAI or
local Qwen **answer model**, which writes suggestions from that text and context.
All current transcription choices are English models.

## Choose and download

While stopped, open **Setup**, select **Transcription model**, select its device,
then click **Download selected model**. The app downloads pinned model files and,
for Nemotron, the matching native runtime. It verifies sizes and SHA-256 hashes
before installing each file. Progress and cancellation remain available while
other Setup controls and Start Interview are locked.

| Model | Model download | Devices | Guidance |
| --- | ---: | --- | --- |
| Moonshine Small | 142 MB | CPU | Current default; smallest supported choice. |
| Moonshine Medium | 269 MB | CPU | More CPU work; compare accuracy on your vocabulary. |
| NVIDIA Nemotron English 0.6B Q8 | 700 MB | CPU or NVIDIA CUDA | Native streaming model; requires Linux x86_64. |

For a supported NVIDIA GPU, **Nemotron on CUDA is the fastest option in the
saved-clip comparison below**. For a small installation that works without a
GPU, keep Moonshine Small on CPU. Medium made slightly fewer errors on this
small sample but used more CPU time. No model was error-free.

Sizes use decimal MB and exclude Python dependencies. Nemotron additionally
downloads about 5 MB for its CPU runtime or 107 MB for CUDA. Extracted runtimes
use roughly 22 MiB or 366 MiB respectively; allow space for the retained archive,
model, and temporary extraction. Models are stored under
`~/.local/share/interview-helper/models/`, with third-party notices.

Installed files persist across launches. Setup choices are not yet saved: choose
the desired model again after reopening. **Use installed model** reuses its files.
Cancelled or failed downloads retain completed verified files for the next
attempt; an incomplete file is downloaded again. Nothing downloads during an
interview or just because the app opens. No audio or personal context is sent
to the model hosts.

The catalog lists supported adapters and pinned models; it is not a general
Hugging Face browser. A custom path must contain a compatible model of the
selected type. An incomplete native runtime reports its path for repair rather
than overwriting an unknown directory.

## CPU and NVIDIA GPU setup

Use the ordinary `.[desktop]` installation for CPU operation. Nemotron's CUDA
transcription runtime is downloaded with the model when you select NVIDIA GPU;
it does not use Python Torch or ONNX Runtime for transcription. The upstream
Linux x86_64 binary targets NVIDIA Turing or newer (compute capability 7.5+),
with a driver compatible with CUDA 12.8. AMD GPUs and Apple Metal are not
supported by this integration. **GPU index 0** selects the first NVIDIA GPU.
If you already ran Nemotron using its CPU-only runtime, restart the application
before switching to CUDA. The app reports this requirement instead of reusing
the wrong loaded library. A CUDA runtime can subsequently run on either device.

**Speech / turn detection** is independent of transcription. CPU remains its
default. To try CUDA detection, use a fresh environment with the GPU dependency
profile, then install its additional model:

```bash
python3 -m venv .venv-gpu
.venv-gpu/bin/python -m pip install '.[gpu-desktop]'
.venv-gpu/bin/interview-helper-install-turn-models --device cuda
.venv-gpu/bin/interview-helper-gui
```

For a wheel, replace `.[gpu-desktop]` with the wheel path followed by
`[gpu-desktop]`. This profile includes substantially larger NVIDIA/Torch
dependencies for detection; the sizes in the table describe transcription
models only. Do not combine `desktop`/`automatic` with `gpu-desktop` in one
environment: CPU `onnxruntime` and `onnxruntime-gpu` install overlapping files.
The GPU profile pins ONNX Runtime 1.26.0 and Torch 2.8.0 for CUDA 12.8/cuDNN 9.

GPU detector startup profiles one synthetic inference to confirm actual CUDA
compute. Some shape/control operations and audio preprocessing still use CPU.
Missing models, libraries, unsupported hardware, or unsuccessful GPU execution
produce errors; explicit GPU selections do not silently fall back to CPU.

Document lookup uses a local SQLite/text index and stays on CPU. A GPU does not
accelerate this implementation. An answer server has its own device settings;
these controls cannot move a hosted OpenAI model onto your graphics card.

For the held-control command, use `--transcription-model moonshine-small`,
`moonshine-medium`, or `nemotron-en`, with `--transcription-device cpu|cuda`.
`--detection-device cpu|cuda` applies when automatic listening is enabled.
`--model-path` and `--nemotron-library` are optional overrides. Downloads must
be completed first. For example:

```python
from interview_helper.model_catalog import install_model
install_model("nemotron-en", device="cuda")
```

## Why these choices: October 8, 2026

Compact size alone does not establish the best model for two live audio streams.
We checked current upstream models before adopting Whisper. Moonshine stays the
existing CPU baseline; Nemotron is the supported native CPU/CUDA alternative.

| Candidate reviewed | Relevant upstream evidence | Decision for this integration |
| --- | --- | --- |
| [Moonshine streaming](https://moonshine-voice.readthedocs.io/en/latest/models/available-models/) | Small and Medium native streaming models; compact quantized downloads. | Include both CPU choices. |
| [Nemotron English](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b) and [NeMo-Speech.cpp](https://github.com/NVIDIA/NeMo-Speech.cpp) | 700 MB Q8 model, native CPU/CUDA runtime, independent streaming sessions. | Evaluate and integrate this GPU alternative. |
| [Phonon-2](https://huggingface.co/FermionResearch/Phonon-2) | About 164 MB; [published server](https://www.fermionresearch.com/docs/speech-streaming/) supports one live stream at a time; CUDA distribution uses a container. | Needs different concurrency/runtime integration for interviewer plus microphone. |
| [Parakeet Redux](https://moondream.ai/blog/introducing-parakeet-redux-and-ultra) | About 178 MB; published previews start after four seconds, then every two seconds. | Preview behavior needs qualification for this app. |
| [Parakeet Realtime EOU 120M](https://huggingface.co/nvidia/parakeet_realtime_eou_120m-v1) | Small streaming option; punctuation/capitalization and accuracy tradeoffs. | Not integrated; no local performance claim. |
| [Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) | Published streaming deployment uses vLLM. | Larger runtime/integration commitment than the native choices. |
| [Whisper](https://github.com/openai/whisper) | Mature multilingual models and ecosystem; chunked transcription needs additional streaming integration. | Provisional prototype not adopted or shipped. |

Publisher benchmark scores are not interchangeable with measurements in this
application. The selection can expand after an adapter supports two sources,
bounded memory, cancellation, explicit finalization, and reproducible tests.

## Transcription measurements

Ten preselected synthetic practice questions (89.925 seconds of audio,
202 normalized words) were replayed on a Ryzen 9 5900X and RTX 3090. Each model
received identical 16 kHz mono PCM in 80 ms chunks, without vocabulary hints.
The replay ran as fast as each adapter accepted audio; Nemotron's bounded
background queue applied backpressure in the test harness. Model load was
measured separately. The first clip includes lazy inference initialization.

| Model/device | Word errors | Total replay time | Median finalization |
| --- | ---: | ---: | ---: |
| Moonshine Small / CPU | 10/202 (4.95%) | 57.35 s | 103 ms |
| Moonshine Medium / CPU | 8/202 (3.96%) | 71.96 s | 395 ms |
| Nemotron / CPU | 9/202 (4.46%) | 26.29 s | 1,229 ms |
| Nemotron / CUDA | 9/202 (4.46%) | 2.27 s | 95 ms |

Finalization is the wall time for `finish()` after submitting the last saved
chunk. Native queue backlog contributes to it. These accelerated replay times
exclude capture, turn detection, retrieval and answer generation; they are not
live speech-to-answer latency. CUDA graph logs confirmed actual GPU compute.
Both native devices also completed two interleaved streams independently.
All four choices produced no text on empty input or two/eight seconds of silence.

Scoring ignores case/punctuation and canonicalizes acronym spacing using a fixed
vocabulary derived from the references. Technical confusions still count:
all models struggled with adjacent spoken acronyms such as ASA/ACL and FMC;
Nemotron also confused “prefilter.” This is one synthetic voice and a small
sample, not a general accuracy ranking or qualification for accents, echo or
real interview conditions.

A second check delivered two of the same clips at normal speed, one chunk every
80 ms. From delivery of the last audio chunk to the final result:

| Model/device | Clip 1 | Clip 2 |
| --- | ---: | ---: |
| Moonshine Small / CPU | 153 ms | 291 ms |
| Moonshine Medium / CPU | 255 ms | 178 ms |
| Nemotron / CPU | 126 ms | 153 ms |
| Nemotron / CUDA | 13 ms | 11 ms |

Synchronous Moonshine processing sometimes delayed chunk delivery. From the
scheduled audio end, Small took 418/291 ms and Medium 717/422 ms; Nemotron stayed
within 1 ms of the table. Two clips are a smoke check, not a latency guarantee.
No capture buffering, speech-end detector delay or answer generation is included.
Native GPU transcription and both GPU detectors were also verified together in
one process; full CUDA libraries are loaded before the native runtime when the
GPU detection profile is installed.

## Detector measurements

On one RTX 3090 workstation, using the same saved nine-second synthetic clip:

| Operation | CPU | CUDA |
| --- | ---: | ---: |
| Silero speech detection, whole clip | 28.5 ms | 124.8 ms |
| Smart Turn, one completion check | 46.8 ms | 3.3 ms |

These are medians of five warm runs. Profiling confirmed CUDA compute for both;
Silero rejected three seconds of silence on both devices. The results show why
GPU is not automatically faster for every operation. Keep CPU detection as the
simple default; GPU Smart Turn can help when turn checks dominate the workload.
This table is not an end-to-end answer latency or real-microphone benchmark.
