"""Curated local models and explicit, verified downloads; importing is offline."""
from __future__ import annotations

from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import tarfile
import tempfile
import threading
from typing import Callable
import urllib.request

DEFAULT_MODEL_ROOT = Path.home() / '.local/share/interview-helper/models'


@dataclass(frozen=True)
class ModelSpec:
    id: str
    label: str
    description: str
    backend: str
    architecture: str
    size_bytes: int
    devices: tuple[str, ...]


MODEL_CATALOG = (
    ModelSpec('moonshine-small', 'Moonshine Small', 'Fast streaming English transcription on CPU.', 'moonshine', 'small', 142300974, ('cpu',)),
    ModelSpec('moonshine-medium', 'Moonshine Medium', 'Larger streaming English model; uses more CPU and memory.', 'moonshine', 'medium', 269141623, ('cpu',)),
    ModelSpec('nemotron-en', 'NVIDIA Nemotron English', 'Streaming English 0.6B Q8. Download: 700 MB model plus 107 MB CUDA runtime (5 MB CPU); extra space needed for extraction.', 'nemotron', '', 699872960, ('cpu', 'cuda')),
)


@dataclass(frozen=True)
class _Asset:
    name: str
    url: str
    size: int
    sha256: str


class DownloadCancelled(RuntimeError):
    """The explicitly requested download was cancelled."""


def model_path(spec: ModelSpec, root: Path = DEFAULT_MODEL_ROOT) -> Path:
    if spec.backend == 'moonshine':
        return root / f'download.moonshine.ai/model/{spec.architecture}-streaming-en/quantized_26_08_21'
    return root / 'nemotron-en' / 'nemotron-speech-streaming-en-0.6b.q8_0.gguf'


def runtime_library_path(device: str, root: Path = DEFAULT_MODEL_ROOT) -> Path:
    if device not in ('cpu', 'cuda'):
        raise ValueError('Device must be cpu or cuda')
    return root / 'runtimes' / f'nemo-speech-0.1.0-linux-x86_64-{device}' / 'lib/libnemo_speech_asr_c.so'


def _check_cancel(cancel: threading.Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise DownloadCancelled('Model download cancelled')


def _hash(path: Path, cancel: threading.Event | None = None) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        while chunk := source.read(1024 * 1024):
            _check_cancel(cancel)
            digest.update(chunk)
    return digest.hexdigest()


def is_installed(spec: ModelSpec, device: str = 'cpu', root: Path = DEFAULT_MODEL_ROOT) -> bool:
    if device not in spec.devices:
        return False
    path = model_path(spec, root)
    if spec.backend == 'moonshine':
        return all((path / a.name).is_file() and (path / a.name).stat().st_size == a.size for a in _MOONSHINE[spec.architecture])
    runtime = runtime_library_path(device, root)
    return (path.is_file() and path.stat().st_size == _NEMOTRON.size
            and runtime.is_file() and (runtime.parent.parent / '.verified.json').is_file())


def _download(asset: _Asset, path: Path, cancel: threading.Event | None,
              progress: Callable[[int, int, str], None] | None) -> None:
    _check_cancel(cancel)
    if path.is_file() and path.stat().st_size == asset.size and _hash(path, cancel) == asset.sha256:
        if progress:
            progress(asset.size, asset.size, asset.name)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(path.parent).free < asset.size + 16 * 1024 * 1024:
        raise OSError('Insufficient free disk space for ' + asset.name)
    temporary: Path | None = None
    try:
        request = urllib.request.Request(asset.url, headers={'User-Agent': 'interview-helper/0.1'})
        with urllib.request.urlopen(request, timeout=15) as response:
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as output:
                temporary = Path(output.name)
                digest = hashlib.sha256()
                received = 0
                while chunk := response.read(256 * 1024):
                    _check_cancel(cancel)
                    received += len(chunk)
                    if received > asset.size:
                        raise RuntimeError('Unexpected download size: ' + asset.name)
                    digest.update(chunk)
                    output.write(chunk)
                    if progress:
                        progress(received, asset.size, asset.name)
                if received != asset.size or digest.hexdigest() != asset.sha256:
                    raise RuntimeError('Model integrity check failed: ' + asset.name)
                output.flush()
                os.fsync(output.fileno())
        _check_cancel(cancel)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _extract(archive: Path, destination: Path, cancel: threading.Event | None) -> None:
    """Materialize regular files, resolving only links inside the pinned archive."""
    with tarfile.open(archive) as source:
        members = {m.name: m for m in source.getmembers()}
        for member in members.values():
            name = PurePosixPath(member.name)
            if name.is_absolute() or '..' in name.parts:
                raise RuntimeError('Unsafe runtime archive path')
            if not (member.isfile() or member.isdir() or member.issym() or member.islnk()):
                raise RuntimeError('Unsafe runtime archive entry')
        total = sum(m.size for m in members.values())
        if shutil.disk_usage(destination).free < total * 3 + 16 * 1024 * 1024:
            raise OSError('Insufficient free disk space to extract runtime')
        for member in members.values():
            _check_cancel(cancel)
            output = destination / member.name
            if member.isdir():
                output.mkdir(parents=True, exist_ok=True)
                continue
            resolved = member
            visited: set[str] = set()
            while resolved.issym() or resolved.islnk():
                if resolved.name in visited:
                    raise RuntimeError('Cyclic runtime archive link')
                visited.add(resolved.name)
                target = PurePosixPath(resolved.linkname)
                if target.is_absolute() or '..' in target.parts:
                    raise RuntimeError('Unsafe runtime archive link')
                key = str(PurePosixPath(resolved.name).parent / target) if resolved.issym() else str(target)
                if key not in members:
                    raise RuntimeError('Missing runtime archive link target')
                resolved = members[key]
            if not resolved.isfile():
                raise RuntimeError('Invalid runtime archive link target')
            output.parent.mkdir(parents=True, exist_ok=True)
            incoming = source.extractfile(resolved)
            if incoming is None:
                raise RuntimeError('Unreadable runtime archive file')
            with incoming, output.open('wb') as target_file:
                while chunk := incoming.read(1024 * 1024):
                    _check_cancel(cancel)
                    target_file.write(chunk)
            output.chmod(0o755 if member.mode & 0o111 else 0o644)


def install_model(model_id: str, *, device: str = 'cpu', root: Path = DEFAULT_MODEL_ROOT,
                  progress: Callable[[int, int, str], None] | None = None,
                  cancel_event: threading.Event | None = None) -> Path:
    spec = next((m for m in MODEL_CATALOG if m.id == model_id), None)
    if spec is None:
        raise ValueError('Unknown model: ' + model_id)
    if device not in spec.devices:
        raise ValueError('This model does not support ' + device)
    if spec.backend == 'nemotron' and (platform.system() != 'Linux' or platform.machine() != 'x86_64'):
        raise RuntimeError('The bundled Nemotron runtime requires Linux x86_64')
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.download.lock').open('a') as lock:
        while True:
            _check_cancel(cancel_event)
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if cancel_event is not None:
                    cancel_event.wait(0.1)
                else:
                    threading.Event().wait(0.1)
        path = model_path(spec, root)
        if spec.backend == 'moonshine':
            for asset in (*_MOONSHINE[spec.architecture], _MOONSHINE_LICENSE):
                _download(asset, path / asset.name, cancel_event, progress)
        else:
            _download(_NEMOTRON_NOTICE, path.parent / _NEMOTRON_NOTICE.name, cancel_event, progress)
            _download(_NEMOTRON, path, cancel_event, progress)
            asset = _RUNTIMES[device]
            archive = root / '.downloads' / asset.name
            _download(asset, archive, cancel_event, progress)
            runtime = runtime_library_path(device, root).parent.parent
            marker = runtime / '.verified.json'
            if not marker.is_file() or not runtime_library_path(device, root).is_file():
                runtime.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(dir=runtime.parent) as temporary:
                    staging = Path(temporary)
                    _extract(archive, staging, cancel_event)
                    extracted = staging / runtime.name
                    (extracted / '.verified.json').write_text(json.dumps({'sha256': asset.sha256}))
                    _check_cancel(cancel_event)
                    if runtime.exists():
                        raise RuntimeError('Incomplete runtime exists; move it aside before reinstalling: ' + str(runtime))
                    extracted.replace(runtime)
        return path

_MOONSHINE: dict[str, tuple[_Asset, ...]] = {
    'small': (
        _Asset('adapter.ort', 'https://download.moonshine.ai/model/small-streaming-en/quantized_26_08_21/adapter.ort', 2870368, 'c665f742364febad597cc9ac1e0b341ffbee0e24a1466e2f3bde95e6e4771762'),
        _Asset('cross_kv.ort', 'https://download.moonshine.ai/model/small-streaming-en/quantized_26_08_21/cross_kv.ort', 5356536, 'e2d3417144e9514055ebfefe8dcc4c0a55a55adcb8530435844c75c53e352bf6'),
        _Asset('decoder_kv.ort', 'https://download.moonshine.ai/model/small-streaming-en/quantized_26_08_21/decoder_kv.ort', 81878600, '1a05465b1dd955858dfcbee039c0020fb5dd982b0f5094c34e61735d518d771b'),
        _Asset('encoder.ort', 'https://download.moonshine.ai/model/small-streaming-en/quantized_26_08_21/encoder.ort', 44148576, '2d4d973e91e8aca08c51e7e7efa28a46ab265b63d809d5294d18b86bcd85b993'),
        _Asset('frontend.model.ort', 'https://download.moonshine.ai/model/small-streaming-en/quantized_26_08_21/frontend.model.ort', 26944, '09b1210ae30dc5f0f3e45f0ebab914c254741323114f53fbbe5ae62cca35058f'),
        _Asset('frontend.weights.ort', 'https://download.moonshine.ai/model/small-streaming-en/quantized_26_08_21/frontend.weights.ort', 7769464, '7ef97521bd4bad3928f5bb6808586f4fcc6e92bd5990394112eed7d4052ec338'),
        _Asset('streaming_config.json', 'https://download.moonshine.ai/model/small-streaming-en/quantized_26_08_21/streaming_config.json', 512, '26f02b6afb22d60871a5efd85c3d38e569cc0ddb6c5eb6e93d3260152ae8a47a'),
        _Asset('tokenizer.bin', 'https://download.moonshine.ai/model/small-streaming-en/quantized_26_08_21/tokenizer.bin', 249974, '6884b35fd6377d4c4d32336a0bc152f36b64d1e45b6503683cdc238250a8472d'),
    ),
    'medium': (
        _Asset('adapter.ort', 'https://download.moonshine.ai/model/medium-streaming-en/quantized_26_08_21/adapter.ort', 3651296, '3f2a287def57cc094367a0eec3c4f5fc36a32ec420e86764b696920991b20281'),
        _Asset('cross_kv.ort', 'https://download.moonshine.ai/model/medium-streaming-en/quantized_26_08_21/cross_kv.ort', 11643776, '642f6e21cd305be79342207c6f9e6b681d469d55bc48c72b27b84846fb71fd1e'),
        _Asset('decoder_kv.ort', 'https://download.moonshine.ai/model/medium-streaming-en/quantized_26_08_21/decoder_kv.ort', 146972408, '193bb366492b74fc4ad338c6778e8d8eb916aaa11b5aa264f9057f4db7759486'),
        _Asset('encoder.ort', 'https://download.moonshine.ai/model/medium-streaming-en/quantized_26_08_21/encoder.ort', 94705376, '12915e76ebac7dd287c5ea63965d06103a53ba1ce242a4a34f318f3958c60c37'),
        _Asset('frontend.model.ort', 'https://download.moonshine.ai/model/medium-streaming-en/quantized_26_08_21/frontend.model.ort', 28720, '95768855c70c8251eeecc05fedf69999da1b8ab16f605c9f457fd3354b0ad6b5'),
        _Asset('frontend.weights.ort', 'https://download.moonshine.ai/model/medium-streaming-en/quantized_26_08_21/frontend.weights.ort', 11889560, '5ac941f490cbe035b335b99a414cc393d62d4c6f9f2423495b286870d271d709'),
        _Asset('streaming_config.json', 'https://download.moonshine.ai/model/medium-streaming-en/quantized_26_08_21/streaming_config.json', 513, '28e83b7a28e91472692a035e0dae3116422ae43aeb2bef5ed822c44ce89b88af'),
        _Asset('tokenizer.bin', 'https://download.moonshine.ai/model/medium-streaming-en/quantized_26_08_21/tokenizer.bin', 249974, '6884b35fd6377d4c4d32336a0bc152f36b64d1e45b6503683cdc238250a8472d'),
    ),
}

_RUNTIMES = {
    'cpu': _Asset('nemo-speech-0.1.0-linux-x86_64-cpu.tar.gz', 'https://github.com/NVIDIA/NeMo-Speech.cpp/releases/download/v0.1.0/nemo-speech-0.1.0-linux-x86_64-cpu.tar.gz', 4583913, '0f74131d631ad2c694cf0ec53490866bb6461147959589a69fb6fc231944065b'),
    'cuda': _Asset('nemo-speech-0.1.0-linux-x86_64-cuda.tar.gz', 'https://github.com/NVIDIA/NeMo-Speech.cpp/releases/download/v0.1.0/nemo-speech-0.1.0-linux-x86_64-cuda.tar.gz', 107310946, 'e68628f396489c98fb353e070efaea5bc4977409ae7734fce56c251a79e29147'),
}

_NEMOTRON = _Asset(
    'nemotron-speech-streaming-en-0.6b.q8_0.gguf',
    'https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b/resolve/ebe59e5a817142986528bbbee5dba8db7b38ed50/nemotron-speech-streaming-en-0.6b.q8_0.gguf',
    699872960,
    'd9a01898d2a611c8764e23a1c2f45e70bbd5a425dc4de93692ac951dd603812d',
)

_MOONSHINE_LICENSE = _Asset('moonshine-LICENSE', 'https://raw.githubusercontent.com/moonshine-ai/moonshine/v0.1.5/LICENSE', 14180, 'fa7d1174dd8af6a7cd280be20b80d10095ed4c19b5b20b61a7715c3ad790dc5f')

_NEMOTRON_NOTICE = _Asset('nemotron-README.md', 'https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b/raw/ebe59e5a817142986528bbbee5dba8db7b38ed50/README.md', 25987, '7701f4b2f1c16542c8ceb2b3a61dd144032898c17f4dc9cc1bbecda6972edec8')
