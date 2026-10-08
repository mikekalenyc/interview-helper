"""Offline checks for model integrity and preservation of existing downloads."""

import hashlib
import io
from pathlib import Path

import pytest

from interview_helper import install_turn_models as installer


@pytest.fixture
def model_asset(monkeypatch: pytest.MonkeyPatch) -> bytes:
    data = b"verified model bytes"
    monkeypatch.setattr(installer, "ASSETS", (
        ("model.onnx", "https://example.invalid/model", hashlib.sha256(data).hexdigest()),
    ))
    return data


def test_verified_install_is_reusable_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model_asset: bytes,
) -> None:
    calls: list[str] = []

    def download(url: str, *, timeout: int) -> io.BytesIO:
        calls.append(url)
        assert timeout == 60
        return io.BytesIO(model_asset)

    monkeypatch.setattr(installer.urllib.request, "urlopen", download)
    destination = tmp_path / "models"
    installer.install(destination)
    installer.install(destination)
    assert (destination / "model.onnx").read_bytes() == model_asset
    assert calls == ["https://example.invalid/model"]
    assert list(destination.iterdir()) == [destination / "model.onnx"]


@pytest.mark.parametrize("interrupted", [False, True])
def test_bad_download_preserves_existing_model_and_removes_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model_asset: bytes,
    interrupted: bool,
) -> None:
    destination = tmp_path / "model.onnx"
    destination.write_bytes(b"previous model")

    class InterruptedDownload(io.BytesIO):
        def read(self, size: int = -1) -> bytes:
            if self.tell():
                raise OSError("connection interrupted")
            return super().read(size)

    def download(url: str, *, timeout: int) -> io.BytesIO:
        if interrupted:
            return InterruptedDownload(model_asset)
        return io.BytesIO(b"corrupted download")

    monkeypatch.setattr(installer.urllib.request, "urlopen", download)
    with pytest.raises((OSError, RuntimeError), match="interrupted|SHA256 mismatch"):
        installer.install(tmp_path)
    assert destination.read_bytes() == b"previous model"
    assert list(tmp_path.iterdir()) == [destination]


def test_cuda_install_preserves_cpu_models_and_adds_gpu_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model_asset: bytes,
) -> None:
    data = b"gpu model"
    monkeypatch.setattr(installer, "GPU_ASSET", (
        "gpu.onnx", "https://example.invalid/gpu", hashlib.sha256(data).hexdigest(),
    ))
    calls: list[str] = []

    def download(url: str, *, timeout: int) -> io.BytesIO:
        calls.append(url)
        return io.BytesIO(data if url.endswith("/gpu") else model_asset)

    monkeypatch.setattr(installer.urllib.request, "urlopen", download)
    installer.install(tmp_path)
    installer.install(tmp_path, device="cuda")
    installer.install(tmp_path, device="cuda")
    assert calls == ["https://example.invalid/model", "https://example.invalid/gpu"]
    assert (tmp_path / "model.onnx").read_bytes() == model_asset
    assert (tmp_path / "gpu.onnx").read_bytes() == data


def test_invalid_device_does_not_create_install_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cpu or cuda"):
        installer.install(tmp_path / "unused", device="auto")
    assert not (tmp_path / "unused").exists()
