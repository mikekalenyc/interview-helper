#!/usr/bin/env python3
"""Explicitly install pinned public CPU/CUDA models and their license notices.

No audio or candidate data is read or uploaded. SHA256 values pin the downloaded
bytes; each file is verified before an atomic replacement of its destination.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import tempfile
import urllib.request

SMART_REVISION = "f766f81d3cfdf7737ac64aad813d91bbfd56bf93"
SILERO_REVISION = "be95df9152c0d7618fa1edfeb296fc3dae32376f"
SMART_CODE_REVISION = "4786657e242dfe77dd138699ac564ee074a2a543"
ASSETS = (
    ("smart-turn-v3.2-cpu.onnx",
     f"https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/{SMART_REVISION}/smart-turn-v3.2-cpu.onnx",
     "2bb026316b14a660486a75b1733cd3fbab8c2fd0314dc9af7be49f8cca967e4f"),
    ("silero-v6.2.onnx",
     f"https://raw.githubusercontent.com/snakers4/silero-vad/{SILERO_REVISION}/src/silero_vad/data/silero_vad.onnx",
     "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"),
    ("LICENSE-smart-turn.txt",
     f"https://raw.githubusercontent.com/pipecat-ai/smart-turn/{SMART_CODE_REVISION}/LICENSE",
     "0d66364067f678c08586ebb60a16a2aed4fa081ec11057df35585759ce0e774f"),
    ("LICENSE-silero.txt",
     f"https://raw.githubusercontent.com/snakers4/silero-vad/{SILERO_REVISION}/LICENSE",
     "2e63e9a38b6e8fc0c7bc37ce174caca1862870856c6daf5697cfb785e925520b"),
)
GPU_ASSET = (
    "smart-turn-v3.2-gpu.onnx",
    f"https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/{SMART_REVISION}/smart-turn-v3.2-gpu.onnx",
    "ab8dc64b88713f90b571c15b714bd1330e6c883cad8763dacf65c9376dc539be",
)


def install(destination: Path, *, device: str = "cpu") -> None:
    if device not in {"cpu", "cuda"}:
        raise ValueError("Detector device must be cpu or cuda")
    destination.mkdir(parents=True, exist_ok=True)
    for name, url, expected in (*ASSETS, *((GPU_ASSET,) if device == "cuda" else ())):
        path = destination / name
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected:
            print(f"Verified {path}")
            continue
        temporary: Path | None = None
        try:
            digest = hashlib.sha256()
            with urllib.request.urlopen(url, timeout=60) as response:
                with tempfile.NamedTemporaryFile(dir=destination, delete=False) as output:
                    temporary = Path(output.name)
                    while chunk := response.read(1024 * 1024):
                        digest.update(chunk)
                        output.write(chunk)
            if digest.hexdigest() != expected:
                raise RuntimeError(f"SHA256 mismatch for {name}; existing model left intact")
            temporary.replace(path)
            print(f"Installed {path}")
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path,
                        default=Path.home() / ".local/share/interview-helper/turn-models")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    install(args.directory, device=args.device)


if __name__ == "__main__":
    main()
