"""Validated daemon configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class AudioConfig:
    sample_rate: int = 16_000
    channels: int = 1
    sample_format: str = "float32le"
    chunk_milliseconds: int = 80
    release_drain_milliseconds: int = 300

    def __post_init__(self) -> None:
        if not 8_000 <= self.sample_rate <= 96_000:
            raise ValueError("Audio sample rate must be between 8kHz and 96kHz")
        if self.channels != 1:
            raise ValueError("The MVP requires mono capture")
        if self.sample_format != "float32le":
            raise ValueError("The MVP requires float32le capture")
        if not 10 <= self.chunk_milliseconds <= 1_000:
            raise ValueError("Audio chunk duration must be between 10ms and 1000ms")
        if not 0 <= self.release_drain_milliseconds <= 2_000:
            raise ValueError("Release drain must be between 0ms and 2000ms")

    @property
    def chunk_bytes(self) -> int:
        frames = self.sample_rate * self.chunk_milliseconds // 1_000
        return frames * self.channels * 4


@dataclass(frozen=True, slots=True)
class InputConfig:
    device: Path
    event_code: str

    def __post_init__(self) -> None:
        if not self.event_code.strip():
            raise ValueError("Input event code cannot be empty")


@dataclass(frozen=True, slots=True)
class MoonshineConfig:
    model_path: Path
    update_interval_seconds: float = 0.2
    keyterms: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.update_interval_seconds <= 0:
            raise ValueError("Moonshine update interval must be positive")
        if any(not term.strip() for term in self.keyterms):
            raise ValueError("Moonshine keyterms cannot be blank")
