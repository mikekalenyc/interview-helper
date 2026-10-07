"""Resident Moonshine streaming adapter with configurable vocabulary."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from interview_helper.config import MoonshineConfig
from interview_helper.core import Transcript


class MoonshineError(RuntimeError):
    """Moonshine could not process a recording."""


class MoonshineSession:
    def __init__(
        self,
        stream: Any,
        *,
        partial_callback: Callable[[str], None],
        complete_callback: Callable[[Transcript], None],
    ) -> None:
        self.stream = stream
        self.partial_callback = partial_callback
        self.complete_callback = complete_callback
        self.closed = False
        self.completed = False
        self.finishing = False
        self.completed_lines: dict[object, str] = {}
        self.completed_order: list[object] = []
        self.partial_lines: dict[object, str] = {}
        self.last_latency_ms = 0
        self.stream.add_listener(self._event)
        self.stream.start()

    def add_pcm(self, pcm: bytes, sample_rate: int) -> None:
        if self.closed or self.completed:
            return
        if not pcm or len(pcm) % 4:
            raise MoonshineError("Invalid float32 PCM chunk")
        samples = np.frombuffer(pcm, dtype="<f4")
        if not np.isfinite(samples).all():
            raise MoonshineError("PCM contains non-finite samples")
        self.stream.add_audio(samples.tolist(), sample_rate)

    def finish(self) -> None:
        if self.closed:
            return
        self.finishing = True
        try:
            if not self.completed:
                self.stream.stop()
                self._emit_complete()
        finally:
            self.close()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.stream.close()

    def _event(self, event: object) -> None:
        line = getattr(event, "line", None)
        text = str(getattr(line, "text", "")).strip()
        if not text or self.completed:
            return
        line_id = getattr(line, "line_id", None)
        key: object = line_id if line_id is not None else ("text", text)
        event_name = type(event).__name__
        if event_name == "LineTextChanged":
            self.partial_lines[key] = text
            self.partial_callback(self._combined_text())
        elif event_name == "LineCompleted":
            if line_id is None:
                self.partial_lines.clear()
            if key not in self.completed_lines:
                self.completed_order.append(key)
            self.completed_lines[key] = text
            self.partial_lines.pop(key, None)
            self.last_latency_ms = max(
                self.last_latency_ms,
                max(0, int(getattr(line, "last_transcription_latency_ms", 0))),
            )
            self.partial_callback(self._combined_text())
            if self.finishing:
                self._emit_complete()

    def _combined_text(self) -> str:
        parts = [self.completed_lines[key] for key in self.completed_order]
        parts.extend(
            text for key, text in self.partial_lines.items()
            if key not in self.completed_lines
        )
        return " ".join(part.strip() for part in parts if part.strip()).strip()

    def _emit_complete(self) -> None:
        if self.completed:
            return
        self.completed = True
        text = self._combined_text()
        if text:
            self.complete_callback(
                Transcript(
                    text=text,
                    transcription_seconds=self.last_latency_ms / 1_000,
                )
            )


class MoonshineTranscriber:
    def __init__(self, config: MoonshineConfig, *, transcriber: Any | None = None) -> None:
        if transcriber is None:
            try:
                from moonshine_voice import ModelArch
                from moonshine_voice.transcriber import Transcriber
            except ImportError as error:
                raise MoonshineError(
                    "Install the project's 'moonshine' dependencies"
                ) from error
            if not config.model_path.is_dir():
                raise MoonshineError(f"Moonshine model is missing: {config.model_path}")
            transcriber = Transcriber(
                config.model_path,
                ModelArch.SMALL_STREAMING,
                update_interval=config.update_interval_seconds,
                options={"return_audio_data": "false"},
            )
        self.transcriber = transcriber
        self.config = config
        set_keyterms = getattr(transcriber, "set_keyterms", None)
        if set_keyterms is not None:
            set_keyterms(config.keyterms)

    def open_stream(
        self,
        *,
        partial_callback: Callable[[str], None],
        complete_callback: Callable[[Transcript], None],
    ) -> MoonshineSession:
        stream = self.transcriber.create_stream(
            update_interval=self.config.update_interval_seconds
        )
        return MoonshineSession(
            stream,
            partial_callback=partial_callback,
            complete_callback=complete_callback,
        )
