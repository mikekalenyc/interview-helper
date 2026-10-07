"""Private local persistence for completed text-only interview exchanges."""

from __future__ import annotations

import json
import os
import uuid
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from interview_helper.answer import AnswerTurn


@dataclass(frozen=True, slots=True)
class HistoryEntry:
    question: str
    answer: str
    created_at: str


@dataclass(frozen=True, slots=True)
class InterviewRecord:
    session_id: str
    started_at: str
    entries: tuple[HistoryEntry, ...]
    spoken: tuple[HistoryEntry, ...] = ()

    @property
    def title(self) -> str:
        if not self.entries:
            return self.started_at
        question = self.entries[0].question.replace("\n", " ").strip()
        return question if len(question) <= 72 else f"{question[:69]}..."


class InterviewHistorySession:
    def __init__(
        self,
        store: InterviewHistoryStore,
        *,
        session_id: str,
        started_at: str,
    ) -> None:
        self.store = store
        self.session_id = session_id
        self.started_at = started_at
        self._entries: list[HistoryEntry] = []
        self._spoken: tuple[HistoryEntry, ...] = ()
        self._lock = threading.RLock()

    def append(self, turn: AnswerTurn) -> InterviewRecord:
        with self._lock:
            return self._append(turn)

    def _append(self, turn: AnswerTurn) -> InterviewRecord:
        self._entries.append(
            HistoryEntry(
                question=turn.question,
                answer=turn.answer,
                created_at=_now(),
            )
        )
        record = InterviewRecord(
            session_id=self.session_id,
            started_at=self.started_at,
            entries=tuple(self._entries),
            spoken=self._spoken,
        )
        self.store.save(record)
        return record

    def set_spoken(self, entries: tuple[HistoryEntry, ...]) -> InterviewRecord:
        """Save finalized, separately labeled speech; corrections replace its text."""
        with self._lock:
            self._spoken = entries
            record = InterviewRecord(self.session_id, self.started_at,
                                     tuple(self._entries), self._spoken)
            self.store.save(record)
            return record


class InterviewHistoryStore:
    def __init__(self, directory: Path | None = None) -> None:
        self.directory = directory or default_history_directory()

    def new_session(self) -> InterviewHistorySession:
        started_at = _now()
        session_id = (
            datetime.now().astimezone().strftime("%Y%m%dT%H%M%S")
            + "-"
            + uuid.uuid4().hex[:10]
        )
        return InterviewHistorySession(
            self,
            session_id=session_id,
            started_at=started_at,
        )

    def save(self, record: InterviewRecord) -> None:
        self.directory.mkdir(parents=True, mode=0o700, exist_ok=True)
        self.directory.chmod(0o700)
        target = self.directory / f"{record.session_id}.json"
        temporary = self.directory / f".{record.session_id}.tmp"
        payload = {
            "version": 2,
            "session_id": record.session_id,
            "started_at": record.started_at,
            "entries": [
                {
                    "question": entry.question,
                    "answer": entry.answer,
                    "created_at": entry.created_at,
                }
                for entry in record.entries
            ],
            "spoken": [
                {"question": entry.question, "answer": entry.answer,
                 "created_at": entry.created_at}
                for entry in record.spoken
            ],
        }
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
            0o600,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(payload, output, ensure_ascii=False, indent=2)
            output.write("\n")
        temporary.chmod(0o600)
        temporary.replace(target)

    def records(self) -> tuple[InterviewRecord, ...]:
        if not self.directory.exists():
            return ()
        records: list[InterviewRecord] = []
        for path in self.directory.glob("*.json"):
            try:
                payload: Any = json.loads(path.read_text(encoding="utf-8"))
                records.append(_record_from_payload(payload))
            except (OSError, UnicodeError, json.JSONDecodeError, ValueError, TypeError):
                continue
        return tuple(sorted(records, key=lambda record: record.started_at, reverse=True))


def default_history_directory() -> Path:
    configured = os.environ.get("XDG_DATA_HOME")
    base = Path(configured) if configured else Path.home() / ".local" / "share"
    return base / "interview-helper" / "interviews"


def _record_from_payload(payload: Any) -> InterviewRecord:
    if not isinstance(payload, dict):
        raise ValueError("History record must be an object")
    entries_payload = payload.get("entries")
    if not isinstance(entries_payload, list):
        raise ValueError("History entries must be a list")
    entries: list[HistoryEntry] = []
    for item in entries_payload:
        if not isinstance(item, dict):
            raise ValueError("History entry must be an object")
        question = item.get("question")
        answer = item.get("answer")
        created_at = item.get("created_at")
        if not all(isinstance(value, str) for value in (question, answer, created_at)):
            raise ValueError("History entry fields must be text")
        assert isinstance(question, str)
        assert isinstance(answer, str)
        assert isinstance(created_at, str)
        entries.append(HistoryEntry(question, answer, created_at))
    session_id = payload.get("session_id")
    started_at = payload.get("started_at")
    if not isinstance(session_id, str) or not isinstance(started_at, str):
        raise ValueError("History session fields must be text")
    spoken: list[HistoryEntry] = []
    spoken_payload = payload.get("spoken", [])
    if not isinstance(spoken_payload, list):
        raise ValueError("Spoken history must be a list")
    for item in spoken_payload:
        if not isinstance(item, dict) or not all(
            isinstance(item.get(key), str) for key in ("question", "answer", "created_at")
        ):
            raise ValueError("Spoken history fields must be text")
        spoken.append(HistoryEntry(item["question"], item["answer"], item["created_at"]))
    return InterviewRecord(session_id, started_at, tuple(entries), tuple(spoken))


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")
