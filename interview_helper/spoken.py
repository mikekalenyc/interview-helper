"""Bounded, editable actual speech for an explicitly enabled live session."""

from __future__ import annotations

import html
import re
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace


MAX_UTTERANCE_CHARACTERS = 4_000
MAX_STORED_CHARACTERS = 40_000
MAX_CONTEXT_CHARACTERS = 4_800


@dataclass(frozen=True, slots=True)
class SpokenTurn:
    id: int
    question: str
    text: str
    timestamp: float


class SpokenConversation:
    """Process-memory-only statements; never accepts generated answers implicitly."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._turns: list[SpokenTurn] = []
        self._next_id = 1

    @staticmethod
    def _validate(text: str) -> str:
        text = text.strip()
        if len(text) > MAX_UTTERANCE_CHARACTERS:
            # Truncation could remove a final qualification or negation.
            raise ValueError("Spoken response exceeds the 4,000-character limit")
        return text

    def _trim(self) -> None:
        while len(self._turns) > 100 or sum(
            len(turn.text) + len(turn.question) for turn in self._turns
        ) > MAX_STORED_CHARACTERS:
            self._turns.pop(0)

    def append(self, text: str, question: str = "") -> SpokenTurn:
        text = self._validate(text)
        if not text:
            raise ValueError("Spoken response must not be empty")
        with self._lock:
            turn = SpokenTurn(self._next_id, question.strip()[:600], text, time.monotonic())
            self._next_id += 1
            self._turns.append(turn)
            self._trim()
            return turn

    def snapshot(self) -> tuple[SpokenTurn, ...]:
        with self._lock:
            return tuple(self._turns)

    def replace(self, turn_id: int, text: str) -> bool:
        text = self._validate(text)
        with self._lock:
            for index, turn in enumerate(self._turns):
                if turn.id == turn_id:
                    if text:
                        self._turns[index] = replace(turn, text=text)
                    else:
                        del self._turns[index]
                    self._trim()
                    return True
        return False

    def clear(self) -> None:
        with self._lock:
            self._turns.clear()


def select_spoken_turns(
    question: str, turns: Sequence[SpokenTurn], *, followup: bool,
) -> tuple[SpokenTurn, ...]:
    """Retain the latest answer as continuity, then relevant older answers.

    Pauses can split one response into several utterances. Keep the contiguous
    group together so an initial claim is not separated from its correction.
    The next question need not repeat the answer's vocabulary to depend on it.
    """
    if not turns:
        return ()
    stop = {"the", "and", "that", "this", "with", "what", "how", "why", "did", "you",
            "your", "was", "were", "have", "had", "would", "could", "about", "which",
            "when", "where", "from", "for", "not", "but", "our", "then", "used"}

    def terms(text: str) -> set[str]:
        return {word for word in re.findall(r"[a-z0-9][a-z0-9_-]+", text.lower())
                if len(word) > 2 and word not in stop}

    query = terms(question)
    if followup:
        query |= terms(turns[-1].question + " " + turns[-1].text)
    groups: list[list[tuple[int, SpokenTurn]]] = []
    for index, turn in enumerate(turns):
        if groups and turn.question and groups[-1][-1][1].question == turn.question:
            groups[-1].append((index, turn))
        else:
            groups.append([(index, turn)])
    ranked = sorted(groups, key=lambda group: (
        int(group[-1][0] == len(turns) - 1),
        len(query & terms(" ".join(t.question + " " + t.text for _, t in group))),
        group[-1][0],
    ), reverse=True)
    selected: list[tuple[int, SpokenTurn]] = []
    # Includes outer labels and room for an explicit omission notice.
    remaining = MAX_CONTEXT_CHARACTERS - 250
    for group in ranked:
        size = sum(len(render_spoken_turn(turn)) + 2 for _, turn in group)
        relevant = bool(query & terms(" ".join(t.question + " " + t.text for _, t in group)))
        latest = group[-1][0] == len(turns) - 1
        if latest and size > remaining:
            # Do not substitute an older statement for an omitted latest reply.
            return ()
        if size <= remaining and (relevant or latest):
            selected.extend(group)
            remaining -= size
    return tuple(turn for _, turn in sorted(selected))


RECENT_EXCHANGE_LIMIT = 5
RECENT_EXCHANGE_CHARACTERS = 12_000


def recent_spoken_exchanges(
    turns: Sequence[SpokenTurn], *, limit: int = RECENT_EXCHANGE_LIMIT,
    max_characters: int = RECENT_EXCHANGE_CHARACTERS,
) -> tuple[SpokenTurn, ...]:
    """Return the last answered questions whole, leaving relevance to the model.

    Unlike keyword selection, nothing is filtered by vocabulary. Answers are
    never truncated; the oldest whole exchanges are dropped to fit the budget.
    """
    groups: list[list[SpokenTurn]] = []
    for turn in turns:
        if groups and turn.question and groups[-1][-1].question == turn.question:
            groups[-1].append(turn)
        else:
            groups.append([turn])
    selected: list[SpokenTurn] = []
    remaining = max_characters
    for group in reversed(groups[-limit:]):
        size = sum(len(render_spoken_turn(turn)) + 2 for turn in group)
        if size > remaining:
            break
        selected[:0] = group
        remaining -= size
    return tuple(selected)


def render_spoken_turn(turn: SpokenTurn) -> str:
    return (f'<utterance id="{turn.id}">\n'
            f"<preceding_question>{html.escape(turn.question)}</preceding_question>\n"
            f"<actual_response>{html.escape(turn.text)}</actual_response>\n"
            "</utterance>")
