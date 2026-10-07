from __future__ import annotations

import stat
from pathlib import Path

from interview_helper.answer import AnswerTurn
from interview_helper.history import InterviewHistoryStore


def test_completed_turns_round_trip_as_one_private_interview(tmp_path: Path) -> None:
    directory = tmp_path / "history"
    store = InterviewHistoryStore(directory)
    session = store.new_session()

    session.append(AnswerTurn("First question", "First answer"))
    session.append(AnswerTurn("Follow-up", "Second answer"))

    records = store.records()
    assert len(records) == 1
    assert [entry.question for entry in records[0].entries] == [
        "First question",
        "Follow-up",
    ]
    assert [entry.answer for entry in records[0].entries] == [
        "First answer",
        "Second answer",
    ]
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    saved = next(directory.glob("*.json"))
    assert stat.S_IMODE(saved.stat().st_mode) == 0o600


def test_empty_session_is_not_written(tmp_path: Path) -> None:
    store = InterviewHistoryStore(tmp_path / "history")
    store.new_session()

    assert store.records() == ()
    assert not store.directory.exists()


def test_corrupt_history_does_not_hide_valid_records(tmp_path: Path) -> None:
    store = InterviewHistoryStore(tmp_path / "history")
    session = store.new_session()
    session.append(AnswerTurn("Valid question", "Valid answer"))
    (store.directory / "broken.json").write_text("not json", encoding="utf-8")

    records = store.records()

    assert len(records) == 1
    assert records[0].entries[0].question == "Valid question"


def test_spoken_history_separate_correctable_and_legacy_compatible(tmp_path: Path) -> None:
    import json
    from interview_helper.history import HistoryEntry
    store = InterviewHistoryStore(tmp_path)
    session = store.new_session()
    session.append(AnswerTurn('What did you do?', 'Suggested replacement'))
    session.set_spoken((HistoryEntry('What did you do?', 'I kept the firewall.', 'today'),))
    record = store.records()[0]
    assert record.entries[0].answer == 'Suggested replacement'
    assert record.spoken[0].answer == 'I kept the firewall.'
    session.set_spoken((HistoryEntry('What did you do?', 'I only observed.', 'today'),))
    session.append(AnswerTurn('Why?', 'Second suggestion'))
    assert store.records()[0].spoken[0].answer == 'I only observed.'
    session.set_spoken(())
    assert store.records()[0].spoken == ()
    path = next(tmp_path.glob('*.json'))
    payload = json.loads(path.read_text())
    del payload['spoken']
    payload['version'] = 1
    path.write_text(json.dumps(payload))
    assert len(store.records()[0].entries) == 2
