"""Two-source lifecycle checks without capture hardware."""

from __future__ import annotations

import threading
from typing import cast

import pytest

from interview_helper.daemon import InterviewDaemon
from interview_helper.duplex import DuplexDaemon


class Child:
    def __init__(self, error: BaseException | None = None) -> None:
        self.running = threading.Event()
        self.stopped = threading.Event()
        self.error = error
        self.closes = 0

    def run(self) -> None:
        self.running.set()
        if self.error:
            raise self.error
        assert self.stopped.wait(3)

    def close(self) -> None:
        self.closes += 1
        self.stopped.set()


def duplex(a: Child, b: Child) -> DuplexDaemon:
    return DuplexDaemon(cast(InterviewDaemon, a), cast(InterviewDaemon, b))


def test_both_run_and_close_joins_and_is_idempotent() -> None:
    a, b = Child(), Child()
    pair = duplex(a, b)
    worker = threading.Thread(target=pair.run)
    worker.start()
    assert a.running.wait(2) and b.running.wait(2)
    pair.close()
    worker.join(2)
    assert not worker.is_alive()
    assert all(not thread.is_alive() for thread in pair._threads)
    pair.close()
    assert a.closes == b.closes == 1
    with pytest.raises(RuntimeError, match='cannot be restarted'):
        pair.run()


@pytest.mark.parametrize('side', [0, 1])
def test_either_startup_failure_closes_both_and_propagates(side: int) -> None:
    children = [Child(), Child()]
    children[side].error = RuntimeError('capture start failed')
    pair = duplex(*children)
    with pytest.raises(RuntimeError, match='capture start failed'):
        pair.run()
    assert all(child.stopped.is_set() for child in children)
    assert all(not thread.is_alive() for thread in pair._threads)


def test_close_before_run_never_starts_capture() -> None:
    a, b = Child(), Child()
    pair = duplex(a, b)
    pair.close()
    pair.run()
    assert not a.running.is_set() and not b.running.is_set()
    assert a.closes == b.closes == 1


def test_either_normal_return_stops_peer() -> None:
    a, b = Child(), Child()
    a.stopped.set()
    pair = duplex(a, b)
    pair.run()
    assert b.stopped.is_set()
    assert all(not thread.is_alive() for thread in pair._threads)


def test_stop_during_startup_joins_both() -> None:
    entered = threading.Event()

    class SlowStart(Child):
        def run(self) -> None:
            entered.set()
            assert self.stopped.wait(3)
            raise AssertionError('closed while connecting')

    a, b = SlowStart(), Child()
    pair = duplex(a, b)
    errors: list[BaseException] = []

    def run() -> None:
        try:
            pair.run()
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=run)
    worker.start()
    assert entered.wait(2)
    pair.close()
    worker.join(2)
    assert not worker.is_alive()
    assert errors == []
    assert a.closes == b.closes == 1
