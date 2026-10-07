"""Coordinated lifetime for independent interviewer and candidate daemons."""

from __future__ import annotations

import threading

from interview_helper.daemon import InterviewDaemon


class DuplexDaemon:
    """Run both capture streams; stop both when either run exits.

    Instances are single-use. Capture reconnection belongs to each child.
    """

    def __init__(self, interviewer: InterviewDaemon, candidate: InterviewDaemon) -> None:
        self.interviewer = interviewer
        self.candidate = candidate
        self._lock = threading.Lock()
        self._stopping = threading.Event()
        self._completed = threading.Event()
        self._closed = threading.Event()
        self._started = False
        self._threads: list[threading.Thread] = []
        self._errors: list[BaseException] = []

    def run(self) -> None:
        with self._lock:
            if self._started:
                raise RuntimeError("Dual audio daemon cannot be restarted")
            self._started = True
            if self._stopping.is_set():
                return
            for label, daemon in (("interviewer", self.interviewer), ("candidate", self.candidate)):
                thread = threading.Thread(target=self._run_child, args=(daemon,),
                                          name=f"dual-audio-{label}", daemon=True)
                self._threads.append(thread)
                try:
                    thread.start()
                except BaseException as error:
                    self._threads.pop()
                    self._errors.append(error)
                    self._completed.set()
                    break
        try:
            self._completed.wait()
        finally:
            self.close()
        if self._errors:
            raise self._errors[0]

    def _run_child(self, daemon: InterviewDaemon) -> None:
        try:
            if not self._stopping.is_set():
                daemon.run()
        except BaseException as error:
            # InterviewDaemon asserts when close wins its initial connect race.
            if not (self._stopping.is_set() and isinstance(error, AssertionError)):
                with self._lock:
                    self._errors.append(error)
        finally:
            self._completed.set()

    def close(self) -> None:
        with self._lock:
            owner = not self._stopping.is_set()
            self._stopping.set()
            threads = tuple(self._threads)
        if not owner:
            self._closed.wait()
            return
        try:
            # Closing both concurrently prevents one stream's pending callback
            # from keeping the other stream alive during shutdown.
            def stop(daemon: InterviewDaemon) -> None:
                try:
                    daemon.close()
                except BaseException as error:
                    with self._lock:
                        self._errors.append(error)

            stoppers = [threading.Thread(target=stop, args=(daemon,), daemon=True)
                        for daemon in (self.interviewer, self.candidate)]
            for thread in stoppers:
                thread.start()
            for thread in stoppers:
                thread.join()
            for thread in threads:
                if thread is not threading.current_thread():
                    thread.join()
        finally:
            self._closed.set()
            self._completed.set()
