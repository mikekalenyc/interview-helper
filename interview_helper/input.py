"""Raw Linux input adapter for held keyboard and mouse controls."""

from __future__ import annotations

import threading
import select
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence, cast


class InputError(RuntimeError):
    """The configured raw input device or event is unusable."""


class HotkeyLearningCancelled(InputError):
    """The user explicitly cancelled hotkey learning."""


@dataclass(frozen=True, slots=True)
class KeyTransition:
    pressed: bool


def transition_for_value(value: int) -> KeyTransition | None:
    if value == 1:
        return KeyTransition(pressed=True)
    if value == 0:
        return KeyTransition(pressed=False)
    return None


class Event(Protocol):
    type: int
    code: int
    value: int


@dataclass(frozen=True, slots=True)
class LearnedHotkey:
    device: Path
    event_code: str
    device_name: str


class Device(Protocol):
    name: str

    def read_loop(self) -> Iterator[Event]: ...

    def close(self) -> None: ...


class LearningDevice(Protocol):
    name: str

    def fileno(self) -> int: ...

    def read(self) -> Sequence[Event]: ...

    def close(self) -> None: ...


DeviceWaiter = Callable[[Sequence[LearningDevice], float], Sequence[LearningDevice]]


def stable_event_paths(directory: Path = Path("/dev/input/by-id")) -> tuple[Path, ...]:
    """Return stable event symlinks without broadening device permissions."""
    if not directory.is_dir():
        return ()
    paths: list[Path] = []
    resolved: set[Path] = set()
    for path in sorted(directory.glob("*-event-*")):
        try:
            target = path.resolve(strict=True)
        except OSError:
            continue
        if target in resolved:
            continue
        resolved.add(target)
        paths.append(path)
    return tuple(paths)


def learning_event_paths(
    selected: Path | None,
    *,
    directory: Path = Path("/dev/input/by-id"),
) -> tuple[Path, ...]:
    """Combine stable discovery with a manually selected fallback path.

    Stable links come first so a selected volatile ``/dev/input/eventN`` path
    resolves back to a persistent path when one exists. A distinct selected
    device is still observed alongside discovery rather than limiting Learn to
    a stale or exclusively grabbed device.
    """
    paths = list(stable_event_paths(directory))
    resolved = {path.resolve(strict=False) for path in paths}
    if selected is not None and selected.resolve(strict=False) not in resolved:
        paths.append(selected)
    return tuple(paths)


def default_hotkey_device(
    paths: Sequence[Path] | None = None,
    *,
    evdev_module: Any | None = None,
) -> Path | None:
    """Choose the first readable stable keyboard without grabbing it."""
    candidates = tuple(paths) if paths is not None else stable_event_paths()
    keyboards = tuple(path for path in candidates if path.name.endswith("event-kbd"))
    if evdev_module is None:
        try:
            import evdev
        except ImportError:
            return None
        evdev_module = evdev
    for path in keyboards:
        try:
            device = cast(LearningDevice, evdev_module.InputDevice(str(path)))
        except (OSError, PermissionError):
            continue
        device.close()
        return path
    return None


def wait_readable(
    devices: Sequence[LearningDevice], timeout: float
) -> Sequence[LearningDevice]:
    readable, _, _ = select.select(list(devices), [], [], timeout)
    return cast(Sequence[LearningDevice], readable)


def learn_hotkey(
    paths: Sequence[Path] | None = None,
    *,
    cancel: threading.Event | None = None,
    timeout: float = 20.0,
    evdev_module: Any | None = None,
    waiter: DeviceWaiter = wait_readable,
) -> LearnedHotkey:
    """Observe one EV_KEY press without grabbing or suppressing input."""
    if timeout <= 0:
        raise ValueError("Hotkey learning timeout must be positive")
    cancel = cancel or threading.Event()
    candidates = tuple(paths) if paths is not None else stable_event_paths()
    if not candidates:
        raise InputError("No stable input event devices were found in /dev/input/by-id")
    if evdev_module is None:
        try:
            import evdev
        except ImportError as error:
            raise InputError("Install the project's 'input' dependencies") from error
        evdev_module = evdev

    opened: list[LearningDevice] = []
    device_paths: dict[int, Path] = {}
    failures: list[str] = []
    try:
        for path in candidates:
            try:
                device = cast(LearningDevice, evdev_module.InputDevice(str(path)))
            except (OSError, PermissionError) as error:
                failures.append(f"{path}: {error}")
                continue
            opened.append(device)
            device_paths[id(device)] = path
        if not opened:
            detail = failures[0] if failures else "no readable devices"
            raise InputError(
                "No selected input device is readable. Install a narrow udev "
                f"rule and retry ({detail})."
            )

        deadline = time.monotonic() + timeout
        ev_key = int(evdev_module.ecodes.EV_KEY)
        while True:
            if cancel.is_set():
                raise HotkeyLearningCancelled("Hotkey learning cancelled")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise InputError("No key press detected before learning timed out")
            for device in waiter(opened, min(remaining, 0.2)):
                try:
                    events = device.read()
                except BlockingIOError:
                    continue
                for event in events:
                    if event.type != ev_key or event.value != 1:
                        continue
                    name = evdev_module.ecodes.bytype[ev_key].get(
                        event.code, str(event.code)
                    )
                    if isinstance(name, (list, tuple)):
                        name = name[0]
                    return LearnedHotkey(
                        device=device_paths[id(device)],
                        event_code=str(name),
                        device_name=device.name,
                    )
    finally:
        for device in opened:
            device.close()


class EvdevHoldControl:
    def __init__(
        self,
        device_path: Path,
        event_code: str,
        *,
        evdev_module: Any | None = None,
    ) -> None:
        if evdev_module is None:
            try:
                import evdev
            except ImportError as error:
                raise InputError("Install the project's 'input' dependencies") from error
            evdev_module = evdev
        assert evdev_module is not None
        self.evdev = evdev_module
        self.device_path = device_path
        self.event_code_name = event_code
        try:
            self.event_code = int(event_code)
        except ValueError:
            try:
                self.event_code = int(getattr(self.evdev.ecodes, event_code))
            except (AttributeError, TypeError, ValueError) as error:
                raise InputError(f"Unknown evdev event code: {event_code}") from error
        try:
            self.device: Device = self.evdev.InputDevice(str(device_path))
        except (OSError, PermissionError) as error:
            raise InputError(
                f"Cannot open {device_path}: {error}. "
                "Install a narrow udev rule and run as your desktop user."
            ) from error

    def run(
        self,
        press_callback: Callable[[], object],
        release_callback: Callable[[], object],
    ) -> None:
        ev_key = int(self.evdev.ecodes.EV_KEY)
        for event in self.device.read_loop():
            if event.type != ev_key or event.code != self.event_code:
                continue
            transition = transition_for_value(event.value)
            if transition is None:
                continue
            if transition.pressed:
                press_callback()
            else:
                release_callback()

    def close(self) -> None:
        self.device.close()


class ManualHoldControl:
    """A non-device held control driven by explicit UI press/release actions."""

    def __init__(self) -> None:
        self._closed = threading.Event()
        self._ready = threading.Event()
        self._callback_lock = threading.Lock()
        self._press_callback: Callable[[], object] | None = None
        self._release_callback: Callable[[], object] | None = None

    @property
    def ready(self) -> bool:
        return self._ready.is_set() and not self._closed.is_set()

    def wait_until_ready(self, timeout: float = 1.0) -> bool:
        return self._ready.wait(timeout) and not self._closed.is_set()

    def run(
        self,
        press_callback: Callable[[], object],
        release_callback: Callable[[], object],
    ) -> None:
        with self._callback_lock:
            self._press_callback = press_callback
            self._release_callback = release_callback
            self._ready.set()
        self._closed.wait()
        with self._callback_lock:
            self._ready.clear()
            self._press_callback = None
            self._release_callback = None

    def press(self) -> bool:
        with self._callback_lock:
            callback = self._press_callback if self.ready else None
        return bool(callback()) if callback is not None else False

    def release(self) -> bool:
        with self._callback_lock:
            callback = self._release_callback if self.ready else None
        return bool(callback()) if callback is not None else False

    def close(self) -> None:
        self._closed.set()
