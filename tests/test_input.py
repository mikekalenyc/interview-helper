import threading
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from interview_helper.input import (
    HotkeyLearningCancelled,
    InputError,
    ManualHoldControl,
    default_hotkey_device,
    learn_hotkey,
    learning_event_paths,
    transition_for_value,
)


def test_evdev_press_repeat_release_semantics() -> None:
    assert transition_for_value(1).pressed is True
    assert transition_for_value(2) is None
    assert transition_for_value(0).pressed is False


def test_manual_hold_control_forwards_ui_press_and_release() -> None:
    control = ManualHoldControl()
    calls: list[str] = []

    def pressed() -> bool:
        calls.append("press")
        return True

    def released() -> bool:
        calls.append("release")
        return True

    worker = threading.Thread(
        target=control.run,
        args=(pressed, released),
    )
    worker.start()
    assert control.wait_until_ready()

    assert control.press() is True
    assert control.release() is True
    assert calls == ["press", "release"]

    control.close()
    worker.join(timeout=1)
    assert not worker.is_alive()
    assert control.press() is False


@dataclass
class FakeEvent:
    type: int
    code: int
    value: int


class FakeLearningDevice:
    def __init__(self, path: str) -> None:
        self.path = path
        self.name = "Test Keyboard"
        self.closed = False

    def fileno(self) -> int:
        return 10

    def read(self) -> list[FakeEvent]:
        return [FakeEvent(1, 30, 2), FakeEvent(1, 30, 1)]

    def close(self) -> None:
        self.closed = True


def test_learn_hotkey_returns_stable_path_and_ignores_repeat() -> None:
    device = FakeLearningDevice("/dev/input/by-id/test-event-kbd")
    evdev = SimpleNamespace(
        InputDevice=lambda _path: device,
        ecodes=SimpleNamespace(EV_KEY=1, bytype={1: {30: "KEY_A"}}),
    )

    learned = learn_hotkey(
        (Path(device.path),),
        evdev_module=evdev,
        waiter=lambda devices, _timeout: devices,
    )

    assert learned.device == Path(device.path)
    assert learned.event_code == "KEY_A"
    assert learned.device_name == "Test Keyboard"
    assert device.closed


def test_learn_hotkey_cancellation_closes_opened_device() -> None:
    cancel = threading.Event()
    cancel.set()
    device = FakeLearningDevice("/dev/input/by-id/test-event-kbd")
    evdev = SimpleNamespace(
        InputDevice=lambda _path: device,
        ecodes=SimpleNamespace(EV_KEY=1, bytype={1: {30: "KEY_A"}}),
    )

    with pytest.raises(HotkeyLearningCancelled):
        learn_hotkey(
            (Path(device.path),),
            cancel=cancel,
            evdev_module=evdev,
            waiter=lambda _devices, _timeout: (),
        )

    assert device.closed


def test_learn_hotkey_times_out_and_closes_device() -> None:
    device = FakeLearningDevice("/dev/input/by-id/test-event-kbd")
    evdev = SimpleNamespace(
        InputDevice=lambda _path: device,
        ecodes=SimpleNamespace(EV_KEY=1, bytype={1: {30: "KEY_A"}}),
    )

    with pytest.raises(InputError, match="timed out"):
        learn_hotkey(
            (Path(device.path),),
            timeout=0.001,
            evdev_module=evdev,
            waiter=lambda _devices, _timeout: (),
        )

    assert device.closed


def test_learn_hotkey_reports_inaccessible_devices() -> None:
    def fail(_path: str) -> FakeLearningDevice:
        raise PermissionError("denied")

    evdev = SimpleNamespace(InputDevice=fail)

    with pytest.raises(InputError, match="narrow udev rule"):
        learn_hotkey((Path("/dev/input/by-id/test-event-kbd"),), evdev_module=evdev)


def test_learning_paths_prefer_stable_link_and_include_distinct_selection(
    tmp_path: Path,
) -> None:
    by_id = tmp_path / "by-id"
    by_id.mkdir()
    event1 = tmp_path / "event1"
    event2 = tmp_path / "event2"
    event1.touch()
    event2.touch()
    stable = by_id / "test-event-kbd"
    stable.symlink_to(event1)

    assert learning_event_paths(event1, directory=by_id) == (stable,)
    assert learning_event_paths(event2, directory=by_id) == (stable, event2)


def test_default_hotkey_device_chooses_first_readable_stable_keyboard() -> None:
    inaccessible = Path("/dev/input/by-id/a-event-kbd")
    keyboard = Path("/dev/input/by-id/b-event-kbd")
    mouse = Path("/dev/input/by-id/c-event-mouse")
    opened: list[str] = []

    def open_device(path: str) -> FakeLearningDevice:
        opened.append(path)
        if path == str(inaccessible):
            raise PermissionError("denied")
        return FakeLearningDevice(path)

    evdev = SimpleNamespace(InputDevice=open_device)

    assert default_hotkey_device(
        (inaccessible, keyboard, mouse), evdev_module=evdev
    ) == keyboard
    assert opened == [str(inaccessible), str(keyboard)]
