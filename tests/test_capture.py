from interview_helper.capture import (
    CaptureError,
    MicrophoneSource,
    MonitorSource,
    ParecMonitorCapture,
    PulseMicrophoneResolver,
    PulseMonitorResolver,
    list_microphone_sources,
)
from interview_helper.config import AudioConfig


def test_resolves_current_default_sink_monitor() -> None:
    calls: list[tuple[str, ...]] = []

    def runner(command: tuple[str, ...]) -> str:
        calls.append(command)
        if command == ("pactl", "get-default-sink"):
            return "bluez_output.headphones\n"
        return (
            "52\talsa_input.dji\tPipeWire\ts24le 2ch 48000Hz\tSUSPENDED\n"
            "99\tbluez_output.headphones.monitor\tPipeWire\t"
            "s16le 1ch 16000Hz\tSUSPENDED\n"
        )

    monitor = PulseMonitorResolver(runner).resolve_default()

    assert monitor == MonitorSource(
        sink="bluez_output.headphones",
        source="bluez_output.headphones.monitor",
    )
    assert calls == [
        ("pactl", "get-default-sink"),
        ("pactl", "list", "short", "sources"),
    ]


def test_rejects_default_sink_without_monitor() -> None:
    def runner(command: tuple[str, ...]) -> str:
        if command == ("pactl", "get-default-sink"):
            return "missing\n"
        return "1\tother.monitor\tPipeWire\ts16le 1ch 16000Hz\tSUSPENDED\n"

    try:
        PulseMonitorResolver(runner).resolve_default()
    except CaptureError as error:
        assert "missing.monitor" in str(error)
    else:
        raise AssertionError("Missing monitor was accepted")


def test_resolves_current_default_microphone() -> None:
    calls: list[tuple[str, ...]] = []

    def runner(command: tuple[str, ...]) -> str:
        calls.append(command)
        if command == ("pactl", "get-default-source"):
            return "alsa_input.usb_dji\n"
        return "52\talsa_input.usb_dji\tPipeWire\ts24le 2ch 48000Hz\tSUSPENDED\n"

    microphone = PulseMicrophoneResolver(runner).resolve_default()

    assert microphone == MonitorSource(
        sink="alsa_input.usb_dji", source="alsa_input.usb_dji"
    )
    assert calls == [
        ("pactl", "get-default-source"),
        ("pactl", "list", "short", "sources"),
    ]


def test_rejects_output_monitor_as_default_microphone() -> None:
    def runner(command: tuple[str, ...]) -> str:
        assert command == ("pactl", "get-default-source")
        return "headphones.monitor\n"

    try:
        PulseMicrophoneResolver(runner).resolve_default()
    except CaptureError as error:
        assert "not a microphone" in str(error)
    else:
        raise AssertionError("Output monitor was accepted as a microphone")


def test_lists_linux_microphones_with_descriptions_and_default() -> None:
    def runner(command: tuple[str, ...]) -> str:
        if command == ("pactl", "get-default-source"):
            return "alsa_input.dji\n"
        return """[
          {"name": "headphones.monitor", "description": "Monitor", "properties": {"device.class": "monitor"}},
          {"name": "alsa_input.onboard", "description": "Built-in Audio Analog Stereo", "properties": {}},
          {"name": "alsa_input.dji", "description": "DJI MIC MINI Analog Stereo", "properties": {}}
        ]"""

    assert list_microphone_sources(runner) == (
        MicrophoneSource("alsa_input.onboard", "Built-in Audio Analog Stereo"),
        MicrophoneSource("alsa_input.dji", "DJI MIC MINI Analog Stereo", True),
    )


def test_resolves_explicit_microphone_without_reading_system_default() -> None:
    calls: list[tuple[str, ...]] = []

    def runner(command: tuple[str, ...]) -> str:
        calls.append(command)
        return "52\talsa_input.dji\tPipeWire\ts24le 2ch 48000Hz\tSUSPENDED\n"

    microphone = PulseMicrophoneResolver(
        runner, source="alsa_input.dji"
    ).resolve_default()

    assert microphone == MonitorSource("alsa_input.dji", "alsa_input.dji")
    assert calls == [("pactl", "list", "short", "sources")]


def test_parec_command_normalizes_audio_for_asr() -> None:
    capture = ParecMonitorCapture(AudioConfig())

    assert capture.command_for(MonitorSource("sink", "sink.monitor")) == (
        "parec",
        "--device=sink.monitor",
        "--raw",
        "--format=float32le",
        "--rate=16000",
        "--channels=1",
        "--latency-msec=80",
    )


def test_audio_chunk_size_is_exact_float32_frames() -> None:
    config = AudioConfig(sample_rate=16_000, chunk_milliseconds=80)
    assert config.chunk_bytes == 5_120
