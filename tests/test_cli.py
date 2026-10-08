from pathlib import Path

import pytest

from interview_helper.answer import ANSWER_TOKEN_LIMIT
from interview_helper.application import ApplicationConfig, ApplicationCallbacks

from interview_helper.cli import parser


def test_run_defaults_to_shared_answer_limit() -> None:
    args = parser().parse_args(
        [
            "run",
            "--input-device",
            "/dev/input/example",
            "--event-code",
            "KEY_M",
        ]
    )

    assert args.answer_max_tokens == ANSWER_TOKEN_LIMIT


def test_technical_library_is_explicit_cli_option() -> None:
    required = ["run", "--input-device", "/dev/input/example", "--event-code", "KEY_M"]
    assert parser().parse_args(required).technical_library is None
    assert parser().parse_args([*required, "--technical-library", "/library"]).technical_library == Path("/library")


def test_openai_configuration_reaches_cli_application(monkeypatch: pytest.MonkeyPatch) -> None:
    import interview_helper.cli as cli
    from interview_helper.application import AnswerProvider
    captured = []

    class Application:
        def __init__(self, config: ApplicationConfig, callbacks: ApplicationCallbacks) -> None:
            captured.append(config)

        def run(self) -> None:
            pass

    monkeypatch.setattr(cli, "InterviewApplication", Application)
    cli.main(["run", "--input-device", "/dev/input/example", "--event-code", "KEY_M",
              "--answer-provider", "openai", "--openai-api-key-file", "/private/key.rtf"])
    assert captured[0].answer_provider is AnswerProvider.OPENAI
    assert captured[0].openai_api_key_file == Path("/private/key.rtf")


@pytest.mark.parametrize("token_override", [None, 240])
def test_cli_forwards_library_and_prints_evidence(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    token_override: int | None,
) -> None:
    import interview_helper.cli as cli

    captured: dict[str, ApplicationConfig] = {}

    class Application:
        def __init__(
            self, config: ApplicationConfig, callbacks: ApplicationCallbacks,
        ) -> None:
            captured["config"] = config
            self.callbacks = callbacks

        def run(self) -> None:
            self.callbacks.evidence("Retrieved: local-library/aws.md")

    monkeypatch.setattr(cli, "InterviewApplication", Application)
    cli.main([
        "run", "--input-device", "/dev/input/example", "--event-code", "KEY_M",
        "--resume", "/resume.md", "--technical-library", "/library",
        *([] if token_override is None else ["--answer-max-tokens", str(token_override)]),
    ])
    assert captured["config"].technical_library == Path("/library")
    assert captured["config"].answer_max_tokens == (
        ANSWER_TOKEN_LIMIT if token_override is None else token_override
    )
    output = capsys.readouterr()
    assert "Retrieved: local-library/aws.md" in output.err
    assert output.out == ""


def test_technical_answers_reach_cli_application(monkeypatch: pytest.MonkeyPatch) -> None:
    import interview_helper.cli as cli
    captured = []

    class Application:
        def __init__(self, config: ApplicationConfig, callbacks: ApplicationCallbacks) -> None:
            captured.append(config)

        def run(self) -> None:
            pass

    monkeypatch.setattr(cli, "InterviewApplication", Application)
    base = ["run", "--input-device", "/dev/input/example", "--event-code", "KEY_M"]
    cli.main([*base, "--resume", "/private/resume.md", "--technical-answers", "/prep/firewall.md"])
    assert captured[0].technical_answers == Path("/prep/firewall.md")
    with pytest.raises(SystemExit):
        cli.main([*base, "--technical-answers", "/prep/firewall.md"])


def test_openai_model_choice_reaches_cli_application(monkeypatch: pytest.MonkeyPatch) -> None:
    import interview_helper.cli as cli
    captured = []

    class Application:
        def __init__(self, config: ApplicationConfig, callbacks: ApplicationCallbacks) -> None:
            captured.append(config)

        def run(self) -> None:
            pass

    monkeypatch.setattr(cli, "InterviewApplication", Application)
    base = ["run", "--input-device", "/dev/input/example", "--event-code", "KEY_M",
            "--answer-provider", "openai"]
    cli.main([*base, "--openai-model", "gpt-6-luna"])
    assert captured[0].openai_model == "gpt-6-luna"
    with pytest.raises(SystemExit):
        parser().parse_args([*base, "--openai-model", "gpt-6-astra"])


@pytest.mark.parametrize("model,device,backend,architecture", [
    ("moonshine-small", "cpu", "moonshine", "small"),
    ("moonshine-medium", "cpu", "moonshine", "medium"),
    ("nemotron-en", "cuda", "nemotron", None),
])
def test_compute_options_reach_application(
    monkeypatch: pytest.MonkeyPatch, model: str, device: str,
    backend: str, architecture: str | None,
) -> None:
    import interview_helper.cli as cli
    captured = []

    class Application:
        def __init__(self, config: ApplicationConfig, callbacks: ApplicationCallbacks) -> None:
            captured.append(config)

        def run(self) -> None:
            pass

    monkeypatch.setattr(cli, "InterviewApplication", Application)
    cli.main(["run", "--input-device", "/dev/input/example", "--event-code", "KEY_M",
              "--transcription-model", model, "--transcription-device", device,
              "--detection-device", "cuda", "--gpu-device-index", "2", "--model-path", "/models/chosen"])
    selected = captured[0]
    assert selected.transcription_backend == backend
    assert selected.transcription_device == device
    assert selected.detection_device == "cuda"
    assert selected.gpu_device_index == 2
    assert selected.model_path == Path("/models/chosen")
    if architecture:
        assert selected.moonshine_architecture == architecture
