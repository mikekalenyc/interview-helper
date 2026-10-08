"""Command-line entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from interview_helper.answer import ANSWER_TOKEN_LIMIT
from interview_helper.openai_client import OPENAI_MODEL, OPENAI_MODELS
from interview_helper.application import (
    ApplicationCallbacks,
    ApplicationConfig,
    AnswerProvider,
    InterviewApplication,
)
from interview_helper.capture import CaptureError, PulseMonitorResolver
from interview_helper.context import ContextError
from interview_helper.daemon import (
    stderr_error,
    stderr_partial,
    stderr_state,
    stdout_answer,
    stdout_transcript,
)
from interview_helper.input import InputError
from interview_helper.moonshine import MoonshineError
from interview_helper.qwen import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL as DEFAULT_QWEN_MODEL,
    QwenError,
)


DEFAULT_MODEL = Path.home() / (
    ".local/share/interview-helper/models/download.moonshine.ai/model/"
    "small-streaming-en/quantized_26_08_21"
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subcommands = result.add_subparsers(dest="command", required=True)
    subcommands.add_parser("audio", help="print the current sink monitor")
    run = subcommands.add_parser("run", help="run the hold-to-transcribe daemon")
    run.add_argument("--input-device", type=Path, required=True)
    run.add_argument("--event-code", required=True, help="for example KEY_F13 or BTN_SIDE")
    run.add_argument("--model-path", type=Path, help="installed model path; never downloads during an interview")
    run.add_argument("--transcription-model", choices=("moonshine-small", "moonshine-medium", "nemotron-en"), default="moonshine-small")
    run.add_argument("--transcription-device", choices=("cpu", "cuda"), default="cpu")
    run.add_argument("--detection-device", choices=("cpu", "cuda"), default="cpu",
                     help="speech/turn detection device (used by automatic listening)")
    run.add_argument("--gpu-device-index", type=int, default=0)
    run.add_argument("--nemotron-library", type=Path, help="optional installed NeMo-Speech.cpp ASR library path")
    run.add_argument(
        "--keyterm",
        action="append",
        default=[],
        help="expected literal term; may be repeated",
    )
    run.add_argument("--show-partials", action="store_true")
    run.add_argument(
        "--resume",
        type=Path,
        help="UTF-8 .txt or Markdown resume; enables grounded answers",
    )
    run.add_argument(
        "--context",
        type=Path,
        action="append",
        default=[],
        help="project .txt/.md file or directory; may be repeated",
    )
    run.add_argument(
        "--technical-answers", type=Path,
        help="prepared technical Q&A (.txt/.md) sent whole with each question; "
             "turns on technical interview mode",
    )
    run.add_argument(
        "--technical-library", type=Path,
        help="folder of local technical reference documents; no online lookup",
    )
    run.add_argument("--qwen-base-url", default=DEFAULT_BASE_URL)
    run.add_argument("--qwen-model", default=DEFAULT_QWEN_MODEL)
    run.add_argument("--answer-provider", choices=("local", "openai"), default="local")
    run.add_argument("--openai-model", choices=tuple(OPENAI_MODELS), default=OPENAI_MODEL)
    run.add_argument("--openai-api-key-file", type=Path,
                     help="OpenAI credential file path; never pass the key itself")
    run.add_argument("--answer-max-tokens", type=int, default=ANSWER_TOKEN_LIMIT)
    return result


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    try:
        if args.command == "audio":
            monitor = PulseMonitorResolver().resolve_default()
            print(f"sink={monitor.sink}")
            print(f"monitor={monitor.source}")
            return
        if args.context and args.resume is None:
            raise ContextError("--context requires --resume")
        if args.technical_answers and args.resume is None:
            raise ContextError("--technical-answers requires --resume")
        if args.answer_max_tokens <= 0:
            raise ValueError("--answer-max-tokens must be positive")
        from interview_helper.model_catalog import MODEL_CATALOG, model_path
        selected_model = next(model for model in MODEL_CATALOG if model.id == args.transcription_model)
        application = InterviewApplication(
            ApplicationConfig(
                input_device=args.input_device,
                event_code=args.event_code,
                model_path=args.model_path or model_path(selected_model),
                transcription_backend=selected_model.backend,
                moonshine_architecture=selected_model.architecture,
                transcription_device=args.transcription_device,
                detection_device=args.detection_device,
                gpu_device_index=args.gpu_device_index,
                nemotron_library=args.nemotron_library,
                keyterms=tuple(args.keyterm),
                resume=args.resume,
                context=tuple(args.context),
                qwen_base_url=args.qwen_base_url,
                qwen_model=args.qwen_model,
                answer_provider=AnswerProvider(args.answer_provider),
                openai_api_key_file=args.openai_api_key_file,
                openai_model=args.openai_model,
                answer_max_tokens=args.answer_max_tokens,
                show_partials=args.show_partials,
                technical_library=args.technical_library,
                technical_answers=args.technical_answers,
            ),
            ApplicationCallbacks(
                state=stderr_state,
                status=lambda message: print(message, file=sys.stderr, flush=True),
                evidence=lambda text: print(text, file=sys.stderr, flush=True),
                partial=stderr_partial,
                transcript=stdout_transcript,
                answer=stdout_answer,
                error=stderr_error,
            ),
        )
        application.run()
    except (CaptureError, ContextError, InputError, MoonshineError, QwenError, ValueError, RuntimeError) as error:
        print(f"interview-helper: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
