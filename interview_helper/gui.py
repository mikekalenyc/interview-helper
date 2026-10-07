"""Dependency-friendly entry point for the optional desktop application."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from interview_helper.openai_client import OPENAI_MODELS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resume",
        type=Path,
        help="preselect an explicitly supplied local text or Markdown resume",
    )
    parser.add_argument(
        "--context", type=Path, action="append", default=[],
        help="preselect a local interview guide or background file; repeat for multiple files",
    )
    parser.add_argument(
        "--technical-answers", type=Path,
        help="preselect prepared technical Q&A; turns on technical interview mode",
    )
    parser.add_argument("--answer-provider", choices=("local", "openai"), default="local")
    parser.add_argument("--openai-model", choices=tuple(OPENAI_MODELS), default=None,
                        help="preselect the OpenAI model shown in Setup")
    parser.add_argument("--openai-api-key-file", type=Path,
                        help="OpenAI credential file path; never pass the key itself")
    args = parser.parse_args()
    try:
        from interview_helper.qt_gui import run_gui
    except ImportError as error:
        if error.name and error.name.startswith("PySide6"):
            print(
                "interview-helper-gui: install the project's 'gui' dependencies",
                file=sys.stderr,
            )
            raise SystemExit(2) from error
        raise
    raise SystemExit(
        run_gui(resume_path=args.resume,
                context_paths=tuple(args.context), answer_provider=args.answer_provider,
                openai_api_key_file=args.openai_api_key_file,
                openai_model=args.openai_model,
                technical_answers_path=args.technical_answers)
    )


if __name__ == "__main__":
    main()
