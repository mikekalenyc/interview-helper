#!/usr/bin/env python3
"""Copy reviewed release inputs to a new directory without touching Git state.

No network calls, credentials, personal profiles, model files, recordings,
workstation plans, or historical evaluation reports are included. Review the
result before committing or uploading; an allowlist is not a secret scanner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


FILES = (
    ".gitignore", "README.md", "LICENSE", "THIRD_PARTY_NOTICES.md",
    "pyproject.toml", "MANIFEST.in", ".github/workflows/checks.yml",
    "docs/SETUP.md", "docs/MODES.md", "docs/PERSONALIZATION.md",
    "docs/REQUIREMENTS.md", "docs/RELEASING.md",
    "scripts/install_turn_models.py", "scripts/build_question_audio.py",
    "scripts/question_player.html", "scripts/benchmark_answers.py",
    "scripts/prepare_github_release.py", "deployment/README.md",
    "deployment/interview-helper.service.example",
    "deployment/99-interview-helper.rules.example",
)
PATTERNS = (
    "interview_helper/*.py", "tests/test_*.py", "examples/*.example.md",
)


def export(source: Path, destination: Path) -> int:
    source = source.resolve()
    destination = destination.resolve()
    if destination.exists():
        raise ValueError(f"Destination already exists; choose a new directory: {destination}")
    paths = {source / name for name in FILES}
    for pattern in PATTERNS:
        matches = tuple(source.glob(pattern))
        if not matches:
            raise ValueError(f"No release inputs matched {pattern}")
        paths.update(matches)
    selected = sorted(paths, key=lambda path: str(path.relative_to(source)))
    for path in selected:
        if not path.is_file():
            raise ValueError(f"Missing release input: {path.relative_to(source)}")
        components = (path, *path.parents[:len(path.relative_to(source).parts) - 1])
        if any(part.is_symlink() for part in components):
            raise ValueError(f"Release input must not be a symlink: {path.relative_to(source)}")
    destination.mkdir(parents=True)
    manifest: dict[str, str] = {}
    for path in selected:
        relative = path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        manifest[relative.as_posix()] = hashlib.sha256(target.read_bytes()).hexdigest()
    (destination / "RELEASE_MANIFEST.json").write_text(
        json.dumps({"sha256": manifest}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return len(selected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path, help="new export directory; never overwritten")
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    count = export(args.source, args.destination)
    print(f"Prepared {count} release inputs in {args.destination}")
    print("Review RELEASE_MANIFEST.json, then test/build before uploading.")


if __name__ == "__main__":
    main()
