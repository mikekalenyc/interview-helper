# Preparing a private release

Initial distribution is private GitHub plus a wheel and source archive. Original
code is all rights reserved. Do not publish to PyPI or change repository visibility
as part of a routine build. The initial repository is
`mikekalenyc/interview-helper`.

## Curated export

The development checkout can contain private preparation, hardware identifiers,
historical reports, and an incomplete Git index. Prepare a new reviewable export
without changing that index:

```bash
python3 scripts/prepare_github_release.py private/github-release-NEW
```

The script copies application modules, synthetic tests, generic examples,
selected documentation, deployment examples, build metadata, and CI. It rejects
an existing destination or symlink inputs. `RELEASE_MANIFEST.json` lists SHA-256
hashes of copied files. Review its contents: this is a source allowlist, not a
guarantee that future source edits contain no sensitive text.

Private candidate files, credentials, recordings, model weights, generated
history, local agent state, workstation plans, historical benchmark reports,
and user-specific Q&A are not export inputs. Existing local files and staged
changes remain in the development checkout. The clean export has its own Git
repository for the initial upload.

For later releases, reconcile changes against the published repository rather
than force-pushing a new unrelated history. Keep the old export until its commit
and asset checksums are recorded.

## Verify the actual release contents

In the clean export, install development dependencies and run:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev,gui]'
.venv/bin/pytest
.venv/bin/mypy interview_helper scripts/prepare_github_release.py
.venv/bin/python -m build
```

Inspect both archives for private or unexpected files and confirm that the
packaged model installer and license notices are present. Install the wheel in
a separate environment, from outside the checkout, and check all three commands:

```bash
interview-helper --help
interview-helper-gui --help
interview-helper-install-turn-models --help
```

The minimal wheel smoke checks base dependencies and entry points. A full desktop
installation additionally needs the `desktop` extra, system libraries, audio
hardware, model downloads, and provider setup. CI tests Python 3.11 and 3.12;
it does not capture hardware audio, fetch models, or make paid model calls.

## Upload

Review the clean export and commit it on `main`. Create the repository explicitly
as private, push that commit, and verify GitHub reports private visibility.
Use the initial `v0.1.0` tag for an experimental prerelease. Attach the built
wheel, source archive, and a SHA-256 checksums file; do not attach profiles or
model weights. Confirm remote commit/tag IDs and attached asset names afterward.

Report local test/build results separately from GitHub Actions results and
physical/audio qualification. A private upload and passing CI do not establish
overall answer accuracy or production readiness.
