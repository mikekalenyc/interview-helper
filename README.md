# Interview Helper

A Linux desktop interview assistant that transcribes interviewer audio locally
and suggests short spoken answers using your own résumé, background, and prepared
Q&A. It also transcribes your microphone so follow-up answers can use what you
actually said. A separate command-line daemon supports hold-to-transcribe.

**Status:** experimental, version 0.1.1. **License:** proprietary, all rights
reserved. Access to this repository does not grant redistribution rights; see
[LICENSE](LICENSE) and [third-party notices](THIRD_PARTY_NOTICES.md).

## What you need

- A Linux desktop with PipeWire's PulseAudio compatibility service or PulseAudio,
  plus the `pactl` and `parec` tools.
- Python 3.11 or newer; Python 3.12 is the current tested development environment.
- Headphones and a microphone for the desktop workflow.
- Local Moonshine speech-recognition and Silero/Smart Turn detection models,
  downloaded explicitly during setup.
- An answer provider: your own OpenAI API credentials and billing, or a separately
  running local Qwen server with an OpenAI-compatible API.
- A résumé in UTF-8 text or Markdown to enable suggested answers. Add factual
  background/project notes and prepared Q&A for your subject and preferred wording.

The desktop does not require raw keyboard access, an `input` group membership,
or a GPU for transcription. A local answer model has separate hardware
requirements. Windows/macOS and a standalone desktop installer are not provided.

## Install and start

For users who have repository access:

```bash
git clone https://github.com/mikekalenyc/interview-helper.git
cd interview-helper
python3 -m venv .venv
.venv/bin/python -m pip install '.[desktop]'
```

Then follow [the setup guide](docs/SETUP.md) to install system packages, download
the models, select an answer provider, and start your first session. The
`desktop` extra includes Qt, Moonshine, ONNX Runtime, and Transformers. Starting
the app does not download models or start recording.

For a wheel downloaded from this repository's Releases page:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install './interview_helper-0.1.1-py3-none-any.whl[desktop]'
```

The package is not published to PyPI. Wheel users still need the same system
packages and explicit model setup described in the guide.

## Prepare your own answers

Read [Conversational and technical modes](docs/MODES.md) for a function-by-function
comparison, preparation steps, example questions/answers, and launch commands.

Copy the [example templates](examples/) to a private folder and replace the
placeholders. The [personalization guide](docs/PERSONALIZATION.md) explains which
files to provide and how each mode uses them.

- **Conversational mode (general interviews):** résumé and background/Q&A are treated as your confirmed
  personal experience. Only relevant excerpts are selected. Include accurate
  role, actions, outcomes, and limits of your experience.
- **Technical interviews:** enable **Technical mode** and select a prepared Q&A
  file. Its full contents, up to 40,000 characters, guide the subject matter and
  plain wording. These technical answers do not prove personal experience.
- **Speaking style:** the current prompts use fixed short, plain spoken wording.
  Technical Q&A can influence phrasing. There is no separate per-user style
  profile, automatic training, or guarantee of reproducing an individual's voice.

[Requirements and remaining work](docs/REQUIREMENTS.md) separates supported
behavior from proposed personalization and distribution improvements.

## During a session

In Setup, select your résumé, context files, microphone, and answer provider.
Press **Start Interview** to begin visible capture of headphone output and your
microphone. Only accepted interviewer questions trigger suggestions. Use
**Answer now** if automatic detection misses a turn, or **Cancel question / answer**
to discard unfinished work. **Interview Done** stops capture and clears live
conversation context. Closing the window also stops capture.

Questions and streamed answers appear in separate panels; **A− / A+** adjust
answer text. **You said** lets you correct or discard microphone transcripts.
**Retrieved sources** shows selected evidence. Completed text is reviewable in
**History**; saved history is not reloaded as context for future sessions.

Use headphones and disable microphone playback into the captured output.
Automatic turn boundaries, overlapping speech, echo, recognition of technical
terms, and answer grounding can still fail. Physical-device validation is
separate from unit tests. No universal accuracy or latency claim is made.

## Data handling

Audio recognition and document lookup run locally. Recordings are not retained
by default. Desktop sessions save completed questions, suggestions, and finalized
candidate speech as local text history under
`${XDG_DATA_HOME:-~/.local/share}/interview-helper/interviews/`.

Local mode sends selected text to the configured local endpoint. OpenAI mode
sends selected résumé/background, prepared technical answers when enabled,
interviewer transcripts, and bounded conversation text to OpenAI. Audio stays
local. Select the provider intentionally and supply your own credentials;
credentials are not included in the package or saved in history.

Keep personal files outside the checkout or under its ignored `private/` folder.
Use the app only where recording and assistance are permitted. Review suggested
answers before relying on them.

## Headless daemon and development

The held-control CLI is optional: install `.[input,moonshine]`, then use
`interview-helper run --help`. It needs access to one selected input device;
[deployment guidance](deployment/README.md) describes narrow udev permissions
and the optional user service. Never run the daemon as root.

```bash
.venv/bin/python -m pip install -e '.[dev,gui]'
.venv/bin/pytest
.venv/bin/mypy interview_helper
.venv/bin/python -m build
```

Tests use synthetic fixtures and do not require recording, input permissions,
model downloads, or paid API calls. The GUI tests use Qt's offscreen platform.
See [release instructions](docs/RELEASING.md) for the curated source export,
package checks, and private GitHub release process.
