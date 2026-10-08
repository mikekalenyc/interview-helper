# Setup

## 1. System and Python dependencies

On a Debian/Ubuntu desktop with a working PipeWire/PulseAudio session:

```bash
sudo apt install python3-venv pulseaudio-utils libegl1 libgl1 libxkbcommon-x11-0 libxcb-cursor0
python3 -m venv .venv
.venv/bin/python -m pip install '.[desktop]'
```

These are setup commands for the user to run. Distribution package names can
differ. The application itself runs as the logged-in desktop user, never root.
Python dependencies are declared in `pyproject.toml`; no separate requirements
file is needed. `desktop` installs all four optional desktop runtime components.
Use the wheel install command in the README when installing a release asset.

Check audio configuration without recording:

```bash
.venv/bin/interview-helper audio
pactl get-default-sink
pactl get-default-source
pactl list short sources
```

The interviewer must play through the selected/default headphone output. The
app captures its monitor source, not a specific conferencing application.
Other sound playing through that output can also be transcribed during a session.
Select a real microphone in Setup. Keep sidetone/microphone playback out of the
captured output to avoid echo.

## 2. Install the local models explicitly

Open the application, go to **Setup**, choose a **Transcription model** and
**Transcription device**, then click **Download selected model**. The app shows
the approximate download size, progress, cancellation, and installed status.
It stores verified files locally and reuses them. Selecting a model alone does
not download it, and an interview never triggers a download.

Start with **Moonshine Small / CPU**. See [model and device choices](MODELS.md)
for Medium and Nemotron, NVIDIA requirements, and optional GPU detection setup.

For a headless installation, the same explicit downloader is callable in Python:

```bash
.venv/bin/python - <<'PY'
from interview_helper.model_catalog import install_model
print(install_model('moonshine-small'))
PY
```

Install the pinned CPU speech/turn-detection models and their license notices:

```bash
.venv/bin/interview-helper-install-turn-models
```

This command works in both source and wheel installations. It verifies SHA-256
hashes and stores files in `~/.local/share/interview-helper/turn-models/`.
Transcription files are in `~/.local/share/interview-helper/models/`.
The default Moonshine path ends with
`download.moonshine.ai/model/small-streaming-en/quantized_26_08_21` beneath the
model directory above. Advanced users may browse to an existing compatible model.
Select its matching model type; a different model file cannot be made compatible
just by renaming it. The app never downloads missing models on startup.

## 3. Prepare personal context

Use the [personalization guide](PERSONALIZATION.md). A nonempty résumé is required
for generated suggestions, including technical mode. Without a résumé, the app
can transcribe without generating answers. Export PDF/DOCX to UTF-8 `.txt`, `.md`,
or `.markdown` first.

Do not load the unfilled templates as personal evidence. Store completed copies
outside the repository or under `private/` and pass only the files you intend
the app to use. GUI setup choices are not currently saved between launches;
launcher arguments can preselect your files and provider.

## 4. Choose an answer provider

For OpenAI, use your own API account with billing and access to a model in the
app's model selector. Choose **OpenAI Luna** in Setup and select a credential
file, or supply `OPENAI_API_KEY` in the launching environment. A key file may
contain a plain-text key or one unbroken key in an RTF document. Only its path is
shown; do not put keys in command arguments, repository files, or screenshots.
An explicitly selected key file takes precedence over the environment.

Example using a private key file and context paths:

```bash
.venv/bin/interview-helper-gui \
  --answer-provider openai \
  --openai-api-key-file "$HOME/.config/interview-helper/api-key.txt" \
  --resume "$HOME/private/interview/resume.md" \
  --context "$HOME/private/interview/background.md" \
  --technical-answers "$HOME/private/interview/technical-answers.md"
```

Omit `--technical-answers` for general interviews. Technical mode is also
selectable in Setup. The current model list is fixed by the application; access
and request compatibility depend on your account. A missing model, rejected
request, or quota error is shown explicitly, with no silent fallback or retry.

For local mode, separately configure a Qwen server accepting the app's
OpenAI-compatible streaming chat-completions request with thinking disabled.
The defaults are `http://127.0.0.1:8082/v1` and
`qwen3.8-27b-uncensored`; change the URL/model in Setup for your server.
The package does not install an answer server or its weights. Hardware and
context capacity depend on the selected model. Other compatible servers need
their own validation; do not assume any endpoint is interchangeable.

## 5. First session and troubleshooting

Open the app, review Setup, then click **Start Interview**. Both audio sources
remain visible. Click **Interview Done** to stop. A first hardware check should
record the exact sink, monitor, and microphone source names and use a short,
consented test question. Do not infer hardware readiness from passing unit tests.

- Missing audio commands: install `pulseaudio-utils` and verify the desktop audio
  server is running. A failed `pactl` connection means the session is unavailable.
- Missing models: run the explicit installation steps above and verify the model
  directory selected in Setup.
- GPU unavailable: select CPU or install the supported NVIDIA driver/runtime.
  An explicit GPU request fails visibly if initialization cannot use it; the app
  does not silently switch transcription or detector inference to CPU.
- Qt platform/plugin error: check the distribution's Qt/XCB/OpenGL runtime
  packages and launch from a graphical desktop session.
- No answer: verify that a résumé is selected and the chosen provider is reachable.
- Cut-off answer: the model hit the configured output cap; the app reports an
  error instead of saving an incomplete answer as a completed turn.
- Incorrect question boundary: use **Answer now** or cancel and repeat. Natural
  pauses and echo remain experimental.

The optional technical-library field selects a directory containing an
application-compatible snapshot manifest, not an arbitrary PDF folder.
Leave it blank for initial setup. Personal
guides and prepared Q&A do not need that library.
