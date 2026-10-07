# Distribution and user requirements

The initial delivery is a private GitHub repository and Python wheel/source
distribution, with all rights reserved for original project code. It is an
experimental Linux application, not a frozen executable or production guarantee.

## Required from each user

- A supported Linux graphical/audio session, Python, and the desktop dependencies.
- Headphones and microphone with a verified output-monitor route.
- Explicitly installed local transcription and turn-detection models.
- An answer backend: personal API credentials/billing or a separately configured
  local model server with sufficient hardware and context capacity.
- A factual résumé; accurate background and Q&A strongly improve useful answers.
- Prepared answers in their preferred wording, with personal facts distinguished
  from general technical knowledge and hypothetical examples.
- Permission for the recording/assistance context, and a decision about using
  a cloud answer provider and retaining completed text history locally.

No real résumé, completed Q&A, credential, recording, model weight, or local
interview history is part of the distributable example profile.

## Implemented capabilities

- Explicit session start/stop with visible local dual-source transcription.
- General mode with bounded factual retrieval; technical mode with complete
  prepared Q&A up to 40,000 characters.
- Local or OpenAI answer provider, one streamed request per accepted question.
- Separate actual-speech context, correction/discard controls, and local text
  history. Previous generated suggestions are not evidence of actual speech.
- Short plain-language answer prompts, plus technical Q&A wording guidance.
- Optional headless held-control daemon; narrow input-device access guidance.

## Proposed work, not implemented in this release

These items need separate product decisions and implementation; preparing this
release does not imply they are already supported.

1. **Per-user speaking profile.** A separate style field/file with tone, sentence
   length, vocabulary, preferred phrases, and several user-approved examples.
   Style must never supply personal facts or override accuracy. Verify against
   held-out questions and unknown-experience cases before claiming style matching.
2. **Persistent profiles and first-run setup.** Save selected context/provider/device
   paths and provide explicit model-install actions and readiness diagnostics.
   Avoid storing raw credentials in ordinary profile files.
3. **Distribution beyond Python.** Decide on supported distributions/architectures,
   dependency versions, desktop shortcut installation, upgrades, and eventual
   AppImage/Flatpak or other installers. Review bundled third-party licenses then.
4. **Broader provider support.** Validate account-visible models and API/server
   capabilities without assuming the current fixed selector works for everyone.
5. **Release qualification.** Test a fresh machine, real question boundaries,
   overlapping speech, echo, technical vocabulary, and grounded answers using
   saved, consented clips. Report meaningful word errors, speech-end/release to
   first and complete answer timing, and false output on silence.

Current unit/package checks do not close the existing natural-conversation,
answer-quality, or physical-hardware acceptance gaps. No new live capture is
required just to package and upload the application.
