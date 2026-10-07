# Third-party components

The application license applies to the project's original code. Dependencies,
adapted code, and separately downloaded models retain their own licenses.

- `interview_helper/turn_detection.py` adapts preprocessing from Daily's
  [Smart Turn](https://github.com/pipecat-ai/smart-turn) (BSD-2-Clause) and
  recurrent-state handling from [Silero VAD](https://github.com/snakers4/silero-vad)
  (MIT). Their complete notices are retained in that source file, included in
  both the source distribution and wheel.
- The explicit turn-model installer downloads pinned public model files and
  their license notices, with SHA-256 verification. Model weights are not bundled
  in this repository or the Python distributions.
- [Moonshine Voice](https://github.com/moonshine-ai/moonshine) and its models are
  installed separately. Consult the license of the selected package/model.
- The optional desktop uses [Qt for Python / PySide6](https://doc.qt.io/qtforpython-6/licenses.html),
  which offers LGPL/GPL and commercial licensing. Other Python dependencies
  retain the licenses in their installed distribution metadata.

The release contains Python application code, not a frozen executable with
bundled dependencies or model weights. A future installer bundling third-party
components needs its own distribution-license review.
