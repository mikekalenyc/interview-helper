from __future__ import annotations

import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from interview_helper.nemotron import NemotronError, NemotronTranscriber


def pcm(value: float = 0.2, samples: int = 1280) -> bytes:
    return np.full(samples, value, dtype="<f4").tobytes()


def wait_for(predicate: Any) -> None:
    until = time.monotonic() + 3
    while not predicate() and time.monotonic() < until:
        time.sleep(0.005)
    assert predicate()


class Native:
    def __init__(self, blocked: bool = False, fail: bool = False) -> None:
        self.streams: list[dict[str, Any]] = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self.fail = fail
        self.destroyed = 0
        if not blocked:
            self.release.set()

    def create(self, path: Path, device: str, index: int) -> object:
        self.configuration = (path, device, index)
        return object()

    def destroy(self, recognizer: Any) -> None:
        self.destroyed += 1

    def open(self, recognizer: Any, keyterms: tuple[str, ...]) -> Any:
        stream: dict[str, Any] = dict(chunks=[], results=deque(), closed=False, keyterms=keyterms)
        self.streams.append(stream)
        return stream

    def push(self, stream: Any, data: bytes) -> None:
        stream['chunks'].append(data)
        n = len(stream['chunks'])
        stream['results'].append((f'partial {n}', False))

    def finish(self, stream: Any) -> None:
        stream['results'].append((f"final {len(stream['chunks'])}", True))

    def next(self, stream: Any) -> tuple[str, bool] | None:
        self.entered.set()
        assert self.release.wait(5)
        if self.fail:
            raise RuntimeError('native decoding failed')
        return stream['results'].popleft() if stream['results'] else None

    def close(self, stream: Any) -> None:
        stream['closed'] = True


@pytest.fixture
def paths(tmp_path: Path) -> tuple[Path, Path]:
    model, library = tmp_path / 'model.gguf', tmp_path / 'lib.so'
    model.write_bytes(b'model')
    library.write_bytes(b'library')
    return model, library


def make(paths: tuple[Path, Path], native: Native, **kwargs: Any) -> NemotronTranscriber:
    return NemotronTranscriber(paths[0], library_path=paths[1], backend=native, **kwargs)


def test_incremental_push_partials_tail_and_one_synchronous_final(paths: tuple[Path, Path]) -> None:
    native = Native()
    owner = make(paths, native, device='cuda', device_index=2, keyterms=('BGP',))
    partials, finals = [], []
    stream = owner.open_stream(partial_callback=partials.append, complete_callback=finals.append)
    a, b = pcm(), pcm(0.7)
    stream.add_pcm(a, 16000)
    wait_for(lambda: partials == ['partial 1'])
    stream.add_pcm(b, 16000)
    stream.finish()
    stream.finish()
    assert [t.text for t in finals] == ['final 2']
    assert native.streams[0]['chunks'] == [a, b]
    assert native.streams[0]['closed']
    assert native.streams[0]['keyterms'] == ('BGP',)
    assert native.configuration[1:] == ('cuda', 2)
    owner.close()
    owner.close()
    wait_for(lambda: native.destroyed == 1)


def test_native_segment_finals_do_not_end_application_turn(paths: tuple[Path, Path]) -> None:
    class Segments(Native):
        def push(self, stream: Any, data: bytes) -> None:
            stream['chunks'].append(data)
            stream['results'].extend([('first', True), ('sec', False), ('second', False)])
        def finish(self, stream: Any) -> None:
            stream['results'].append(('second', True))
    owner = make(paths, Segments())
    partials, finals = [], []
    stream = owner.open_stream(partial_callback=partials.append, complete_callback=finals.append)
    stream.add_pcm(pcm(), 16000)
    wait_for(lambda: len(partials) == 3)
    assert partials == ['first', 'first sec', 'first second']
    assert not finals
    stream.finish()
    assert finals[0].text == 'first second'
    owner.close()


@pytest.mark.parametrize('value', [0.0, 1e-9])
def test_synthetic_silence_no_false_transcript(paths: tuple[Path, Path], value: float) -> None:
    owner = make(paths, Native())
    results = []
    stream = owner.open_stream(partial_callback=results.append, complete_callback=lambda t: results.append(t.text))
    stream.add_pcm(pcm(value), 16000)
    stream.finish()
    assert results == ['']
    owner.close()


@pytest.mark.parametrize('data,rate', [(b'',16000), (b'x',16000), (pcm(),8000), (pcm(float('nan')),16000)])
def test_invalid_audio(paths: tuple[Path, Path], data: bytes, rate: int) -> None:
    owner = make(paths, Native())
    stream = owner.open_stream(partial_callback=lambda _: None, complete_callback=lambda _: None)
    with pytest.raises(NemotronError):
        stream.add_pcm(data, rate)
    owner.close()


@pytest.mark.parametrize('finishing', [False, True])
def test_close_during_decode_defers_release_and_suppresses_results(paths: tuple[Path, Path], finishing: bool) -> None:
    native = Native(blocked=True)
    owner = make(paths, native)
    results = []
    stream = owner.open_stream(partial_callback=results.append, complete_callback=results.append)
    stream.add_pcm(pcm(), 16000)
    assert native.entered.wait(2)
    thread = threading.Thread(target=stream.finish)
    if finishing:
        thread.start()
    owner.close()
    assert not native.destroyed
    assert not native.streams[0]['closed']
    if finishing:
        thread.join(1)
        assert not thread.is_alive()
    native.release.set()
    stream._worker.join(2)
    assert native.destroyed == 1 and native.streams[0]['closed']
    assert not results


@pytest.mark.parametrize('operation', ['add', 'finish'])
def test_worker_errors_are_surfaced(paths: tuple[Path, Path], operation: str) -> None:
    owner = make(paths, Native(fail=True))
    stream = owner.open_stream(partial_callback=lambda _: None, complete_callback=lambda _: None)
    stream.add_pcm(pcm(), 16000)
    assert stream._done.wait(2)
    with pytest.raises(NemotronError, match='native decoding failed'):
        stream.add_pcm(pcm(), 16000) if operation == 'add' else stream.finish()
    owner.close()


def test_callbacks_outside_shared_model_lock_and_source_isolation(paths: tuple[Path, Path]) -> None:
    native = Native()
    owner = make(paths, native)
    finals = []
    candidate = owner.open_stream(partial_callback=lambda _: None, complete_callback=finals.append)
    candidate.add_pcm(pcm(0.8),16000)
    def complete(value: Any) -> None:
        candidate.finish()
        finals.append(value)
    interviewer = owner.open_stream(partial_callback=lambda _: None, complete_callback=complete)
    interviewer.add_pcm(pcm(0.2),16000)
    interviewer.add_pcm(pcm(0.3),16000)
    interviewer.finish()
    assert [t.text for t in finals] == ['final 1', 'final 2']
    assert native.streams[0]['chunks'] == [pcm(0.8)]
    owner.close()


def test_buffer_limits(paths: tuple[Path, Path]) -> None:
    native = Native(blocked=True)
    owner = make(paths, native)
    stream = owner.open_stream(partial_callback=lambda _: None, complete_callback=lambda _: None)
    stream.add_pcm(pcm(),16000)
    assert native.entered.wait(2)
    stream._queue_limit = len(pcm())
    stream.add_pcm(pcm(),16000)
    with pytest.raises(NemotronError, match='limit'):
        stream.add_pcm(pcm(),16000)
    stream.close()
    native.release.set()
    owner.close()


def test_finish_timeout(paths: tuple[Path, Path]) -> None:
    native = Native(blocked=True)
    owner = make(paths,native)
    stream = owner.open_stream(partial_callback=lambda _: None, complete_callback=lambda _: None)
    stream._finish_timeout = 0.02
    stream.add_pcm(pcm(),16000)
    with pytest.raises(NemotronError, match='Timed out'):
        stream.finish()
    native.release.set()
    owner.close()


@pytest.mark.parametrize('options', [{'device':'auto'}, {'device_index':-1}])
def test_invalid_device(paths: tuple[Path,Path], options: Any) -> None:
    with pytest.raises(NemotronError,match='device index'):
        make(paths,Native(),**options)


def test_boolean_device_index_rejected(paths: tuple[Path, Path]) -> None:
    with pytest.raises(NemotronError, match='device index'):
        make(paths, Native(), device_index=True)


def test_runtime_cpu_to_cuda_requires_restart(paths: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    from interview_helper import nemotron
    from types import SimpleNamespace
    loaded = SimpleNamespace(path=paths[1])
    monkeypatch.setattr(nemotron, '_runtime', loaded)
    assert nemotron._get_runtime(paths[1], 'cpu') is loaded
    with pytest.raises(NemotronError, match='Restart the app'):
        nemotron._get_runtime(paths[1], 'cuda')


def test_loaded_cuda_runtime_can_serve_cpu_and_cuda(paths: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    from interview_helper import nemotron
    from types import SimpleNamespace
    (paths[1].parent / 'libggml-cuda.so').touch()
    loaded = SimpleNamespace(path=paths[1])
    monkeypatch.setattr(nemotron, '_runtime', loaded)
    assert nemotron._get_runtime(paths[1], 'cpu') is loaded
    assert nemotron._get_runtime(paths[1], 'cuda') is loaded


def test_ctypes_result_string_copied_before_native_release() -> None:
    import ctypes as C
    from types import SimpleNamespace
    from interview_helper.nemotron import _Native
    native = _Native.__new__(_Native)
    released = []
    def next_result(stream: Any, out: Any) -> int:
        out._obj.value = 123
        return 0
    native.lib = SimpleNamespace(
        nemo_speech_asr_stream_next=next_result,
        nemo_speech_asr_result_transcript=lambda handle, alt: b' native text ',
        nemo_speech_asr_result_is_final=lambda handle: True,
        nemo_speech_asr_result_destroy=lambda handle: released.append(handle.value),
    )
    assert native.next(C.c_void_p(1)) == (' native text ', True)
    assert released == [123]


def test_ctypes_native_errors_include_thread_local_message() -> None:
    from types import SimpleNamespace
    from interview_helper.nemotron import _Native
    native = _Native.__new__(_Native)
    native.lib = SimpleNamespace(nemo_speech_asr_last_error=lambda: b'GPU allocation failed')
    with pytest.raises(NemotronError, match='decode: GPU allocation failed'):
        native._check(3, 'decode')


def test_full_cuda_libraries_load_before_native_runtime(
    paths: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys
    from types import SimpleNamespace
    from interview_helper import nemotron
    order = []
    (paths[1].parent / 'libggml-cuda.so').touch()
    monkeypatch.setitem(sys.modules, 'onnxruntime', SimpleNamespace(
        get_available_providers=lambda: ['CPUExecutionProvider', 'CUDAExecutionProvider'],
        preload_dlls=lambda **kwargs: order.append(('preload', kwargs)),
    ))
    monkeypatch.setattr(nemotron, '_runtime', None)
    monkeypatch.setattr(nemotron, '_Native', lambda path: order.append(('native', path)) or object())
    nemotron._get_runtime(paths[1], 'cuda')
    assert order == [('preload', {'directory': ''}), ('native', paths[1])]


def test_native_cuda_does_not_require_gpu_ort_in_cpu_installation(
    paths: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys
    from types import SimpleNamespace
    from interview_helper import nemotron
    (paths[1].parent / 'libggml-cuda.so').touch()
    monkeypatch.setitem(sys.modules, 'onnxruntime', SimpleNamespace(
        get_available_providers=lambda: ['CPUExecutionProvider'],
        preload_dlls=lambda **kwargs: pytest.fail('CPU installation must not load GPU detector libraries'),
    ))
    nemotron._prepare_shared_cuda(paths[1])
