import hashlib
import io
from pathlib import Path
import tarfile
import threading

import pytest

from interview_helper import model_catalog as catalog


def asset(data: bytes) -> catalog._Asset:
    return catalog._Asset('model', 'https://example.invalid/model', len(data), hashlib.sha256(data).hexdigest())


def test_download_verifies_and_reuses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = b'verified bytes'
    calls = []
    def open_url(request: object, timeout: int) -> io.BytesIO:
        calls.append(request)
        return io.BytesIO(data)
    monkeypatch.setattr(catalog.urllib.request, 'urlopen', open_url)
    path = tmp_path / 'model'
    events = []
    for _ in range(2):
        catalog._download(asset(data), path, None, lambda *args: events.append(args))
    assert path.read_bytes() == data
    assert len(calls) == 1
    assert events[-1] == (len(data), len(data), 'model')


@pytest.mark.parametrize('cancel', [False, True])
def test_failure_preserves_existing_and_cleans_partial(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cancel: bool) -> None:
    path = tmp_path / 'model'
    path.write_bytes(b'previous')
    event = threading.Event()
    def open_url(request: object, timeout: int) -> io.BytesIO:
        if cancel:
            event.set()
        return io.BytesIO(b'corrupt')
    monkeypatch.setattr(catalog.urllib.request, 'urlopen', open_url)
    with pytest.raises(RuntimeError):
        catalog._download(asset(b'verified bytes'), path, event, None)
    assert path.read_bytes() == b'previous'
    assert list(tmp_path.iterdir()) == [path]


def test_disk_space_checked_before_network(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(catalog.shutil, 'disk_usage', lambda _: type('Space', (), {'free': 0})())
    with pytest.raises(OSError, match='disk space'):
        catalog._download(asset(b'data'), tmp_path / 'model', None, None)


@pytest.mark.parametrize('name,link', [('../escape', ''), ('/absolute', ''), ('runtime/link', '../../escape'), ('runtime/link', '/escape')])
def test_archive_rejects_escaping_paths(tmp_path: Path, name: str, link: str) -> None:
    archive = tmp_path / 'archive.tar.gz'
    with tarfile.open(archive, 'w:gz') as out:
        member = tarfile.TarInfo(name)
        if link:
            member.type = tarfile.SYMTYPE
            member.linkname = link
        out.addfile(member)
    with pytest.raises(RuntimeError, match='Unsafe'):
        catalog._extract(archive, tmp_path, None)


def test_internal_library_links_are_materialized(tmp_path: Path) -> None:
    archive = tmp_path / 'archive.tar.gz'
    with tarfile.open(archive, 'w:gz') as out:
        member = tarfile.TarInfo('runtime/lib/real.so'); member.size = 4
        out.addfile(member, io.BytesIO(b'code'))
        member = tarfile.TarInfo('runtime/lib/link.so'); member.type = tarfile.SYMTYPE; member.linkname = 'real.so'
        out.addfile(member)
    catalog._extract(archive, tmp_path, None)
    path = tmp_path / 'runtime/lib/link.so'
    assert path.read_bytes() == b'code' and not path.is_symlink()


def test_cancel_before_install_and_invalid_device(tmp_path: Path) -> None:
    event = threading.Event(); event.set()
    with pytest.raises(catalog.DownloadCancelled):
        catalog.install_model('moonshine-small', root=tmp_path, cancel_event=event)
    with pytest.raises(ValueError, match='support'):
        catalog.install_model('moonshine-small', device='cuda', root=tmp_path)
    with pytest.raises(ValueError, match='Unknown'):
        catalog.install_model('unknown', root=tmp_path)


def test_installed_check_never_hashes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = catalog.MODEL_CATALOG[0]
    model = catalog.model_path(spec, tmp_path)
    model.mkdir(parents=True)
    fake = asset(b'data')
    monkeypatch.setitem(catalog._MOONSHINE, 'small', (fake,))
    (model / fake.name).write_bytes(b'data')
    monkeypatch.setattr(catalog, '_hash', lambda *_: pytest.fail('UI must not hash'))
    assert catalog.is_installed(spec, root=tmp_path)


def test_waiting_download_can_be_cancelled(tmp_path: Path) -> None:
    import fcntl
    event = threading.Event()
    errors: list[Exception] = []
    def run() -> None:
        try:
            catalog.install_model('moonshine-small', root=tmp_path, cancel_event=event)
        except Exception as exc:
            errors.append(exc)
    with (tmp_path / '.download.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        thread = threading.Thread(target=run)
        thread.start()
        event.set()
        thread.join(timeout=2)
    assert not thread.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], catalog.DownloadCancelled)
