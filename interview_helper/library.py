"""Offline, source-attributed retrieval from a downloaded technical library."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import unicodedata
from urllib.parse import urlsplit


@dataclass(frozen=True)
class ReferencePassage:
    title: str
    url: str
    text: str
    downloaded_at: str
    locator: str


_STOP = set('a an the and or to of in on for with between is are was were how what why when where which can could would should do does did you your i me my it this that tell explain about use using used from through into need needs own placing place reaching reach stop stops now have has their them they its then instead'.split())
_ALIASES = {'tgw': 'transit gateway', 'vpc': 'virtual private cloud',
            'gwlb': 'gateway load balancer', 'az': 'availability zone',
            'bgp': 'border gateway protocol', 'nacl': 'network access control list',
            'zpa': 'zscaler private access', 'mtls': 'mutual tls'}
_VERSION = '5'
_FOLLOWUP_WORDS = set(
    'work works working happen happens handle handles fail fails failure failures '
    'benefit benefits risk risks tradeoff tradeoffs advantage advantages disadvantage '
    'disadvantages more detail details example examples elaborate practical practice '
    'there they them those these its their then also affect affects impact impacts '
    'limitation limitations result results outcome outcomes'.split()
)


def _tokens(text: str) -> list[str]:
    text = unicodedata.normalize('NFKC', text).lower()
    text = re.sub(r'\b(?:network access control lists?|network acls?|nacls?)\b', 'nacl', text)
    text = re.sub(r'\bsecurity groups?\b', 'securitygroup', text)
    text = re.sub(r'\b(?:limitations?|limits?|restrictions?|quotas?)\b', 'limitation', text)
    for acronym, phrase in _ALIASES.items():
        text = re.sub(r'\b' + re.escape(phrase) + r'\b', acronym, text)
    return list(dict.fromkeys(t for t in re.findall(r'[a-z0-9]+', text) if t not in _STOP and len(t) > 1))


def is_generic_followup(question: str) -> bool:
    return (bool(question.strip()) and len(question.split()) <= 12
            and set(_tokens(question)) <= _FOLLOWUP_WORDS)


def _inside(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if Path(value).is_absolute() or not path.is_relative_to(root):
        raise ValueError('Technical library path escapes its root: ' + value)
    return path


def _read(path: Path, maximum: int) -> str:
    with path.open('rb') as stream:
        data = stream.read(maximum + 1)
    if len(data) > maximum:
        raise ValueError(f'Technical library file exceeds {maximum} bytes: {path}')
    return data.decode('utf-8')


def _chunks(text: str, title: str) -> list[tuple[str, str]]:
    """Keep page and Markdown headings attached; remove obvious catalog noise."""
    chunks: list[tuple[str, str]] = []
    heading = title
    page = ''
    buffer = ''

    def flush() -> None:
        nonlocal buffer
        if len(buffer.strip()) >= 40:
            chunks.append((heading + '\n' + buffer.strip(), page or heading))
        buffer = ''

    for raw in unicodedata.normalize('NFKC', text).splitlines():
        line = raw.strip()
        match = re.match(r'## PDF page (\d+)\s*$', line)
        if match:
            flush()
            page = 'PDF physical page ' + match.group(1)
            heading = title
            continue
        if re.match(r'^#{1,6}\s+', line):
            flush()
            heading = re.sub(r'^#+\s+', '', line)[:240]
            continue
        if (not line or re.search(r'\.{4,}', line)
                or re.fullmatch(r'[ivxlcdm]+', line)
                or line in {title, 'Amazon VPC', 'AWS Transit Gateway'}
                or line.startswith(('Content type:', 'Source:', 'Downloaded (UTC):',
                                        '[Skip to content]', '> Documentation Index',
                                        '> Fetch the complete', '> Use this file', 'Last updated'))):
            continue
        while line:
            if len(buffer) >= 2100:
                flush()
            room = 2200 - len(buffer)
            if len(line) <= room:
                buffer += line + '\n'
                break
            split = line.rfind(' ', 0, room)
            if split < room // 2:
                split = room
            buffer += line[:split]
            line = line[split:].lstrip()
            flush()
    flush()
    return chunks


class TechnicalLibrary:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    @classmethod
    def open(cls, root: Path, *, cache_path: Path | None = None) -> TechnicalLibrary:
        root = root.expanduser().resolve()
        connection: sqlite3.Connection | None = None
        try:
            snapshot = _inside(root, _read(root / 'LATEST.txt', 1024).strip())
            manifest_path = _inside(root, str((snapshot / 'manifest.json').relative_to(root)))
            manifest_text = _read(manifest_path, 2_000_000)
            manifest = json.loads(manifest_text)
            docs = manifest.get('documents') if isinstance(manifest, dict) else None
            if not isinstance(docs, list) or not docs or len(docs) > 2000:
                raise ValueError('Manifest must contain 1–2000 documents')
            files: list[tuple[Path, str, str, str]] = []
            fingerprint = hashlib.sha256((_VERSION + str(root) + str(snapshot) + manifest_text).encode())
            total = 0
            for doc in docs:
                if not isinstance(doc, dict) or doc.get('status') != 'downloaded':
                    raise ValueError('Manifest contains a missing or failed document; refresh the library')
                title, url, date, directory = (doc.get(k) for k in ('title', 'source_url', 'downloaded_at', 'directory'))
                if not all(isinstance(v, str) and v for v in (title, url, date, directory)):
                    raise ValueError('Manifest document requires title, source_url, downloaded_at and directory')
                assert isinstance(title, str) and isinstance(url, str) and isinstance(date, str) and isinstance(directory, str)
                if urlsplit(url).scheme not in ('http', 'https') or not urlsplit(url).hostname:
                    raise ValueError('Manifest source URL must be HTTP(S)')
                folder = _inside(root, directory)
                if not folder.is_relative_to(snapshot):
                    raise ValueError('Manifest document must belong to the selected snapshot')
                paths = sorted((folder / 'sections').glob('*.md')) if doc.get('vendor') == 'AWS' else [folder / 'reference.md']
                if not paths:
                    raise ValueError(f'No extracted references found for {title}')
                for path in paths:
                    path = _inside(root, str(path.relative_to(root)))
                    stat = path.stat()
                    total += stat.st_size
                    if stat.st_size > 4_000_000 or total > 100_000_000 or len(files) >= 10_000:
                        raise ValueError('Technical library exceeds indexing limits')
                    fingerprint.update(f'{path}:{stat.st_size}:{stat.st_mtime_ns}'.encode())
                    files.append((path, title, url, date))
            if cache_path is None:
                key = hashlib.sha256(str(root).encode()).hexdigest()[:16]
                cache_path = Path.home() / '.cache' / 'interview-helper' / f'technical-library-{key}.sqlite3'
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(cache_path, check_same_thread=False)
            connection.execute('CREATE TABLE IF NOT EXISTS metadata (fingerprint TEXT)')
            existing = connection.execute('SELECT fingerprint FROM metadata').fetchone()
            if existing != (fingerprint.hexdigest(),):
                with connection:
                    connection.execute('DROP TABLE IF EXISTS passages')
                    connection.execute('CREATE VIRTUAL TABLE passages USING fts5(terms, title UNINDEXED, url UNINDEXED, text UNINDEXED, downloaded_at UNINDEXED, locator UNINDEXED)')
                    seen: set[str] = set()
                    count = 0
                    for path, title, url, date in files:
                        for text, locator in _chunks(_read(path, 4_000_000), title):
                            digest = hashlib.sha256(text.encode()).hexdigest()
                            if digest in seen:
                                continue
                            seen.add(digest)
                            count += 1
                            if count > 60_000:
                                raise ValueError('Technical library exceeds passage limit')
                            connection.execute('INSERT INTO passages VALUES (?, ?, ?, ?, ?, ?)',
                                               (' '.join(_tokens(text)), title, url, text, date, locator))
                    if not count:
                        raise ValueError('Technical library has no usable extracted text')
                    connection.execute('DELETE FROM metadata')
                    connection.execute('INSERT INTO metadata VALUES (?)', (fingerprint.hexdigest(),))
            return cls(connection)
        except (OSError, UnicodeError, ValueError, sqlite3.Error) as exc:
            if connection is not None:
                connection.close()
            raise ValueError(f'Cannot open technical library at {root}: {exc}') from exc

    def close(self) -> None:
        self._connection.close()

    def search(self, question: str, *, recent_question: str = '', max_characters: int = 10000,
               limit: int = 5) -> tuple[ReferencePassage, ...]:
        max_characters = min(max_characters, 20000)
        current = _tokens(question[:4000])[:40]
        terms = current
        # Reuse the previous subject only for short, generic follow-ups. A named
        # technical topic (including a new acronym) always stands on its own.
        if is_generic_followup(question) and recent_question.strip():
            terms = list(dict.fromkeys([t for t in _tokens(recent_question[:2000]) if t not in _INTENT_WORDS] + current))[:40]
        if not terms or limit <= 0 or max_characters <= 0:
            return ()
        intents = set(terms) & _INTENT_WORDS
        subjects = set(terms) - _INTENT_WORDS - _FOLLOWUP_WORDS - {'compare', 'comparison', 'difference', 'differences', 'versus', 'vs'}
        if not subjects:
            return ()
        phrases = _query_phrases(question, subjects)
        # A limitations follow-up about several named controls needs evidence
        # for each, not two passages about whichever term dominates the index.
        entities = [term for term in terms if term in {*_ALIASES, 'securitygroup'}]
        entities = list(dict.fromkeys(entities))
        if 'limitation' in intents and len(entities) > 1 and limit > 1:
            selected: list[ReferencePassage] = []
            for entity in entities[:min(limit, 10)]:
                selected.extend(self.search(
                    entity + ' limitation', max_characters=max_characters // min(len(entities), limit, 10), limit=1,
                ))
            return tuple(selected)
        # FTS operators and punctuation never reach MATCH; all tokens are quoted.
        query = ' OR '.join('"' + term + '"' for term in sorted(subjects))
        rows = self._connection.execute(
            'SELECT title, url, text, downloaded_at, locator, bm25(passages) FROM passages WHERE passages MATCH ? ORDER BY bm25(passages) LIMIT 300', (query,)).fetchall()
        # Preserve candidates containing a multiword subject before reranking;
        # a broad OR query can otherwise crowd them out with generic matches.
        # AND uses only indexed words; original adjacency is checked below.
        for phrase in phrases[:8]:
            parts = phrase.split()
            if any(len(part) < 2 for part in parts):
                continue
            conjunction = ' AND '.join('"' + part + '"' for part in parts)
            rows.extend(self._connection.execute(
                'SELECT title, url, text, downloaded_at, locator, bm25(passages) '
                'FROM passages WHERE passages MATCH ? ORDER BY bm25(passages) LIMIT 100',
                (conjunction,),
            ).fetchall())
        rows = list({(row[1], row[2], row[4]): row for row in rows}.values())
        rows = [row for row in rows if subjects & set(_tokens(row[2]))]
        rows.sort(key=lambda row: (_subject_score(row[2], subjects, intents, phrases), -row[5]), reverse=True)
        result: list[ReferencePassage] = []
        seen: list[set[str]] = []
        remaining = min(max_characters, 20000)
        for title, url, text, date, locator, _score in rows:
            words = set(_tokens(text))
            if any(len(words & previous) / max(1, len(words | previous)) > .85 for previous in seen):
                continue
            passage = _relevant_excerpt(text, terms, min(remaining, max(1, max_characters // min(limit, 10))), phrases)
            result.append(ReferencePassage(title, url, passage, date, locator))
            seen.append(words)
            remaining -= len(passage)
            if len(result) >= min(limit, 10) or remaining <= 0:
                break
        return tuple(result)


def _relevant_excerpt(text: str, terms: list[str], limit: int,
                      phrases: tuple[str, ...] = ()) -> str:
    """Keep query-bearing text when the prompt budget is smaller than a page."""
    if len(text) <= limit:
        return text
    starts = [0, *[match.end() for match in re.finditer(r'\n|[.!?]\s+', text)]]
    candidates = []
    for start in starts:
        if start >= len(text):
            continue
        excerpt = text[start:start + limit]
        if start + limit < len(text):
            boundaries = [m.end() for m in re.finditer(r'[.!?](?:\s|$)', excerpt)]
            if boundaries and boundaries[-1] >= limit // 2:
                excerpt = excerpt[:boundaries[-1]]
            elif '\n' in excerpt:
                excerpt = excerpt.rsplit('\n', 1)[0]
        candidates.append(excerpt)
    subjects = set(terms) - _INTENT_WORDS - _FOLLOWUP_WORDS - {'compare', 'comparison', 'difference', 'differences', 'versus', 'vs'}
    intents = set(terms) & _INTENT_WORDS
    return max(candidates, key=lambda excerpt: (
        _subject_score(excerpt, subjects, intents, phrases),
        -next((index for index, word in enumerate(_tokens(excerpt)) if word in terms), 10000),
    ))


_INTENT_WORDS = {'limitation', 'compare', 'comparison', 'difference', 'differences', 'versus', 'vs', 'benefit', 'benefits', 'risk', 'risks', 'tradeoffs'}


def _query_phrases(question: str, subjects: set[str]) -> tuple[str, ...]:
    """Preserve adjacent subject words; the FTS index stores unique tokens.

    Check phrases against source prose, not FTS positions: deduplicated index
    terms cannot establish adjacency in the original document.
    """
    words = re.findall(r'[a-z0-9]+', question.lower())
    return tuple(dict.fromkeys(
        left + ' ' + right for left, right in zip(words, words[1:])
        if left in subjects and (right in subjects or len(right) == 1)
    ))


def _subject_score(text: str, subjects: set[str], intents: set[str],
                   phrases: tuple[str, ...] = ()) -> float:
    """Prefer sections about the subjects over pages mentioning them in passing."""
    lines = [set(_tokens(line)) for line in text.splitlines()]
    words = set(_tokens(text))
    headings = [set(_tokens(line)) for line in text.splitlines()
                if len(line.split()) <= 12 and not line.rstrip().endswith(('.', ','))
                and not line.lstrip().startswith(('•', '-'))]
    prose = ' '.join(re.findall(r'[a-z0-9]+', text.lower()))
    phrase_score = sum(30 for phrase in phrases
                       if re.search(r'\b' + re.escape(phrase) + r's?\b', prose))
    # Definitions often label a subject followed by a dash/colon. Prefer that
    # direct explanation over a procedural page repeating the subject in CLI
    # arguments or a table. Keep source punctuation for this signal.
    phrase_score += sum(25 for phrase in phrases if len(phrase.split()[-1]) > 1 and re.search(
        r'\b' + re.escape(phrase).replace(r'\ ', r'\s+') + r's?\s*[:–—]', text.lower(),
    ))
    return (phrase_score + 8 * len(subjects & words)
            + sum(2 * min(2, sum(term in line for line in lines)) for term in subjects)
            + 5 * sum(any(term in heading for heading in headings) for term in subjects)
            + 4 * len(intents & words)
            + 4 * sum(any(term in heading for heading in headings) for term in intents))
