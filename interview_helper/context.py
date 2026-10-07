"""Local candidate-profile loading and deterministic relevance selection."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path


SUPPORTED_SUFFIXES = {".txt", ".md", ".markdown"}
_WORD = re.compile(r"[a-z0-9][a-z0-9+#.-]*")
_STOP_WORDS = {
    "about", "and", "are", "can", "did", "for", "from", "have", "how",
    "into", "that", "the", "this", "was", "what", "when", "where", "which",
    "with", "would", "you", "your",
}


class ContextError(ValueError):
    """Candidate context could not be loaded safely."""


def structured_candidate_text(text: str) -> str:
    """Annotate supplied evidence without rewriting or inferring any facts.

    These conservative lexical hints are not a truth validator. In particular,
    an unclassified statement remains usable evidence, and an outcome label
    means only that the source explicitly uses result wording. Preserve lines
    and their order so qualifications, headings, and Q&A remain attached.
    """
    annotated: list[str] = []
    for line in text.splitlines(keepends=True):
        words = line.lower()
        # Unknowns and qualifications outrank apparent result/metric cues.
        if re.search(
            r"\b(?:unknown|unmeasured|unverified|not|never|no|didn't|"
            r"could not|couldn't|cannot|can't|without measuring)\b", words,
        ):
            label = "Qualified/unknown evidence; preserve qualification"
        elif re.search(
            r"\b(?:goal|goals|aim|aimed|aims|intended|intention|target|"
            r"targeted|expected|planned|hope|hoped|would|could|should|might|may|"
            r"to reduce|to improve|to increase|to prevent|to avoid|"
            r"to ensure|to enable|to minimize|to eliminate|"
            r"ensuring|preventing)\b", words,
        ):
            label = "Action/goal or intended benefit; not proof of achieved outcome"
        elif not words.rstrip().endswith("?") and re.search(
            r"\b(?:measured|observed|achieved|resulted in)\b|"
            r"\b(?:result|outcome|results|outcomes)\s*:\s*\S|"
            r"\b(?:reduced|increased|improved|decreased)\b[^.!?\n]*\d",
            words,
        ):
            label = "Source-stated outcome/measurement; retain exact scope"
        else:
            label = ""
        if label:
            annotated.append(f"[{label}]\n")
        annotated.append(line)
    return "".join(annotated)


@dataclass(frozen=True, slots=True)
class ContextChunk:
    source: str
    kind: str
    text: str
    order: int
    heading: tuple[str, ...] = ()
    topic: str = ""


TECHNICAL_ANSWERS_LIMIT = 40_000


@dataclass(frozen=True, slots=True)
class TechnicalAnswers:
    """Prepared technical Q&A sent whole in technical mode; never personal evidence."""

    source: str
    text: str

    @classmethod
    def load(cls, path: Path) -> TechnicalAnswers:
        path = path.expanduser()
        _validate_file(path)
        text = _read_text(path)
        if len(text) > TECHNICAL_ANSWERS_LIMIT:
            raise ContextError(
                f"Technical answers file exceeds {TECHNICAL_ANSWERS_LIMIT:,} characters: {path}"
            )
        return cls(path.name, text)


@dataclass(frozen=True, slots=True)
class CandidateProfile:
    chunks: tuple[ContextChunk, ...]

    @classmethod
    def load(
        cls,
        resume: Path,
        context_paths: tuple[Path, ...] = (),
        *,
        chunk_characters: int = 1_800,
    ) -> CandidateProfile:
        if chunk_characters < 200:
            raise ContextError("Context chunk size must be at least 200 characters")
        documents = [(resume, "resume")]
        for path in context_paths:
            documents.extend((item, "project") for item in _expand_path(path))

        chunks: list[ContextChunk] = []
        names = Counter(path.name for path, _ in documents)
        occurrences: Counter[str] = Counter()
        for path, kind in documents:
            if kind == "resume":
                _validate_file(path)
            text = _read_text(path)
            occurrences[path.name] += 1
            source = path.name
            if names[source] > 1:
                source = f"{occurrences[source]}: {source}"
            sections = (
                _markdown_sections(text, kind)
                if path.suffix.lower() in {".md", ".markdown"}
                else [((), kind, text)]
            )
            for heading, section_kind, body in sections:
                prefix = " > ".join(heading)
                # Keep full ancestry as metadata; cap its prompt rendering so
                # a long question heading cannot crowd out the answer body.
                if len(prefix) > chunk_characters // 2:
                    prefix = prefix[:chunk_characters // 2 - 1] + "…"
                prefix = f"{prefix}\n\n" if prefix else ""
                for part in _chunk_text(body, max(1, chunk_characters - len(prefix))):
                    chunks.append(ContextChunk(
                        source=source, kind=section_kind, text=prefix + part,
                        order=len(chunks), heading=heading,
                        topic=heading[1] if len(heading) > 1 else (
                            heading[0] if heading else ""
                        ),
                    ))
        if not any(chunk.kind == "resume" for chunk in chunks):
            raise ContextError("Resume did not contain usable text")
        return cls(tuple(chunks))

    def fast_chunks(
        self, question: str, *, recent_question: str = "", max_characters: int = 2_400,
    ) -> tuple[ContextChunk, ...]:
        """Compact evidence from user-authored files, all treated as factual.

        The separate vendor library is not part of this profile. Legacy kind
        metadata still supports the older retrieval API; this explicitly trusted
        path promotes supplied Q/A and background to experience evidence.
        """
        if max_characters <= 0:
            raise ContextError("Context character limit must be positive")
        query = _fast_terms(question)
        recent = _fast_terms(recent_question)
        introduction = bool(re.search(
            r"\b(about yourself|your background|your career|through your resume)\b",
            question.lower(),
        ))
        if introduction:
            selected_resume: list[ContextChunk] = []
            remaining = max_characters
            for chunk in self.chunks:
                if chunk.kind == "resume" and remaining > 0:
                    excerpt = _clip_chunk(chunk, remaining)
                    selected_resume.append(excerpt)
                    remaining -= len(excerpt.text)
            return tuple(selected_resume)
        followup = bool(re.search(r"\b(it|that|those|they|them|this|its)\b", question.lower()))
        followup = followup or bool(query) and query <= {
            "result", "results", "outcome", "outcomes", "impact", "more", "detail",
            "details", "limitations", "benefits", "risks", "tradeoffs", "elaborate",
        }
        if followup or not query:
            query |= recent
        if not query:
            return ()
        words = {c.order: _fast_terms(c.text) for c in self.chunks}
        frequency = Counter(t for values in words.values() for t in values)

        def overlap(values: set[str]) -> float:
            return sum(math.log(1 + len(self.chunks) / frequency[t])
                       for t in query & values if t in frequency)

        def score(chunk: ContextChunk) -> float:
            heading = _fast_terms(chunk.heading[-1]) if chunk.heading else set()
            topic = _fast_terms(chunk.topic)
            # Headings disambiguate a passage; they must not multiply rare
            # question phrasing enough to overwhelm its actual subject match.
            base = overlap(words[chunk.order])
            return base + min(2.0, overlap(heading) * 0.5) + min(1.5, overlap(topic) * 0.5)

        ranked = sorted(self.chunks, key=lambda c: (-score(c), c.order))
        ranked = [c for c in ranked if score(c) > 0]
        if not ranked:
            return ()
        first = ranked[0]
        # Keep a matching answer and its actual background together before
        # considering additional topics. Do not spend the budget on resume order.
        companions = [c for c in ranked if c != first and c.source == first.source
                      and c.topic and c.topic == first.topic]
        candidates = [first, *companions, *[c for c in ranked[1:] if c not in companions]]
        selected: list[ContextChunk] = []
        remaining = max_characters
        for chunk in candidates:
            if remaining < min(160, max_characters):
                break
            allowance = remaining
            if not selected and companions:
                allowance = max(1, remaining * 2 // 3)
            clipped = _clip_chunk(chunk, allowance)
            if clipped.text:
                selected.append(replace(clipped, kind="resume" if chunk.kind == "resume" else "experience"))
                remaining -= len(clipped.text)
            if len(selected) >= 3:
                break
        return tuple(selected)

    def relevant_chunks(
        self,
        question: str,
        *,
        max_characters: int = 24_000,
        max_resume_characters: int = 12_000,
        recent_question: str = "",
    ) -> tuple[ContextChunk, ...]:
        """Select evidence; limits count text including Markdown ancestry."""
        if max_characters <= 0 or max_resume_characters <= 0:
            raise ContextError("Context character limits must be positive")
        resume_size = sum(len(c.text) for c in self.chunks if c.kind == "resume")
        if (sum(len(c.text) for c in self.chunks) <= max_characters
                and resume_size <= max_resume_characters):
            return self.chunks
        terms = _terms(question)
        # Previous questions resolve references, but must not pull the answer
        # back to an old topic after an explicit new question.
        followup = bool(re.search(r"\b(it|that|those|they|them|this)\b", question.lower()))
        recent = _terms(recent_question) if followup or len(terms) <= 2 else set()
        frequency = Counter(term for c in self.chunks for term in _terms(c.text))

        def score(chunk: ContextChunk) -> float:
            words = _terms(chunk.text)
            def overlap(query: set[str]) -> float:
                return sum(math.log(1 + len(self.chunks) / frequency[t])
                           for t in query & words)
            return overlap(terms) + overlap(recent) * (0.8 if len(terms) <= 2 else 0.15)

        selected: list[ContextChunk] = []
        used = 0
        resume_used = 0

        def add(chunk: ContextChunk) -> bool:
            nonlocal used, resume_used
            size = len(chunk.text)
            if chunk in selected or used + size > max_characters:
                return False
            if chunk.kind == "resume" and resume_used + size > max_resume_characters:
                return False
            selected.append(chunk)
            used += size
            if chunk.kind == "resume":
                resume_used += size
            return True

        # Leave room for guide evidence even when the resume is large.
        for chunk in self.chunks:
            if chunk.kind == "resume" and used + len(chunk.text) <= max_characters // 2:
                add(chunk)
        ranked = sorted(self.chunks, key=lambda c: (-score(c), c.order))
        for chunk in ranked:
            if score(chunk) <= 0 or not add(chunk):
                continue
            if chunk.kind in {"practice", "reference"} and chunk.topic:
                linked = [c for c in self.chunks
                          if c.source == chunk.source and c.topic == chunk.topic
                          and c.kind == "experience"]
                for companion in sorted(linked, key=lambda c: (-score(c), c.order)):
                    add(companion)
        return tuple(selected)


def _terms(text: str) -> set[str]:
    return {word for word in _WORD.findall(text.lower())
            if len(word) >= 3 and word not in _STOP_WORDS}


def _section_kind(heading: tuple[str, ...], default: str) -> str:
    if default == "resume":
        return default
    lowered = tuple(title.lower() for title in heading)
    # A practice/reference parent cannot be promoted to personal experience
    # by a nested heading such as "Resume bullet" in a sample answer.
    if any(any(term in title for term in (
        "hypothetical", "practice", "draft answer", "interview questions",
    )) for title in lowered):
        return "practice"
    if any(any(term in title for term in (
        "reference", "technical notes", "technical questions", "gotcha",
        "structured interview context",
    )) for title in lowered):
        return "reference"
    if any(any(term in title for term in (
        "table of contents", "how this material", "editing conventions",
        "limitations", "how to use",
    )) for title in lowered):
        return "reference"
    for title in reversed(lowered):
        if any(term in title for term in (
            "resume bullet", "backstory", "actual experience", "experience narrative",
        )):
            return "experience"
        if "questions & answers" in title or "questions and answers" in title:
            return "reference"
    # Guide introduction is instructional material, not candidate evidence.
    if len(heading) <= 1 and any("guide" in title for title in lowered):
        return "reference"
    return default


def _markdown_sections(text: str, default: str) -> list[tuple[tuple[str, ...], str, str]]:
    sections: list[tuple[tuple[str, ...], str, str]] = []
    stack: list[tuple[int, str]] = []
    lines: list[str] = []
    fence = ""
    fence_length = 0

    def flush() -> None:
        body = "\n".join(lines).strip()
        if body:
            heading = tuple(title for _, title in stack)
            kind = _section_kind(heading, default)
            if kind == "project" and ("**Topic ID:**" in body or not body.strip("-*_ \n")):
                kind = "reference"
            sections.append((heading, kind, body))
        lines.clear()

    for line in text.splitlines():
        stripped = line.lstrip()
        fence_match = re.match(r"(`{3,}|~{3,})(.*)$", stripped)
        if fence_match:
            marker = fence_match[1]
            if not fence:
                fence = marker[0]
                fence_length = len(marker)
            elif (fence == marker[0] and len(marker) >= fence_length
                  and not fence_match[2].strip()):
                fence = ""
            lines.append(line)
            continue
        match = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line) if not fence else None
        if match:
            flush()
            level, title = len(match[1]), match[2]
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
        else:
            lines.append(line)
    flush()
    return sections


def _expand_path(path: Path) -> tuple[Path, ...]:
    if not path.exists():
        raise ContextError(f"Context path does not exist: {path}")
    if path.is_file():
        _validate_file(path)
        return (path,)
    if not path.is_dir():
        raise ContextError(f"Context path is not a regular file or directory: {path}")
    files = tuple(
        sorted(
            item
            for item in path.rglob("*")
            if item.is_file() and item.suffix.lower() in SUPPORTED_SUFFIXES
        )
    )
    if not files:
        raise ContextError(
            f"Context directory contains no .txt or Markdown files: {path}"
        )
    return files


def _validate_file(path: Path) -> None:
    if not path.exists():
        raise ContextError(f"Context file does not exist: {path}")
    if not path.is_file():
        raise ContextError(f"Expected a context file, got: {path}")
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ContextError(
            f"Unsupported context format {path.suffix or '(none)'} for {path}; "
            "use UTF-8 .txt, .md, or .markdown"
        )


def _read_text(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except UnicodeDecodeError as error:
        raise ContextError(f"Context file is not valid UTF-8: {path}") from error
    except OSError as error:
        raise ContextError(f"Could not read context file {path}: {error}") from error
    if not text:
        raise ContextError(f"Context file is empty: {path}")
    return text


def _chunk_text(text: str, limit: int) -> tuple[str, ...]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        pieces = [
            paragraph[index : index + limit]
            for index in range(0, len(paragraph), limit)
        ]
        for piece in pieces:
            candidate = f"{current}\n\n{piece}" if current else piece
            if len(candidate) <= limit:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                current = piece
    if current:
        chunks.append(current)
    return tuple(chunks)


_FAST_STOP = _STOP_WORDS | {
    "tell", "explain", "does", "use", "used", "work", "worked", "were", "why",
    "actually", "also", "could", "should", "there", "their", "them", "they",
    "walk", "through", "exactly", "say", "said", "please", "give", "talk",
}


def _fast_terms(text: str) -> set[str]:
    text = text.lower()
    text = re.sub(r"service interruptions?|service disruptions?|outages?", "troubleshooting", text)
    text = re.sub(r"\b(role|responsibilities|responsibility)\b", "personally own", text)
    text = re.sub(r"\b(owned|ownership)\b", "own", text)
    text = re.sub(r"\bregions\b", "region", text)
    text = re.sub(r"\bdata[ -]?cent(?:er|re)s?\b", "datacenter", text)
    words: set[str] = set()
    for raw in re.findall(r"[a-z0-9][a-z0-9+#./-]*", text):
        word = raw.rstrip(".-/")
        words.add(word)
        # Keep compounds as phrases and also match their ordinary components.
        # Numeric technical identifiers (us-east-1, IPv4/CIDR, ASNs, models)
        # stay whole, so different regions/networks do not gain spurious hits.
        if "-" in word and not any(char.isdigit() for char in word):
            words.update(word.split("-"))
    return {word for word in words if len(word) >= 3 and word not in _FAST_STOP}


def _clip_chunk(chunk: ContextChunk, limit: int) -> ContextChunk:
    if len(chunk.text) <= limit:
        return chunk
    # The original rendered ancestry is repeated on each chunk. Keep a compact
    # heading while preserving its complete value in metadata.
    body = chunk.text
    if chunk.heading and "\n\n" in body:
        body = body.split("\n\n", 1)[1]
    heading = " > ".join(chunk.heading)
    heading_limit = min(limit // 3, 180)
    if len(heading) > heading_limit:
        heading = heading[:max(0, heading_limit - 1)] + "…"
    prefix = heading + "\n\n" if heading else ""
    available = max(0, limit - len(prefix))
    excerpt = body[:available]
    if len(body) > available:
        boundaries = [m.end() for m in re.finditer(r"\n\s*\n|[.!?](?:\s|$)", excerpt)]
        if boundaries and boundaries[-1] >= available // 2:
            excerpt = excerpt[:boundaries[-1]]
        elif " " in excerpt:
            excerpt = excerpt.rsplit(" ", 1)[0]
    return replace(chunk, text=(prefix + excerpt).rstrip()[:limit])
