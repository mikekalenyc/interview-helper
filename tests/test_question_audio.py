import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("build_question_audio", Path(__file__).parents[1] / "scripts/build_question_audio.py")
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
audio_signature = module.audio_signature
cache_matches = module.cache_matches
read_questions = module.read_questions
spoken_text = module.spoken_text


def group(number: int = 1) -> str:
    return f"""#### Q{number}. Original `AWS` question?

Original **answer** paragraph.

```text
VPC → transit
```

**Follow-up question:** Original follow-up?

**Follow-up answer:**

Original follow-up **answer**.
"""


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "guide.md"
    path.write_text(text)
    return path


def test_sections_pairing_order_and_source_fidelity(tmp_path: Path) -> None:
    path = write(tmp_path, "## First\n\n### Backstory\n\nExclude backstory.\n\n" + group() + group(2) + "\n---\n\n## Second\n\n### Backstory\n\nExclude next backstory.\n\n" + group())
    questions = read_questions(path)
    assert [q["number"] for q in questions] == [1, 2, 3]
    assert [q["local_number"] for q in questions] == [1, 2, 1]
    assert [q["section"] for q in questions] == ["First", "First", "Second"]
    assert questions[0]["text"] == "Original `AWS` question?"
    assert questions[0]["answer"] == "Original **answer** paragraph.\n\n```text\nVPC → transit\n```"
    assert all(q["followup_text"] == "Original follow-up?" for q in questions)
    assert all(q["followup_answer"] == "Original follow-up **answer**." for q in questions)


@pytest.mark.parametrize("body", [group(2), group() + group(), group() + group(3), group().replace("**Follow-up answer:**", "Answer:"), group().replace("Original follow-up?", ""), group().replace("Original follow-up **answer**.", ""), group().replace("Q1.", "Qx."), group() + "\n**Follow-up question:** extra", group() + "\n**Follow-up answer** malformed"])
def test_invalid_groups_rejected(tmp_path: Path, body: str) -> None:
    with pytest.raises(ValueError):
        read_questions(write(tmp_path, "## Section\n\n" + body))


def test_legacy_fifty_questions(tmp_path: Path) -> None:
    text = "## Legacy\n\n" + "\n\n".join(f"{n}. Question {n}?" for n in range(1, 51))
    questions = read_questions(write(tmp_path, text))
    assert len(questions) == 50
    assert questions[-1] == {"number": 50, "section": "Legacy", "text": "Question 50?"}
    with pytest.raises(ValueError):
        read_questions(write(tmp_path, text.replace("50. Question 50?", "")))


def test_independent_cache_and_hash_invalidation(tmp_path: Path) -> None:
    audio = tmp_path / "question.wav"
    signature = audio_signature("question", "model", "voice", "config")
    prior = {"signature": signature, "followup_signature": "other"}
    assert not cache_matches(prior, "", signature, audio)
    audio.touch()
    assert cache_matches(prior, "", signature, audio)
    assert not cache_matches(prior, "followup_", signature, audio)
    for index in range(4):
        inputs = ["question", "model", "voice", "config"]
        inputs[index] += "changed"
        assert audio_signature(*inputs) != signature


def test_pronunciation_changes_only_speech() -> None:
    text = "How do TGW, ZPA, CIDR, GCP, ASN and DR work?"
    assert spoken_text(text) == "How do T G W, Z P A, cider, G C P, A S N and disaster recovery work?"
