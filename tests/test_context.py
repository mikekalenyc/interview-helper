from pathlib import Path

import pytest

from interview_helper.context import CandidateProfile, ContextError, structured_candidate_text


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_evidence_annotations_preserve_all_supplied_words_and_order() -> None:
    original = (
        "# Interview Questions & Answers\r\n"
        "What was your role?\r\n"
        "I owned routing; InfoSec owned policy.\r\n\r\n"
        "I designed failover to reduce downtime.\r\n"
        "The measured downtime fell from 12 to 4 minutes.\r\n"
        "Other outcomes are unknown."
    )
    formatted = structured_candidate_text(original)
    without_annotations = "".join(
        line for line in formatted.splitlines(keepends=True)
        if not line.startswith("[")
    )
    assert without_annotations == original
    assert "Source-stated outcome/measurement" in formatted
    assert "Qualified/unknown evidence" in formatted


@pytest.mark.parametrize("statement", [
    "Designed redundancy to reduce downtime by 30%.",
    "Implemented checks to ensure reliable delivery.",
    "I added monitoring, preventing potential failures.",
    "Our target was a measured reduction of 30%.",
    "The design could have improved uptime by 5%.",
    "Expected outcome: reduced downtime by 30%.",
])
def test_design_intentions_are_never_promoted_to_results(statement: str) -> None:
    formatted = structured_candidate_text(statement)
    assert "not proof of achieved outcome" in formatted
    assert "Source-stated outcome/measurement" not in formatted
    assert formatted.endswith(statement)


@pytest.mark.parametrize("statement", [
    "We did not achieve the measured target.",
    "No documented outcome: reduced latency by 10%.",
    "Reduced downtime by 30% is unverified.",
    "We never achieved the result.",
    "A reduction of 30% was not achieved.",
])
def test_unknown_results_keep_their_qualification(statement: str) -> None:
    formatted = structured_candidate_text(statement)
    assert "Qualified/unknown evidence" in formatted
    assert "Source-stated outcome/measurement" not in formatted


def test_explicit_metrics_are_preserved_without_inventing_missing_metrics() -> None:
    result = "Led deployment automation and reduced rollout time by 40%."
    assert "Source-stated outcome/measurement" in structured_candidate_text(result)
    background = "I deployed redundant routers.\nWhich results did you achieve?"
    assert structured_candidate_text(background) == background
    assert structured_candidate_text("") == ""


def test_loads_resume_and_selects_relevant_project_chunks(tmp_path: Path) -> None:
    resume = write(
        tmp_path / "resume.md",
        "Platform engineer at Example Corp. Python and Kubernetes.",
    )
    projects = tmp_path / "projects"
    projects.mkdir()
    write(projects / "search.md", "Built a search service using OpenSearch.")
    write(
        projects / "deploy.md",
        "Led Kubernetes deployment automation and reduced rollout time by 40%.",
    )

    profile = CandidateProfile.load(resume, (projects,), chunk_characters=200)
    selected = profile.relevant_chunks(
        "Tell me about Kubernetes deployment automation",
        max_characters=300,
        max_resume_characters=200,
    )

    assert selected[0].kind == "resume"
    assert selected[1].source == "deploy.md"
    assert "40%" in selected[1].text


@pytest.mark.parametrize("name", ["resume.pdf", "resume.docx", "resume"])
def test_rejects_unsupported_resume_formats(tmp_path: Path, name: str) -> None:
    resume = write(tmp_path / name, "candidate")
    with pytest.raises(ContextError, match="Unsupported context format"):
        CandidateProfile.load(resume)


def test_rejects_missing_and_empty_context(tmp_path: Path) -> None:
    with pytest.raises(ContextError, match="does not exist"):
        CandidateProfile.load(tmp_path / "missing.md")

    resume = write(tmp_path / "resume.md", "  ")
    with pytest.raises(ContextError, match="empty"):
        CandidateProfile.load(resume)


def test_context_directory_must_contain_supported_files(tmp_path: Path) -> None:
    resume = write(tmp_path / "resume.md", "candidate")
    context = tmp_path / "projects"
    context.mkdir()
    write(context / "diagram.pdf", "not parsed")
    with pytest.raises(ContextError, match="contains no"):
        CandidateProfile.load(resume, (context,))


def test_markdown_types_topic_and_answer_survive_splits(tmp_path: Path) -> None:
    resume = write(tmp_path / "resume.txt", "Engineer")
    guide = write(tmp_path / "guide.md", """# Interview Guide
## Routing
### Resume Bullet
Designed the routing system.
### Backstory
Owned validation of failover.
### Interview Questions & Answers
#### Why BGP?
""" + "Technical answer. " * 60 + """
##### Actual experience
This is still a practice answer.
## Table of Contents
Navigation only.
""")
    profile = CandidateProfile.load(resume, (guide,), chunk_characters=250)
    practice = [c for c in profile.chunks if c.kind == "practice"]
    assert len(practice) > 2
    assert all(c.topic == "Routing" for c in practice)
    assert all("Interview Questions & Answers" in c.heading for c in practice)
    assert all(len(c.text) <= 250 for c in profile.chunks)
    assert any(c.kind == "experience" and "Owned validation" in c.text for c in profile.chunks)
    assert profile.chunks[-1].kind == "reference"


def test_late_partial_answer_brings_related_experience(tmp_path: Path) -> None:
    resume = write(tmp_path / "resume.txt", "Engineer")
    guide = write(tmp_path / "guide.md", "# Guide\n" + "\n".join(
        f"## Topic {i}\nUnrelated storage content. " + "disk " * 100
        for i in range(20)
    ) + """
## Connectivity
### Supplied experience narrative
I validated redundant circuits.
### Practice questions and draft answers
#### Why use BGP communities?
Communities control exported prefixes.
""")
    profile = CandidateProfile.load(resume, (guide,), chunk_characters=400)
    chosen = profile.relevant_chunks("Explain BGP communities and a missing technical detail", max_characters=800)
    assert any("Communities control" in c.text for c in chosen)
    assert any(c.kind == "experience" and "redundant circuits" in c.text for c in chosen)
    assert not any("Unrelated storage" in c.text for c in chosen)
    assert sum(len(c.text) for c in chosen) <= 800


def test_followup_uses_previous_question_but_new_topic_wins() -> None:
    from interview_helper.context import ContextChunk
    profile = CandidateProfile((
        ContextChunk("resume", "resume", "Engineer", 0),
        ContextChunk("bgp", "reference", "BGP routing redundant peers", 1),
        ContextChunk("dns", "reference", "DNS caching resolver", 2),
        ContextChunk("padding", "reference", "unrelated " * 100, 3),
    ))
    followup = profile.relevant_chunks("How does that work?", recent_question="Explain BGP routing", max_characters=45)
    assert any(c.source == "bgp" for c in followup)
    changed = profile.relevant_chunks("Explain DNS caching resolver", recent_question="BGP routing redundant peers", max_characters=45)
    assert any(c.source == "dns" for c in changed)
    assert not any(c.source == "bgp" for c in changed)


def test_long_headings_and_mixed_fences_are_bounded(tmp_path: Path) -> None:
    resume = write(tmp_path / "resume.txt", "Engineer")
    title = "Long title " * 100
    guide = write(tmp_path / "guide.md", f"# Technical reference\n## {title}\n" + """````markdown
### Actual experience
~~~
```nested
### Resume Bullet
````
This remains reference.
""")
    chunks = CandidateProfile.load(resume, (guide,), chunk_characters=200).chunks[1:]
    assert all(c.kind == "reference" for c in chunks)
    assert all(c.heading == ("Technical reference", title.strip()) for c in chunks)
    assert all(len(c.text) <= 200 for c in chunks)
    assert "Actual experience" in "".join(c.text for c in chunks)


@pytest.mark.parametrize("title", ["AWS Cloud Networking Gotcha Questions & Answers", "Technical Questions"])
def test_reference_guide_is_not_candidate_experience(tmp_path: Path, title: str) -> None:
    resume = write(tmp_path / "resume.txt", "Engineer")
    guide = write(tmp_path / "guide.md", f"# {title}\n## Routing\n### Backstory\nI deployed BGP.\n")
    assert CandidateProfile.load(resume, (guide,)).chunks[-1].kind == "reference"


def test_small_context_and_duplicate_sources(tmp_path: Path) -> None:
    resume = write(tmp_path / "resume.txt", "Engineer")
    folders = (tmp_path / "first", tmp_path / "second")
    for folder in folders:
        folder.mkdir()
        write(folder / "notes.md", "Plain unstructured notes")
    profile = CandidateProfile.load(resume, folders)
    assert profile.relevant_chunks("Unrelated question") == profile.chunks
    assert len({c.source for c in profile.chunks}) == 3
    assert all(str(tmp_path) not in c.source for c in profile.chunks)


def test_topic_cross_reference_metadata_is_not_experience(tmp_path: Path) -> None:
    resume = write(tmp_path / "resume.txt", "Engineer")
    guide = write(tmp_path / "guide.md", "# Resume Interview Guide\n## Routing\n**Topic ID:** B01\n**Related technical reference:** Q1\n### Supplied experience narrative\nI tested routing.\n")
    chunks = CandidateProfile.load(resume, (guide,)).chunks
    assert chunks[1].kind == "reference"
    assert chunks[2].kind == "experience"


def test_fast_factual_answer_and_linked_background(tmp_path: Path) -> None:
    resume = write(tmp_path / "resume.txt", "Engineer")
    guide = write(tmp_path / "guide.md", """# Cloud Guide
## Multi-region AWS
### Backstory
I worked on routing design.
### Interview Questions & Answers
#### Which AWS region did you use?
The primary region was us-east-1.
## Other topic
""" + "Unrelated information. " * 200)
    profile = CandidateProfile.load(resume, (guide,))
    selected = profile.fast_chunks("Which AWS regions did you use?", max_characters=500)
    assert "us-east-1" in selected[0].text
    assert any("routing design" in c.text for c in selected)
    assert all(c.kind == "experience" for c in selected)
    assert sum(len(c.text) for c in selected) <= 500


def test_fast_role_and_interruption_paraphrases(tmp_path: Path) -> None:
    resume = write(tmp_path / "resume.txt", "Engineer")
    guide = write(tmp_path / "guide.md", """# Guide
## ExampleNet
### Interview Questions & Answers
#### What were the design principles behind ExampleNet?
Centralize reusable services.
#### What did you personally own?
I owned routing architecture, while InfoSec owned security policy.
## Cloud Troubleshooting
### Backstory
I diagnosed asymmetric packet paths during outages.
""")
    profile = CandidateProfile.load(resume, (guide,))
    assert "InfoSec" in profile.fast_chunks("What was your role in ExampleNet?")[0].text
    assert "asymmetric" in profile.fast_chunks("How did you handle service interruptions?")[0].text


def test_fast_relevant_resume_tail_displaces_prefix() -> None:
    from interview_helper.context import ContextChunk
    chunks = tuple(ContextChunk("resume", "resume", "Generic background. " * 80, i) for i in range(6))
    profile = CandidateProfile(chunks + (ContextChunk("resume", "resume", "Led QUICKSILVER migration.", 6),))
    assert "QUICKSILVER" in profile.fast_chunks("Explain QUICKSILVER migration", max_characters=300)[0].text


def test_fast_clips_long_answer_with_ancestry_and_followup(tmp_path: Path) -> None:
    resume = write(tmp_path / "resume.txt", "Engineer")
    guide = write(tmp_path / "guide.md", "# Guide\n## BGP\n### How does BGP failover work?\n" + "Redundant peers retain connectivity. " * 70)
    profile = CandidateProfile.load(resume, (guide,))
    selected = profile.fast_chunks("How does that work?", recent_question="Explain BGP failover", max_characters=320)
    assert selected and selected[0].heading[-1] == "How does BGP failover work?"
    assert sum(len(c.text) for c in selected) <= 320
    assert "Redundant peers" in selected[0].text
    assert selected[0].text.endswith(".")


@pytest.mark.parametrize('question', ['Tell me about yourself.', 'Walk me through your background.'])
def test_fast_introduction_includes_bounded_resume(question: str) -> None:
    from interview_helper.context import ContextChunk
    profile = CandidateProfile((ContextChunk('resume', 'resume', 'Network engineer. ' * 100, 0),))
    selected = profile.fast_chunks(question, max_characters=300)
    assert selected and 'Network engineer.' in selected[0].text
    assert sum(len(c.text) for c in selected) <= 300


def test_fast_result_followup_retrieves_facts_and_topic_change_stands_alone() -> None:
    from interview_helper.context import ContextChunk
    profile = CandidateProfile((
        ContextChunk('guide', 'experience', 'AWS migration reduced downtime by 30%.', 0),
        ContextChunk('guide', 'experience', 'Cloudflare protected public applications.', 1),
    ))
    selected = profile.fast_chunks('What was the result?', recent_question='Tell me about the AWS migration')
    assert selected and '30%' in selected[0].text
    changed = profile.fast_chunks('Explain Cloudflare', recent_question='Tell me about the AWS migration')
    assert len(changed) == 1 and 'Cloudflare' in changed[0].text


def test_fast_terms_punctuation_compounds_and_identifiers() -> None:
    from interview_helper.context import _fast_terms
    assert _fast_terms("migration. owned.") == _fast_terms("migration own")
    assert _fast_terms("data-center") == _fast_terms("data center") == _fast_terms("datacenter")
    compound = _fast_terms("managed-data-center-to-cloud")
    assert {"managed-datacenter-to-cloud", "managed", "datacenter", "cloud"} <= compound
    identifiers = _fast_terms("10.20.0.0/16, 10.20.0.1. us-east-1. AS64512 C++.")
    assert {"10.20.0.0/16", "10.20.0.1", "us-east-1", "as64512", "c++"} <= identifiers
    assert "east" not in identifiers
    assert not _fast_terms("Walk me through exactly what you say, please.")


@pytest.mark.parametrize("question", [
    "Walk me through the managed-data-center-to-cloud migration. What architecture did you own?",
    "You say you led a datacenter migration. Explain the original and target architecture, exactly.",
    "Explain your data center migration and personal responsibilities.",
])
def test_fast_migration_subject_beats_question_heading_framing(tmp_path: Path, question: str) -> None:
    resume = write(tmp_path / "resume.txt", "Network engineer.")
    distractor = write(tmp_path / "questions.md", """# Interview Guide
## Multi-region AWS
### Exactly what do you say about your network architecture?
I designed multi-region AWS routing.
## AWS Inspection
### Walk me through what you personally owned in the architecture.
I owned firewall inspection policies.
## BGP
### Explain BGP route selection.
BGP selected routes using local preference.
""")
    background = write(tmp_path / "background.md", "# Background\n## Early career\n"
                       + "Maintained unrelated equipment. " * 70 + """
## Datacenter migration
### Original and target architecture
The original managed data center used a spine-and-leaf fabric and site-to-site tunnels.
The target environment for the migration was Azure.
### Responsibilities
I owned tunnel review, checked active connections and identified partner contacts.
""")
    profile = CandidateProfile.load(resume, (distractor, background), chunk_characters=300)
    selected = profile.fast_chunks(question, max_characters=850)
    assert selected[0].source == "background.md"
    evidence = "\n".join(c.text for c in selected)
    assert "spine-and-leaf" in evidence
    assert "Azure" in evidence
    assert "checked active connections" in evidence
    assert sum(len(c.text) for c in selected) <= 850
    for new_question, expected in [
        ("Explain AWS inspection firewall policies.", "firewall inspection"),
        ("Explain BGP route selection.", "local preference"),
    ]:
        changed = profile.fast_chunks(new_question, recent_question=question, max_characters=300)
        assert changed[0].source == "questions.md"
        assert expected in changed[0].text
        assert "Azure" not in "\n".join(c.text for c in changed)


def test_technical_answers_load_whole_and_reject_unusable_files(tmp_path: Path) -> None:
    from interview_helper.context import TECHNICAL_ANSWERS_LIMIT, TechnicalAnswers
    prepared = tmp_path / "firewall.md"
    prepared.write_text("1. **What is FMC?**\n\n**Answer:** The manager.\n", encoding="utf-8")
    loaded = TechnicalAnswers.load(prepared)
    assert loaded.source == "firewall.md"
    assert "The manager." in loaded.text
    with pytest.raises(ContextError, match="does not exist"):
        TechnicalAnswers.load(tmp_path / "missing.md")
    unsupported = tmp_path / "firewall.docx"
    unsupported.write_bytes(b"x")
    with pytest.raises(ContextError, match="Unsupported"):
        TechnicalAnswers.load(unsupported)
    oversized = tmp_path / "huge.md"
    oversized.write_text("x" * (TECHNICAL_ANSWERS_LIMIT + 1), encoding="utf-8")
    with pytest.raises(ContextError, match="exceeds"):
        TechnicalAnswers.load(oversized)
