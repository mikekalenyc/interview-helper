import threading

import pytest

from interview_helper.spoken import SpokenConversation, select_spoken_turns


def test_correction_deletion_clear_and_immutable_snapshot() -> None:
    conversation = SpokenConversation()
    first = conversation.append("  I helped with Azure.  ", "What did you own?")
    original = conversation.snapshot()
    assert conversation.replace(first.id, "I only observed; I did not own it.")
    assert original[0].text == "I helped with Azure."
    assert conversation.snapshot()[0].timestamp == first.timestamp
    assert conversation.replace(first.id, "")
    assert conversation.snapshot() == ()
    assert not conversation.replace(first.id, "stale")
    second = conversation.append("new")
    conversation.clear()
    assert conversation.append("another").id > second.id


def test_storage_limits_and_no_silent_truncation() -> None:
    conversation = SpokenConversation()
    with pytest.raises(ValueError):
        conversation.append(" ")
    with pytest.raises(ValueError):
        conversation.append("x" * 4001)
    for _ in range(120):
        conversation.append("x" * 4000, "q" * 600)
    turns = conversation.snapshot()
    assert len(turns) <= 100
    assert sum(len(t.text) + len(t.question) for t in turns) <= 40_000
    assert turns[-1].text == "x" * 4000
    with pytest.raises(ValueError):
        conversation.replace(turns[-1].id, "x" * 4001)
    assert conversation.snapshot() == turns


def test_selection_keeps_whole_qualifications_relevant_older_and_new_topics() -> None:
    conversation = SpokenConversation()
    older = conversation.append("I helped with ASA tunnel ownership.", "What did you do?")
    conversation.append("I worked with Azure.")
    latest = conversation.append("ASA " * 980 + "I would do this; I have not actually done it.")
    selected = select_spoken_turns("What risks?", conversation.snapshot(), followup=True)
    assert latest in selected and older in selected
    assert sum(len(t.text) + len(t.question) for t in selected) <= 6000
    assert selected[-1].text.endswith("I have not actually done it.")
    # The latest response remains continuity even on an explicit new topic;
    # older unrelated statements are not pulled in and document lookup stays separate.
    assert select_spoken_turns("Explain Kubernetes pods", conversation.snapshot(), followup=False) == (latest,)


def test_latest_answer_keeps_pause_segments_and_changed_position_without_word_overlap() -> None:
    conversation = SpokenConversation()
    previous = 'Describe a disagreement.'
    first = conversation.append('I initially opposed the inspection design.', previous)
    middle = conversation.append('I thought internal traffic was already trusted.', previous)
    final = conversation.append('After discussing the threats, I supported the proposal.', previous)
    assert select_spoken_turns('An architect objects to segmentation overhead. Would you grant an exception?',
                               conversation.snapshot(), followup=False) == (first, middle, final)


def test_oversized_latest_answer_is_not_replaced_by_older_or_partial_claims() -> None:
    conversation = SpokenConversation()
    conversation.append('I opposed inspection.', 'Older question')
    conversation.append('I opposed inspection. ' * 100, 'Current question')
    conversation.append('Later I changed my mind. ' * 100, 'Current question')
    assert select_spoken_turns('How did you approach inspection?', conversation.snapshot(), followup=False) == ()


def test_concurrent_append_produces_unique_ids() -> None:
    conversation = SpokenConversation()
    threads = [threading.Thread(target=lambda: [conversation.append("response") for _ in range(20)])
               for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    turns = conversation.snapshot()
    assert len(turns) == len({turn.id for turn in turns}) == 80


def test_recent_exchanges_keep_last_five_without_keyword_filtering() -> None:
    from interview_helper.spoken import recent_spoken_exchanges
    conversation = SpokenConversation()
    for number in range(1, 8):
        conversation.append(f"answer {number}", f"question {number}")
    conversation.append("and a correction to six", "question 7")
    selected = recent_spoken_exchanges(conversation.snapshot())
    # Questions 3-7 survive whole; question 7's two utterances stay together.
    assert [turn.question for turn in selected] == [
        "question 3", "question 4", "question 5", "question 6", "question 7", "question 7",
    ]
    assert selected[-1].text == "and a correction to six"


def test_recent_exchanges_drop_oldest_whole_exchange_never_truncate() -> None:
    from interview_helper.spoken import recent_spoken_exchanges, render_spoken_turn
    conversation = SpokenConversation()
    conversation.append("old " * 900, "Explain IPsec phase one")
    conversation.append("short answer", "What about UDP?")
    turns = conversation.snapshot()
    budget = len(render_spoken_turn(turns[1])) + 2 + 100
    selected = recent_spoken_exchanges(turns, max_characters=budget)
    assert selected == (turns[1],)
    assert recent_spoken_exchanges(turns, max_characters=10) == ()
