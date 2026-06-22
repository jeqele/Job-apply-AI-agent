"""Tests for persisted chat sessions in CV content storage."""

from job_apply_ai.cv_modifier.cv_content_store import (
    append_active_chat_messages,
    get_active_chat_messages,
    get_active_chat_session_id,
    get_chat_sessions,
    normalize_store,
    set_active_chat_session,
    start_chat_session,
)


def test_legacy_chat_history_migrates_to_session():
    store = normalize_store({
        "tailored_content": {"professional_title": "Engineer"},
        "chat_history": [
            {"role": "user", "content": "Shorten the summary"},
            {"role": "assistant", "content": "Done"},
        ],
    })

    sessions = get_chat_sessions(store, "cv")
    assert len(sessions) == 1
    assert sessions[0]["title"] == "Shorten the summary"
    assert get_active_chat_messages(store, "cv") == store["chat_history"]


def test_start_new_session_preserves_previous_messages():
    store = normalize_store({
        "chat_history": [{"role": "user", "content": "First request"}],
    })
    append_active_chat_messages(
        store,
        "cv",
        [{"role": "assistant", "content": "Updated CV"}],
    )

    first_session_id = get_active_chat_session_id(store, "cv")
    start_chat_session(store, "cv")

    assert get_active_chat_messages(store, "cv") == []
    assert len(get_chat_sessions(store, "cv")) == 2
    assert get_active_chat_session_id(store, "cv") != first_session_id

    assert set_active_chat_session(store, "cv", first_session_id)
    assert get_active_chat_messages(store, "cv")[0]["content"] == "First request"


def test_cover_letter_sessions_are_independent_from_cv():
    store = normalize_store({})
    append_active_chat_messages(store, "cv", [{"role": "user", "content": "CV edit"}])
    append_active_chat_messages(store, "cover_letter", [{"role": "user", "content": "Letter edit"}])

    assert get_active_chat_messages(store, "cv")[0]["content"] == "CV edit"
    assert get_active_chat_messages(store, "cover_letter")[0]["content"] == "Letter edit"
    assert get_active_chat_session_id(store, "cv") != get_active_chat_session_id(store, "cover_letter")
