"""Deterministic per-request context projection for the LCM engine."""

from __future__ import annotations

from copy import deepcopy
from unittest.mock import Mock

import pytest

import hermes_lcm.engine as lcm_engine
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.tokens import count_messages_tokens


@pytest.fixture
def engine(tmp_path):
    config = LCMConfig(database_path=str(tmp_path / "selection.db"))
    config.fresh_tail_count = 2
    instance = LCMEngine(config=config)
    instance._session_id = "selection-session"
    instance.context_length = 4_000
    instance.threshold_tokens = 3_000
    try:
        yield instance
    finally:
        instance.shutdown()


def test_same_selection_is_deterministic_and_retry_ingest_is_idempotent(engine, monkeypatch):
    reconcile = Mock(
        side_effect=AssertionError("unfiltered selection must not reconcile its cursor")
    )
    monkeypatch.setattr(engine, "_reconcile_ingest_cursor_from_store", reconcile)
    conversation = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "current question"},
    ]
    request = [
        {"role": "system", "content": "stable system"},
        *deepcopy(conversation),
    ]

    first = engine.select_context(
        deepcopy(request),
        conversation_messages=deepcopy(conversation),
        budget_tokens=4_000,
    )
    rows_after_first = engine._store.get_session_count(engine._session_id)
    second = engine.select_context(
        deepcopy(request),
        conversation_messages=deepcopy(conversation),
        budget_tokens=4_000,
    )

    assert first is not None
    assert second == first
    assert rows_after_first == len(conversation)
    assert engine._store.get_session_count(engine._session_id) == rows_after_first


def test_host_shaped_prefill_before_canonical_is_preserved_and_never_ingested(engine):
    conversation = [
        {"role": "user", "content": "canonical prompt"},
        {"role": "assistant", "content": "canonical answer"},
    ]
    system = {
        "role": "system",
        "content": "request-only system",
        "cache_control": {"type": "ephemeral"},
    }
    prefill = [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "request-only prefill"}],
            "provider_metadata": {"nested": {"keep": [1, 2, 3]}},
        },
    ]
    request = [system, *deepcopy(prefill), *deepcopy(conversation)]
    request_before = deepcopy(request)
    prefill_before = deepcopy(prefill)

    selected = engine.select_context(
        request,
        conversation_messages=deepcopy(conversation),
        budget_tokens=4_000,
    )
    stored = engine._store.get_session_messages(engine._session_id)

    assert selected == request_before
    assert request == request_before
    assert selected[1:2] == prefill_before
    assert [row["content"] for row in stored] == [
        "canonical prompt",
        "canonical answer",
    ]
    assert all("request-only prefill" not in row["content"] for row in stored)


def test_fresh_selection_filters_all_replayed_scaffolds_before_durable_ingest(engine):
    engine._dag.add_node(
        SummaryNode(
            session_id=engine._session_id,
            depth=0,
            summary="current DAG summary",
            token_count=8,
            source_token_count=80,
            source_ids=[1],
            source_type="messages",
            created_at=1.0,
            expand_hint="lcm_expand 1",
        )
    )
    lcm_note = {
        "role": "system",
        "content": (
            "[Note: This conversation uses Lossless Context Management (LCM). "
            "Earlier turns have been compacted into hierarchical summaries below.]"
        ),
    }
    stale_summary = {
        "role": "user",
        "content": (
            "[Recent Summary (d0, node 999)]\n"
            "stale replayed summary\n"
            "[Expand for details: lcm_expand 999]"
        ),
    }
    objective = {
        "role": "user",
        "content": (
            "[Current user objective preserved from compacted history]\n"
            "stale preserved objective"
        ),
    }
    fresh = [
        {"role": "user", "content": "fresh question"},
        {"role": "assistant", "content": "fresh answer"},
    ]
    conversation = [lcm_note, stale_summary, objective, *fresh]
    request = [
        {"role": "system", "content": "request system"},
        *deepcopy(conversation),
    ]

    selected = engine.select_context(
        request,
        conversation_messages=deepcopy(conversation),
        budget_tokens=4_000,
    )
    stored = engine._store.get_session_messages(engine._session_id)

    assert selected is not None
    assert [row["content"] for row in stored] == ["fresh question", "fresh answer"]
    selected_text = "\n".join(str(message.get("content", "")) for message in selected)
    assert "stale replayed summary" not in selected_text
    assert "stale preserved objective" not in selected_text
    assert "Lossless Context Management" not in selected_text
    assert selected_text.count("current DAG summary") == 1


def test_selection_keeps_role_ineligible_scaffold_lookalikes_with_tool_pairing(engine):
    assistant = {
        "role": "assistant",
        "content": (
            "[Current user objective preserved from compacted history]\n"
            "assistant-authored canonical content"
        ),
        "tool_calls": [
            {
                "id": "call_scaffold_lookalike",
                "type": "function",
                "function": {"name": "inspect", "arguments": "{}"},
            }
        ],
    }
    tool = {
        "role": "tool",
        "tool_call_id": "call_scaffold_lookalike",
        "tool_name": "inspect",
        "content": (
            "[Recent Summary (d0, node 91)]\n"
            "canonical tool output\n"
            "[Expand for details: lcm_expand 91]"
        ),
    }
    user_prose = {
        "role": "user",
        "content": (
            "[Recent Summary (d0, node 92)] is syntax I am discussing; "
            "[Expand for details: lcm_expand 92] is not a trailing scaffold hint."
        ),
    }
    conversation = [
        {"role": "user", "content": "run the paired inspection"},
        assistant,
        tool,
        user_prose,
    ]
    request = [{"role": "system", "content": "stable system"}, *deepcopy(conversation)]

    selected = engine.select_context(
        request,
        conversation_messages=deepcopy(conversation),
        budget_tokens=4_000,
    )
    stored = engine._store.get_session_messages(engine._session_id)

    assert selected == request
    assert [row["content"] for row in stored] == [
        message["content"] for message in conversation
    ]
    assert selected[2]["tool_calls"][0]["id"] == selected[3]["tool_call_id"]


def test_selection_filters_multi_summary_projection_scaffold(engine):
    multi_summary = {
        "role": "user",
        "content": (
            "[Durable Summary (d2, node 10)]\n"
            "durable projection\n"
            "[Expand for details: lcm_expand 10]\n\n---\n\n"
            "[Recent Summary (d0, node 11)]\n"
            "recent projection\n"
            "[Expand for details: lcm_expand 11]"
        ),
    }
    fresh = [{"role": "user", "content": "fresh canonical prompt"}]
    conversation = [multi_summary, *fresh]
    request = [{"role": "system", "content": "stable system"}, *deepcopy(conversation)]

    selected = engine.select_context(
        request,
        conversation_messages=deepcopy(conversation),
        budget_tokens=4_000,
    )

    assert selected == [{"role": "system", "content": "stable system"}, *fresh]
    assert [row["content"] for row in engine._store.get_session_messages(engine._session_id)] == [
        "fresh canonical prompt"
    ]


def test_filtered_selection_realigns_unfiltered_cursor_and_coexists_with_legacy_ingest(engine):
    older_turn = [
        {"role": "user", "content": "older durable question"},
        {"role": "assistant", "content": "older durable answer"},
    ]
    engine._ingest_messages(deepcopy(older_turn))
    scaffolds = [
        {
            "role": "system",
            "content": (
                "[Note: This conversation uses Lossless Context Management (LCM). "
                "Earlier turns have been compacted into hierarchical summaries below.]"
            ),
        },
        {
            "role": "user",
            "content": (
                "[Recent Summary (d0, node 999)]\n"
                "stale replayed summary\n"
                "[Expand for details: lcm_expand 999]"
            ),
        },
    ]
    new_tail = [
        {"role": "user", "content": "new tail question"},
        {"role": "assistant", "content": "new tail answer"},
    ]
    conversation = [*scaffolds, *older_turn, *new_tail]
    request = [{"role": "system", "content": "request system"}, *deepcopy(conversation)]

    # Simulate a cursor left aligned to the older unfiltered active view.
    engine._ingest_cursor = len([*scaffolds, *older_turn])
    engine._ingest_cursor_needs_reconcile = False

    first = engine.select_context(
        deepcopy(request),
        conversation_messages=deepcopy(conversation),
        budget_tokens=4_000,
    )
    rows_after_first = engine._store.get_session_messages(engine._session_id)

    assert first is not None
    assert [row["content"] for row in rows_after_first] == [
        "older durable question",
        "older durable answer",
        "new tail question",
        "new tail answer",
    ]
    assert all(
        scaffold["content"] not in {row["content"] for row in rows_after_first}
        for scaffold in scaffolds
    )
    assert all(message in first for message in new_tail)
    assert engine._ingest_cursor_needs_reconcile is True

    # Selection retries use the filtered view, while legacy preflight/ingest can
    # still observe the unfiltered canonical view during host coexistence.
    second = engine.select_context(
        deepcopy(request),
        conversation_messages=deepcopy(conversation),
        budget_tokens=4_000,
    )
    assert second == first
    assert engine._store.get_session_count(engine._session_id) == len(rows_after_first)

    engine._ingest_messages(deepcopy(conversation))
    assert engine._store.get_session_count(engine._session_id) == len(rows_after_first)
    assert engine._ingest_cursor == len(conversation)
    assert engine._ingest_cursor_needs_reconcile is False

    third = engine.select_context(
        deepcopy(request),
        conversation_messages=deepcopy(conversation),
        budget_tokens=4_000,
    )
    assert third == first
    assert engine._store.get_session_count(engine._session_id) == len(rows_after_first)


def test_existing_dag_summary_replaces_replayed_scaffold_and_keeps_fresh_tail(engine):
    engine._dag.add_node(
        SummaryNode(
            session_id=engine._session_id,
            depth=0,
            summary="durable summary from the current DAG",
            token_count=12,
            source_token_count=120,
            source_ids=[1, 2],
            source_type="messages",
            created_at=1.0,
            expand_hint="lcm_expand 1",
        )
    )
    old_scaffold = {
        "role": "user",
        "content": (
            "[Recent Summary (d0, node 999)]\n"
            "stale replayed summary\n"
            "[Expand for details: lcm_expand 999]"
        ),
    }
    conversation = [
        old_scaffold,
        {"role": "user", "content": "fresh question"},
        {"role": "assistant", "content": "fresh answer"},
    ]
    request = [
        {"role": "system", "content": "stable system"},
        *deepcopy(conversation),
    ]

    selected = engine.select_context(
        request,
        conversation_messages=deepcopy(conversation),
        budget_tokens=4_000,
    )

    assert selected is not None
    contents = [str(message.get("content", "")) for message in selected]
    assert any("durable summary from the current DAG" in content for content in contents)
    assert any(content == "fresh question" for content in contents)
    assert any(content == "fresh answer" for content in contents)
    assert not any("stale replayed summary" in content for content in contents)
    non_system = [message for message in selected if message.get("role") != "system"]
    assert non_system and non_system[0]["role"] == "user"


def test_budget_counts_request_envelope_and_prefill_and_selection_is_stable(engine):
    conversation = [
        {"role": "user", "content": "old " * 300},
        {"role": "assistant", "content": "old answer " * 300},
        {"role": "user", "content": "fresh prompt"},
    ]
    system = {
        "role": "system",
        "content": "immutable envelope " * 12,
        "cache_control": {"type": "ephemeral"},
    }
    prefill = {
        "role": "assistant",
        "content": "immutable prefill " * 8,
        "provider_metadata": {"prefill": True},
    }
    request = [system, prefill, *deepcopy(conversation)]
    budget = count_messages_tokens([system, conversation[-1], prefill]) + 8

    first = engine.select_context(
        deepcopy(request),
        conversation_messages=deepcopy(conversation),
        budget_tokens=budget,
    )
    second = engine.select_context(
        deepcopy(request),
        conversation_messages=deepcopy(conversation),
        budget_tokens=budget,
    )

    assert first == second
    assert first is not None
    assert first[0] == system
    assert first[1] == prefill
    assert count_messages_tokens(first) <= budget
    assert not any(message.get("content") == conversation[0]["content"] for message in first)


@pytest.mark.parametrize(
    "suffix",
    [
        {"role": "user", "content": "request-only user suffix"},
        {"role": "tool", "tool_call_id": "call-1", "content": "request-only tool suffix"},
        {"role": "assistant", "content": "request-only assistant suffix"},
    ],
    ids=["user", "tool", "assistant"],
)
def test_any_request_suffix_after_canonical_fails_open_without_ingest(engine, suffix):
    canonical = [{"role": "user", "content": "canonical turn"}]
    request = [
        {"role": "system", "content": "stable system"},
        *deepcopy(canonical),
        suffix,
    ]

    selected = engine.select_context(
        request,
        conversation_messages=deepcopy(canonical),
        budget_tokens=4_000,
    )

    assert selected is None
    assert engine._store.get_session_count(engine._session_id) == 0


def test_unsafe_or_ambiguous_alignment_fails_open_without_request_ingest(engine):
    canonical = [{"role": "user", "content": "canonical turn"}]
    mutated_request = [
        {"role": "system", "content": "stable system"},
        {"role": "user", "content": "request-only mutation"},
    ]
    ambiguous_request = [
        {"role": "system", "content": "stable system"},
        *deepcopy(canonical),
        *deepcopy(canonical),
    ]

    assert engine.select_context(
        mutated_request,
        conversation_messages=deepcopy(canonical),
        budget_tokens=4_000,
    ) is None
    assert engine.select_context(
        ambiguous_request,
        conversation_messages=deepcopy(canonical),
        budget_tokens=4_000,
    ) is None
    assert engine._store.get_session_count(engine._session_id) == 0


def test_historical_api_content_aligns_but_only_clean_canonical_content_is_ingested(engine):
    historical = {
        "role": "user",
        "content": "clean historical prompt",
        "api_content": "clean historical prompt\n\n[historical provider context]",
    }
    historical_assistant = {
        "role": "assistant",
        "content": "clean historical answer",
        "api_content": "clean historical answer\n\n[historical wire answer]",
    }
    current = {"role": "user", "content": "current prompt"}
    conversation = [historical, historical_assistant, current]
    request_history = {
        "role": "user",
        "content": historical["api_content"],
        "cache_control": {"type": "ephemeral"},
    }
    request = [
        {"role": "system", "content": "stable system"},
        request_history,
        {
            "role": "assistant",
            "content": historical_assistant["api_content"],
        },
        deepcopy(current),
    ]

    selected = engine.select_context(
        request,
        conversation_messages=conversation,
        incoming_message=current,
        budget_tokens=engine.context_length,
    )

    assert selected == request
    assert selected[1] == request_history
    rows = engine._store.get_session_messages(engine._session_id)
    assert [row["content"] for row in rows] == [
        "clean historical prompt",
        "clean historical answer",
        "current prompt",
    ]
    assert all("provider context" not in row["content"] for row in rows)


def test_current_api_content_aligns_and_preserves_request_side_content(engine):
    current = {
        "role": "user",
        "content": "clean current prompt",
        "api_content": "clean current prompt\n\n[current provider context]",
    }
    request_current = {
        "role": "user",
        "content": current["api_content"],
        "provider_metadata": {"request": "current"},
    }
    request = [{"role": "system", "content": "stable system"}, request_current]

    selected = engine.select_context(
        request,
        conversation_messages=[current],
        incoming_message=current,
        budget_tokens=engine.context_length,
    )

    assert selected == request
    assert selected[-1] == request_current
    rows = engine._store.get_session_messages(engine._session_id)
    assert [row["content"] for row in rows] == ["clean current prompt"]


def test_current_string_moa_append_aligns_from_api_content_base(engine):
    current = {
        "role": "user",
        "content": "clean current prompt",
        "api_content": "clean current prompt\n\n[current provider context]",
    }
    request_current = {
        "role": "user",
        "content": current["api_content"] + "\n\n[Mixture of Agents guidance]",
        "provider_metadata": {"moa": True},
    }
    request = [{"role": "system", "content": "stable system"}, request_current]

    selected = engine.select_context(
        request,
        conversation_messages=[current],
        incoming_message=current,
        budget_tokens=engine.context_length,
    )

    assert selected == request
    assert selected[-1] == request_current
    rows = engine._store.get_session_messages(engine._session_id)
    assert [row["content"] for row in rows] == ["clean current prompt"]


def test_current_structured_moa_append_aligns_as_trailing_text_parts(engine):
    base_content = [
        {"type": "text", "text": "inspect this image"},
        {"type": "image_url", "image_url": {"url": "https://example.test/image.png"}},
    ]
    current = {"role": "user", "content": base_content}
    request_current = {
        "role": "user",
        "content": [
            *deepcopy(base_content),
            {"type": "text", "text": "\n\n[Mixture of Agents guidance]"},
            {"type": "text", "text": "[additional provider-only guidance]"},
        ],
        "provider_metadata": {"moa": True},
    }
    request = [{"role": "system", "content": "stable system"}, request_current]

    selected = engine.select_context(
        request,
        conversation_messages=[current],
        incoming_message=current,
        budget_tokens=engine.context_length,
    )

    assert selected == request
    assert selected[-1] == request_current
    rows = engine._store.get_session_messages(engine._session_id)
    assert len(rows) == 1
    assert "Mixture of Agents" not in rows[0]["content"]


def test_historical_append_only_content_mismatch_fails_open_without_ingest(engine):
    historical = {"role": "user", "content": "historical prompt"}
    current = {"role": "user", "content": "current prompt"}
    request = [
        {"role": "system", "content": "stable system"},
        {"role": "user", "content": "historical prompt\n\n[untrusted append]"},
        deepcopy(current),
    ]

    assert engine.select_context(
        request,
        conversation_messages=[historical, current],
        incoming_message=current,
        budget_tokens=engine.context_length,
    ) is None
    assert engine._store.get_session_count(engine._session_id) == 0


def test_tool_metadata_mismatch_fails_open_without_ingest(engine):
    canonical = [
        {
            "role": "assistant",
            "content": "calling inspect",
            "tool_calls": [{
                "id": "call-exact",
                "type": "function",
                "function": {"name": "inspect", "arguments": '{"depth":1}'},
            }],
        },
        {
            "role": "tool",
            "tool_call_id": "call-exact",
            "tool_name": "inspect",
            "content": "result",
        },
    ]
    request = [{"role": "system", "content": "stable system"}, *deepcopy(canonical)]
    request[1]["tool_calls"][0]["function"]["arguments"] = '{"depth":2}'

    assert engine.select_context(
        request,
        conversation_messages=canonical,
        budget_tokens=engine.context_length,
    ) is None
    assert engine._store.get_session_count(engine._session_id) == 0


def test_request_envelope_prefill_and_annotations_are_not_ingested(engine):
    conversation = [
        {"role": "user", "content": "canonical prompt"},
        {"role": "assistant", "content": "canonical answer"},
    ]
    annotated_history = deepcopy(conversation)
    annotated_history[0]["cache_control"] = {"type": "ephemeral"}
    system = {"role": "system", "content": "request-only system"}
    prefill = {
        "role": "assistant",
        "content": "request-only prefill",
        "provider_metadata": {"prefill": True},
    }
    request = [system, prefill, *annotated_history]

    selected = engine.select_context(
        deepcopy(request),
        conversation_messages=deepcopy(conversation),
        budget_tokens=4_000,
    )
    stored = engine._store.get_session_messages(engine._session_id)

    assert selected == request
    assert selected[2]["cache_control"] == {"type": "ephemeral"}
    assert [row["content"] for row in stored] == [
        "canonical prompt",
        "canonical answer",
    ]


def test_selection_avoids_projection_side_effects(engine, monkeypatch):
    forbidden = [
        "compress",
        "_assemble_context",
        "_summarize_leaf_chunk_with_rescue",
        "_maybe_condense",
        "_build_proactive_recall_message",
        "_stub_large_tool_results_for_active_replay",
        "_run_pre_compaction_extraction",
        "_purge_embeddings_for_nodes",
    ]
    spies = {}
    for name in forbidden:
        spy = Mock(side_effect=AssertionError(f"selection called forbidden {name}"))
        monkeypatch.setattr(engine, name, spy)
        spies[name] = spy
    for name in (
        "summarize_with_escalation",
        "extract_before_compaction",
        "maybe_externalize_tool_output",
    ):
        spy = Mock(side_effect=AssertionError(f"selection called forbidden {name}"))
        monkeypatch.setattr(lcm_engine, name, spy)
        spies[name] = spy

    conversation = [{"role": "user", "content": "ordinary canonical turn"}]
    selected = engine.select_context(
        [{"role": "system", "content": "system"}, *deepcopy(conversation)],
        conversation_messages=deepcopy(conversation),
        budget_tokens=4_000,
    )

    assert selected is not None
    for spy in spies.values():
        spy.assert_not_called()


def test_canonical_payload_protection_is_idempotent_but_request_payload_is_projection_only(
    tmp_path,
    monkeypatch,
):
    config = LCMConfig(
        database_path=str(tmp_path / "protected-selection.db"),
        large_output_externalization_enabled=True,
        large_output_externalization_threshold_chars=64,
        large_output_externalization_path=str(tmp_path / "externalized"),
    )
    instance = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    instance._session_id = "protected-selection"
    instance.context_length = 8_000
    instance.threshold_tokens = 6_000
    try:
        monkeypatch.setattr(
            instance,
            "_stub_large_tool_results_for_active_replay",
            Mock(side_effect=AssertionError("projection must not externalize replay")),
        )
        canonical_payload = "canonical-payload-" * 40
        request_only_payload = "request-only-prefill-" * 40
        canonical = [
            {
                "role": "assistant",
                "content": "calling tool",
                "tool_calls": [{
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "demo", "arguments": "{}"},
                }],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": canonical_payload},
        ]
        prefill = {"role": "assistant", "content": request_only_payload}
        request = [
            {"role": "system", "content": "system"},
            prefill,
            *deepcopy(canonical),
        ]

        first = instance.select_context(
            deepcopy(request),
            conversation_messages=deepcopy(canonical),
            budget_tokens=8_000,
        )
        second = instance.select_context(
            deepcopy(request),
            conversation_messages=deepcopy(canonical),
            budget_tokens=8_000,
        )
        stored = instance._store.get_session_messages(instance._session_id)

        assert first == second
        assert instance._store.get_session_count(instance._session_id) == 2
        assert stored[1]["content"].startswith("[Externalized tool output:")
        assert all(request_only_payload not in row["content"] for row in stored)
        assert first[1] == prefill
    finally:
        instance.shutdown()


def test_ignored_and_stateless_sessions_remain_noop(engine):
    request = [{"role": "user", "content": "do not persist"}]
    for attribute in ("_session_ignored", "_session_stateless"):
        setattr(engine, attribute, True)
        try:
            assert engine.select_context(
                deepcopy(request),
                conversation_messages=deepcopy(request),
                budget_tokens=4_000,
            ) is None
        finally:
            setattr(engine, attribute, False)
    assert engine._store.get_session_count(engine._session_id) == 0


def test_system_inside_canonical_snapshot_is_envelope_only_and_not_persisted(engine):
    conversation = [
        {"role": "system", "content": "legacy canonical system"},
        {"role": "user", "content": "durable user turn"},
    ]

    selected = engine.select_context(
        deepcopy(conversation),
        conversation_messages=deepcopy(conversation),
        budget_tokens=4_000,
    )
    stored = engine._store.get_session_messages(engine._session_id)

    assert selected == conversation
    assert [row["content"] for row in stored] == ["durable user turn"]


def test_ingest_failure_returns_none_fail_open(engine, monkeypatch):
    canonical = [{"role": "user", "content": "canonical turn"}]
    monkeypatch.setattr(
        engine,
        "_ingest_messages",
        Mock(side_effect=RuntimeError("storage unavailable")),
    )

    selected = engine.select_context(
        [{"role": "system", "content": "system"}, *deepcopy(canonical)],
        conversation_messages=deepcopy(canonical),
        budget_tokens=4_000,
    )

    assert selected is None
    assert engine._ingest_failure_count == 1


def test_dag_frontier_keeps_fresh_duplicate_identity_and_request_metadata(engine):
    durable = [
        {"role": "user", "content": "same literal"},
        {"role": "assistant", "content": "compacted bridge"},
        {"role": "user", "content": "same literal"},
    ]
    engine._ingest_messages(deepcopy(durable))
    rows = engine._store.get_session_messages(engine._session_id)
    engine._last_compacted_store_id = rows[1]["store_id"]
    engine._dag.add_node(
        SummaryNode(
            session_id=engine._session_id,
            depth=0,
            summary="summary covering the older duplicate",
            token_count=8,
            source_token_count=80,
            source_ids=[rows[0]["store_id"], rows[1]["store_id"]],
            source_type="messages",
            created_at=1.0,
            expand_hint="lcm_expand 1",
        )
    )
    canonical_fresh = durable[-1]
    request_fresh = {
        **canonical_fresh,
        "cache_control": {"type": "ephemeral"},
        "provider_metadata": {"trace": {"request_id": "fresh-duplicate"}},
    }

    selected = engine.select_context(
        [{"role": "system", "content": "system"}, deepcopy(request_fresh)],
        conversation_messages=[deepcopy(canonical_fresh)],
        budget_tokens=4_000,
    )

    assert selected is not None
    assert any(
        "summary covering the older duplicate" in str(message.get("content"))
        for message in selected
    )
    assert request_fresh in selected
    assert selected[-1] == request_fresh


def test_projection_store_ids_right_aligns_repeated_identities_monotonically(engine):
    durable = [
        {"role": "user", "content": "same literal"},
        {"role": "assistant", "content": "ordered bridge"},
        {"role": "user", "content": "same literal"},
        {"role": "assistant", "content": "ordered bridge"},
        {"role": "user", "content": "same literal"},
    ]
    engine._ingest_messages(deepcopy(durable))
    rows = engine._store.get_session_messages(engine._session_id)
    replay = [
        {"role": "user", "content": "unmatched leading scaffold"},
        *deepcopy(durable[-3:]),
        {"role": "assistant", "content": "unmatched trailing scaffold"},
    ]
    replay_before = deepcopy(replay)
    rows_before = deepcopy(rows)
    cursor_before = engine._ingest_cursor
    frontier_before = engine._last_compacted_store_id

    mapped = engine._projection_store_ids(replay)

    expected_ids = [row["store_id"] for row in rows[-3:]]
    assert mapped == [None, *expected_ids, None]
    mapped_ids = [store_id for store_id in mapped if store_id is not None]
    assert len(set(mapped_ids)) == len(mapped_ids)
    assert mapped_ids == sorted(mapped_ids)
    assert replay == replay_before
    assert engine._store.get_session_messages(engine._session_id) == rows_before
    assert engine._ingest_cursor == cursor_before
    assert engine._last_compacted_store_id == frontier_before


def test_dag_frontier_excludes_already_compacted_canonical_prefix(engine):
    conversation = [
        {"role": "user", "content": "compacted question"},
        {"role": "assistant", "content": "compacted answer"},
        {"role": "user", "content": "fresh frontier tail"},
    ]
    engine._ingest_messages(deepcopy(conversation))
    rows = engine._store.get_session_messages(engine._session_id)
    engine._last_compacted_store_id = rows[1]["store_id"]
    engine._dag.add_node(
        SummaryNode(
            session_id=engine._session_id,
            depth=0,
            summary="summary covering compacted prefix",
            token_count=8,
            source_token_count=80,
            source_ids=[rows[0]["store_id"], rows[1]["store_id"]],
            source_type="messages",
            created_at=1.0,
            expand_hint="lcm_expand 1",
        )
    )

    selected = engine.select_context(
        [{"role": "system", "content": "system"}, *deepcopy(conversation)],
        conversation_messages=deepcopy(conversation),
        budget_tokens=4_000,
    )

    assert selected is not None
    contents = [message.get("content") for message in selected]
    assert any("summary covering compacted prefix" in str(content) for content in contents)
    assert "fresh frontier tail" in contents
    assert "compacted question" not in contents
    assert "compacted answer" not in contents


@pytest.mark.parametrize(
    "scaffold_content",
    [
        (
            "[Current user objective preserved from compacted history]\n"
            "this exact text is the current request"
        ),
        (
            "[Recent Summary (d0, node 404)]\n"
            "this exact text is the current request\n"
            "[Expand for details: lcm_expand 404]"
        ),
    ],
    ids=["preserved-objective", "summary-shaped"],
)
def test_trusted_current_scaffold_shape_survives_projection_and_ingest_once(
    engine,
    scaffold_content,
):
    older_generated = {"role": "user", "content": scaffold_content}
    current = {"role": "user", "content": scaffold_content}
    conversation = [older_generated, current]
    request_current = {
        "role": "user",
        "content": scaffold_content,
        "provider_metadata": {"request": "current"},
    }
    request = [
        {"role": "system", "content": "stable system"},
        deepcopy(older_generated),
        request_current,
    ]

    selected = engine.select_context(
        request,
        conversation_messages=conversation,
        incoming_message=current,
        budget_tokens=4_000,
    )
    stored = engine._store.get_session_messages(engine._session_id)

    assert selected is not None
    assert [message.get("content") for message in selected].count(scaffold_content) == 1
    assert selected[-1] == request_current
    assert [row["content"] for row in stored] == [scaffold_content]


def test_ambiguous_equal_incoming_copy_fails_open_without_ingest(engine):
    repeated = {
        "role": "user",
        "content": (
            "[Current user objective preserved from compacted history]\n"
            "ambiguous equal copy"
        ),
    }
    conversation = [deepcopy(repeated), deepcopy(repeated)]

    selected = engine.select_context(
        [{"role": "system", "content": "system"}, *deepcopy(conversation)],
        conversation_messages=conversation,
        incoming_message=deepcopy(repeated),
        budget_tokens=4_000,
    )

    assert selected is None
    assert engine._store.get_session_count(engine._session_id) == 0


def test_budget_drops_summary_before_trusted_current_turn(engine):
    engine._dag.add_node(
        SummaryNode(
            session_id=engine._session_id,
            depth=0,
            summary="summary that fits only when the current turn is omitted " * 8,
            token_count=80,
            source_token_count=800,
            source_ids=[1],
            source_type="messages",
            created_at=1.0,
            expand_hint="lcm_expand 1",
        )
    )
    current = {"role": "user", "content": "keep this current request"}
    system = {"role": "system", "content": "immutable system envelope"}
    request_current = {
        **current,
        "provider_metadata": {"exact": ["current", {"keep": True}]},
    }
    request = [system, request_current]
    summary_message = {
        "role": "user",
        "content": (
            "[Recent Summary (d0, node 1)]\n"
            f"{'summary that fits only when the current turn is omitted ' * 8}\n"
            "[Expand for details: lcm_expand 1]"
        ),
    }
    budget = count_messages_tokens([system, current, summary_message]) - 1

    selected = engine.select_context(
        request,
        conversation_messages=[current],
        incoming_message=current,
        budget_tokens=budget,
    )

    assert selected == request
    assert count_messages_tokens(selected) <= budget


def test_projection_uses_threshold_as_prompt_safe_cap_when_host_budget_is_context_length(engine):
    engine._dag.add_node(
        SummaryNode(
            session_id=engine._session_id,
            depth=0,
            summary="summary must be dropped before the active turn " * 30,
            token_count=180,
            source_token_count=1_800,
            source_ids=[1],
            source_type="messages",
            created_at=1.0,
            expand_hint="lcm_expand 1",
        )
    )
    older = [
        {"role": "user", "content": "older question " * 120},
        {"role": "assistant", "content": "older answer " * 120},
    ]
    current = {"role": "user", "content": "retain this active turn"}
    request_current = {
        **current,
        "provider_metadata": {"active": True},
    }
    system = {"role": "system", "content": "immutable system envelope"}
    request = [system, *deepcopy(older), request_current]
    engine.context_length = 4_000
    engine.threshold_tokens = count_messages_tokens([system, request_current]) + 8

    selected = engine.select_context(
        request,
        conversation_messages=[*older, current],
        incoming_message=current,
        budget_tokens=engine.context_length,
    )

    assert selected == [system, request_current]
    assert count_messages_tokens(selected) <= engine.threshold_tokens
    assert selected[-1] == request_current
    assert not any("summary must be dropped" in str(message.get("content")) for message in selected)
    assert not any(message.get("content") == older[0]["content"] for message in selected)


def test_projection_fails_open_when_prompt_safe_threshold_is_not_positive(engine):
    current = {"role": "user", "content": "current turn"}
    engine.threshold_tokens = 0

    assert engine.select_context(
        [{"role": "system", "content": "system"}, deepcopy(current)],
        conversation_messages=[current],
        incoming_message=current,
        budget_tokens=engine.context_length,
    ) is None


def test_trusted_current_turn_over_budget_fails_open(engine):
    current = {"role": "user", "content": "oversized current request " * 80}
    system = {"role": "system", "content": "immutable system envelope"}
    request = [system, deepcopy(current)]
    budget = count_messages_tokens(request) - 1

    assert engine.select_context(
        request,
        conversation_messages=[current],
        incoming_message=current,
        budget_tokens=budget,
    ) is None


def test_trusted_current_tool_loop_is_atomic_and_preserves_request_metadata(engine):
    current = {"role": "user", "content": "run the current tool loop"}
    canonical_active = [
        current,
        {
            "role": "assistant",
            "content": "first attempt",
            "tool_calls": [{
                "id": "call-current",
                "type": "function",
                "function": {"name": "inspect", "arguments": '{"attempt":1}'},
            }],
        },
        {
            "role": "tool",
            "tool_call_id": "call-current",
            "tool_name": "inspect",
            "content": "retry required",
        },
        {"role": "assistant", "content": "retry completed"},
    ]
    request_active = deepcopy(canonical_active)
    request_active[1]["provider_metadata"] = {"trace": {"attempt": 1}}
    request_active[2]["cache_control"] = {"type": "ephemeral"}
    system = {"role": "system", "content": "immutable system envelope"}
    request = [system, *request_active]
    exact_budget = count_messages_tokens(request)

    selected = engine.select_context(
        request,
        conversation_messages=canonical_active,
        incoming_message=current,
        budget_tokens=exact_budget,
    )

    assert selected == request
    assert engine.select_context(
        request,
        conversation_messages=canonical_active,
        incoming_message=current,
        budget_tokens=exact_budget - 1,
    ) is None


def test_compatibility_budget_reserves_complete_newest_tool_turn_before_summary(engine):
    engine._dag.add_node(
        SummaryNode(
            session_id=engine._session_id,
            depth=0,
            summary="summary pressure " * 40,
            token_count=80,
            source_token_count=800,
            source_ids=[1],
            source_type="messages",
            created_at=1.0,
            expand_hint="lcm_expand 1",
        )
    )
    active = [
        {"role": "user", "content": "inspect the newest complete turn"},
        {
            "role": "assistant",
            "content": "calling inspect",
            "tool_calls": [{
                "id": "call-compat",
                "type": "function",
                "function": {"name": "inspect", "arguments": "{}"},
            }],
        },
        {
            "role": "tool",
            "tool_call_id": "call-compat",
            "tool_name": "inspect",
            "content": "complete result",
        },
    ]
    system = {"role": "system", "content": "immutable system envelope"}
    request = [system, *deepcopy(active)]

    selected = engine.select_context(
        request,
        conversation_messages=deepcopy(active),
        budget_tokens=count_messages_tokens(request),
    )

    assert selected == request
    assert [message["role"] for message in selected] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    assert not any("summary pressure" in str(message.get("content")) for message in selected)


def test_compatibility_budget_fails_open_when_only_eligible_tail_is_orphan_tool(engine):
    conversation = [
        {"role": "user", "content": "inspect the compacted prefix"},
        {
            "role": "assistant",
            "content": "calling inspect",
            "tool_calls": [{
                "id": "call-orphan",
                "type": "function",
                "function": {"name": "inspect", "arguments": "{}"},
            }],
        },
        {
            "role": "tool",
            "tool_call_id": "call-orphan",
            "tool_name": "inspect",
            "content": "frontier leaves this result eligible",
        },
    ]
    engine._ingest_messages(deepcopy(conversation))
    rows = engine._store.get_session_messages(engine._session_id)
    engine._last_compacted_store_id = rows[1]["store_id"]
    engine._dag.add_node(
        SummaryNode(
            session_id=engine._session_id,
            depth=0,
            summary="summary covering the call origin",
            token_count=8,
            source_token_count=80,
            source_ids=[rows[0]["store_id"], rows[1]["store_id"]],
            source_type="messages",
            created_at=1.0,
            expand_hint="lcm_expand 1",
        )
    )

    assert engine.select_context(
        [{"role": "system", "content": "system"}, *deepcopy(conversation)],
        conversation_messages=deepcopy(conversation),
        budget_tokens=4_000,
    ) is None


def test_compatibility_budget_fails_open_when_complete_newest_turn_cannot_fit(engine):
    newest = [
        {"role": "user", "content": "large newest prompt " * 40},
        {"role": "assistant", "content": "large newest response " * 40},
    ]
    system = {"role": "system", "content": "immutable system envelope"}
    request = [system, *deepcopy(newest)]

    assert engine.select_context(
        request,
        conversation_messages=deepcopy(newest),
        budget_tokens=count_messages_tokens(request) - 1,
    ) is None


def test_compatibility_budget_keeps_normal_newest_turn_atomic_before_summary(engine):
    engine._dag.add_node(
        SummaryNode(
            session_id=engine._session_id,
            depth=0,
            summary="short summary",
            token_count=4,
            source_token_count=40,
            source_ids=[1],
            source_type="messages",
            created_at=1.0,
            expand_hint="lcm_expand 1",
        )
    )
    newest = [
        {"role": "user", "content": "retain the complete newest group " * 30},
        {"role": "assistant", "content": "retained atomically"},
    ]
    system = {"role": "system", "content": "immutable system envelope"}
    request = [system, *deepcopy(newest)]

    selected = engine.select_context(
        request,
        conversation_messages=deepcopy(newest),
        budget_tokens=count_messages_tokens(request),
    )

    assert selected == request
    assert not any("short summary" in str(message.get("content")) for message in selected)
