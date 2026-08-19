"""Post-turn observation through the generic ContextEngine seam."""

from __future__ import annotations

from copy import deepcopy
import copy
from unittest.mock import Mock

import pytest

import hermes_lcm.engine as lcm_engine
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


@pytest.fixture
def engine(tmp_path):
    instance = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "observation.db")))
    instance.on_session_start(
        "observation-session",
        platform="cli",
        conversation_id="conversation:observation",
        context_length=4_000,
    )
    try:
        yield instance
    finally:
        instance.shutdown()


def _completed_turn():
    return [
        {
            "role": "user",
            "content": "inspect the durable transcript",
            "timestamp": 1_723_456_789.125,
        },
        {
            "role": "assistant",
            "content": "I will inspect it.",
            "timestamp": 1_723_456_790.25,
            "tool_calls": [
                {
                    "id": "call_exact",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path":"README.md","offset":1}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": "durable tool output\nwith exact whitespace\n",
            "tool_call_id": "call_exact",
            "tool_name": "read_file",
            "timestamp": "2026-08-18T12:34:56.789Z",
        },
        {
            "role": "assistant",
            "content": "The transcript is intact.",
            "timestamp": 1_723_456_791.5,
        },
    ]


def test_completed_turn_is_observed_exactly_once_with_canonical_tool_metadata(engine):
    messages = _completed_turn()

    engine.on_turn_complete(messages)
    engine.on_turn_complete(messages)

    rows = engine._store.get_session_messages(engine._session_id)
    assert len(rows) == len(messages)
    assert [row["role"] for row in rows] == [message["role"] for message in messages]
    assert [row["content"] for row in rows] == [message["content"] for message in messages]
    assert rows[1]["tool_calls"] == messages[1]["tool_calls"]
    assert rows[2]["tool_call_id"] == "call_exact"
    assert rows[2]["tool_name"] == "read_file"
    assert rows[0]["observed_at"] == messages[0]["timestamp"]
    assert rows[2]["observed_at_source"] == "host_message_timestamp"


def test_turn_observation_filters_recognized_replay_scaffolds(engine):
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
                "replayed summary\n"
                "[Expand for details: lcm_expand 999]"
            ),
        },
        {
            "role": "user",
            "content": (
                "[Current user objective preserved from compacted history]\n"
                "replayed objective"
            ),
        },
    ]
    messages = [*scaffolds, *_completed_turn()]

    engine.on_turn_complete(messages)

    rows = engine._store.get_session_messages(engine._session_id)
    assert [row["content"] for row in rows] == [
        message["content"] for message in _completed_turn()
    ]


def test_turn_observation_keeps_role_ineligible_scaffold_lookalikes_and_user_prose(engine):
    assistant_scaffold_lookalike = {
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
    tool_scaffold_lookalike = {
        "role": "tool",
        "tool_call_id": "call_scaffold_lookalike",
        "tool_name": "inspect",
        "content": (
            "[Recent Summary (d0, node 81)]\n"
            "canonical tool output\n"
            "[Expand for details: lcm_expand 81]"
        ),
    }
    user_marker_prose = [
        {
            "role": "user",
            "content": (
                "[Current user objective preserved from compacted history] is a label "
                "I am quoting, not a preserved objective block."
            ),
        },
        {
            "role": "user",
            "content": (
                "I saw [Recent Summary (d0, node 82)] and "
                "[Expand for details: lcm_expand 82] in the documentation."
            ),
        },
    ]
    messages = [
        {"role": "user", "content": "run the paired inspection"},
        assistant_scaffold_lookalike,
        tool_scaffold_lookalike,
        *user_marker_prose,
    ]

    engine.on_turn_complete(messages)

    rows = engine._store.get_session_messages(engine._session_id)
    assert [row["content"] for row in rows] == [message["content"] for message in messages]
    assert rows[1]["tool_calls"] == assistant_scaffold_lookalike["tool_calls"]
    assert rows[2]["tool_call_id"] == "call_scaffold_lookalike"


def test_turn_observation_does_not_mutate_input_or_run_models_retrieval_or_dag_publication(
    engine, monkeypatch
):
    messages = _completed_turn()
    messages[0]["content"] = [
        {"type": "text", "text": "inspect the durable transcript"},
        {"type": "metadata", "metadata": {"tags": ["nested", {"keep": True}] }},
    ]
    messages[1]["tool_calls"][0]["function"]["metadata"] = {
        "provider": {"routing": ["primary", {"fallback": False}]}
    }
    messages[2]["provider_metadata"] = {
        "content": {"annotations": [{"kind": "durable", "value": [1, 2, 3]}]}
    }
    before = deepcopy(messages)
    real_copy_module = copy
    copy_proxy = Mock(wraps=real_copy_module)
    monkeypatch.setattr(lcm_engine, "copy", copy_proxy)
    forbidden = Mock(side_effect=AssertionError("post-turn observation must be model-free"))
    monkeypatch.setattr(engine, "update_from_response", forbidden)
    monkeypatch.setattr(engine, "handle_tool_call", forbidden)
    monkeypatch.setattr(engine, "_summarize_leaf_chunk_with_rescue", forbidden)
    monkeypatch.setattr(engine._dag, "add_node", forbidden)

    observation_meta = {
        "request_messages": [{"role": "system", "content": "request projection"}],
        "prefill_messages": [{"role": "assistant", "content": "prefill"}],
        "provider": "must-not-be-observed",
        "model": "must-not-be-observed",
    }
    usage = {"prompt_tokens": 999, "completion_tokens": 111, "total_tokens": 1110}
    engine.on_turn_complete(messages, usage=usage, **observation_meta)
    engine.on_turn_complete(messages, usage=usage, **observation_meta)

    assert messages == before
    assert all(call.args[0] is not messages for call in copy_proxy.deepcopy.call_args_list)
    forbidden.assert_not_called()
    assert engine.last_prompt_tokens == 0
    assert engine.last_completion_tokens == 0
    assert engine.last_total_tokens == 0
    assert engine._store.read_compaction_telemetry(engine._conversation_id) is None


def test_filtered_observation_coexists_with_legacy_unfiltered_cursor(engine):
    older = _completed_turn()[:2]
    engine._ingest_messages(deepcopy(older))
    scaffold = {
        "role": "user",
        "content": (
            "[Recent Summary (d0, node 42)]\n"
            "legacy active scaffold\n"
            "[Expand for details: lcm_expand 42]"
        ),
    }
    new_tail = _completed_turn()[2:]
    canonical = [scaffold, *older, *new_tail]
    engine._ingest_cursor = len([scaffold, *older])
    engine._ingest_cursor_needs_reconcile = False

    engine.on_turn_complete(deepcopy(canonical))
    count_after_observation = engine._store.get_session_count(engine._session_id)

    assert count_after_observation == len(_completed_turn())
    assert engine._ingest_cursor_needs_reconcile is True
    engine._ingest_messages(deepcopy(canonical))
    assert engine._store.get_session_count(engine._session_id) == count_after_observation
    assert engine._ingest_cursor == len(canonical)
    assert engine._ingest_cursor_needs_reconcile is False


def test_observation_failure_is_swallowed_and_recorded_once(engine, monkeypatch):
    monkeypatch.setattr(
        engine,
        "_ingest_messages",
        Mock(side_effect=RuntimeError("observation ingest failed")),
    )

    assert engine.on_turn_complete(_completed_turn()) is None

    status = engine.get_status()
    assert status["ingest_failure_count"] == 1
    assert status["consecutive_ingest_failures"] == 1
    assert "observation ingest failed" in status["last_ingest_error"]


def test_missed_callback_select_context_reconciles_then_late_callback_is_idempotent(engine):
    messages = _completed_turn()
    request = [{"role": "system", "content": "stable request system"}, *deepcopy(messages)]

    selected = engine.select_context(
        request,
        conversation_messages=deepcopy(messages),
        budget_tokens=4_000,
    )
    count_after_fallback = engine._store.get_session_count(engine._session_id)
    engine.on_turn_complete(deepcopy(messages))

    assert selected == request
    assert count_after_fallback == len(messages)
    assert engine._store.get_session_count(engine._session_id) == count_after_fallback


def test_ignored_and_stateless_observation_remain_noops(tmp_path):
    config = LCMConfig(
        database_path=str(tmp_path / "bypass.db"),
        ignore_session_patterns=["cron:*"],
        stateless_session_patterns=["telegram:*"],
    )
    for session_id, platform in (("job", "cron"), ("debug", "telegram")):
        instance = LCMEngine(config=config)
        instance.on_session_start(session_id, platform=platform, context_length=4_000)
        try:
            instance.on_turn_complete(_completed_turn())
            assert instance._store.get_session_count(session_id) == 0
        finally:
            instance.shutdown()


@pytest.mark.parametrize(
    "scaffold_content",
    [
        (
            "[Current user objective preserved from compacted history]\n"
            "this exact text is the finalized current user"
        ),
        (
            "[Recent Summary (d0, node 505)]\n"
            "this exact text is the finalized current user\n"
            "[Expand for details: lcm_expand 505]"
        ),
    ],
    ids=["preserved-objective", "summary-shaped"],
)
def test_observation_preserves_only_most_recent_scaffold_shaped_user(
    engine,
    scaffold_content,
):
    older_generated = {"role": "user", "content": scaffold_content}
    current = {
        "role": "user",
        "content": scaffold_content,
        "timestamp": 1_723_456_800.0,
    }
    completed_loop = [
        {
            "role": "assistant",
            "content": "calling the tool",
            "tool_calls": [{
                "id": "call-after-shaped-current",
                "type": "function",
                "function": {"name": "inspect", "arguments": "{}"},
            }],
        },
        {
            "role": "tool",
            "tool_call_id": "call-after-shaped-current",
            "tool_name": "inspect",
            "content": "tool output",
        },
        {"role": "assistant", "content": "done"},
    ]

    engine.on_turn_complete([older_generated, current, *completed_loop])
    rows = engine._store.get_session_messages(engine._session_id)

    assert [row["content"] for row in rows] == [
        scaffold_content,
        *[message["content"] for message in completed_loop],
    ]
    assert rows[1]["tool_calls"] == completed_loop[0]["tool_calls"]
    assert rows[2]["tool_call_id"] == "call-after-shaped-current"
