"""Automatic optional replay cleanup must make active context smaller."""

from pathlib import Path

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.tokens import count_messages_tokens


def _engine(tmp_path: Path) -> LCMEngine:
    config = LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        fresh_tail_count=32,
        leaf_chunk_tokens=20_000,
    )
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    engine.on_session_start(
        "optional-cleanup-session",
        platform="telegram",
        conversation_id="optional-cleanup-conversation",
        context_length=1_000_000,
    )
    engine.threshold_tokens = 750_000
    return engine


def _optional_stub(text: str) -> str:
    return f"[Externalized LCM ingest payload: {text}]"


def _prepare_preflight(engine, monkeypatch, replay, original_tokens, replay_tokens):
    monkeypatch.setattr(engine, "_ingest_messages", lambda _messages: replay)
    monkeypatch.setattr(engine, "_should_force_overflow_recovery", lambda **_kwargs: False)
    monkeypatch.setattr(
        "hermes_lcm.compaction.count_messages_tokens",
        lambda value: replay_tokens if value is replay else original_tokens,
    )


def test_preflight_defers_optional_cleanup_that_would_grow(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    messages = [{"role": "user", "content": "x"}]
    replay = [{"role": "user", "content": _optional_stub("longer stub")}]
    _prepare_preflight(engine, monkeypatch, replay, original_tokens=55_812, replay_tokens=55_961)

    assert engine.should_compress_preflight(messages) is False
    assert engine._last_compression_status == "deferred"
    assert "would not reduce active context" in engine._last_compression_noop_reason


def test_real_externalization_growth_is_deferred(tmp_path):
    engine = LCMEngine(
        config=LCMConfig(
            database_path=str(tmp_path / "real-lcm.db"),
            fresh_tail_count=32,
            leaf_chunk_tokens=20_000,
            large_output_externalization_enabled=True,
            large_output_externalization_threshold_chars=40,
            large_output_externalization_path=str(tmp_path / "externalized"),
        ),
        hermes_home=str(tmp_path / "real-home"),
    )
    engine.on_session_start(
        "real-optional-cleanup-session",
        platform="telegram",
        context_length=1_000_000,
    )
    messages = [{"role": "user", "content": "raw payload " + "x" * 200}]
    replay = engine._ingest_messages(messages)

    assert engine._replay_diff_is_optional_cleanup_only(messages, replay) is True
    assert count_messages_tokens(replay) >= count_messages_tokens(messages)
    assert engine.should_compress_preflight(messages) is False
    assert engine._last_compression_status == "deferred"
    assert engine._store.count_session_load_messages(engine.current_session_id) == 1
    assert any((tmp_path / "externalized").iterdir())


def test_preflight_defers_optional_cleanup_that_is_equal_size(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    messages = [{"role": "user", "content": "x"}]
    replay = [{"role": "user", "content": _optional_stub("equal-sized stub")}]
    _prepare_preflight(engine, monkeypatch, replay, original_tokens=20, replay_tokens=20)

    assert engine.should_compress_preflight(messages) is False
    assert engine._last_compression_status == "deferred"


def test_preflight_preserves_overflow_recovery_for_optional_cleanup(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    messages = [{"role": "user", "content": "x"}]
    replay = [{"role": "user", "content": _optional_stub("larger stub")}]
    _prepare_preflight(engine, monkeypatch, replay, original_tokens=10, replay_tokens=11)
    monkeypatch.setattr(engine, "_should_force_overflow_recovery", lambda **_kwargs: True)

    assert engine.should_compress_preflight(messages) is True


def test_preflight_allows_optional_cleanup_that_shrinks(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    messages = [{"role": "user", "content": "large raw payload"}]
    replay = [{"role": "user", "content": _optional_stub("short")}]
    _prepare_preflight(engine, monkeypatch, replay, original_tokens=20, replay_tokens=10)

    assert engine.should_compress_preflight(messages) is True


def test_preflight_preserves_mandatory_sensitive_cleanup_even_when_larger(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    messages = [{"role": "tool", "content": "secret", "tool_call_id": "call-1"}]
    replay = [
        {
            "role": "tool",
            "content": "[LCM sensitive redaction: value removed by policy]",
            "tool_call_id": "call-1",
        }
    ]
    _prepare_preflight(engine, monkeypatch, replay, original_tokens=10, replay_tokens=11)

    assert engine.should_compress_preflight(messages) is True


def test_sensitive_externalization_cleanup_is_not_treated_as_optional(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    engine._config.sensitive_patterns_enabled = True
    engine._config.sensitive_patterns = ["api_key"]
    messages = [
        {
            "role": "user",
            "content": "api_key=sk-sensitive-secret-123456789 " + "x" * 200,
        }
    ]
    replay = [{"role": "user", "content": _optional_stub("externalized secret")}]
    _prepare_preflight(engine, monkeypatch, replay, original_tokens=10, replay_tokens=11)

    assert engine._replay_diff_is_optional_cleanup_only(messages, replay) is False
    assert engine.should_compress_preflight(messages) is True


def test_mixed_length_replay_is_not_misclassified_as_optional(tmp_path):
    engine = _engine(tmp_path)
    original = [
        {"role": "user", "content": "new"},
        {"role": "assistant", "content": "replay"},
        {"role": "user", "content": "newer"},
    ]
    replay = original[:-1]

    assert engine._replay_diff_is_optional_cleanup_only(original, replay) is False


def test_compress_returns_original_when_optional_cleanup_does_not_shrink(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    messages = [{"role": "user", "content": "x"}]
    replay = [{"role": "user", "content": _optional_stub("longer stub")}]
    monkeypatch.setattr(engine, "_ingest_messages", lambda _messages: replay)
    monkeypatch.setattr(engine, "_should_force_overflow_recovery", lambda **_kwargs: False)
    monkeypatch.setattr(
        "hermes_lcm.compaction.count_messages_tokens",
        lambda value: 11 if value is replay else 10,
    )
    monkeypatch.setattr(
        "hermes_lcm.engine.summarize_with_escalation",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("summarizer must not run")),
    )

    assert engine.compress(messages) == messages
    assert engine._last_compression_status == "deferred"
    assert "would not reduce active context" in engine._last_compression_noop_reason
    assert engine.compression_count == 0


def test_compress_defers_before_eligible_leaf_can_reach_summarizer(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    engine._config.fresh_tail_count = 1
    engine._config.leaf_chunk_tokens = 1
    messages = [
        {"role": "user", "content": "eligible old backlog"},
        {"role": "assistant", "content": "eligible old response"},
        {"role": "user", "content": "fresh tail"},
    ]
    replay = [
        {"role": "user", "content": _optional_stub("nonshrinking externalization")},
        messages[1],
        messages[2],
    ]
    monkeypatch.setattr(engine, "_ingest_messages", lambda _messages: replay)
    monkeypatch.setattr(engine, "_should_force_overflow_recovery", lambda **_kwargs: False)
    monkeypatch.setattr(
        "hermes_lcm.compaction.count_messages_tokens",
        lambda value: 11 if value is replay else 10,
    )
    monkeypatch.setattr(
        "hermes_lcm.engine.summarize_with_escalation",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("summarizer must not run")),
    )

    assert engine.compress(messages) == messages
    assert engine._last_compression_status == "deferred"
    assert engine.compression_count == 0


def test_compress_preserves_overflow_recovery_for_nonshrinking_cleanup(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    messages = [{"role": "user", "content": "x"}]
    replay = [{"role": "user", "content": _optional_stub("larger stub")}]
    monkeypatch.setattr(engine, "_ingest_messages", lambda _messages: replay)
    monkeypatch.setattr(engine, "_should_force_overflow_recovery", lambda **_kwargs: True)
    monkeypatch.setattr(
        "hermes_lcm.compaction.count_messages_tokens",
        lambda value: 11 if value is replay else 10,
    )
    monkeypatch.setattr(
        "hermes_lcm.engine.summarize_with_escalation",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("summarizer must not run")),
    )

    assert engine.compress(messages) == replay
    assert engine._last_compression_status == "overflow_recovery"


def test_manual_compress_may_adopt_nonshrinking_optional_cleanup(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    messages = [{"role": "user", "content": "x"}]
    replay = [{"role": "user", "content": _optional_stub("longer stub")}]
    monkeypatch.setattr(engine, "_ingest_messages", lambda _messages: replay)
    monkeypatch.setattr(engine, "_should_force_overflow_recovery", lambda **_kwargs: False)
    monkeypatch.setattr(
        "hermes_lcm.compaction.count_messages_tokens",
        lambda value: 11 if value is replay else 10,
    )

    assert engine.compress(messages, force=True) == replay
    assert engine._last_compression_status == "sanitized"
