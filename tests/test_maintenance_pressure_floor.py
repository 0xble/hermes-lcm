"""Opportunistic maintenance should require configured token pressure."""

from pathlib import Path

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


def _engine(tmp_path: Path, **overrides) -> LCMEngine:
    config = LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        large_output_externalization_path=str(tmp_path / "externalized"),
        **overrides,
    )
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    engine.on_session_start(
        "maintenance-floor-session",
        platform="telegram",
        conversation_id="maintenance-floor-conversation",
        context_length=1_000_000,
    )
    return engine


def test_maintenance_pressure_floor_defaults_disabled(tmp_path):
    engine = _engine(tmp_path)
    engine.threshold_tokens = 750_000

    assert engine._config.maintenance_min_pressure_ratio == 0.0
    assert engine._maintenance_pressure_met(1) is True


def test_maintenance_pressure_floor_blocks_below_configured_fraction(tmp_path):
    engine = _engine(tmp_path, maintenance_min_pressure_ratio=0.5)
    engine.threshold_tokens = 750_000

    assert engine._maintenance_pressure_met(374_999) is False
    assert engine._maintenance_pressure_met(375_000) is True


def test_unknown_threshold_never_blocks_maintenance(tmp_path):
    engine = _engine(tmp_path, maintenance_min_pressure_ratio=0.5)
    engine.threshold_tokens = 0

    assert engine._maintenance_pressure_met(1) is True


def test_maintenance_pressure_ratio_loads_from_environment(monkeypatch):
    monkeypatch.setenv("LCM_MAINTENANCE_MIN_PRESSURE_RATIO", "0.4")

    assert LCMConfig.from_env().maintenance_min_pressure_ratio == pytest.approx(0.4)


def test_divergent_replay_leaf_maintenance_respects_pressure_floor(tmp_path, monkeypatch):
    engine = _engine(tmp_path, maintenance_min_pressure_ratio=0.5)
    engine.threshold_tokens = 100
    messages = [{"role": "user", "content": "original"}]
    replay = [{"role": "user", "content": "rewritten"}]

    monkeypatch.setattr(engine, "_ingest_messages", lambda _messages: replay)
    monkeypatch.setattr(engine, "_replay_diff_requests_ingest_cleanup", lambda *_args: False)
    monkeypatch.setattr(engine, "_should_force_overflow_recovery", lambda **_kwargs: False)
    monkeypatch.setattr(
        engine,
        "_leaf_compaction_candidate_status",
        lambda *_args, **_kwargs: (True, "eligible raw backlog outside fresh tail"),
    )
    monkeypatch.setattr(engine, "_has_ignored_backlog_outside_fresh_tail", lambda _messages: False)
    monkeypatch.setattr(engine, "_should_run_deferred_maintenance", lambda *_args, **_kwargs: False)
    monkeypatch.setattr("hermes_lcm.compaction.count_messages_tokens", lambda value: 20 if value is replay else 10)

    assert engine.should_compress_preflight(messages) is False


def test_divergent_replay_leaf_maintenance_runs_at_configured_floor(tmp_path, monkeypatch):
    engine = _engine(tmp_path, maintenance_min_pressure_ratio=1.0)
    engine.threshold_tokens = 100
    messages = [{"role": "user", "content": "original"}]
    replay = [{"role": "user", "content": "rewritten"}]

    monkeypatch.setattr(engine, "_ingest_messages", lambda _messages: replay)
    monkeypatch.setattr(engine, "_replay_diff_requests_ingest_cleanup", lambda *_args: False)
    monkeypatch.setattr(engine, "_should_force_overflow_recovery", lambda **_kwargs: False)
    monkeypatch.setattr(
        engine,
        "_leaf_compaction_candidate_status",
        lambda *_args, **_kwargs: (True, "eligible raw backlog outside fresh tail"),
    )
    monkeypatch.setattr(engine, "_has_ignored_backlog_outside_fresh_tail", lambda _messages: False)
    monkeypatch.setattr(engine, "_should_run_deferred_maintenance", lambda *_args, **_kwargs: False)
    monkeypatch.setattr("hermes_lcm.compaction.count_messages_tokens", lambda _value: 100)

    assert engine.should_compress_preflight(messages) is True


def test_host_pressure_blocks_model_work_when_lcm_estimate_is_above_floor(tmp_path, monkeypatch):
    engine = _engine(
        tmp_path,
        maintenance_min_pressure_ratio=1.0,
        fresh_tail_count=1,
        leaf_chunk_tokens=1,
    )
    engine.threshold_tokens = 100
    messages = [
        {"role": "user", "content": "eligible old backlog"},
        {"role": "assistant", "content": "eligible old response"},
        {"role": "user", "content": "fresh tail"},
    ]
    monkeypatch.setattr(engine, "_should_force_overflow_recovery", lambda **_kwargs: False)
    monkeypatch.setattr("hermes_lcm.compaction.count_messages_tokens", lambda _value: 150)
    summary_calls = []

    def fail_if_called(*_args, **_kwargs):
        summary_calls.append(True)
        raise AssertionError("subthreshold automatic maintenance must not summarize")

    monkeypatch.setattr(engine, "_summarize_leaf_chunk_with_rescue", fail_if_called)

    assert engine.should_compress_preflight(messages) is True
    assert engine.compress(messages, current_tokens=20) == messages
    assert engine._last_compression_status == "deferred"
    assert engine.compression_count == 0
    assert summary_calls == []


def test_maintenance_intent_does_not_adopt_late_cleanup(tmp_path, monkeypatch):
    engine = _engine(
        tmp_path,
        maintenance_min_pressure_ratio=1.0,
        fresh_tail_count=1,
        leaf_chunk_tokens=1,
    )
    engine.threshold_tokens = 100
    messages = [
        {"role": "user", "content": "eligible old backlog"},
        {"role": "assistant", "content": "eligible old response"},
        {"role": "user", "content": "fresh tail"},
    ]
    replay = [
        {"role": "user", "content": "late replay change"},
        messages[1],
        messages[2],
    ]
    ingests = iter([messages, replay])
    monkeypatch.setattr(engine, "_ingest_messages", lambda _messages: next(ingests))
    monkeypatch.setattr(engine, "_should_force_overflow_recovery", lambda **_kwargs: False)
    monkeypatch.setattr("hermes_lcm.compaction.count_messages_tokens", lambda _value: 150)
    monkeypatch.setattr(
        engine,
        "_summarize_leaf_chunk_with_rescue",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("maintenance intent must not summarize")
        ),
    )

    assert engine.should_compress_preflight(messages) is True
    assert engine.compress(messages, current_tokens=20) == messages
    assert engine._last_compression_status == "deferred"
    assert engine.compression_count == 0


@pytest.mark.parametrize(
    ("current_tokens", "force"),
    [(100, False), (20, True)],
    ids=["exact-floor", "manual-force"],
)
def test_host_pressure_guard_preserves_explicit_admission(
    tmp_path,
    monkeypatch,
    current_tokens,
    force,
):
    engine = _engine(
        tmp_path,
        maintenance_min_pressure_ratio=1.0,
        fresh_tail_count=1,
        leaf_chunk_tokens=1,
    )
    engine.threshold_tokens = 100
    messages = [
        {"role": "user", "content": "eligible old backlog"},
        {"role": "assistant", "content": "eligible old response"},
        {"role": "user", "content": "fresh tail"},
    ]
    monkeypatch.setattr(engine, "_should_force_overflow_recovery", lambda **_kwargs: False)
    monkeypatch.setattr("hermes_lcm.compaction.count_messages_tokens", lambda _value: 150)
    monkeypatch.setattr(
        engine,
        "_summarize_leaf_chunk_with_rescue",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("admitted model work reached summarizer")
        ),
    )

    assert engine.should_compress_preflight(messages) is True
    with pytest.raises(AssertionError, match="admitted model work reached summarizer"):
        engine.compress(messages, current_tokens=current_tokens, force=force)


def test_session_reset_clears_preflight_handoffs(tmp_path):
    engine = _engine(tmp_path, maintenance_min_pressure_ratio=1.0)
    engine._preflight_intent = "maintenance"
    engine._preflight_session_id = engine.current_session_id
    engine._preflight_message_list_id = 42
    engine._preflight_cleanup_only_due_to_boundary_cooldown = True

    engine._reset_session_scoped_runtime_state()

    assert engine._preflight_intent is None
    assert engine._preflight_session_id is None
    assert engine._preflight_message_list_id is None
    assert engine._preflight_cleanup_only_due_to_boundary_cooldown is False


def test_preflight_intent_does_not_apply_to_different_message_list(tmp_path, monkeypatch):
    engine = _engine(
        tmp_path,
        maintenance_min_pressure_ratio=1.0,
        fresh_tail_count=1,
        leaf_chunk_tokens=1,
    )
    engine.threshold_tokens = 100
    preflight_messages = [
        {"role": "user", "content": "old preflight backlog"},
        {"role": "assistant", "content": "old preflight response"},
        {"role": "user", "content": "old fresh tail"},
    ]
    direct_messages = [
        {"role": "user", "content": "different direct backlog"},
        {"role": "assistant", "content": "different direct response"},
        {"role": "user", "content": "different fresh tail"},
    ]
    monkeypatch.setattr(engine, "_should_force_overflow_recovery", lambda **_kwargs: False)
    monkeypatch.setattr("hermes_lcm.compaction.count_messages_tokens", lambda _value: 150)
    monkeypatch.setattr(
        engine,
        "_summarize_leaf_chunk_with_rescue",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("different direct compress reached summarizer")
        ),
    )

    assert engine.should_compress_preflight(preflight_messages) is True
    engine._preflight_cleanup_only_due_to_boundary_cooldown = True
    with pytest.raises(AssertionError, match="different direct compress reached summarizer"):
        engine.compress(direct_messages, current_tokens=20)


def test_empty_compress_consumes_preflight_intent(tmp_path, monkeypatch):
    engine = _engine(
        tmp_path,
        maintenance_min_pressure_ratio=1.0,
        fresh_tail_count=1,
        leaf_chunk_tokens=1,
    )
    engine.threshold_tokens = 100
    messages = [
        {"role": "user", "content": "eligible old backlog"},
        {"role": "assistant", "content": "eligible old response"},
        {"role": "user", "content": "fresh tail"},
    ]
    monkeypatch.setattr(engine, "_should_force_overflow_recovery", lambda **_kwargs: False)
    monkeypatch.setattr("hermes_lcm.compaction.count_messages_tokens", lambda _value: 150)
    monkeypatch.setattr(
        engine,
        "_summarize_leaf_chunk_with_rescue",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("unmarked direct compress reached summarizer")
        ),
    )

    assert engine.should_compress_preflight(messages) is True
    assert engine.compress([]) == []
    with pytest.raises(AssertionError, match="unmarked direct compress reached summarizer"):
        engine.compress(messages, current_tokens=20)


def test_divergent_replay_ignored_backlog_respects_pressure_floor(tmp_path, monkeypatch):
    engine = _engine(tmp_path, maintenance_min_pressure_ratio=0.5)
    engine.threshold_tokens = 100
    messages = [{"role": "user", "content": "original"}]
    replay = [{"role": "user", "content": "rewritten"}]

    monkeypatch.setattr(engine, "_ingest_messages", lambda _messages: replay)
    monkeypatch.setattr(engine, "_replay_diff_requests_ingest_cleanup", lambda *_args: False)
    monkeypatch.setattr(engine, "_should_force_overflow_recovery", lambda **_kwargs: False)
    monkeypatch.setattr(
        engine,
        "_leaf_compaction_candidate_status",
        lambda *_args, **_kwargs: (False, "below leaf chunk threshold"),
    )
    monkeypatch.setattr(engine, "_has_ignored_backlog_outside_fresh_tail", lambda _messages: True)
    monkeypatch.setattr(engine, "_should_run_deferred_maintenance", lambda *_args, **_kwargs: False)
    monkeypatch.setattr("hermes_lcm.compaction.count_messages_tokens", lambda value: 20 if value is replay else 10)

    assert engine.should_compress_preflight(messages) is False
