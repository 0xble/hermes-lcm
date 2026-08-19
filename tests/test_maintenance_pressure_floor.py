"""Opportunistic maintenance should require configured token pressure."""

from copy import deepcopy
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


def test_divergent_replay_leaf_maintenance_never_runs_from_preflight(tmp_path, monkeypatch):
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

    assert engine.should_compress_preflight(messages) is False


def test_subthreshold_eligible_leaf_preflight_is_false_with_default_floor(
    tmp_path,
    monkeypatch,
):
    engine = _engine(tmp_path, maintenance_min_pressure_ratio=0.0)
    engine.threshold_tokens = 100
    messages = [{"role": "user", "content": "eligible backlog"}]

    monkeypatch.setattr(engine, "_ingest_messages", lambda _messages: messages)
    monkeypatch.setattr(engine, "_should_force_overflow_recovery", lambda **_kwargs: False)
    monkeypatch.setattr(
        engine,
        "_leaf_compaction_candidate_status",
        lambda *_args, **_kwargs: (True, "eligible raw backlog outside fresh tail"),
    )
    monkeypatch.setattr("hermes_lcm.compaction.count_messages_tokens", lambda _value: 99)

    assert engine.should_compress_preflight(messages) is False


def test_divergent_replay_over_threshold_never_admits_model_work_from_preflight(
    tmp_path,
    monkeypatch,
):
    engine = _engine(tmp_path)
    engine.threshold_tokens = 100
    messages = [{"role": "user", "content": "original"}]
    replay = [{"role": "user", "content": "rewritten"}]

    monkeypatch.setattr(engine, "_ingest_messages", lambda _messages: replay)
    monkeypatch.setattr(engine, "_replay_diff_requests_ingest_cleanup", lambda *_args: False)
    monkeypatch.setattr(engine, "_should_force_overflow_recovery", lambda **_kwargs: False)
    monkeypatch.setattr("hermes_lcm.compaction.count_messages_tokens", lambda _value: 1_000)

    assert engine.should_compress_preflight(messages) is False


def test_host_pressure_blocks_model_work_across_hermes_deepcopy_boundary(tmp_path, monkeypatch):
    engine = _engine(
        tmp_path,
        maintenance_min_pressure_ratio=1.0,
        fresh_tail_count=1,
        leaf_chunk_tokens=1,
    )
    engine.threshold_tokens = 100
    engine._config.extraction_enabled = True
    messages = [
        {"role": "user", "content": "eligible old backlog"},
        {"role": "assistant", "content": "eligible old response"},
        {"role": "user", "content": "fresh tail"},
    ]
    monkeypatch.setattr(engine, "_should_force_overflow_recovery", lambda **_kwargs: False)
    monkeypatch.setattr("hermes_lcm.compaction.count_messages_tokens", lambda _value: 150)
    model_work_calls = []

    def fail_if_called(stage):
        def fail(*_args, **_kwargs):
            model_work_calls.append(stage)
            raise AssertionError(
                f"subthreshold automatic maintenance must not run {stage}"
            )

        return fail

    monkeypatch.setattr(engine, "_run_pre_compaction_extraction", fail_if_called("extraction"))
    monkeypatch.setattr(engine, "_summarize_leaf_chunk_with_rescue", fail_if_called("summarization"))
    monkeypatch.setattr(engine, "_maybe_condense", fail_if_called("condensation"))
    monkeypatch.setattr(engine._dag, "add_node", fail_if_called("DAG publication"))

    assert engine.should_compress_preflight(messages) is False
    copied_messages = deepcopy(messages)
    assert engine.compress(copied_messages, current_tokens=20) == copied_messages
    assert engine._last_compression_status == "deferred"
    assert engine.compression_count == 0
    assert model_work_calls == []


def test_subthreshold_host_pressure_does_not_adopt_unrequested_replay_diff(
    tmp_path,
    monkeypatch,
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

    assert engine.should_compress_preflight(messages) is False
    assert engine.compress(messages, current_tokens=20) == messages
    assert engine._last_compression_status == "deferred"
    assert engine.compression_count == 0


@pytest.mark.parametrize(
    ("current_tokens", "force"),
    [(100, False), (20, True)],
    ids=["exact-threshold", "manual-force"],
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

    if not force:
        assert engine.should_compress(current_tokens) is True
    with pytest.raises(AssertionError, match="admitted model work reached summarizer"):
        engine.compress(messages, current_tokens=current_tokens, force=force)


def test_configured_critical_pressure_uses_host_pressure_for_model_admission(
    tmp_path,
    monkeypatch,
):
    engine = _engine(
        tmp_path,
        maintenance_min_pressure_ratio=1.0,
        critical_budget_pressure_ratio=0.5,
        fresh_tail_count=1,
        leaf_chunk_tokens=1,
    )
    engine.context_length = 200
    engine.threshold_tokens = 150
    messages = [
        {"role": "user", "content": "eligible old backlog"},
        {"role": "assistant", "content": "eligible old response"},
        {"role": "user", "content": "fresh tail"},
    ]
    monkeypatch.setattr(engine, "_should_force_overflow_recovery", lambda **_kwargs: False)
    monkeypatch.setattr(
        engine,
        "_summarize_leaf_chunk_with_rescue",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("critical host pressure reached summarizer")
        ),
    )

    engine.last_prompt_tokens = 100
    assert engine.should_compress(0) is False
    assert engine.should_compress(99) is False
    assert engine.should_compress(100) is True
    with pytest.raises(AssertionError, match="critical host pressure reached summarizer"):
        engine.compress(deepcopy(messages), current_tokens=100)


def test_ignored_backlog_and_debt_are_deferred_without_model_work(tmp_path, monkeypatch):
    engine = _engine(
        tmp_path,
        deferred_maintenance_enabled=True,
        fresh_tail_count=1,
        leaf_chunk_tokens=1,
    )
    engine.threshold_tokens = 100
    messages = [
        {"role": "user", "content": "ignored old backlog"},
        {"role": "user", "content": "fresh tail"},
    ]
    engine._lifecycle.record_debt(
        engine._conversation_id,
        kind="raw_backlog",
        size_estimate=10,
    )
    model_work_calls = []

    def fail_if_called(stage):
        def fail(*_args, **_kwargs):
            model_work_calls.append(stage)
            raise AssertionError(f"preflight must not run {stage}")

        return fail

    monkeypatch.setattr(engine, "_should_force_overflow_recovery", lambda **_kwargs: False)
    monkeypatch.setattr(engine, "_has_ignored_backlog_outside_fresh_tail", lambda _messages: True)
    monkeypatch.setattr(engine, "_run_pre_compaction_extraction", fail_if_called("extraction"))
    monkeypatch.setattr(engine, "_summarize_leaf_chunk_with_rescue", fail_if_called("summarization"))
    monkeypatch.setattr(engine, "_maybe_condense", fail_if_called("condensation"))
    monkeypatch.setattr(engine._dag, "add_node", fail_if_called("DAG publication"))

    assert engine.should_compress_preflight(messages) is False
    assert engine._has_raw_backlog_debt() is True
    assert model_work_calls == []


def test_legacy_direct_compress_without_host_tokens_remains_admitted(tmp_path, monkeypatch):
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
            AssertionError("legacy direct compress reached summarizer")
        ),
    )

    with pytest.raises(AssertionError, match="legacy direct compress reached summarizer"):
        engine.compress(messages, current_tokens=None)


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
