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
