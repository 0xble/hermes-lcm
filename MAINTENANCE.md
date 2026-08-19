# Maintained Hermes-LCM fork

This repository tracks `stephenschoettler/hermes-lcm` while carrying Brian-owned
policy changes. The upstream repository remains authoritative for all unmodified
LCM code. The fork's `main` branch is the maintained candidate; installing it
into Hermes is a separate promotion step.

## Maintenance contract

- **Fork:** `0xble/hermes-lcm`
- **Upstream:** `stephenschoettler/hermes-lcm`
- **Maintained branch:** `main`
- **Forbidden push target:** `upstream`
- **Runtime install surface:** Hermes user plugin `hermes-lcm`
- **Runtime data:** `$HERMES_HOME/lcm.db` is not repository content and must be
  backed up before upgrades that may affect its schema or lifecycle.
- **Release/install rule:** install from an exact pushed fork commit and verify
  the active plugin source, commit, enabled state, context-engine binding, and
  policy behavior after installation.

## Patch register

### LCM-001 — Remove cross-session LCM recall guidance

- **Status:** Active
- **Stable commit subject:** `fix(policy): remove cross-session LCM recall guidance`
- **Summary:** Removes the `lcm_recall` cross-conversation routing instruction
  from the injected Hermes-LCM recall policy. LCM compaction and bounded
  current-session recovery remain documented. The `lcm_recall` implementation
  and tool schema are not removed by this patch; this is an intentional prompt
  routing restriction.
- **Surfaces:**
  `skills/hermes-lcm/references/recall-policy.md`,
  `tests/test_recall_guidance.py`
- **Upstream tracking:** Fork-only policy divergence; no upstream issue or PR
  created. Check upstream source and release notes before each synchronization.
- **Upstream PR:** `None after checked 2026-08-16`.

- **Regression:**
  `python -m pytest tests/test_recall_guidance.py -q`; additionally verify the
  loaded policy contains no `lcm_recall` or cross-conversation routing sentence.
- **Rollback:** Restore the removed routing bullet and its corresponding test
  assertions, then run the focused regression. Do not remove or alter the
  underlying `lcm_recall` tool implementation as part of rollback.
- **Retirement:** Retire only after released upstream provides an equivalent
  policy boundary that prevents ordinary model routing through cross-session
  LCM recall, or after Brian explicitly chooses to restore that route.

### LCM-002 — Preserve safe compaction deferrals across the host boundary

- **Status:** Active; upstream contribution pending.
- **Stable commit subject:** `fix(compaction): expose safe no-progress deferrals`
- **Summary:** Reports threshold-triggered no-progress decisions as `deferred` rather than the ambiguous `noop` status, while retaining `noop` for ordinary below-threshold cleanup/pass-through cases. The public no-op predicate continues to recognize both statuses.
- **Surfaces:**
  `compaction.py`, `engine.py`, `tests/test_lcm_engine.py`
- **Upstream tracking:** Related to hermes-lcm#168 and hermes-lcm#188; verify current upstream state before publication.
- **Upstream PR:** `https://github.com/stephenschoettler/hermes-lcm/pull/531` (open; head `0xble:fix/safe-compaction-deferral`, commit `67e97dfb9c50dae4ff91291ea350cc746d6a642b`).
- **Regression:**
  `pytest tests/test_lcm_engine.py -q`; specifically verifies threshold pressure with only protected/insufficient backlog returns unchanged messages, `deferred` status, and no false compaction request.
- **Rollback:** Revert the focused commit and rerun `pytest tests/test_lcm_engine.py -q`; Hermes host compatibility remains backward-compatible because `last_compression_was_noop` accepts both `noop` and `deferred`.
- **Retirement:** Retire when released upstream exposes an equivalent safe-deferral status and the Hermes host no longer needs the fork-specific behavior.

### LCM-003 — Gate opportunistic maintenance on token pressure

- **Status:** Active; promoted to the Personal runtime from exact commit `bd17a915ae7d2777257c42d4ddec324af8616d20` on 2026-08-18 with `LCM_MAINTENANCE_MIN_PRESSURE_RATIO=1.0`.
- **Stable commit subject:** `fix(compaction): gate unnecessary replay maintenance`
- **Summary:** Adds an opt-in pressure floor for the divergent-replay leaf and ignored-backlog maintenance arms. The source default remains `0.0`; the Personal runtime sets `LCM_MAINTENANCE_MIN_PRESSURE_RATIO=1.0`. Overflow recovery and deterministic cleanup remain outside this gate.
- **Surfaces:** `compaction.py`, `config.py`, `tools.py`, `README.md`, `docs/operator-guide.md`, `tests/test_maintenance_pressure_floor.py`.
- **Upstream tracking:** Direct port of hermes-lcm PR #516, commit `034d52ac8a7ebbbb17cd0ab368cec101c0255bae`; open and unshipped when checked 2026-08-18.
- **Upstream PR:** `https://github.com/stephenschoettler/hermes-lcm/pull/516`.
- **Regression:** `python -m pytest tests/test_maintenance_pressure_floor.py -q` verifies the disabled default, environment loading, boundary semantics, and both divergent opportunistic arms.
- **Rollback:** Clear the runtime pressure-ratio setting, revert this focused commit, and rerun the focused regression. Do not change the normal context threshold.
- **Retirement:** Retire when a released upstream version provides an equivalent pressure floor with default-preserving behavior and both opportunistic arms covered.

### LCM-004 — Require optional replay cleanup to reduce active context

- **Status:** Active; promoted to the Personal runtime from exact commit `bd17a915ae7d2777257c42d4ddec324af8616d20` on 2026-08-18.
- **Stable commit subject:** `fix(compaction): gate unnecessary replay maintenance`
- **Summary:** Classifies only externalization-only replay differences as optional. Automatic optional cleanup is admitted and returned only when it strictly reduces active-context tokens; otherwise LCM reports a benign `deferred` result before Hermes must reject it. Manual compression, overflow recovery, sensitive redaction, quarantine, ignored-message cleanup, mixed-length reconciliation, and Hermes's final anti-growth guard remain unchanged.
- **Surfaces:** `compaction.py`, `tests/test_optional_cleanup_admission.py`, and shrinking-fixture updates in `tests/test_lcm_engine.py` and `tests/test_active_tool_stubbing.py`.
- **Upstream tracking:** No upstream implementation or tracker enforces this host/plugin monotonic cleanup contract after repository-wide check on 2026-08-18. Related symptom reports: hermes-lcm issues #513 and #532.
- **Upstream PR:** `None after checked 2026-08-18`.
- **Regression:** `python -m pytest tests/test_optional_cleanup_admission.py -q` verifies the observed 55,812-to-55,961-token refusal, equal-size deferral, shrinking cleanup admission, mandatory and mixed sensitive/externalization cleanup, manual compression, overflow recovery, durability, and no summarizer call even with eligible backlog. Existing externalization/filtering fixtures use payloads that are actually smaller after cleanup.
- **Rollback:** Revert this focused commit and rerun the focused regression. Keep Hermes's host anti-growth guard intact.
- **Retirement:** Retire when released upstream distinguishes optional from mandatory replay cleanup and proves automatic optional candidates strictly shrink before host adoption.

### LCM-005 — Enforce host pressure at model-spend admission

- **Status:** Superseded in fork source by LCM-006. Managed-process load state was not conclusively established before supersession, so promotion status is intentionally left unresolved.
- **Stable commit subject:** `fix(compaction): enforce host maintenance pressure`
- **Summary:** Attempted to carry automatic-preflight authority into `compress()` through a one-shot handoff keyed by `id(messages)`. Hermes deep-copies the transcript between preflight and compression, so the identity check discarded the host-pressure intent and could admit model-backed work from LCM's larger internal estimate.
- **Surfaces:** `compaction.py`, `engine.py`, `reset_state.py`, `tests/test_maintenance_pressure_floor.py`, and `tests/test_optional_cleanup_admission.py`.
- **Upstream tracking:** Extended the policy direction of open hermes-lcm issue #496 and PR #516, but the identity-bound handoff was not durable across the real Hermes host boundary.
- **Regression:** Superseded by LCM-006's production-shaped deep-copy regression.
- **Rollback:** Not applicable independently; use the LCM-006 rollback guidance.
- **Retirement:** Retained as a superseded patch record so the identity-bound design is not reintroduced.

### LCM-006 — Enforce host pressure directly at compression admission

- **Status:** Active in fork source; not promoted to a managed runtime.
- **Stable commit subject:** `fix(compaction): admit model work from host pressure directly`
- **Summary:** Treats supplied `current_tokens` as authoritative inside `compress()`. Below `maintenance_min_pressure_ratio * threshold_tokens`, automatic non-overflow calls return the original transcript before extraction, summarization, condensation, or DAG publication. Only deterministic replay cleanup requested by ingest may be sanitized and adopted, with optional cleanup still required to strictly shrink. Exact-floor admission, `force=True`, overflow recovery, and legacy direct `compress(current_tokens=None)` remain admitted. No cross-call object identity or preflight-intent state is used.
- **Surfaces:** `compaction.py`, `engine.py`, `reset_state.py`, `tests/test_maintenance_pressure_floor.py`, `tests/test_optional_cleanup_admission.py`, and obsolete identity-assertion removal in `tests/test_active_tool_stubbing.py`.
- **Upstream tracking:** Corrects the fork-only LCM-005 implementation while retaining the policy direction of hermes-lcm issue #496 and PR #516.
- **Regression:** `/Users/brianle/.hermes/hermes-agent/venv/bin/python -m pytest tests/test_maintenance_pressure_floor.py tests/test_optional_cleanup_admission.py tests/test_active_tool_stubbing.py -q` covers the Hermes preflight/deep-copy/compress boundary, model-stage exclusion, deterministic cleanup, strict shrink, boundary cooldown, exact-floor and force admission, and legacy direct compression.
- **Rollback:** Revert this focused patch as a unit. Clearing `LCM_MAINTENANCE_MIN_PRESSURE_RATIO` restores source-default admission behavior without changing the normal context threshold.
- **Retirement:** Retire when a released upstream version enforces supplied host pressure directly at model-spend admission and proves the behavior across a copied transcript boundary.

### LCM-007 — Project deterministic LCM context per provider request

- **Status:** Active in fork source; not installed or promoted to a managed runtime.
- **Stable commit subject:** `feat(context): project deterministic LCM requests`
- **Summary:** Implements Hermes `select_context()` as a thin fail-open adapter over deterministic request projection. Canonical `conversation_messages` alone drive idempotent durable reconciliation after recognized replay scaffold is filtered from aligned canonical/request pairs before ingest and cursor/store-id mapping. Alignment requires exact replay-significant role/tool metadata per occurrence, accepts exact canonical content or nonempty user/assistant `api_content`, and permits only the trusted incoming occurrence to carry Hermes's append-only string or structured trailing-text provider context (including MoA). Historical append matches, metadata drift, suffixes, and ambiguity fail open; ingest strips the replay sidecar and projection preserves request-side current-turn content/metadata. The trusted incoming canonical occurrence is resolved by object identity (or one unambiguous equality match), exempted from scaffold filtering, and retained with its same-turn assistant/tool suffix as an atomic required slice. Projection reads the existing DAG frontier and caps output at the smaller positive host budget and `threshold_tokens`; a nonpositive threshold or over-budget required slice fails open. Immutable material and the required active turn are budgeted before optional DAG summaries and complete older turn/tool groups. No model-backed summarization, extraction, condensation, proactive recall, embedding/provider resolution, or projection-only externalization runs. Canonical payload protection remains enabled and idempotent.
- **Surfaces:** `context_selection.py`, `engine.py`, `tests/test_context_selection.py`, `README.md`, `docs/operator-guide.md`, and this register.
- **Host dependency:** Requires the Hermes `ContextEngine.select_context()` contract and `_apply_context_engine_selection()` request seam. Older hosts retain their existing compression/preflight behavior.
- **Regression:** `/Users/brianle/.hermes/hermes-agent/venv/bin/python -m pytest tests/test_context_selection.py -q` covers deterministic retries, exact historical/current `api_content`, trusted current-turn string/list MoA appends, historical-append and tool-metadata rejection, host-positioned prefill preservation, canonical-only ingest, current-turn scaffold exemption with older duplicate filtering, ambiguous incoming-copy and arbitrary-suffix fail-open, payload protection, no model/projection side effects, threshold-capped active-turn-first atomic budgeting (including tool loops and immutable prefix), nonpositive-threshold fail-open, one-time DAG summary projection, cache metadata stability, ignored/stateless sessions, and fail-open errors.
- **Rollback:** Revert the focused Slice 2 source changes. The host treats an absent/base `select_context()` implementation as a no-op and continues with its unchanged request. Do not delete canonical transcript rows or LCM durable data.
- **Retirement:** Retire when a released upstream LCM version provides equivalent canonical-only deterministic request projection through the Hermes selection hook.

### LCM-008 — Observe finalized turns through the generic context-engine seam

- **Status:** Active in fork source; not installed or promoted to a managed runtime.
- **Stable commit subject:** `feat(context): observe finalized turns idempotently`
- **Summary:** Implements best-effort `ContextEngine.on_turn_complete()` observation as an idempotent, deterministic, model-free ingest of the host's finalized canonical transcript. It filters recognized LCM replay scaffold before persistence while exempting the most recent canonical user occurrence (the standard finalized current-turn boundary), preserves canonical message/tool fields through existing ingest protection, reconciles filtered and legacy unfiltered cursor views, accepts but ignores usage/meta, and records swallowed ingest failures exactly once. Duplicate callbacks, missed-callback lazy selection fallback, and later retries do not duplicate rows or telemetry. No summarizer, extractor, condensation, retrieval/provider/model work, or DAG publication is admitted. Modern hosts with the base hook skip duplicate LCM `post_llm_call` registration; older hosts retain that hook and its clone/session rebinding as a compatibility fallback.
- **Surfaces:** `context_selection.py`, `engine.py`, `engine_registry.py`, `__init__.py`, `tests/test_turn_observation.py`, `tests/test_packaging_install.py`, `README.md`, `docs/operator-guide.md`, and this register.
- **Host dependency and coverage:** Uses Hermes `ContextEngine.on_turn_complete()` when the host base exposes it. Coverage is intentionally best-effort because abnormal host early exits can bypass standard turn finalization; `select_context()` remains lazy reconciliation fallback when observation is missed.
- **Regression:** `/Users/brianle/.hermes/hermes-agent/venv/bin/python -m pytest tests/test_turn_observation.py tests/test_context_selection.py tests/test_packaging_install.py -q` covers exact-once canonical ingest, duplicate callbacks, older scaffold filtering with exact-shaped current-user preservation through assistant/tool completion, filtered/unfiltered cursor coexistence, input immutability, model/provider exclusion, usage/telemetry isolation, single-count fail-open errors, missed-callback fallback, modern registration suppression, and simulated legacy fallback/rebinding.
- **Rollback:** Revert the focused Slice 3 source changes. Modern hosts then fall back to selection/compression ingest behavior; older hosts continue using the retained legacy hook. Do not delete durable LCM rows or alter Hermes runtime/configuration as part of source rollback.
- **Retirement:** Retire when a released upstream LCM version provides equivalent best-effort, canonical-only post-turn observation and safely removes duplicate modern legacy-hook registration while preserving old-host compatibility.

### LCM-009 — Trust Hermes host pressure for model-maintenance admission

- **Status:** Active in fork source; not installed or promoted to a managed runtime.
- **Stable commit subject:** `fix(compaction): retire model-backed preflight admission`
- **Summary:** Narrows `should_compress_preflight(messages)` to deterministic ingest cleanup and forced overflow recovery. Rough active/replay estimates, eligible leaves, divergent non-cleanup replay, ignored backlog, and deferred raw-backlog debt can no longer authorize extraction, summarization, condensation, or DAG publication, even when `maintenance_min_pressure_ratio` is zero. Normal threshold work enters through `should_compress(prompt_tokens)`, trusting Hermes's existing host-owned request-pressure estimate rather than claiming provider-measured telemetry. An explicitly configured `critical_budget_pressure_ratio` is also evaluated from that host estimate and bypasses the direct-compress maintenance-floor defense at that critical boundary. Optional cleanup still requires strict shrink; mandatory sanitation, overflow, cooldown safety exceptions, `force=True`, and legacy direct compression remain intact across Hermes's deep-copy boundary.
- **Surfaces:** `compaction.py`, `config.py`, `tests/test_maintenance_pressure_floor.py`, `tests/test_optional_cleanup_admission.py`, `tests/test_active_tool_stubbing.py`, `tests/test_lcm_engine.py`, `README.md`, `docs/operator-guide.md`, and this register.
- **Host dependency:** Relies on Hermes ordering that evaluates `should_compress(prompt_tokens)` with its existing `estimate_request_tokens_rough` request-pressure estimate before invoking preflight. LCM trusts that host-owned estimate and does not claim provider-measured token telemetry. Older hosts retain deterministic preflight ingest/protection and overflow recovery but receive no rough-estimate model admission.
- **Regression:** `/Users/brianle/.hermes/hermes-agent/venv/bin/python -m pytest tests/test_maintenance_pressure_floor.py tests/test_optional_cleanup_admission.py tests/test_active_tool_stubbing.py tests/test_lcm_engine.py -q` covers default-floor subthreshold leaves, over-threshold replay estimates, ignored/debt deferral, model-stage exclusion, cleanup strict shrink, overflow/cooldown behavior, host normal and critical boundaries, force/legacy behavior, and copied transcripts.
- **Rollback:** Revert only the LCM-009 source, test, and documentation changes. Do not alter plugin configuration or durable LCM data. LCM-007 request projection and LCM-008 finalized-turn observation remain independent.
- **Retirement:** Retire when released upstream LCM confines preflight to deterministic cleanup/overflow and admits all model-backed maintenance from Hermes host pressure, including an explicit critical-pressure boundary and direct-compress guard exemption.

## Synchronization procedure

1. Inspect `git status`, current `HEAD`, `origin`, and `upstream` before work.
2. Preserve unrelated dirty work in an exact named stash before synchronization.
3. Fetch `origin` and `upstream` separately with pruning.
4. Rebase the fork-only commits onto upstream `main`; abort on ambiguous
   conflicts rather than guessing.
5. Run the focused policy regression and the repository's applicable test gate.
6. Push only to the fork's `origin/main` with `--force-with-lease` when history
   was rewritten; never push to `upstream`.
7. Record the exact pushed commit used for installation.

## Promotion and rollback

Promotion must be verified independently from source publication:

1. Back up the active LCM database and record the installed plugin state.
2. Install the fork from an exact pushed commit using Hermes's supported plugin
   installer.
3. Verify fresh-process plugin discovery, enabled state, plugin path, commit,
   and `context.engine: lcm`.
4. Verify the active injected recall policy no longer contains the removed
   cross-session `lcm_recall` instruction.
5. If verification fails, reinstall the recorded prior plugin commit/source,
   restore configuration only if it changed, and verify the rollback.

The official upstream installation must not remain enabled alongside this fork:
that would create ambiguous plugin ownership and duplicate lifecycle hooks.
