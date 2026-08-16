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
