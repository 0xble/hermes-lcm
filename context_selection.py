"""Deterministic, request-only context selection for LCM."""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List

from .tokens import count_message_tokens, count_messages_tokens


logger = logging.getLogger(__name__)


class ContextSelectionMixin:
    """Project canonical conversation state into one provider request."""

    @staticmethod
    def _trusted_incoming_message_index(
        conversation_messages: List[Dict[str, Any]],
        incoming_message: Dict[str, Any],
    ) -> int | None:
        """Resolve the one canonical incoming occurrence without guessing."""
        identity_matches = [
            index
            for index, message in enumerate(conversation_messages)
            if message is incoming_message
        ]
        if len(identity_matches) == 1:
            return identity_matches[0]
        if identity_matches:
            return None
        equality_matches = [
            index
            for index, message in enumerate(conversation_messages)
            if message == incoming_message
        ]
        return equality_matches[0] if len(equality_matches) == 1 else None

    @staticmethod
    def _older_tail_turn_groups(
        messages: List[Dict[str, Any]],
    ) -> list[list[dict[str, Any]]]:
        """Partition older tail at user boundaries for atomic budgeting."""
        if not messages:
            return []
        starts = [
            index
            for index, message in enumerate(messages)
            if message.get("role") == "user"
        ]
        if not starts:
            return [messages]
        boundaries = ([0] if starts[0] else []) + starts + [len(messages)]
        return [
            messages[start:end]
            for start, end in zip(boundaries, boundaries[1:])
            if start < end
        ]

    @staticmethod
    def _has_complete_tool_groups(messages: List[Dict[str, Any]]) -> bool:
        """Reject an older budget group that would expose orphan tool traffic."""
        pending: set[str] = set()
        for message in messages:
            role = str(message.get("role") or "")
            if role == "assistant":
                if pending:
                    return False
                pending = {
                    str(tool_call.get("id") or "")
                    for tool_call in (message.get("tool_calls") or [])
                    if isinstance(tool_call, dict) and str(tool_call.get("id") or "")
                }
            elif role == "tool":
                tool_call_id = str(message.get("tool_call_id") or "")
                if not tool_call_id or tool_call_id not in pending:
                    return False
                pending.remove(tool_call_id)
            elif pending:
                return False
        return not pending

    @staticmethod
    def _alignment_replay_metadata_matches(
        canonical: Dict[str, Any],
        request: Dict[str, Any],
    ) -> bool:
        """Require exact replay-significant fields while allowing API annotations."""
        return all(
            canonical.get(field) == request.get(field)
            for field in ("role", "tool_call_id", "tool_name", "tool_calls")
        )

    @staticmethod
    def _canonical_content_bases(message: Dict[str, Any]) -> list[Any]:
        """Return canonical and persist-what-was-sent content variants."""
        bases = [message.get("content")]
        api_content = message.get("api_content")
        if (
            message.get("role") in {"user", "assistant"}
            and isinstance(api_content, (str, list))
            and bool(api_content)
            and api_content != bases[0]
        ):
            bases.append(api_content)
        return bases

    @staticmethod
    def _is_append_only_provider_content(request_content: Any, base: Any) -> bool:
        """Recognize the two current-turn append shapes used by provider setup."""
        if isinstance(base, str) and isinstance(request_content, str):
            return request_content.startswith(base + "\n\n")
        if not isinstance(base, list) or not isinstance(request_content, list):
            return False
        if len(request_content) <= len(base) or request_content[: len(base)] != base:
            return False
        appended = request_content[len(base) :]
        if not all(
            isinstance(part, dict)
            and part.get("type", "text") == "text"
            and isinstance(part.get("text"), str)
            and bool(part.get("text"))
            for part in appended
        ):
            return False
        return str(appended[0]["text"]).startswith("\n\n")

    def _canonical_request_occurrence_matches(
        self,
        canonical: Dict[str, Any],
        request: Dict[str, Any],
        *,
        trusted_incoming: bool,
    ) -> bool:
        if not self._alignment_replay_metadata_matches(canonical, request):
            return False
        request_content = request.get("content")
        bases = self._canonical_content_bases(canonical)
        if any(request_content == base for base in bases):
            return True
        return trusted_incoming and any(
            self._is_append_only_provider_content(request_content, base)
            for base in bases
        )

    @staticmethod
    def _clean_canonical_message(message: Dict[str, Any]) -> Dict[str, Any]:
        """Keep canonical transcript content, never its API replay sidecar."""
        clean = copy.deepcopy(message)
        clean.pop("api_content", None)
        return clean

    def _align_request_to_canonical_conversation(
        self,
        request_messages: List[Dict[str, Any]],
        conversation_messages: List[Dict[str, Any]],
        *,
        trusted_incoming_index: int | None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]] | None:
        if not conversation_messages:
            return None
        width = len(conversation_messages)
        starts = [
            start
            for start in range(len(request_messages) - width + 1)
            if all(
                self._canonical_request_occurrence_matches(
                    canonical,
                    request_messages[start + index],
                    trusted_incoming=index == trusted_incoming_index,
                )
                for index, canonical in enumerate(conversation_messages)
            )
        ]
        if len(starts) != 1:
            return None
        start = starts[0]
        # The generic host hook does not identify agent.prefill_messages. Accept
        # only the host insertion position (after at most one system envelope)
        # and conservative user/assistant shapes; these cannot authenticate the
        # prefix as configured prefill, so any other/ambiguous shape fails open.
        if start + width != len(request_messages):
            return None
        prefix = request_messages[:start]
        prefill = prefix[1:] if prefix and prefix[0].get("role") == "system" else prefix
        if any(message.get("role") not in {"user", "assistant"} for message in prefill):
            return None
        return (
            copy.deepcopy(prefix),
            copy.deepcopy(request_messages[start : start + width]),
            copy.deepcopy(request_messages[start + width :]),
        )

    def _projection_store_ids(
        self,
        replay_messages: List[Dict[str, Any]],
    ) -> list[int | None]:
        """Map replay to the rightmost monotonic durable subsequence, read-only."""
        stored_rows = self._store.get_session_messages(self._session_id)
        stored_identities = [
            self._message_replay_identity(row, stored_row=True)
            for row in stored_rows
        ]
        result: list[int | None] = [None] * len(replay_messages)
        cursor = len(stored_rows) - 1
        for replay_index in range(len(replay_messages) - 1, -1, -1):
            identity = self._message_replay_identity(replay_messages[replay_index])
            probe = cursor
            while probe >= 0 and stored_identities[probe] != identity:
                probe -= 1
            if probe < 0:
                continue
            result[replay_index] = int(stored_rows[probe]["store_id"])
            cursor = probe - 1
        return result

    def _ingest_filtered_durable_view(
        self,
        durable_messages: List[Dict[str, Any]],
        canonical_messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Ingest a filtered cursor view without trusting another view's offset.

        ``_ingest_cursor`` is an offset into the exact active list last observed.
        Selection and post-turn observation can remove non-durable replay
        scaffolds before ingest, so an offset learned from the unfiltered
        canonical list cannot be reused safely. Reconcile the filtered view now,
        then retain the signal so a coexisting legacy caller can realign to the
        unfiltered view. An unchanged view does not schedule reconciliation.
        """
        cursor_view_changed = len(durable_messages) != len(canonical_messages)
        if cursor_view_changed:
            self._ingest_cursor_needs_reconcile = True
        try:
            return self._ingest_messages(durable_messages)
        finally:
            if cursor_view_changed:
                # Also preserve the signal when ingest fails after reconciliation;
                # the next caller must not trust that partially advanced cursor.
                self._ingest_cursor_needs_reconcile = True

    def select_context(
        self,
        request_messages: List[Dict[str, Any]],
        *,
        conversation_messages: List[Dict[str, Any]] | None = None,
        incoming_message: Dict[str, Any] | None = None,
        budget_tokens: int = 0,
    ) -> List[Dict[str, Any]] | None:
        """Fail-open host adapter for deterministic request projection."""
        try:
            return self._select_context_projection(
                request_messages,
                conversation_messages=conversation_messages,
                incoming_message=incoming_message,
                budget_tokens=budget_tokens,
            )
        except Exception:
            logger.warning(
                "LCM context selection failed; using the unmodified request",
                exc_info=True,
            )
            return None

    def _select_context_projection(
        self,
        request_messages: List[Dict[str, Any]],
        *,
        conversation_messages: List[Dict[str, Any]] | None = None,
        incoming_message: Dict[str, Any] | None = None,
        budget_tokens: int = 0,
    ) -> List[Dict[str, Any]] | None:
        """Return a deterministic request projection or fail open with ``None``."""
        if self._session_ignored or self._session_stateless:
            return None
        if not isinstance(conversation_messages, list):
            return None
        trusted_incoming_index = None
        if incoming_message is not None:
            if not isinstance(incoming_message, dict):
                return None
            trusted_incoming_index = self._trusted_incoming_message_index(
                conversation_messages,
                incoming_message,
            )
            if trusted_incoming_index is None:
                return None
        alignment = self._align_request_to_canonical_conversation(
            request_messages,
            conversation_messages,
            trusted_incoming_index=trusted_incoming_index,
        )
        if alignment is None:
            return None
        prefix, aligned_request, suffix = alignment
        aligned_pairs = [
            (index, canonical, request_copy)
            for index, (canonical, request_copy) in enumerate(
                zip(conversation_messages, aligned_request)
            )
            if (
                index == trusted_incoming_index
                or not self._is_replayed_context_scaffold_message(canonical)
            )
        ]
        filtered_indices = [index for index, _canonical, _request_copy in aligned_pairs]
        filtered_conversation = [canonical for _index, canonical, _request_copy in aligned_pairs]
        aligned_request = [request_copy for _index, _canonical, request_copy in aligned_pairs]
        leading_system_count = 0
        for message in filtered_conversation:
            if message.get("role") != "system":
                break
            leading_system_count += 1
        if any(
            message.get("role") == "system"
            for message in filtered_conversation[leading_system_count:]
        ):
            return None
        if leading_system_count:
            prefix.extend(aligned_request[:leading_system_count])
        durable_indices = filtered_indices[leading_system_count:]
        durable_messages = [
            self._clean_canonical_message(message)
            for message in filtered_conversation[leading_system_count:]
        ]
        durable_request = aligned_request[leading_system_count:]
        try:
            replay_messages = self._ingest_filtered_durable_view(
                durable_messages,
                conversation_messages,
            )
        except Exception as exc:
            self._record_ingest_failure("context selection ingest", exc)
            raise
        self._record_ingest_success()
        nodes = self._dag.get_session_nodes(self._session_id)
        frontier = max(0, int(self._last_compacted_store_id or 0))
        projection_store_ids = (
            self._projection_store_ids(replay_messages)
            if frontier > 0 and nodes
            else [None] * len(replay_messages)
        )

        tail_records: list[tuple[int, dict[str, Any]]] = []
        for canonical_index, canonical, request_copy, replay, store_id in zip(
            durable_indices,
            durable_messages,
            durable_request,
            replay_messages,
            projection_store_ids,
        ):
            is_required_active = (
                trusted_incoming_index is not None
                and canonical_index >= trusted_incoming_index
            )
            if (
                canonical_index != trusted_incoming_index
                and self._is_replayed_context_scaffold_message(canonical)
            ):
                continue
            if not is_required_active and store_id is not None and store_id <= frontier:
                continue
            if is_required_active:
                tail_records.append((canonical_index, request_copy))
            elif self._message_replay_identity(canonical) == self._message_replay_identity(replay):
                tail_records.append((canonical_index, request_copy))
            else:
                tail_records.append((canonical_index, copy.deepcopy(replay)))

        summary_parts: list[str] = []
        for depth in sorted({node.depth for node in nodes}, reverse=True):
            depth_nodes = self._dag.get_uncondensed_at_depth(
                self._session_id,
                depth,
            )
            for node in sorted(
                depth_nodes,
                key=lambda candidate: (candidate.created_at, candidate.node_id),
            ):
                depth_label = {
                    0: "Recent",
                    1: "Session Arc",
                    2: "Durable",
                }.get(depth, f"Depth-{depth}")
                summary_parts.append(
                    f"[{depth_label} Summary (d{depth}, node {node.node_id})]\n"
                    f"{node.summary}\n"
                    f"[Expand for details: {node.expand_hint}]"
                )
        projected: list[dict[str, Any]] = list(prefix)
        summary_message = None
        if summary_parts:
            summary_message = {
                "role": "user",
                "content": "\n\n---\n\n".join(summary_parts),
            }

        threshold_cap = int(getattr(self, "threshold_tokens", 0) or 0)
        if threshold_cap <= 0:
            return None
        budget_cap = int(budget_tokens or 0)
        cap = min(threshold_cap, budget_cap) if budget_cap > 0 else threshold_cap
        if cap > 0:
            immutable_tokens = count_messages_tokens([*prefix, *suffix])
            if trusted_incoming_index is not None:
                required_active = [
                    message
                    for index, message in tail_records
                    if index >= trusted_incoming_index
                ]
                older_tail = [
                    message
                    for index, message in tail_records
                    if index < trusted_incoming_index
                ]
            else:
                # Compatibility hosts do not identify the current canonical
                # occurrence. Reserve the complete newest eligible turn group.
                eligible_tail = [message for _index, message in tail_records]
                turn_groups = self._older_tail_turn_groups(eligible_tail)
                if not turn_groups:
                    return None
                required_active = turn_groups[-1]
                if not self._has_complete_tool_groups(required_active):
                    return None
                older_tail = [
                    message
                    for group in turn_groups[:-1]
                    for message in group
                ]
            required_tokens = count_messages_tokens(required_active)
            if immutable_tokens + required_tokens > cap:
                return None
            summary_tokens = (
                count_message_tokens(summary_message)
                if summary_message is not None
                else 0
            )
            used_tokens = immutable_tokens + required_tokens
            if used_tokens + summary_tokens > cap:
                summary_message = None
                summary_tokens = 0
            used_tokens += summary_tokens
            selected_groups_reversed: list[list[dict[str, Any]]] = []
            for group in reversed(self._older_tail_turn_groups(older_tail)):
                if not self._has_complete_tool_groups(group):
                    break
                group_tokens = count_messages_tokens(group)
                if used_tokens + group_tokens > cap:
                    break
                selected_groups_reversed.append(group)
                used_tokens += group_tokens
            tail = [
                message
                for group in reversed(selected_groups_reversed)
                for message in group
            ]
            tail.extend(required_active)
        else:
            tail = [message for _index, message in tail_records]

        if summary_message is not None:
            projected.append(summary_message)
        projected.extend(tail)
        projected.extend(suffix)
        return projected or None
