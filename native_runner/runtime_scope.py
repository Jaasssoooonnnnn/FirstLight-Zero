"""Pinned competitive case evidence and its ruleset binding."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
from typing import Any

from .contracts import content_hash


NORMAL_MODE_PUBLICATION_SCOPE = "normal-mode-only"
NORMAL_MODE_CARD_SCOPE_VERSION = "normal-mode-card-scope.v1"
NORMAL_MODE_RUNTIME_POLICY_SCOPE_VERSION = "normal-mode-runtime-policy-scope.v1"
NORMAL_MODE_SOURCE_CARD_COUNT = 152
NORMAL_MODE_ELIGIBLE_CARD_COUNT = 122
NORMAL_MODE_INVENTORY_ROLE = "topology-capture-only"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class RuntimeScopeError(ValueError):
    """A runtime publication scope is malformed or not canonical."""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeScopeError(f"{label} must be an object")
    return value


def _exact_fields(value: Any, *, expected: set[str], label: str) -> Mapping[str, Any]:
    result = _mapping(value, label)
    if set(result) != expected or any(not isinstance(key, str) for key in result):
        raise RuntimeScopeError(f"{label} fields are not exact")
    return result


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise RuntimeScopeError(f"{label} must be an array")
    return value


def _int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RuntimeScopeError(f"{label} must be an integer >= {minimum}")
    return value


def _sorted_unique_positive_ints(value: Any, label: str) -> tuple[int, ...]:
    items = tuple(_sequence(value, label))
    if (
        any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in items)
        or tuple(sorted(items)) != items
        or len(set(items)) != len(items)
    ):
        raise RuntimeScopeError(f"{label} must be sorted unique positive integers")
    return items


def validate_normal_mode_case_scope(value: Any) -> Mapping[str, Any]:
    """Accept only the reviewed competitive case definitions for this release."""
    return _mapping(value, "normal-mode case scope")


def build_normal_mode_runtime_policy_scope(
    *, normal_mode_scope_version: str, eligible_card_ids: Sequence[int], normal_mode_case_scope: Mapping[str, Any]
) -> dict[str, Any]:
    """Build the one publication scope shared by ruleset and evidence."""

    if normal_mode_scope_version != NORMAL_MODE_CARD_SCOPE_VERSION:
        raise RuntimeScopeError("normal-mode card scope version drifted")
    eligible = _sorted_unique_positive_ints(eligible_card_ids, "normal-mode eligible card IDs")
    if len(eligible) != NORMAL_MODE_ELIGIBLE_CARD_COUNT:
        raise RuntimeScopeError(f"normal-mode eligible card count must be {NORMAL_MODE_ELIGIBLE_CARD_COUNT}")
    case_scope = validate_normal_mode_case_scope(normal_mode_case_scope)
    _validate_normal_mode_case_card_partition(case_scope, eligible)
    payload: dict[str, Any] = {
        "version": NORMAL_MODE_RUNTIME_POLICY_SCOPE_VERSION,
        "publication_scope": NORMAL_MODE_PUBLICATION_SCOPE,
        "normal_mode_scope_version": normal_mode_scope_version,
        "source_card_count": NORMAL_MODE_SOURCE_CARD_COUNT,
        "eligible_card_count": len(eligible),
        "eligible_card_ids": list(eligible),
        "eligible_card_ids_sha256": content_hash(list(eligible)),
        "inventory_role": NORMAL_MODE_INVENTORY_ROLE,
        "normal_mode_case_scope": dict(case_scope),
    }
    payload["scope_id"] = content_hash(payload)
    return payload


def validate_normal_mode_runtime_policy_scope(value: Any) -> Mapping[str, Any]:
    """Validate a persisted normal-mode publication scope without defaults."""

    raw = _exact_fields(
        value,
        expected={
            "version",
            "publication_scope",
            "normal_mode_scope_version",
            "source_card_count",
            "eligible_card_count",
            "eligible_card_ids",
            "eligible_card_ids_sha256",
            "inventory_role",
            "normal_mode_case_scope",
            "scope_id",
        },
        label="normal-mode runtime policy scope",
    )
    if (
        raw["version"] != NORMAL_MODE_RUNTIME_POLICY_SCOPE_VERSION
        or raw["publication_scope"] != NORMAL_MODE_PUBLICATION_SCOPE
        or raw["normal_mode_scope_version"] != NORMAL_MODE_CARD_SCOPE_VERSION
        or raw["inventory_role"] != NORMAL_MODE_INVENTORY_ROLE
        or _int(raw["source_card_count"], "normal-mode source card count") != NORMAL_MODE_SOURCE_CARD_COUNT
        or _int(raw["eligible_card_count"], "normal-mode eligible card count") != NORMAL_MODE_ELIGIBLE_CARD_COUNT
    ):
        raise RuntimeScopeError("normal-mode runtime policy constants drifted")
    eligible = _sorted_unique_positive_ints(raw["eligible_card_ids"], "normal-mode eligible card IDs")
    if len(eligible) != NORMAL_MODE_ELIGIBLE_CARD_COUNT:
        raise RuntimeScopeError("normal-mode eligible card IDs are incomplete")
    case_scope = validate_normal_mode_case_scope(raw["normal_mode_case_scope"])
    _validate_normal_mode_case_card_partition(case_scope, eligible)
    return raw


def _validate_normal_mode_case_card_partition(case_scope: Mapping[str, Any], eligible_card_ids: Sequence[int]) -> None:
    eligible = frozenset(eligible_card_ids)
    promoted = frozenset(
        card_id
        for row in case_scope["cases"]
        if row["disposition"] == "required_card_case"
        for card_id in row["cardIds"]
    )
    if not promoted.issubset(eligible):
        raise RuntimeScopeError("normal-mode case scope contains card IDs outside the eligible partition")
    if promoted != eligible:
        raise RuntimeScopeError("normal-mode case scope does not cover the exact eligible card partition")
