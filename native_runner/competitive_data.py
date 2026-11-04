"""Load reviewed static facts for the supported competitive resource release.

No executable formulas or parser rules are stored in these files. The JSON is
the old compiler's output, including the historical catalog identities needed
to load existing checkpoints. Changed resources require an explicit rebuild.
"""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Mapping

from .contracts import ContractError, frozen_mapping
from .paths import PACKAGE_ROOT

DATA_ROOT = PACKAGE_ROOT / "data" / "competitive"


@lru_cache(maxsize=1)
def competitive_manifest() -> Mapping[str, Any]:
    return frozen_mapping(json.loads((DATA_ROOT / "manifest.json").read_bytes()))


@lru_cache(maxsize=8)
def _read_data(name: str) -> Mapping[str, Any]:
    try:
        raw = (DATA_ROOT / f"{name}.json").read_bytes()
    except OSError as error:
        raise ContractError(f"unknown competitive dataset: {name}") from error
    return frozen_mapping(json.loads(raw))


def load_competitive_data(name: str, workspace_root: str | Path | None = None) -> Mapping[str, Any]:
    return _read_data(name)
