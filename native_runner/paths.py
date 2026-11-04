"""Code lives in this repository; game assets may live in a shared workspace."""

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_ROOT.parent
_shared_workspace = REPOSITORY_ROOT.parent
WORKSPACE_ROOT = Path(
    os.environ.get(
        "CR_WORKSPACE_ROOT", _shared_workspace if (_shared_workspace / "runtime-update").is_dir() else REPOSITORY_ROOT
    )
).resolve()
