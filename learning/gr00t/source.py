from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from learning.common import require
from learning.gr00t import SOURCE_COMMIT
from learning.offline import enforce_offline


def activate_source(root: Path) -> None:
    """Only the exact clean, deployment-installed upstream checkout may supply Eagle code."""
    enforce_offline()
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    require(
        root.is_absolute() and root.is_dir() and not root.is_symlink(), "Invalid GR00T source root"
    )
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    require(result.stdout.strip() == SOURCE_COMMIT, "Unreviewed GR00T source revision")
    changed = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all", "--", "gr00t"],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    require(not changed.stdout.strip(), "GR00T source/processor checkout has modifications")
    require(
        not any(path.is_symlink() for path in (root / "gr00t").rglob("*")),
        "Source may not redirect reviewed code through symlinks",
    )
    require("gr00t" not in sys.modules, "GR00T must not be imported before source verification")
    # Upstream loads its bundled Eagle classes through Transformers' local module cache.
    # Never reuse a caller-controlled cache for those reviewed source files.
    module_cache = tempfile.TemporaryDirectory(prefix="physicalai-gr00t-modules-")
    os.environ["HF_MODULES_CACHE"] = module_cache.name
    _module_caches.append(module_cache)
    sys.path.insert(0, str(root))


_module_caches: list[tempfile.TemporaryDirectory] = []
