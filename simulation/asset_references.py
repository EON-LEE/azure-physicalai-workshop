"""Keep USD dependencies in the approved Azure asset bundle or bundled Kit materials."""

import re
from pathlib import Path

BUILTIN_MATERIALS = frozenset({"OmniPBR.mdl", "OmniGlass.mdl", "UsdPreviewSurface.mdl"})


def local_reference(root: Path, layer: Path, reference: str) -> Path | None:
    if not reference or reference in BUILTIN_MATERIALS:
        return None
    if (
        re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", reference)
        or reference.startswith(("/", "\\"))
        or "\\" in reference
    ):
        raise ValueError("External or absolute USD asset references are not allowed.")
    resolved = (layer.parent / reference).resolve()
    if not resolved.is_relative_to(root.resolve()) or not resolved.is_file():
        raise ValueError("A USD dependency is outside or missing from the approved asset bundle.")
    return resolved
