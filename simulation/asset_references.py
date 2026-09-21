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


def validate_usd_bundle(root: Path, root_asset: Path) -> None:
    from pxr import Sdf, Usd

    if not root_asset.resolve().is_relative_to(root.resolve()):
        raise ValueError("The robot layer must be inside the approved asset bundle.")
    if not Usd.Stage.IsSupportedFile(str(root_asset)):
        raise ValueError("The approved root asset is not a supported USD file.")
    pending = [root_asset]
    seen = set()
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        layer = Sdf.Layer.FindOrOpen(str(path))
        if layer is None:
            raise ValueError("The approved robot USD layer could not be opened.")

        def check(reference, layer_path=path):
            resolved = local_reference(root, layer_path, reference)
            if resolved is not None and resolved.suffix.lower() in {".usd", ".usda", ".usdc"}:
                pending.append(resolved)

        for reference in layer.GetExternalReferences():
            check(reference)

        def attribute(spec_path, current_layer=layer, validator=check):
            item = current_layer.GetObjectAtPath(spec_path)
            if not isinstance(item, Sdf.AttributeSpec):
                return
            values = [item.default]
            values.extend(
                current_layer.QueryTimeSample(spec_path, moment)
                for moment in current_layer.ListTimeSamplesForPath(spec_path)
            )
            for value in values:
                if isinstance(value, Sdf.AssetPath):
                    validator(value.path)
                elif item.typeName == Sdf.ValueTypeNames.AssetArray and value is not None:
                    for asset in value:
                        validator(asset.path)

        layer.Traverse(Sdf.Path.absoluteRootPath, attribute)
