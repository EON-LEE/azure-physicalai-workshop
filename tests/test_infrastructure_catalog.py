import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_offline_infrastructure_check_compiles_every_declared_template():
    source = ast.parse((ROOT / "scripts" / "validate_infra.py").read_text(encoding="utf-8"))
    loops = [
        node
        for node in ast.walk(source)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "name"
        and isinstance(node.iter, ast.Tuple)
    ]
    assert len(loops) == 1
    configured = ast.literal_eval(loops[0].iter)
    expected = {path.stem for path in (ROOT / "infra").glob("*.bicep")}
    assert len(configured) == len(set(configured)), "Do not silently compile duplicate templates."
    assert set(configured) == expected, (
        "New deployment templates must not bypass infrastructure CI."
    )
