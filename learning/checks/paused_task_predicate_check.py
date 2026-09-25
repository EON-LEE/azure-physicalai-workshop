"""Golden fixture cross-check against the exact reviewed runtime predicate source; no simulator."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import subprocess
from pathlib import Path
from zipfile import ZipFile

from learning.checks.fixtures import JOINTS
from learning.common import require, vector
from learning.contract import bounded_joints
from learning.paused.task import PREDICATE_SOURCE, MeasuredTaskPredicate


def run(source_archive: Path | None = None) -> dict:
    if source_archive is not None:
        with ZipFile(source_archive) as archive:
            info = archive.getinfo(PREDICATE_SOURCE["path"])
            require(info.file_size <= 128 * 1024, "Oversized pinned source")
            source = archive.read(info)
    else:
        source = subprocess.run(
            ["git", "show", f"{PREDICATE_SOURCE['commit']}:{PREDICATE_SOURCE['path']}"],
            check=True,
            capture_output=True,
            timeout=10,
        ).stdout
    blob = hashlib.sha1(
        f"blob {len(source)}\0".encode() + source, usedforsecurity=False
    ).hexdigest()
    require(blob == PREDICATE_SOURCE["git_blob"], "Pinned runtime predicate Git blob changed")
    tree = ast.parse(source)
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == PREDICATE_SOURCE["symbol"]
    ]
    require(len(classes) == 1, "Missing exact reviewed runtime TaskWatchdog")
    namespace = {
        "vector": vector,
        "require": require,
        "bounded_joints": bounded_joints,
        "dist": math.dist,
        "isfinite": math.isfinite,
    }
    exec(
        compile(ast.Module(body=classes, type_ignores=[]), PREDICATE_SOURCE["path"], "exec"),
        namespace,
    )
    tested = 0
    for held_steps in (2, 3, 4, 10):
        for opening in (0.034, 0.035, 0.04):
            initial, goal = (0.3, 0.0, 0.1), (0.3, 0.0, 0.2)
            runtime = namespace["TaskWatchdog"](initial, goal)
            evaluator = MeasuredTaskPredicate(initial, goal)
            points = [((0.3, 0.0, 0.2), (0.3, 0.0, 0.2), JOINTS) for _ in range(held_steps)] + [
                ((0.3, 0.0, 0.3), (0.3, 0.0, 0.2), (*JOINTS[:7], opening, opening))
                for _ in range(30)
            ]
            for tcp, part, joints in points:
                expected = runtime.observe(tcp=tcp, part=part, joints=joints)
                actual = evaluator.observe(tcp=tcp, part=part, joints=joints)
                require(expected == actual, "Runtime and evaluator physical outcomes diverged")
                require(
                    all(
                        getattr(runtime, key) == getattr(evaluator, key)
                        for key in (
                            "grasp_verified",
                            "grasp_seconds",
                            "settled_seconds",
                            "goal_error_m",
                        )
                    ),
                    "Runtime and evaluator threshold state diverged",
                )
                tested += 1
    return {
        "check": "pinned-runtime-task-watchdog-golden-fixture-parity",
        "source": PREDICATE_SOURCE,
        "fixture_states_compared": tested,
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "test_only": True,
        "actual_simulator_executed": False,
        "contact_sensor_claimed": False,
        "optimizer_steps": 0,
        "cloud_calls": 0,
        "learning_quality_verified": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path)
    print(json.dumps(run(parser.parse_args().source_archive), indent=2))
