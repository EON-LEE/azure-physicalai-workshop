"""Real AML SDK and Python 3.11 static-bootstrap check; offline metadata fixtures only."""

from __future__ import annotations

import argparse
import base64
import json
import os
import shlex
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

from learning.checks.smolvla_aml_check import example_config
from learning.common import canonical, digest, read_json, require, write_json
from learning.paused.contract import PausedControlProfile
from learning.smolvla import azure
from learning.smolvla.checkpoint_runner import POLICY_SCHEMA
from learning.smolvla.checkpoints import DEFAULT_LIMITS
from learning.smolvla.embedded_source import prepare_context
from learning.smolvla.image_bootstrap import BOOTSTRAP_RELATIVE, materialize_source


def run() -> dict:
    from importlib.metadata import version

    from azure.ai.ml import load_job
    from azure.ai.ml.operations import ComponentOperations

    require(version("azure-ai-ml") == "1.35.0", "Use the pinned Azure ML SDK")
    require(
        sys.version_info[:2] == (3, 11),
        "Run this check in the qualified native Python 3.11 environment",
    )
    native_resolver = ComponentOperations._resolve_dependencies_for_component.__globals__[
        "_try_resolve_code_for_component"
    ]
    with tempfile.TemporaryDirectory(prefix="embedded-source-check-") as temporary:
        root = Path(temporary)
        context = root / "image-context"
        receipt = prepare_context(context)
        config = example_config()
        config.update(
            schema=azure.CONFIG_SCHEMA,
            execution_timing="paused_simulation",
            real_time_admission=False,
            criteria_sha256="d" * 64,
            frozen_plan_sha256="e" * 64,
            control_profile_sha256=PausedControlProfile("f" * 64).sha256,
            job_deadline_utc="2030-01-01T00:00:00Z",
            source_delivery=read_json(context / "source-delivery.json"),
            checkpointing={
                "schema": POLICY_SCHEMA,
                "limits": asdict(DEFAULT_LIMITS),
                "resume": None,
            },
        )
        legacy = {key: value for key, value in config.items() if key != "source_delivery"}
        counts = {}
        for label, value in (("embedded", config), ("legacy", legacy)):
            directory = root / label
            azure.create_plan(value, directory, deterministic_job_name=f"offline-{label}")
            job = load_job(source=directory / "job.json")
            calls = []

            def resolver(code, *, azureml_type, calls=calls):
                calls.append(azureml_type)
                return (
                    "/subscriptions/22222222-2222-4222-8222-222222222222/resourceGroups/unit/"
                    "providers/Microsoft.MachineLearningServices/workspaces/unit/"
                    "codes/offline/versions/1"
                )

            for child in job.jobs.values():
                if label == "embedded":
                    require(child.code is None, "SDK unexpectedly inferred an embedded code asset")
                native_resolver(child.component, resolver)
            counts[label] = len(calls)
        require(
            counts == {"embedded": 0, "legacy": 2},
            "Actual SDK code-resolution behavior differs from the intended delivery modes",
        )
        plan = azure.read_plan(root / "embedded")
        command = read_json(root / "embedded" / "job.json")["jobs"]["train"]["command"]
        tokens = shlex.split(command)
        payload = tokens[tokens.index("--config-base64") + 1]
        arguments = {
            "config_base64": payload,
            "config_sha256": tokens[tokens.index("--config-sha256") + 1],
            "snapshot_sha256": plan["snapshot_sha256"],
            "static_sha256": receipt["static_sha256"],
        }
        source = context / "source"
        workspace = root / "task-local-source"
        materialize_source(
            source, context / "static-manifest.json", destination=workspace, **arguments
        )
        azure.verify_code(workspace, plan["snapshot_sha256"], config=config)
        unrelated = root / "aml-overridden-working-directory"
        unrelated.mkdir()
        cold_environment = dict(os.environ)
        cold_environment.pop("AZUREML_RUN_ID", None)
        cold_environment.pop("PYTHONHOME", None)
        bootstrap = source / BOOTSTRAP_RELATIVE
        static = subprocess.run(
            [
                sys.executable,
                "-I",
                "-B",
                str(bootstrap),
                "verify-static",
                "--static-sha256",
                receipt["static_sha256"],
            ],
            cwd=unrelated,
            env=cold_environment,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        static_result = json.loads(static.stdout)
        require(
            static_result["static_sha256"] == receipt["static_sha256"], "Cold static check failed"
        )
        expired = dict(config, job_deadline_utc="2020-01-01T00:00:00Z")
        expired_bytes = canonical(expired) + b"\n"
        rejected = subprocess.run(
            [
                sys.executable,
                "-I",
                "-B",
                str(bootstrap),
                "run",
                "export",
                "--static-sha256",
                receipt["static_sha256"],
                "--config-base64",
                base64.b64encode(expired_bytes).decode(),
                "--config-sha256",
                digest(expired_bytes),
                "--snapshot-sha256",
                plan["snapshot_sha256"],
                "--input",
                str(root / "missing-data"),
                "--output",
                str(root / "must-not-exist"),
            ],
            cwd=unrelated,
            env=cold_environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        require(
            rejected.returncode != 0
            and "deadline expired" in rejected.stderr
            and not (root / "must-not-exist").exists(),
            "Cold bootstrap did not reject expiry before native component work",
        )
        return {
            "check": "actual-sdk-1.35.0-no-code-and-native-python-3.11-bootstrap",
            "test_only": True,
            "fixture_config": True,
            "static_file_count": receipt["file_count"],
            "static_sha256": receipt["static_sha256"],
            "full_snapshot_sha256": plan["snapshot_sha256"],
            "actual_sdk_code_resolutions": counts,
            "materialized_native_snapshot_verified": True,
            "arbitrary_aml_working_directory_verified": True,
            "cold_expired_deadline_rejected": True,
            "encoded_config_bytes": len(base64.b64decode(payload)),
            "train_command_bytes": len(command.encode()),
            "image_builds": 0,
            "cloud_calls": 0,
            "jobs_submitted": 0,
            "model_weights_loaded": False,
            "learning_quality_verified": False,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = run()
    if args.report is not None:
        write_json(args.report, report)
    print(json.dumps(report, indent=2))
