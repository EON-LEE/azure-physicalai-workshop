"""Offline real-SDK/native-3.11 command bootstrap qualification. No model or cloud workload."""

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
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from learning.checks.smolvla_aml_check import example_config
from learning.common import canonical, digest, read_json, require, write_json
from learning.gr00t.azure import workspace_id
from learning.paused.command import EXECUTION, P1_TRAINING_COHORT, validate_command_job
from learning.paused.contract import PausedControlProfile
from learning.smolvla import azure
from learning.smolvla.checkpoint_runner import POLICY_SCHEMA
from learning.smolvla.checkpoints import DEFAULT_LIMITS
from learning.smolvla.embedded_source import prepare_context
from learning.smolvla.image_bootstrap import BOOTSTRAP_RELATIVE, materialize_source


def run(*, p1_additional20: bool = False) -> dict:
    from azure.ai.ml import MLClient, load_job
    from azure.ai.ml._restclient.arm_ml_service.models import UriFolderJobOutput
    from azure.ai.ml.operations import JobOperations

    require(version("azure-ai-ml") == "1.35.0", "Use the pinned Azure ML SDK")
    require(sys.version_info[:2] == (3, 11), "Use the native Linux Python 3.11 environment")
    with (
        tempfile.TemporaryDirectory(prefix="command-source-check-") as temporary,
        patch(
            "azure.core.pipeline.transport.RequestsTransport.send",
            side_effect=AssertionError("Offline check attempted a network request"),
        ),
    ):
        root = Path(temporary)
        context = root / "image-context"
        receipt = prepare_context(context, direct=True)
        config = example_config()
        config.update(
            schema=azure.CONFIG_SCHEMA,
            run_id="offline-direct-command",
            execution_timing="paused_simulation",
            real_time_admission=False,
            criteria_sha256="d" * 64,
            frozen_plan_sha256="e" * 64,
            control_profile_sha256=PausedControlProfile("f" * 64).sha256,
            compute_size="Standard_NC24ads_A100_v4",
            compute_tier="LowPriority",
            output_prefix=f"tenants/{config['tenant_id']}/owners/{config['owner_id']}/learning/outputs",
            job_deadline_utc="2030-01-01T00:00:00Z",
            source_delivery=read_json(context / "source-delivery.json"),
            job_execution=EXECUTION,
            checkpointing={
                "schema": POLICY_SCHEMA,
                "limits": asdict(DEFAULT_LIMITS),
                "resume": None,
            },
        )
        if p1_additional20:
            config["training_cohort"] = dict(P1_TRAINING_COHORT)
            config["parameters"]["resume_mode"] = "weights_only"
        plan_dir = root / "plan"
        azure.create_plan(config, plan_dir, deterministic_job_name=config["run_id"])
        plan = azure.read_plan(plan_dir)
        job = load_job(source=plan_dir / "job.json")
        require(
            job.type == "command" and job.code is None and not job.inputs and not job.outputs,
            "Actual SDK inferred a code/data mount",
        )
        resolved = []
        environment_id = workspace_id(config) + "/environments/offline-native/versions/1"

        def resolver(asset, *, azureml_type):
            resolved.append(azureml_type)
            require(azureml_type == "environments", "Unexpected code/data asset resolver call")
            require(asset.image == config["environment_image"], "SDK environment image changed")
            return environment_id

        adapter = SimpleNamespace(
            _resolve_compute_id=lambda resolver, compute: (
                workspace_id(config) + "/computes/" + config["compute"]
            ),
        )
        JobOperations._resolve_arm_id_for_command_job(adapter, job, resolver)
        wire = job._to_job()._to_rest_object()
        properties = wire.properties
        require(
            properties.code_id is None and not properties.inputs and not properties.outputs,
            "Actual AML wire contains custom source or data preparation",
        )
        wire.id, wire.name = workspace_id(config) + "/jobs/" + config["run_id"], config["run_id"]
        properties.status = "Completed"
        properties.outputs = {
            "default": UriFolderJobOutput(
                uri="azureml://datastores/workspaceartifactstore/ExperimentRun/dcid."
                + config["run_id"],
                mode="ReadWriteMount",
            ),
        }
        client = MLClient(
            SimpleNamespace(get_token=lambda *args, **kwargs: require(False, "Unexpected token")),
            config["subscription_id"],
            config["resource_group"],
            config["workspace"],
        )
        with patch.object(client.jobs, "_get_job", return_value=wire) as fetch:
            actual = client.jobs.get(config["run_id"])
        fetch.assert_called_once_with(config["run_id"])
        require(
            actual.compute == config["compute"] and actual.environment == "offline-native:1",
            "Actual public SDK GET did not exercise workspace-local reference normalization",
        )
        environment_reads = []

        def environment(name, *, version):
            environment_reads.append((name, version))
            return SimpleNamespace(image=config["environment_image"])

        binding = validate_command_job(
            SimpleNamespace(environments=SimpleNamespace(get=environment)),
            config,
            actual,
            snapshot_sha256=plan["snapshot_sha256"],
            expected_status="Completed",
        )
        require(environment_reads == [("offline-native", "1")], "Unbound environment read")
        tokens = shlex.split(read_json(plan_dir / "job.json")["command"])
        arguments = {
            "config_base64": tokens[tokens.index("--config-base64") + 1],
            "config_sha256": tokens[tokens.index("--config-sha256") + 1],
            "snapshot_sha256": plan["snapshot_sha256"],
            "static_sha256": receipt["static_sha256"],
        }
        materialized = root / "materialized"
        materialize_source(
            context / "source",
            context / "static-manifest.json",
            destination=materialized,
            **arguments,
        )
        azure.verify_code(materialized, plan["snapshot_sha256"], config=config)
        unrelated = root / "azureml-overridden-cwd"
        unrelated.mkdir()
        environment = dict(os.environ)
        for name in ("AZUREML_RUN_ID", "PYTHONHOME", "PYTHONPATH"):
            environment.pop(name, None)
        bootstrap = context / "source" / BOOTSTRAP_RELATIVE
        checked = subprocess.run(
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
            env=environment,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        require(
            json.loads(checked.stdout)["static_sha256"] == receipt["static_sha256"],
            "Cold static source qualification failed",
        )
        environment["PYTHONPATH"] = str(materialized)
        imports = subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                "import learning.paused.command, learning.paused.command_model; "
                "import learning.paused.components; "
                "from learning.smolvla.azure import verify_code; "
                "from learning.common import read_json; from pathlib import Path; "
                "root=Path(__import__('os').environ['PYTHONPATH']); "
                f"verify_code(root, '{plan['snapshot_sha256']}', "
                "config=read_json(root/'run-config.json')); "
                "print('native-imports-and-source-verified')",
            ],
            cwd=unrelated,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        require("native-imports-and-source-verified" in imports.stdout, "Native imports failed")
        expired = dict(config, job_deadline_utc="2020-01-01T00:00:00Z")
        expired_bytes = canonical(expired) + b"\n"
        rejected = subprocess.run(
            [
                sys.executable,
                "-I",
                "-B",
                str(bootstrap),
                "run",
                "command-train",
                "--static-sha256",
                receipt["static_sha256"],
                "--config-base64",
                base64.b64encode(expired_bytes).decode(),
                "--config-sha256",
                digest(expired_bytes),
                "--snapshot-sha256",
                plan["snapshot_sha256"],
            ],
            cwd=unrelated,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        require(
            rejected.returncode != 0 and "deadline expired" in rejected.stderr,
            "Cold command bootstrap did not reject the old authority",
        )
        return {
            "check": "actual-aml-sdk-1.35-command-and-native-python-3.11-source",
            "test_only": True,
            "fixture_config": True,
            "training_cohort": config.get("training_cohort"),
            "static_file_count": receipt["file_count"],
            "static_sha256": receipt["static_sha256"],
            "snapshot_sha256": plan["snapshot_sha256"],
            "actual_sdk_code_asset_resolutions": 0,
            "actual_sdk_data_asset_resolutions": 0,
            "actual_sdk_environment_resolutions": len(resolved),
            "actual_sdk_wire_custom_inputs": 0,
            "actual_sdk_wire_custom_outputs": 0,
            "actual_sdk_get_roundtrip_job_type": binding["azure_job_type"],
            "actual_public_sdk_get_exercised": True,
            "actual_sdk_compute_reference": actual.compute,
            "actual_sdk_environment_reference": actual.environment,
            "native_materialized_imports_verified": True,
            "overridden_cwd_verified": True,
            "cold_expired_deadline_rejected": True,
            "runtime_config_bytes": len(base64.b64decode(arguments["config_base64"])),
            "command_bytes": len(read_json(plan_dir / "job.json")["command"].encode()),
            "cloud_calls": 0,
            "image_builds": 0,
            "jobs_submitted": 0,
            "model_weights_loaded": False,
            "learning_quality_verified": False,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--p1-additional20", action="store_true")
    args = parser.parse_args()
    report = run(p1_additional20=args.p1_additional20)
    if args.report is not None:
        write_json(args.report, report)
    print(json.dumps(report, indent=2))
