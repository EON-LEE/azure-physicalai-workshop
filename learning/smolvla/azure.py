from __future__ import annotations

import argparse
from pathlib import Path

from learning.common import read_json, require
from learning.deadlines import JobDeadline, validate_deadline
from learning.gr00t import azure as shared
from learning.smolvla import POLICY_TYPE, UPSTREAM

CONFIG_SCHEMA = "physicalai.smolvla-azure/v2"
PLAN_SCHEMA = "physicalai.smolvla-azure-plan/v2"
LEGACY_CONFIG_SCHEMA = "physicalai.smolvla-azure/v1"
LEGACY_PLAN_SCHEMA = "physicalai.smolvla-azure-plan/v1"
LEGACY_CODE_FILES = shared.CODE_FILES + (
    "learning/smolvla/__init__.py",
    "learning/smolvla/adaptation.py",
    "learning/smolvla/artifacts.py",
    "learning/smolvla/prepare.py",
    "learning/smolvla/dataset.py",
    "learning/smolvla/train.py",
    "learning/smolvla/inference.py",
    "learning/smolvla/ipc.py",
    "learning/smolvla/evaluation.py",
    "learning/smolvla/azure.py",
    "learning/smolvla/components.py",
    "learning/smolvla/probe.py",
    "learning/smolvla/pyproject.toml",
    "learning/smolvla/uv.lock",
    "learning/smolvla/Dockerfile",
    "learning/smolvla/rollout.py",
)
CODE_FILES = LEGACY_CODE_FILES + ("learning/deadlines.py",)


def validate_config(config: dict) -> None:
    require(
        config.get("schema") in (CONFIG_SCHEMA, LEGACY_CONFIG_SCHEMA),
        "Unsupported SmolVLA Azure configuration schema",
    )
    base = dict(config)
    if config["schema"] == CONFIG_SCHEMA:
        validate_deadline(base.pop("job_deadline_utc", None))
    inputs = None
    if config.get("kind") == "train":
        inputs = {
            "demonstrations": "uri_folder",
            "parent_model": "uri_folder",
            "backbone": "uri_folder",
        }
    shared.validate_config(base, schema=config["schema"], upstream=UPSTREAM, input_types=inputs)
    if config["kind"] == "train":
        require(
            config["parameters"]["gradient_accumulation_steps"] == 1,
            "Pinned LeRobot CLI requires gradient_accumulation_steps=1",
        )


def job_deadline(config: dict) -> JobDeadline:
    require(
        config.get("schema") == CONFIG_SCHEMA,
        "New workload admission requires v2 with an explicit approved job deadline",
    )
    return JobDeadline(config.get("job_deadline_utc"))


def build_job(config: dict, snapshot_sha256: str, job_name: str) -> dict:
    return shared.build_job(
        config,
        snapshot_sha256,
        job_name,
        validator=validate_config,
        policy_type=POLICY_TYPE,
        command_module="learning.smolvla.components",
        include_backbone=True,
    )


def create_plan(
    config: dict, output: Path, *, deterministic_job_name: str, source_root=None
) -> str:
    return shared.create_plan(
        config,
        output,
        deterministic_job_name=deterministic_job_name,
        source_root=source_root,
        validator=validate_config,
        job_builder=build_job,
        code_files=CODE_FILES if config["schema"] == CONFIG_SCHEMA else LEGACY_CODE_FILES,
        plan_schema=PLAN_SCHEMA if config["schema"] == CONFIG_SCHEMA else LEGACY_PLAN_SCHEMA,
    )


def read_plan(path: Path) -> dict:
    legacy = read_json(path / "plan.json").get("schema") == LEGACY_PLAN_SCHEMA
    plan = shared.read_plan(
        path,
        validator=validate_config,
        job_builder=build_job,
        code_files=LEGACY_CODE_FILES if legacy else CODE_FILES,
        plan_schema=LEGACY_PLAN_SCHEMA if legacy else PLAN_SCHEMA,
    )
    require(
        plan["config"]["schema"] == (LEGACY_CONFIG_SCHEMA if legacy else CONFIG_SCHEMA),
        "Policy plan/config deadline schemas differ",
    )
    return plan


def verify_code(root: Path, expected_sha256: str) -> None:
    shared.verify_code(root, expected_sha256, code_files=CODE_FILES)


class PolicyJobs(shared.Gr00tJobs):
    policy_type = POLICY_TYPE
    config_validator = staticmethod(validate_config)
    plan_reader = staticmethod(read_plan)

    def __init__(self, client, config: dict, *, storage_client) -> None:
        super().__init__(client, config, storage_client=storage_client)
        self._deadline = job_deadline(self.config) if config["schema"] == CONFIG_SCHEMA else None

    def check_license(self) -> None:
        require(
            self.config["upstream"] == UPSTREAM
            and UPSTREAM["model_license"] == UPSTREAM["backbone_license"] == "Apache-2.0",
            "Explicit Smol model/backbone/source/license pins changed",
        )

    def check_admission(self) -> None:
        self.check_license()
        require(self._deadline is not None, "New admission requires an explicit v2 job deadline")
        self._deadline.check()

    def reconcile_deadline(self, job_name: str) -> dict:
        """Explicit mutation: the caller must first acquire its durable cancellation claim."""
        require(self._deadline is not None, "Deadline reconciliation requires v2 authority")
        current = self.status(job_name)
        expired = self._deadline.remaining_seconds() <= 0
        result = {**current, "cancellation_requested": False}
        if expired and current["azure_status"] not in (
            "Completed",
            "Failed",
            "Canceled",
            "CancelRequested",
        ):
            result = self.cancel(job_name)
        return {
            **result,
            "job_deadline_utc": self._deadline.value,
            "deadline_expired": expired,
        }


def clients_for_managed_identity(config: dict, *, caller_client_id: str):
    return shared.clients_for_managed_identity(
        config,
        caller_client_id=caller_client_id,
        validator=validate_config,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Offline-only explicitly selected SmolVLA AML plan."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--plan-dir", required=True, type=Path)
    parser.add_argument("--job-name", required=True)
    args = parser.parse_args()
    print(create_plan(read_json(args.config), args.plan_dir, deterministic_job_name=args.job_name))
