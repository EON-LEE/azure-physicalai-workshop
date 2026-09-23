from __future__ import annotations

import argparse
from pathlib import Path

from learning.common import read_json, require
from learning.gr00t import azure as shared
from learning.smolvla import POLICY_TYPE, UPSTREAM

CONFIG_SCHEMA = "physicalai.smolvla-azure/v1"
PLAN_SCHEMA = "physicalai.smolvla-azure-plan/v1"
CODE_FILES = shared.CODE_FILES + (
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


def validate_config(config: dict) -> None:
    inputs = None
    if config.get("kind") == "train":
        inputs = {
            "demonstrations": "uri_folder",
            "parent_model": "uri_folder",
            "backbone": "uri_folder",
        }
    shared.validate_config(config, schema=CONFIG_SCHEMA, upstream=UPSTREAM, input_types=inputs)
    if config["kind"] == "train":
        require(
            config["parameters"]["gradient_accumulation_steps"] == 1,
            "Pinned LeRobot CLI requires gradient_accumulation_steps=1",
        )


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
        code_files=CODE_FILES,
        plan_schema=PLAN_SCHEMA,
    )


def read_plan(path: Path) -> dict:
    return shared.read_plan(
        path,
        validator=validate_config,
        job_builder=build_job,
        code_files=CODE_FILES,
        plan_schema=PLAN_SCHEMA,
    )


def verify_code(root: Path, expected_sha256: str) -> None:
    shared.verify_code(root, expected_sha256, code_files=CODE_FILES)


class PolicyJobs(shared.Gr00tJobs):
    policy_type = POLICY_TYPE
    config_validator = staticmethod(validate_config)
    plan_reader = staticmethod(read_plan)

    def check_license(self) -> None:
        require(
            self.config["upstream"] == UPSTREAM
            and UPSTREAM["model_license"] == UPSTREAM["backbone_license"] == "Apache-2.0",
            "Explicit Smol model/backbone/source/license pins changed",
        )


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
