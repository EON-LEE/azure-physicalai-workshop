"""Explicitly attested command-v3 admission with unchanged paused prediction/IPC controls."""

from __future__ import annotations

import argparse
from pathlib import Path

from learning.common import ContractError, read_json, require
from learning.contract import Scope
from learning.paused.command_artifacts import ADMISSION_KIND, MODEL_SCHEMA, validate_model
from learning.paused.contract import EXECUTION_TIMING, PausedControlProfile
from learning.paused.ipc import serve
from learning.paused.model import LocalPausedSmolVLAPolicy


class LocalCommandPausedSmolVLAPolicy(LocalPausedSmolVLAPolicy):
    model_admission, artifact_schema = ADMISSION_KIND, MODEL_SCHEMA

    def __init__(
        self,
        root: Path,
        *,
        backbone_root: Path,
        scope: Scope,
        model_sha256: str,
        expected_control_profile_sha256: str,
        expected_criteria_sha256: str,
        expected_frozen_plan_sha256: str,
    ) -> None:
        metadata = validate_model(root, expected_scope=scope, expected_model_sha256=model_sha256)
        require(
            metadata["criteria_sha256"] == expected_criteria_sha256
            and metadata["frozen_plan_sha256"] == expected_frozen_plan_sha256,
            "Command v3 model belongs to different frozen criteria/conditions",
        )
        self._load(
            root,
            backbone_root=backbone_root,
            scope=scope,
            model_sha256=model_sha256,
            expected_control_profile_sha256=expected_control_profile_sha256,
            metadata=metadata,
            profile=PausedControlProfile(**metadata["control_profile"]),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Explicit command-v3 paused Smol model process.")
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--backbone-root", required=True, type=Path)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--binding", required=True, type=Path)
    parser.add_argument("--socket-path", required=True, type=Path)
    parser.add_argument("--allowed-client-uid", type=int)
    args = parser.parse_args()
    binding = read_json(args.binding)
    require(
        binding.get("execution_timing") == EXECUTION_TIMING
        and binding.get("real_time_admission") is False,
        "Explicit paused model-server binding is required",
    )
    try:
        policy = LocalCommandPausedSmolVLAPolicy(
            args.model_root,
            backbone_root=args.backbone_root,
            scope=Scope(**binding["scope"]),
            model_sha256=args.model_sha256,
            expected_control_profile_sha256=PausedControlProfile(
                **binding["control_profile"]
            ).sha256,
            expected_criteria_sha256=binding["criteria_sha256"],
            expected_frozen_plan_sha256=binding["frozen_plan_sha256"],
        )
        serve(policy, args.socket_path, allowed_client_uid=args.allowed_client_uid)
    except (ContractError, OSError) as exc:
        raise SystemExit(f"Command-v3 paused Smol process stopped: {exc}") from exc


if __name__ == "__main__":
    main()
