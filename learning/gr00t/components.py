from __future__ import annotations

import argparse
from pathlib import Path

from learning.common import canonical, digest, file_digest, read_json, require, write_json
from learning.contract import ControlProfile, Scope
from learning.gr00t.artifacts import task_contract
from learning.gr00t.azure import (
    clients_for_managed_identity,
    running_job_binding,
    validate_config,
    verify_code,
)
from learning.gr00t.bootstrap import evaluate_bootstrap
from learning.gr00t.dataset import export_dataset, validate_export
from learning.gr00t.evaluation import evaluate_pair
from learning.gr00t.train import Gr00tTrainOptions, run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Private Azure ML GR00T components.")
    parser.add_argument("command", choices=("export", "train", "compare", "bootstrap"))
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--runtime-config", required=True, type=Path)
    parser.add_argument("--snapshot-sha256", required=True)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--after", type=Path)
    parser.add_argument("--plan", type=Path)
    args = parser.parse_args()
    verify_code(Path.cwd(), args.snapshot_sha256)
    config = read_json(args.runtime_config)
    validate_config(config)
    scope = Scope(config["tenant_id"], config["owner_id"])
    if args.command == "export":
        result = export_dataset(
            args.input,
            args.output / "dataset",
            expected_scope=scope,
            expected_manifest_sha256=config["inputs"]["demonstrations"]["sha256"],
        )
        require(
            ControlProfile(**result["control_profile"]).sha256 == config["control_profile_sha256"],
            "Actual capture profile differs from the approved job",
        )
        from learning.contract import DemonstrationSource

        for episode in result["episodes"]:
            task = DemonstrationSource(**episode["demonstration"])
            require(
                digest(canonical(task_contract(task))) == config["task_sha256"],
                "Captured task differs from the approved job",
            )
    elif args.command == "train":
        require(args.parent is not None, "Missing explicit parent checkpoint input")
        dataset = args.input / "dataset"
        export_sha = file_digest(dataset / "export.json")
        exported = validate_export(dataset, scope=scope, expected_sha256=export_sha)
        require(
            exported["raw_manifest_sha256"] == config["inputs"]["demonstrations"]["sha256"],
            "Training export is not derived from the approved raw dataset",
        )
        client, _ = clients_for_managed_identity(
            config, caller_client_id=config["managed_identity_client_id"]
        )
        run_training(
            dataset,
            args.parent,
            args.output,
            source_root=Path("/opt/isaac-gr00t"),
            scope=scope,
            export_sha256=export_sha,
            parent_model_sha256=config["inputs"]["parent_model"]["sha256"],
            code_snapshot_sha256=args.snapshot_sha256,
            azure_config=config,
            client=client,
            options=Gr00tTrainOptions(**config["parameters"]),
        )
    elif args.command == "compare":
        require(
            args.parent is not None and args.after is not None and args.plan is not None,
            "Paired evaluation requires both real policy artifacts and the frozen plan",
        )
        plan = read_json(args.plan)
        require(
            plan["policy_before_sha256"] == config["inputs"]["policy_before"]["sha256"]
            and plan["policy_after_sha256"] == config["inputs"]["policy_after"]["sha256"],
            "Paired policies differ from the approved Azure job",
        )
        report = evaluate_pair(
            plan,
            args.input,
            args.parent,
            args.after,
            scope=scope,
            expected_plan_sha256=config["inputs"]["plan"]["sha256"],
            expected_results_sha256=config["inputs"]["evidence"]["sha256"],
        )
        client, _ = clients_for_managed_identity(
            config, caller_client_id=config["managed_identity_client_id"]
        )
        report.update(running_job_binding(client, config))
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(args.output / "report.json", report)
        if not report["quality_gate"]:
            raise SystemExit(1)
    else:
        require(
            args.parent is not None and args.plan is not None,
            "Missing privileged P0 bootstrap inputs",
        )
        plan = read_json(args.plan)
        require(
            plan["candidate_model_sha256"] == config["inputs"]["candidate"]["sha256"],
            "Bootstrap candidate differs from the approved job",
        )
        report = evaluate_bootstrap(
            plan,
            args.input,
            args.parent,
            scope=scope,
            expected_plan_sha256=config["inputs"]["plan"]["sha256"],
            expected_results_sha256=config["inputs"]["evidence"]["sha256"],
        )
        client, _ = clients_for_managed_identity(
            config, caller_client_id=config["managed_identity_client_id"]
        )
        report.update(running_job_binding(client, config))
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(args.output / "report.json", report)
        if not report["quality_gate_passed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
