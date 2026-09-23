from __future__ import annotations

import argparse
from pathlib import Path

from learning.common import canonical, digest, file_digest, read_json, require, write_json
from learning.contract import ControlProfile, Scope
from learning.gr00t.azure import running_job_binding
from learning.smolvla.artifacts import validate_model
from learning.smolvla.azure import clients_for_managed_identity, validate_config, verify_code
from learning.smolvla.dataset import convert_dataset
from learning.smolvla.evaluation import evaluate_bootstrap, evaluate_pair
from learning.smolvla.train import TrainOptions, run_training
from learning.train import validate_conversion


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Explicitly selected private Azure SmolVLA components."
    )
    parser.add_argument("command", choices=("export", "train", "compare", "bootstrap"))
    for name in ("input", "output", "runtime-config"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--snapshot-sha256", required=True)
    for name in ("parent", "backbone", "after", "plan"):
        parser.add_argument(f"--{name}", type=Path)
    args = parser.parse_args()
    verify_code(Path.cwd(), args.snapshot_sha256)
    config = read_json(args.runtime_config)
    validate_config(config)
    scope = Scope(config["tenant_id"], config["owner_id"])
    if args.command == "export":
        conversion = convert_dataset(
            args.input,
            args.output / "dataset",
            expected_scope=scope,
            expected_manifest_sha256=config["inputs"]["demonstrations"]["sha256"],
        )
        require(
            ControlProfile(**conversion["control_profile"]).sha256
            == config["control_profile_sha256"],
            "Actual teacher control profile differs from the approved plan",
        )
        for episode in conversion["episodes"]:
            task = {
                name: episode["demonstration"][name]
                for name in ("task_id", "instruction", "goal_id")
            }
            require(
                digest(canonical(task)) == config["task_sha256"], "Actual teaching task differs"
            )
        return
    client, _ = clients_for_managed_identity(
        config, caller_client_id=config["managed_identity_client_id"]
    )
    if args.command == "train":
        require(
            args.parent is not None and args.backbone is not None,
            "Missing approved vendor/parent and backbone",
        )
        parent = validate_model(
            args.parent,
            expected_scope=scope,
            expected_model_sha256=config["inputs"]["parent_model"]["sha256"],
            for_inference=False,
        )
        require(
            parent["backbone_manifest_sha256"] == config["inputs"]["backbone"]["sha256"],
            "Parent and planned backbone versions differ",
        )
        dataset = args.input / "dataset"
        conversion_sha = file_digest(dataset / "conversion.json")
        conversion = validate_conversion(dataset, scope)
        require(
            conversion["raw_manifest_sha256"] == config["inputs"]["demonstrations"]["sha256"],
            "Converted training data does not derive from approved real demonstrations",
        )
        run_training(
            dataset,
            args.parent,
            args.backbone,
            args.output,
            scope=scope,
            parent_model_sha256=config["inputs"]["parent_model"]["sha256"],
            conversion_sha256=conversion_sha,
            code_snapshot_sha256=args.snapshot_sha256,
            config=config,
            client=client,
            options=TrainOptions(**config["parameters"]),
        )
        return
    require(
        args.parent is not None and args.plan is not None, "Missing immutable evaluation inputs"
    )
    plan = read_json(args.plan)
    if args.command == "compare":
        require(args.after is not None, "P0/P1 requires two actual trained Smol policies")
        require(
            plan["policy_before_sha256"] == config["inputs"]["policy_before"]["sha256"]
            and plan["policy_after_sha256"] == config["inputs"]["policy_after"]["sha256"],
            "Paired policy identities differ from the approved evaluation",
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
        passed = report["quality_gate"]
    else:
        require(
            plan["candidate_model_sha256"] == config["inputs"]["candidate"]["sha256"],
            "Bootstrap candidate differs from the approved artifact",
        )
        report = evaluate_bootstrap(
            plan,
            args.input,
            args.parent,
            scope=scope,
            expected_plan_sha256=config["inputs"]["plan"]["sha256"],
            expected_results_sha256=config["inputs"]["evidence"]["sha256"],
        )
        passed = report["quality_gate_passed"]
    report.update(running_job_binding(client, config))
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "report.json", report)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
