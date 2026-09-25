from __future__ import annotations

import argparse
import sys
from pathlib import Path

from learning.common import canonical, digest, file_digest, read_json, require
from learning.contract import Scope
from learning.deadlines import run_bounded
from learning.gr00t.azure import running_job_binding
from learning.paused.artifacts import validate_model
from learning.paused.contract import PausedControlProfile
from learning.paused.dataset import convert_dataset, validate_conversion
from learning.paused.train import run_training
from learning.smolvla.azure import (
    clients_for_managed_identity,
    is_paused,
    job_deadline,
    validate_config,
    verify_code,
)
from learning.smolvla.train import TrainOptions


def main() -> None:
    parser = argparse.ArgumentParser(description="Explicit paused-simulation Azure components.")
    parser.add_argument("command", choices=("export", "train", "compare", "bootstrap"))
    for name in ("input", "output", "runtime-config"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--snapshot-sha256", required=True)
    for name in ("parent", "backbone", "after", "plan"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--run-component", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    config = read_json(args.runtime_config)
    validate_config(config)
    require(is_paused(config), "Paused component requires explicit new-mode authority")
    deadline = job_deadline(config)
    deadline.check()
    verify_code(Path.cwd(), args.snapshot_sha256, config=config)
    deadline.check()
    if not args.run_component:
        run_bounded(
            [sys.executable, "-m", "learning.paused.components", *sys.argv[1:], "--run-component"],
            deadline,
        )
        return
    scope = Scope(config["tenant_id"], config["owner_id"])
    client, _ = clients_for_managed_identity(
        config, caller_client_id=config["managed_identity_client_id"]
    )
    running_job_binding(client, config)
    deadline.check()
    if args.command == "export":
        require(config["kind"] == "train", "Wrong component kind")
        conversion = convert_dataset(
            args.input,
            args.output / "dataset",
            expected_scope=scope,
            expected_manifest_sha256=config["inputs"]["demonstrations"]["sha256"],
            expected_criteria_sha256=config["criteria_sha256"],
            expected_frozen_plan_sha256=config["frozen_plan_sha256"],
        )
        require(
            PausedControlProfile(**conversion["control_profile"]).sha256
            == config["control_profile_sha256"],
            "Actual paused data control profile differs from the reviewed job",
        )
        for episode in conversion["episodes"]:
            task = {
                name: episode["demonstration"][name]
                for name in ("task_id", "instruction", "goal_id")
            }
            require(
                digest(canonical(task)) == config["task_sha256"], "Actual teaching task differs"
            )
        deadline.check()
        return
    if args.command == "train":
        require(
            config["kind"] == "train" and args.parent is not None and args.backbone is not None,
            "Missing approved paused training inputs",
        )
        parent = validate_model(
            args.parent,
            expected_scope=scope,
            expected_model_sha256=config["inputs"]["parent_model"]["sha256"],
            for_inference=False,
        )
        deadline.check()
        require(
            parent["backbone_manifest_sha256"] == config["inputs"]["backbone"]["sha256"],
            "Parent and approved backbone versions differ",
        )
        dataset = args.input / "dataset"
        conversion = validate_conversion(dataset, scope)
        require(
            conversion["raw_manifest_sha256"] == config["inputs"]["demonstrations"]["sha256"]
            and PausedControlProfile(**conversion["control_profile"]).sha256
            == config["control_profile_sha256"],
            "Converted raw provenance or new servo fingerprint differs",
        )
        deadline.check()
        run_training(
            dataset,
            args.parent,
            args.backbone,
            args.output,
            scope=scope,
            parent_model_sha256=config["inputs"]["parent_model"]["sha256"],
            conversion_sha256=file_digest(dataset / "conversion.json"),
            code_snapshot_sha256=args.snapshot_sha256,
            config=config,
            client=client,
            options=TrainOptions(**config["parameters"]),
        )
        deadline.check()
        return
    from learning.paused.evaluation import evaluate_bootstrap, evaluate_pair

    require(args.parent is not None and args.plan is not None, "Missing frozen evaluation inputs")
    plan = read_json(args.plan)
    require(
        plan["criteria_sha256"] == config["criteria_sha256"]
        and plan["frozen_plan_sha256"] == config["frozen_plan_sha256"]
        and plan["control_profile_sha256"] == config["control_profile_sha256"],
        "Evaluation profile/criteria/conditions changed",
    )
    kwargs = {
        "scope": scope,
        "expected_plan_sha256": config["inputs"]["plan"]["sha256"],
        "expected_results_sha256": config["inputs"]["evidence"]["sha256"],
    }
    if args.command == "compare":
        require(
            args.after is not None
            and config["kind"] == "compare"
            and plan["policy_before_sha256"] == config["inputs"]["policy_before"]["sha256"]
            and plan["policy_after_sha256"] == config["inputs"]["policy_after"]["sha256"],
            "Missing or rebound actual policy pair",
        )
        report = evaluate_pair(plan, args.input, args.parent, args.after, **kwargs)
    else:
        require(
            config["kind"] == "bootstrap_compare"
            and plan["candidate_model_sha256"] == config["inputs"]["candidate"]["sha256"],
            "Wrong or rebound bootstrap candidate",
        )
        report = evaluate_bootstrap(plan, args.input, args.parent, **kwargs)
    deadline.check()
    from learning.common import write_json

    report.update(running_job_binding(client, config))
    args.output.mkdir(parents=True, exist_ok=True)
    deadline.check()
    write_json(args.output / "report.json", report)
    require(report["quality_gate_passed"], "Paused physical evidence did not pass quality gates")


if __name__ == "__main__":
    main()
