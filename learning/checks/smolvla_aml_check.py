"""Real SDK schema check with placeholder resource IDs; never calls Azure."""

import argparse
import copy
import json
from dataclasses import asdict
from pathlib import Path

from learning.checks.groot_fixtures import PROFILE, TASK
from learning.common import canonical, digest, read_json, require, write_json
from learning.gr00t.artifacts import task_contract
from learning.smolvla import UPSTREAM
from learning.smolvla.azure import create_plan, read_plan
from learning.smolvla.train import TrainOptions


def example_config() -> dict:
    root = Path(__file__).resolve().parents[2]
    config = read_json(root / "learning" / "examples" / "azure-train.json")
    config.update(
        schema="physicalai.smolvla-azure/v1",
        compute_tier="LowPriority",
        model_name="smolvla-franka",
        upstream=UPSTREAM,
        specification_sha256="a" * 64,
        control_profile_sha256=PROFILE.sha256,
        task_sha256=digest(canonical(task_contract(TASK))),
        parameters=asdict(
            TrainOptions(max_steps=100, checkpoint_steps=10, compute_tier="LowPriority")
        ),
    )
    asset = config["inputs"]["demonstrations"]
    config["inputs"].update(
        {
            name: {**asset, "name": f"approved-{name}", "uri": asset["uri"] + "/" + name}
            for name in ("parent_model", "backbone")
        }
    )
    return config


def run(output: Path, *, job_deadline_utc: str | None = None) -> dict:
    from azure.ai.ml import load_job

    require(not output.exists(), "Choose a new offline schema-check directory")
    config = example_config()
    if job_deadline_utc is not None:
        config.update(schema="physicalai.smolvla-azure/v2", job_deadline_utc=job_deadline_utc)
    successes = []
    for kind in ("train", "compare", "bootstrap_compare"):
        current = copy.deepcopy(config)
        current["kind"] = kind
        if kind != "train":
            current["parameters"] = {"timeout_seconds": 300}
            names = (
                ("policy_before", "policy_after", "evidence", "plan")
                if kind == "compare"
                else ("candidate", "evidence", "plan")
            )
            asset = config["inputs"]["demonstrations"]
            current["inputs"] = {
                name: {
                    **asset,
                    "name": f"approved-{name}",
                    "uri": asset["uri"] + "/" + name,
                    "type": "uri_file" if name == "plan" else "uri_folder",
                }
                for name in names
            }
        directory = output / kind
        checksum = create_plan(current, directory, deterministic_job_name=f"smol-{kind}")
        require(read_plan(directory)["plan_sha256"] == checksum, "Offline plan revalidation failed")
        job = load_job(source=directory / "job.json")
        require(job.name == f"smol-{kind}", "SDK did not preserve deterministic job identity")
        successes.append({"kind": kind, "plan_sha256": checksum, "jobs": list(job.jobs)})
    report = {
        "check": "real-azure-ai-ml-1.35.0-schema",
        "policy_type": "smolvla",
        "config_schema": config["schema"],
        "schemas": successes,
        "cloud_calls": 0,
        "jobs_submitted": 0,
        "learning_quality_verified": False,
    }
    write_json(output / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--job-deadline-utc")
    args = parser.parse_args()
    print(json.dumps(run(args.output, job_deadline_utc=args.job_deadline_utc), indent=2))
