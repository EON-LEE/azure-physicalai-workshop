from __future__ import annotations

import argparse
import sys
from pathlib import Path

from learning.azure import verify_snapshot
from learning.common import ContractError, file_digest, read_json, require, sha256, write_json
from learning.contract import Scope, validate_dataset
from learning.convert import convert_dataset
from learning.evaluation import evaluate_results
from learning.inference import validate_model_artifact
from learning.train import TrainOptions, train_policy


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pinned Azure ML command components.")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "convert", "train", "gate"):
        command = commands.add_parser(name)
        command.add_argument("--source", type=Path, required=True)
        command.add_argument("--tenant-id", required=True)
        command.add_argument("--owner-id", required=True)
        if name != "validate":
            command.add_argument("--output", type=Path, required=True)
            command.add_argument("--snapshot-sha256", required=True)
        if name in ("validate", "convert"):
            command.add_argument("--manifest-sha256", required=True)
        if name == "convert":
            command.add_argument("--image-size", type=int, default=224)
        if name == "train":
            command.add_argument("--model-name", required=True)
            command.add_argument("--model-version", required=True)
            for option, default in (
                ("steps", 20000),
                ("batch-size", 8),
                ("seed", 12345),
                ("chunk-size", 10),
                ("n-action-steps", 1),
                ("timeout-seconds", 3600),
            ):
                command.add_argument(f"--{option}", type=int, default=default)
        if name == "gate":
            command.add_argument("--model-name", required=True)
            command.add_argument("--model-version", required=True)
            command.add_argument("--model-sha256", required=True)
            command.add_argument("--evidence", type=Path, required=True)
            command.add_argument("--evidence-sha256", required=True)
            command.add_argument("--plan", type=Path, required=True)
            command.add_argument("--plan-sha256", required=True)
    args = parser.parse_args(argv)
    scope = Scope(args.tenant_id, args.owner_id)
    try:
        if args.command != "validate":
            verify_snapshot(Path.cwd(), args.snapshot_sha256)
        if args.command == "validate":
            dataset = validate_dataset(
                args.source,
                expected_scope=scope,
                require_live=True,
                expected_manifest_sha256=args.manifest_sha256,
            )
            print(f"Validated {len(dataset.episodes)} scoped Isaac episodes.")
        elif args.command == "convert":
            convert_dataset(
                args.source,
                args.output / "dataset",
                expected_scope=scope,
                expected_manifest_sha256=args.manifest_sha256,
                image_size=args.image_size,
            )
        elif args.command == "train":
            source = args.source / "dataset"
            train_policy(
                source,
                args.output / "model",
                expected_scope=scope,
                expected_conversion_sha256=file_digest(source / "conversion.json"),
                model_name=args.model_name,
                model_version=args.model_version,
                code_snapshot_sha256=args.snapshot_sha256,
                options=TrainOptions(
                    steps=args.steps,
                    batch_size=args.batch_size,
                    seed=args.seed,
                    chunk_size=args.chunk_size,
                    n_action_steps=args.n_action_steps,
                    timeout_seconds=args.timeout_seconds,
                ),
            )
        else:
            require(
                file_digest(args.evidence / "results.json") == sha256(args.evidence_sha256),
                "Evaluator result manifest checksum mismatch",
            )
            model = validate_model_artifact(
                args.source,
                expected_scope=scope,
                expected_model_sha256=args.model_sha256,
            )
            require(
                model["model_name"] == args.model_name
                and model["model_version"] == args.model_version,
                "Candidate model version differs from the approved Azure job",
            )
            verdict = evaluate_results(
                read_json(args.plan),
                args.evidence,
                model,
                expected_scope=scope,
                expected_model_sha256=args.model_sha256,
                expected_plan_sha256=args.plan_sha256,
            )
            args.output.mkdir(parents=True, exist_ok=True)
            write_json(args.output / "gate.json", verdict)
            if not verdict["passed"]:
                print(f"Learning release gate rejected: {verdict['reasons']}", file=sys.stderr)
                return 1
    except (ContractError, OSError) as exc:
        print(f"Learning component failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
