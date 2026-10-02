from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from learning.common import (
    canonical,
    digest,
    finite,
    integer,
    keys,
    read_json,
    require,
    safe_path,
    sha256,
    token,
    utc,
    vector,
    write_json,
)
from learning.contract import (
    CAMERAS,
    CameraSample,
    Provenance,
    Scope,
    ValidatedDataset,
    png_dimensions,
)

PLAN_SCHEMA = "physicalai.heldout-plan/v1"
RESULT_SCHEMA = "physicalai.heldout-results/v1"
GATE_SCHEMA = "physicalai.learning-gate/v1"


@dataclass(frozen=True)
class EvaluationCase:
    episode_id: str
    environment_id: str
    revision: str
    seed: int
    defect_label: bool
    expected_destination_id: str
    expected_destination_position_m: tuple[float, float, float]
    tolerance_m: float = 0.04

    def validate(self) -> None:
        for value in (self.episode_id, self.environment_id, self.expected_destination_id):
            token(value, "evaluation case identifier")
        sha256(self.revision, "evaluation environment revision")
        integer(self.seed, "held-out seed", high=2**32 - 1)
        require(type(self.defect_label) is bool, "Evaluator defect label must be explicit")
        vector(self.expected_destination_position_m, 3, "destination position")
        require(0 < finite(self.tolerance_m, "goal tolerance") <= 0.04, "Unapproved goal tolerance")


@dataclass(frozen=True)
class GateThresholds:
    min_episodes: int = 20
    min_success_rate: float = 0.9
    min_baseline_success_rate: float = 0.9
    max_success_regression: float = 0.05
    max_latency_p95_ms: float = 80.0

    def validate(self) -> None:
        integer(self.min_episodes, "minimum held-out episodes", 20, 10000)
        for name in ("min_success_rate", "min_baseline_success_rate"):
            require(0 < finite(getattr(self, name), name) <= 1, f"Invalid {name}")
        require(0 <= finite(self.max_success_regression, "regression") <= 0.1, "Invalid regression")
        require(0 < finite(self.max_latency_p95_ms, "latency") <= 100, "Invalid latency threshold")


@dataclass(frozen=True)
class EpisodeOutcome:
    controller: str
    episode_id: str
    environment_id: str
    revision: str
    seed: int
    started_at_utc: str
    ended_at_utc: str
    elapsed_monotonic_ns: int
    physics_steps: int
    control_steps: int
    final_physics_step: int
    final_monotonic_ns: int
    final_pose_m: tuple[float, float, float]
    destination_id: str
    terminated: bool
    truncated: bool
    failure_reason: str | None
    safety_violations: tuple[str, ...]
    latencies_ms: tuple[float, ...]


def validate_runtime(runtime: dict) -> None:
    keys(runtime, {"azure_resource_id", "run_id", "provenance"}, "evaluation runtime")
    require(
        isinstance(runtime["azure_resource_id"], str)
        and re.fullmatch(
            r"/subscriptions/[0-9a-f-]{36}/resourceGroups/[A-Za-z0-9_.-]+/providers/"
            r"Microsoft\.Compute/virtualMachines/[A-Za-z0-9_.-]+",
            runtime["azure_resource_id"],
        ),
        "Evaluation requires the explicit approved Azure simulator VM resource ID",
    )
    token(runtime["run_id"], "evaluation run ID")
    values = keys(runtime["provenance"], set(Provenance.__dataclass_fields__), "runtime provenance")
    Provenance(**values).validate()


def build_evaluation_plan(
    dataset: ValidatedDataset,
    model: dict,
    cases: list[EvaluationCase],
    *,
    model_sha256: str,
    baseline_revision: str,
    expected_runtime: dict,
    thresholds: GateThresholds | None = None,
) -> dict:
    thresholds = thresholds or GateThresholds()
    thresholds.validate()
    sha256(model_sha256, "model manifest checksum")
    require(
        re.fullmatch(r"[0-9a-f]{40}", baseline_revision) is not None,
        "Pin the baseline controller commit",
    )
    validate_runtime(expected_runtime)
    require(
        model["scope"] == dataset.manifest["scope"]
        and model["raw_manifest_sha256"] == dataset.manifest_sha256,
        "Checkpoint and held-out dataset provenance differ",
    )
    expected = {
        episode.metadata["episode_id"]: episode.metadata for episode in dataset.split("test")
    }
    require(len(cases) == len(expected), "Evaluation must include the entire held-out test split")
    train_ids = {episode["episode_id"] for episode in model["training"]["episodes"]}
    train_seeds = {episode["seed"] for episode in model["training"]["episodes"]}
    seen_ids, seen_seeds = set(), set()
    for case in cases:
        case.validate()
        require(case.episode_id in expected, "Unknown held-out episode")
        require(
            case.episode_id not in seen_ids and case.seed not in seen_seeds, "Duplicate test case"
        )
        require(
            case.episode_id not in train_ids and case.seed not in train_seeds,
            "Training/held-out episode or seed leakage",
        )
        metadata = expected[case.episode_id]
        require(
            (case.environment_id, case.revision, case.seed)
            == (metadata["environment_id"], metadata["revision"], metadata["seed"]),
            "Held-out environment or seed changed",
        )
        seen_ids.add(case.episode_id)
        seen_seeds.add(case.seed)
    return {
        "schema": PLAN_SCHEMA,
        "scope": dataset.manifest["scope"],
        "raw_manifest_sha256": dataset.manifest_sha256,
        "model_sha256": model_sha256,
        "model_name": model["model_name"],
        "model_version": model["model_version"],
        "baseline_revision": baseline_revision,
        "expected_runtime": expected_runtime,
        "expected_episode_count": len(cases),
        "cases": [asdict(case) for case in cases],
        "thresholds": asdict(thresholds),
    }


def validate_plan(plan: dict, model: dict, expected_scope: Scope) -> tuple[dict, GateThresholds]:
    expected_scope.validate()
    keys(
        plan,
        {
            "schema",
            "scope",
            "raw_manifest_sha256",
            "model_sha256",
            "model_name",
            "model_version",
            "baseline_revision",
            "expected_runtime",
            "expected_episode_count",
            "cases",
            "thresholds",
        },
        "evaluation plan",
    )
    require(plan["schema"] == PLAN_SCHEMA, "Unsupported held-out plan")
    require(
        plan["scope"] == model["scope"] == asdict(expected_scope),
        "Evaluation tenant/owner mismatch",
    )
    require(
        plan["raw_manifest_sha256"] == model["raw_manifest_sha256"]
        and plan["model_name"] == model["model_name"]
        and plan["model_version"] == model["model_version"],
        "Evaluation model/dataset identity mismatch",
    )
    sha256(plan["model_sha256"])
    require(
        isinstance(plan["baseline_revision"], str)
        and re.fullmatch(r"[0-9a-f]{40}", plan["baseline_revision"]),
        "Baseline revision must be pinned",
    )
    validate_runtime(plan["expected_runtime"])
    count = integer(plan["expected_episode_count"], "expected episodes", 1, 10000)
    require(
        isinstance(plan["cases"], list) and len(plan["cases"]) == count, "Wrong plan case count"
    )
    threshold_values = keys(
        plan["thresholds"], set(GateThresholds.__dataclass_fields__), "gate thresholds"
    )
    thresholds = GateThresholds(**threshold_values)
    thresholds.validate()
    train_ids = {ep["episode_id"] for ep in model["training"]["episodes"]}
    train_seeds = {ep["seed"] for ep in model["training"]["episodes"]}
    expected, seeds = {}, set()
    for value in plan["cases"]:
        case = EvaluationCase(**keys(value, set(EvaluationCase.__dataclass_fields__), "test case"))
        case.validate()
        require(
            case.episode_id not in expected and case.seed not in seeds, "Duplicate held-out case"
        )
        require(
            case.episode_id not in train_ids and case.seed not in train_seeds,
            "Training/held-out episode or seed leakage",
        )
        expected[case.episode_id] = value
        seeds.add(case.seed)
    return expected, thresholds


class EvaluationRecorder:
    """Evaluator-only artifact writer; never pass its labels or plan into policy observations."""

    def __init__(self, root: Path, *, plan: dict, runtime: dict) -> None:
        validate_runtime(runtime)
        require(runtime == plan["expected_runtime"], "Unapproved evaluation runtime")
        require(not root.exists(), "Evaluation output already exists")
        root.mkdir(parents=True)
        self.root, self.plan, self.runtime = root, plan, runtime
        self.records: list[dict] = []
        self.seen: set[tuple[str, str]] = set()
        self.finalized = False

    def record(self, outcome: EpisodeOutcome, final_images: dict[str, CameraSample]) -> None:
        require(not self.finalized, "Evaluation already finalized")
        require(outcome.controller in ("baseline", "learned"), "Unknown controller")
        token(outcome.episode_id, "held-out episode ID")
        key = (outcome.controller, outcome.episode_id)
        require(key not in self.seen, "Duplicate evaluation result")
        require(set(final_images) == set(CAMERAS), "Final simulator images are required")
        require(
            outcome.episode_id in {case["episode_id"] for case in self.plan["cases"]},
            "Result is not in the held-out plan",
        )
        images = {}
        for camera, sample in final_images.items():
            width, height = png_dimensions(sample.png)
            rel = f"{outcome.controller}/{outcome.episode_id}/{camera}.png"
            path = safe_path(self.root, rel, must_exist=False)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(sample.png)
            images[camera] = {
                "path": rel,
                "sha256": digest(sample.png),
                "width": width,
                "height": height,
                "rendering_frame": sample.rendering_frame,
                "physics_step": sample.physics_step,
                "monotonic_ns": sample.monotonic_ns,
            }
        record = {**asdict(outcome), "final_images": images}
        write_json(self.root / f"{outcome.controller}/{outcome.episode_id}/outcome.json", record)
        self.seen.add(key)
        self.records.append(record)

    def finalize(self) -> Path:
        require(not self.finalized, "Evaluation already finalized")
        expected = {
            (controller, case["episode_id"])
            for controller in ("baseline", "learned")
            for case in self.plan["cases"]
        }
        require(self.seen == expected, "Incomplete paired evaluation; no success report is emitted")
        path = self.root / "results.json"
        write_json(
            path,
            {
                "schema": RESULT_SCHEMA,
                "scope": self.plan["scope"],
                "plan_sha256": digest(canonical(self.plan)),
                "model_sha256": self.plan["model_sha256"],
                "baseline_revision": self.plan["baseline_revision"],
                "runtime": self.runtime,
                "episodes": self.records,
            },
        )
        self.finalized = True
        return path


def _validate_outcome(record: dict, case: dict, root: Path, interval: int) -> bool:
    keys(record, set(EpisodeOutcome.__dataclass_fields__) | {"final_images"}, "episode result")
    require(record["controller"] in ("baseline", "learned"), "Unknown evaluation controller")
    require(
        all(
            record[name] == case[name]
            for name in ("episode_id", "environment_id", "revision", "seed")
        ),
        "Evaluation case identity mismatch",
    )
    require(
        utc(record["ended_at_utc"]) > utc(record["started_at_utc"]), "Invalid episode UTC interval"
    )
    duration = integer(record["elapsed_monotonic_ns"], "episode duration", 1)
    integer(record["physics_steps"], "physics steps", 1)
    integer(record["final_physics_step"], "final physics step", record["physics_steps"])
    integer(record["final_monotonic_ns"], "final timestamp", 1)
    controls = integer(record["control_steps"], "control steps", high=100000)
    minimum_steps = controls * interval if record["controller"] == "learned" else controls
    require(record["physics_steps"] >= minimum_steps, "Control count exceeds actual physics steps")
    latencies = record["latencies_ms"]
    require(
        isinstance(latencies, (list, tuple)) and len(latencies) == controls,
        "Latency sample count must equal actual control steps",
    )
    require(
        all(finite(value, "latency") >= 0 for value in latencies)
        and sum(latencies) * 1e6 <= duration,
        "Invalid latency measurements",
    )
    require(
        type(record["terminated"]) is bool
        and type(record["truncated"]) is bool
        and record["terminated"] != record["truncated"],
        "Evaluation episode did not end exactly once",
    )
    require(
        record["failure_reason"] is None
        or (isinstance(record["failure_reason"], str) and 0 < len(record["failure_reason"]) <= 512),
        "Invalid failure reason",
    )
    require(
        controls > 0 or record["failure_reason"] is not None, "Zero-action success is forbidden"
    )
    require(
        isinstance(record["safety_violations"], (list, tuple))
        and all(isinstance(item, str) and item for item in record["safety_violations"]),
        "Invalid safety evidence",
    )
    pose = vector(record["final_pose_m"], 3, "final object pose")
    token(record["destination_id"], "actual destination")
    for camera, image in keys(record["final_images"], set(CAMERAS), "final images").items():
        keys(
            image,
            {
                "path",
                "sha256",
                "width",
                "height",
                "rendering_frame",
                "physics_step",
                "monotonic_ns",
            },
            "final image",
        )
        require(
            image["path"] == f"{record['controller']}/{record['episode_id']}/{camera}.png",
            "Cross-episode evaluation image",
        )
        payload = safe_path(root, image["path"]).read_bytes()
        require(digest(payload) == sha256(image["sha256"]), "Evaluation image checksum mismatch")
        require(
            png_dimensions(payload) == (image["width"], image["height"]), "Wrong evidence shape"
        )
        integer(image["rendering_frame"], "final render frame")
        integer(image["physics_step"], "final image physics step")
        integer(image["monotonic_ns"], "final image timestamp", 1)
        require(
            0 <= record["final_physics_step"] - image["physics_step"] < interval
            and 0 <= record["final_monotonic_ns"] - image["monotonic_ns"] <= 1_000_000_000,
            "Final pose evidence uses a stale/future image",
        )
    inside = all(
        abs(actual - target) <= case["tolerance_m"]
        for actual, target in zip(pose, case["expected_destination_position_m"], strict=True)
    )
    return (
        record["terminated"]
        and not record["truncated"]
        and record["failure_reason"] is None
        and not record["safety_violations"]
        and record["destination_id"] == case["expected_destination_id"]
        and inside
    )


def evaluate_results(
    plan: dict,
    result_root: Path,
    model: dict,
    *,
    expected_scope: Scope,
    expected_model_sha256: str,
    expected_plan_sha256: str,
) -> dict:
    require(
        digest(canonical(plan)) == sha256(expected_plan_sha256), "Evaluation plan checksum mismatch"
    )
    require(plan["model_sha256"] == sha256(expected_model_sha256), "Evaluation checkpoint changed")
    cases, thresholds = validate_plan(plan, model, expected_scope)
    results = keys(
        read_json(safe_path(result_root, "results.json"), max_bytes=64 * 1024 * 1024),
        {
            "schema",
            "scope",
            "plan_sha256",
            "model_sha256",
            "baseline_revision",
            "runtime",
            "episodes",
        },
        "evaluation results",
    )
    require(results["schema"] == RESULT_SCHEMA, "Unsupported evaluation result format")
    require(
        results["scope"] == plan["scope"]
        and results["plan_sha256"] == expected_plan_sha256
        and results["model_sha256"] == expected_model_sha256
        and results["baseline_revision"] == plan["baseline_revision"]
        and results["runtime"] == plan["expected_runtime"],
        "Unbound evaluation evidence",
    )
    validate_runtime(results["runtime"])
    require(
        isinstance(results["episodes"], list) and len(results["episodes"]) == 2 * len(cases),
        "Exact baseline and learned episode counts are required",
    )
    seen, failures = set(), []
    counts = {"baseline": 0, "learned": 0}
    latencies = {"baseline": [], "learned": []}
    safety_count = 0
    interval = model["observation"]["physics_hz"] // model["observation"]["fps"]
    for record in results["episodes"]:
        require(isinstance(record, dict), "Invalid episode record")
        key = (record.get("controller"), record.get("episode_id"))
        require(key not in seen and key[1] in cases, "Duplicate/unknown evaluation episode")
        passed = _validate_outcome(record, cases[key[1]], result_root, interval)
        seen.add(key)
        counts[key[0]] += int(passed)
        latencies[key[0]].extend(record["latencies_ms"])
        safety_count += len(record["safety_violations"])
        if not passed:
            failures.append(
                {
                    "controller": key[0],
                    "episode_id": key[1],
                    "reason": record["failure_reason"] or "destination_pose_predicate_failed",
                    "final_pose_m": record["final_pose_m"],
                    "destination_id": record["destination_id"],
                    "truncated": record["truncated"],
                    "safety_violations": record["safety_violations"],
                }
            )
    expected_pairs = {(controller, ep) for controller in counts for ep in cases}
    require(seen == expected_pairs, "Incomplete paired evaluation")
    rates = {controller: count / len(cases) for controller, count in counts.items()}
    p95 = {}
    for controller, values in latencies.items():
        p95[controller] = sorted(values)[math.ceil(len(values) * 0.95) - 1] if values else None
    reasons = []
    provenance = results["runtime"]["provenance"]
    if provenance["source_kind"] != "isaac_sim" or provenance["capture_host"] != "azure_gpu":
        reasons.append("no_live_azure_gpu_episodes")
    if (
        model["training"]["test_only"]
        or model["training"]["device"] != "cuda"
        or not model["training"].get("azureml_run_id")
        or not model["training"].get("gpu_model")
    ):
        reasons.append("no_production_gpu_trained_checkpoint")
    if len(cases) < thresholds.min_episodes:
        reasons.append("insufficient_heldout_episodes")
    if rates["learned"] < thresholds.min_success_rate:
        reasons.append("learned_success_rate_below_threshold")
    if rates["baseline"] < thresholds.min_baseline_success_rate:
        reasons.append("baseline_success_rate_below_threshold")
    if rates["baseline"] - rates["learned"] > thresholds.max_success_regression:
        reasons.append("learned_policy_regresses_against_paired_baseline")
    if p95["learned"] is None or p95["learned"] > thresholds.max_latency_p95_ms:
        reasons.append("learned_policy_latency_above_threshold")
    if safety_count:
        reasons.append("safety_violation")
    return {
        "schema": GATE_SCHEMA,
        "passed": not reasons,
        "learning_quality_verified": not reasons,
        "reasons": reasons,
        "model_sha256": expected_model_sha256,
        "plan_sha256": expected_plan_sha256,
        "baseline_revision": plan["baseline_revision"],
        "episode_count_per_controller": len(cases),
        "total_episode_count": 2 * len(cases),
        "success_counts": counts,
        "success_rates": rates,
        "latency_p95_ms": p95,
        "safety_violation_count": safety_count,
        "failures": failures,
        "thresholds": asdict(thresholds),
        "runtime": results["runtime"],
    }
