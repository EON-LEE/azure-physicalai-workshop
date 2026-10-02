"""CPU audit-contract fixtures; no model inference, simulator or policy-quality claims."""

import importlib
import json
import math
import subprocess
import sys
from copy import deepcopy
from dataclasses import asdict

import pytest

from learning.checks.fixtures import JOINTS, SCOPE
from learning.common import ContractError, canonical, digest
from learning.contract import JOINT_NAMES, JOINT_UNITS
from tests.learning.test_paused_contract import profile


def audit():
    return importlib.import_module("learning.paused.train_audit")


@pytest.fixture
def documents():
    episodes = [
        {
            "episode_id": f"train-{seed}",
            "environment_id": f"scene-{seed}",
            "revision": "a" * 64,
            "seed": seed,
            "split": "train",
            "frame_count": count,
        }
        for seed, count in ((10001, 426), (10002, 425))
    ]
    raw = {
        "schema": "physicalai.demonstrations/v3",
        "purpose": "demonstration",
        "scope": asdict(SCOPE),
        "joint_names": list(JOINT_NAMES),
        "joint_units": list(JOINT_UNITS),
        "fps": 10,
        "physics_hz": 60,
        "control_profile": asdict(profile()),
        "criteria_sha256": "d" * 64,
        "frozen_plan_sha256": "e" * 64,
        "episodes": episodes,
    }
    conversion = {
        "schema": "physicalai.lerobot-conversion/v3",
        "test_only": False,
        "scope": asdict(SCOPE),
        "split": "train",
        "joint_names": list(JOINT_NAMES),
        "joint_units": list(JOINT_UNITS),
        "fps": 10,
        "physics_hz": 60,
        "image_size": 256,
        "image_transform": "rgb-bilinear-square/v1",
        "raw_manifest_sha256": digest(canonical(raw)),
        "control_profile": asdict(profile()),
        "criteria_sha256": "d" * 64,
        "frozen_plan_sha256": "e" * 64,
        "episodes": [
            {**ep, "episode_index": index, "terminated": True, "truncated": False}
            for index, ep in enumerate(episodes)
        ],
    }
    model = {
        "scope": asdict(SCOPE),
        "control_profile": asdict(profile()),
        "criteria_sha256": "d" * 64,
        "frozen_plan_sha256": "e" * 64,
        "backbone_manifest_sha256": "f" * 64,
        "training": {
            "raw_manifest_sha256": digest(canonical(raw)),
            "conversion_sha256": digest(canonical(conversion)),
            "episodes": [
                {key: ep[key] for key in ("episode_id", "environment_id", "revision", "seed")}
                for ep in episodes
            ],
        },
    }
    plan = {
        "schema": "physicalai.smolvla-train-audit-plan/v1",
        "audit_id": "train-only-audit",
        "scope": asdict(SCOPE),
        "model_sha256": digest(canonical(model)),
        "backbone_sha256": "f" * 64,
        "raw_manifest_sha256": digest(canonical(raw)),
        "conversion_sha256": digest(canonical(conversion)),
        "control_profile_sha256": profile().sha256,
        "criteria_sha256": "d" * 64,
        "frozen_plan_sha256": "e" * 64,
        "audit_script_sha256": "1" * 64,
        "static_source_sha256": "2" * 64,
        "environment_image": "unit.azurecr.io/smolvla@sha256:" + "3" * 64,
        "random_seed": 42,
        "max_wall_seconds": 1800,
        "max_report_bytes": 16 * 1024**2,
    }
    return plan, raw, conversion, model


def test_plan_and_input_contract_close_exact_train_joint_profile_and_model_bindings(documents):
    plan, raw, conversion, model = documents
    audit().validate_plan(plan)
    audit().validate_input_bindings(plan, raw, conversion, model)
    assert len(audit().sample_anchors(conversion)) == 6


@pytest.mark.parametrize(
    "change",
    [
        "heldout",
        "validation",
        "wrong_joint",
        "wrong_units",
        "rebound_model",
        "wrong_profile",
        "truncated",
        "renamed_episode",
        "wrong_pixels",
        "nontrain_plan",
    ],
)
def test_invalid_or_relabelled_training_inputs_stop_before_optional_model_import(documents, change):
    plan, raw, converted, model = deepcopy(documents)
    if change in ("heldout", "validation"):
        raw["episodes"][0]["seed"] = 30001 if change == "heldout" else 20001
    elif change == "wrong_joint":
        raw["joint_names"][-1] = "total_gripper_gap"
    elif change == "wrong_units":
        raw["joint_units"][-1] = "rad"
    elif change == "rebound_model":
        model["training"]["raw_manifest_sha256"] = "0" * 64
    elif change == "wrong_profile":
        raw["control_profile"]["hold_steps"] = 1
    elif change == "truncated":
        converted["episodes"][0].update(terminated=False, truncated=True)
    elif change == "renamed_episode":
        model["training"]["episodes"][0]["episode_id"] = "relabelled"
    elif change == "wrong_pixels":
        converted["image_transform"] = "different-resizer"
    else:
        converted["split"] = "test"
    with pytest.raises(ContractError):
        audit().validate_input_bindings(plan, raw, converted, model)


def test_fixed_samples_include_first_middle_final_for_every_original_train_episode(documents):
    _, _, converted, _ = documents
    anchors = audit().sample_anchors(converted)
    assert [(row["episode_id"], row["frame_index"]) for row in anchors] == [
        ("train-10001", 0),
        ("train-10001", 212),
        ("train-10001", 425),
        ("train-10002", 0),
        ("train-10002", 212),
        ("train-10002", 424),
    ]
    assert [row["dataset_index"] for row in anchors] == [0, 212, 425, 426, 638, 850]


def test_horizon_keeps_native_repeat_last_padding_and_all_fifty_rows():
    frames = [{"commanded_joint_targets": [*JOINTS[:8], 0.02 + i * 0.0001]} for i in range(55)]
    values, mask = audit().target_horizon(frames, 53)
    assert len(values) == len(mask) == 50
    assert mask == [False, False] + [True] * 48
    assert values[0] != values[1] and values[1:] == [values[1]] * 49
    assert len(values[0]) == 9
    assert frames[-1]["commanded_joint_targets"][-1] == 0.0254


def test_full_horizon_guards_are_not_reduced_to_consumed_row_zero():
    targets = [list(JOINTS) for _ in range(50)]
    predictions = deepcopy(targets)
    predictions[49][8] = 0.045
    value = audit().summarize_predictions(predictions, targets, [False] * 50, JOINTS)
    assert value["all_rows_within_joint_limits"] is False
    assert value["row_zero_within_joint_limits"] is True
    assert value["joint_violations"] == [
        {
            "row": 49,
            "joint_index": 8,
            "joint": JOINT_NAMES[8],
            "value": 0.045,
            "low": 0,
            "high": 0.04,
        }
    ]
    assert predictions[49][8] == 0.045


def test_padding_is_excluded_only_from_error_summary_never_the_fifty_row_guard():
    targets = [list(JOINTS) for _ in range(50)]
    predictions = deepcopy(targets)
    predictions[-1][8] = -0.001
    value = audit().summarize_predictions(predictions, targets, [False] + [True] * 49, JOINTS)
    assert value["valid_target_rows"] == 1 and value["mae_per_joint"] == [0.0] * 9
    assert value["all_rows_within_joint_limits"] is False


def test_nonfinite_predictions_remain_explicit_diagnostics_not_fabricated_finite_targets():
    targets = [list(JOINTS) for _ in range(50)]
    values = deepcopy(targets)
    values[7][8] = float("nan")
    summary = audit().summarize_predictions(values, targets, [False] * 50, JOINTS)
    assert summary["all_finite"] is False
    assert summary["mae_per_joint"] is None
    encoded = audit().json_tensor(values)
    assert encoded[7][8] == {"nonfinite": "nan"}
    assert math.isnan(values[7][8])
    json.dumps(encoded, allow_nan=False)


def test_p0_exposure_is_anchor_coverage_not_fifty_times_more_independent_observations():
    counts = [
        426,
        425,
        429,
        424,
        422,
        437,
        430,
        437,
        426,
        433,
        439,
        428,
        424,
        428,
        440,
        418,
        431,
        437,
        427,
        426,
    ]
    value = audit().training_exposure(counts, steps=1000, batch_size=1)
    assert value["frames"] == 8587 and value["anchor_samples"] == 1000
    assert value["anchor_passes"] == pytest.approx(1000 / 8587)
    assert value["completed_passes"] == 0
    assert value["anchors_with_tail_padding"] == 20 * 49
    assert value["repeated_tail_target_rows_per_pass"] == 20 * sum(range(1, 50))


def test_prospective_plan_covers_every_anchor_and_limits_full_state_transfers():
    value = audit().prospective_training([426] * 20)
    assert value["parameters"]["batch_size"] == 8
    assert value["parameters"]["max_steps"] == 20 * math.ceil(8520 / 8)
    assert value["parameters"]["checkpoint_steps"] * 10 == value["parameters"]["max_steps"]
    assert value["exposure"]["completed_passes"] == 20
    assert value["checkpoints"]["count"] == 10
    assert value["parameters"]["timeout_seconds"] < value["job_timeout_seconds"] <= 86400
    assert value["automatic_submission"] is False and value["quality_release"] is False


def test_audit_records_are_create_only_bounded_and_not_success_on_partial_io(tmp_path):
    target = tmp_path / "sample.json"
    audit().write_bounded(target, {"value": 1}, maximum=100)
    with pytest.raises(FileExistsError):
        audit().write_bounded(target, {"value": 2}, maximum=100)
    with pytest.raises(ContractError, match="bound"):
        audit().write_bounded(tmp_path / "oversize.json", {"value": "x" * 100}, maximum=30)
    assert not (tmp_path / "oversize.json").exists()
    assert json.loads(target.read_text()) == {"value": 1}


@pytest.mark.parametrize(
    ("dtype", "value"),
    [
        ("torch.float32", 0.03999999910593033),
        ("torch.float16", 0.040008544921875),
        ("torch.bfloat16", 0.0400390625),
    ],
)
def test_tensor_record_preserves_dtype_and_exact_value_before_reporting(dtype, value):
    class NativeTensorDouble:
        shape = (1, 50, 9)
        device = "cuda:0"

        def __init__(self):
            self.dtype = dtype

        def detach(self):
            return self

        def cpu(self):
            return self

        def tolist(self):
            return [[[value] * 9] * 50]

        def float(self):
            raise AssertionError("Audit must not coerce native precision.")

    record = audit().tensor_record(NativeTensorDouble())
    assert record["dtype"] == dtype and record["shape"] == [1, 50, 9]
    assert record["values"][0][0][8] == value


def test_native_preparation_compares_effective_state_not_training_batch_rank():
    from types import SimpleNamespace

    calls = []

    class Effective:
        def __eq__(self, other):
            return isinstance(other, Effective)

    def state(batch):
        calls.append(batch["shape"])
        return Effective()

    loaded = SimpleNamespace(policy=SimpleNamespace(prepare_state=state))
    tensors = SimpleNamespace(equal=lambda first, second: first == second)
    audit().check_native_state(loaded, {"shape": [1, 1, 9]}, {"shape": [1, 9]}, tensors)
    assert calls == [[1, 1, 9], [1, 9]]


def test_import_and_help_do_not_load_azure_models_or_any_motion_runtime():
    code = """
import importlib.abc, sys
class NoMotion(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        blocked = {'simulation','isaacsim','omni','carb','pxr','torch','lerobot','azure'}
        if fullname.split('.')[0] in blocked:
            raise AssertionError('Unexpected heavyweight/motion import: '+fullname)
sys.meta_path.insert(0, NoMotion())
from learning.paused import train_audit
train_audit.main(['--help'])
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    assert "TRAIN-only" in result.stdout
