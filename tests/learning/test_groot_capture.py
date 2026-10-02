from dataclasses import asdict, replace

import pytest

from learning import contract
from learning.capture import EpisodeWriter
from learning.checks.fixtures import JOINTS, PROVENANCE, SCOPE, frame
from learning.common import ContractError, canonical, digest


def make_profile():
    return contract.ControlProfile(servo_profile_sha256="d" * 64)


def make_source(kind="human_teleop", source_policy_sha256=None):
    return contract.DemonstrationSource(
        kind=kind,
        task_id="place-part",
        instruction="Place the part in the kitting tray.",
        goal_id="kit-tray",
        source_policy_sha256=source_policy_sha256,
    )


def sample(index, *, terminal=False):
    value = frame(index, terminal=terminal)
    controls = tuple(
        contract.AppliedControl(
            physics_step=value.physics_step + step,
            monotonic_ns=value.monotonic_ns + step * 10_000_000,
            commanded_joint_targets=JOINTS,
            commanded_joint_velocities=(0.0,) * 9,
            gravity_efforts=(1.0,) * 7 + (0.0, 0.0),
        )
        for step in range(1, 7)
    )
    return replace(value, applied_controls=controls)


def writer(root):
    return EpisodeWriter(
        root,
        dataset_id="teaching",
        scope=SCOPE,
        episode=contract.EpisodeSpec("teaching", "customer-line", "f" * 64, 17, "train"),
        provenance=replace(PROVENANCE, simulator_version="6.0.0"),
        control_profile=make_profile(),
        demonstration=make_source(),
    )


def test_v2_records_exact_actual_hold_and_demonstrator_without_changing_v1(tmp_path):
    capture = writer(tmp_path / "v2")
    capture.append(sample(0))
    capture.append(sample(1, terminal=True))
    capture.finalize()
    data = contract.validate_dataset(tmp_path / "v2", expected_scope=SCOPE)
    assert data.manifest["schema"] == "physicalai.demonstrations/v2"
    assert data.manifest["control_profile"] == asdict(make_profile())
    assert data.episodes[0].metadata["demonstration"] == asdict(make_source())
    assert data.episodes[0].frames[0]["applied_controls"][5]["physics_step"] == 6
    assert make_profile().sha256 == digest(canonical(asdict(make_profile())))


@pytest.mark.parametrize(
    "invalid",
    [
        "changed-target",
        "velocity",
        "finger-gravity",
        "missing-tick",
        "duplicate-tick",
        "future-observation",
        "nonfinite",
    ],
)
def test_v2_rejects_false_control_alignment(tmp_path, invalid):
    capture = writer(tmp_path / "invalid")
    value = sample(0)
    controls = list(value.applied_controls)
    if invalid == "changed-target":
        controls[3] = replace(controls[3], commanded_joint_targets=(0.01, *JOINTS[1:]))
    elif invalid == "velocity":
        controls[3] = replace(controls[3], commanded_joint_velocities=(0.1,) * 9)
    elif invalid == "finger-gravity":
        controls[3] = replace(controls[3], gravity_efforts=(1.0,) * 9)
    elif invalid == "missing-tick":
        controls.pop()
    elif invalid == "duplicate-tick":
        controls[3] = controls[2]
    elif invalid == "future-observation":
        value = replace(value, monotonic_ns=controls[-1].monotonic_ns)
    else:
        controls[3] = replace(controls[3], gravity_efforts=(float("nan"),) * 9)
    with pytest.raises(ContractError):
        capture.append(replace(value, applied_controls=tuple(controls)))
    assert not (tmp_path / "invalid" / "manifest.json").exists()


def test_v2_refuses_legacy_60hz_and_partial_source_identity(tmp_path):
    with pytest.raises(ContractError):
        EpisodeWriter(
            tmp_path / "decimated",
            dataset_id="bad",
            scope=SCOPE,
            episode=contract.EpisodeSpec("e", "line", "a" * 64, 1, "train"),
            provenance=PROVENANCE,
            control_profile=make_profile(),
            demonstration=make_source(),
            fps=60,
        )
    with pytest.raises(ContractError):
        make_source("learned").validate()
    with pytest.raises(ContractError):
        make_source("human_teleop", "a" * 64).validate()


def test_gr00t_franka_mapping_is_not_so101_or_combined_gripper():
    from learning.gr00t.dataset import modality_spec

    schema = modality_spec()
    for group in ("state", "action"):
        assert schema[group]["arm"]["start"] == 0
        assert schema[group]["arm"]["end"] == 7
        assert schema[group]["fingers"]["start"] == 7
        assert schema[group]["fingers"]["end"] == 9
        assert schema[group]["fingers"]["absolute"] is True
    assert set(schema["video"]) == {"inspection", "overview"}
    assert schema["annotation"] == {"human.task_description": {"original_key": "task_index"}}


def test_gr00t_export_rejects_v1_instead_of_relabeling_60hz(make_capture):
    from learning.gr00t.dataset import prepare_export

    with pytest.raises(ContractError, match="v2"):
        prepare_export(make_capture(), expected_scope=SCOPE, allow_test_fixture=True)


@pytest.mark.parametrize("case", ["arm-tracking", "finger-contact", "target-slew"])
def test_v2_labels_must_fit_the_same_ten_hz_tracking_and_slew_guard(tmp_path, case):
    capture = writer(tmp_path / "unsafe-label")
    value = sample(0)
    if case == "target-slew":
        capture.append(value)
        value = sample(1, terminal=True)
        targets = (0.1, *JOINTS[1:])
        value = replace(value, joint_positions=targets)
    elif case == "arm-tracking":
        targets = (0.1, *JOINTS[1:])
    else:
        targets = (*JOINTS[:7], 0.0, 0.0)
    controls = tuple(
        replace(control, commanded_joint_targets=targets) for control in value.applied_controls
    )
    with pytest.raises(ContractError, match="tracking|slew"):
        capture.append(replace(value, commanded_joint_targets=targets, applied_controls=controls))


pytest_plugins = ("learning.checks.pytest_fixtures",)
