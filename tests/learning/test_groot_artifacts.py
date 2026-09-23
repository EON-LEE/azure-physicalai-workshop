from dataclasses import asdict

import pytest

from learning.checks.fixtures import SCOPE
from learning.checks.groot_fixtures import PROFILE, TASK, teaching
from learning.common import ContractError, file_digest, write_json


def test_v21_export_plan_preserves_all_frames_and_only_physical_features(tmp_path):
    from learning.gr00t.dataset import export_rows, prepare_export, v21_info

    root = teaching(tmp_path / "capture")
    validated = prepare_export(root, expected_scope=SCOPE, allow_test_fixture=True)
    rows = export_rows(validated.split("train")[0], episode_index=0, start_index=0, task_index=0)
    assert len(rows) == 18
    assert set(rows[0]) == {
        "observation.state",
        "action",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    }
    assert rows[1]["timestamp"] == 0.1 and rows[-1]["timestamp"] == 1.7
    assert rows[1]["action"][0] == 0.001
    assert len(rows[1]["action"]) == 9
    info = v21_info(episodes=1, frames=18, tasks=1, image_size=224)
    assert info["codebase_version"] == "v2.1"
    assert info["fps"] == 10
    assert info["data_path"] == "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    assert set(info["features"]) == set(rows[0]) | {
        "observation.images.inspection",
        "observation.images.overview",
    }


def test_safe_model_loader_checks_digest_before_any_ml_import(tmp_path):
    from learning.gr00t.artifacts import validate_model

    write_json(tmp_path / "model.json", {"arbitrary": "not-a-policy"})
    with pytest.raises(ContractError, match="checksum"):
        validate_model(tmp_path, expected_scope=SCOPE, expected_model_sha256="0" * 64)


def test_pretrained_model_cannot_be_mislabeled_franka_before_policy(tmp_path):
    from learning.gr00t.artifacts import model_contract, validate_model

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    write_json(
        checkpoint / "config.json",
        {
            "model_type": "gr00t_n1_5",
            "architectures": ["GR00T_N1_5"],
            "action_horizon": 16,
            "action_dim": 32,
            "action_head_cfg": {"action_horizon": 16, "action_dim": 32},
        },
    )
    (checkpoint / "model.safetensors").write_bytes(b"synthetic-not-weights")
    value = model_contract(
        scope=SCOPE,
        profile=PROFILE,
        task=TASK,
        checkpoint=checkpoint,
        role="pretrained",
        training=None,
    )
    write_json(tmp_path / "model.json", value)
    with pytest.raises(ContractError, match="trained"):
        validate_model(
            tmp_path,
            expected_scope=SCOPE,
            expected_model_sha256=file_digest(tmp_path / "model.json"),
            for_inference=True,
        )


def test_upstream_config_and_dependencies_are_separate_from_act():
    from learning.gr00t import MODEL_REVISION, SOURCE_COMMIT
    from learning.gr00t.franka_modality import config_spec

    assert SOURCE_COMMIT == "4af2b622892f7dcb5aae5a3fb70bcb02dc217b96"
    assert MODEL_REVISION == "869830fc749c35f34771aa5209f923ac57e4564e"
    spec = config_spec()
    assert spec["embodiment_tag"] == "new_embodiment"
    assert spec["state"] == ["state.arm", "state.fingers"]
    assert spec["action"] == ["action.arm", "action.fingers"]
    assert spec["action_indices"] == list(range(16))
    assert spec["video"] == ["video.inspection", "video.overview"]
    assert spec["control_profile"] == asdict(PROFILE) | {"servo_profile_sha256": None}
