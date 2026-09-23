import pytest

from learning.checks.fixtures import JOINTS
from learning.common import ContractError


def test_n17_contract_uses_actual_artifact_horizon_not_stale_n15_or_readme_defaults():
    from learning.gr00t.n17 import ACTION_HORIZON, MODEL_REVISION, SOURCE_COMMIT, modality_config

    assert ACTION_HORIZON == 40
    assert MODEL_REVISION == "2fc962b973bccdd5d8ce4f67cc63b264d6886495"
    assert SOURCE_COMMIT == "23ace64f17aa5015259b8609d371eb61a357c776"
    modalities = modality_config()
    assert modalities["action"]["delta_indices"] == list(range(40))
    assert modalities["action"]["modality_keys"] == ["arm", "fingers"]
    assert [item["rep"] for item in modalities["action"]["action_configs"]] == ["ABSOLUTE"] * 2
    assert modalities["video"]["modality_keys"] == ["inspection", "overview"]


def test_n17_unpacks_tuple_batched_physical_outputs_without_relabeling_n15():
    from learning.gr00t.n17 import physical_actions

    actions = {"arm": [[list(JOINTS[:7])] * 40], "fingers": [[list(JOINTS[7:])] * 40]}
    assert physical_actions((actions, {})) == (JOINTS,) * 40
    for invalid in (
        actions,
        ({"action.arm": [list(JOINTS[:7])] * 16, "action.fingers": [list(JOINTS[7:])] * 16}, {}),
        ({"arm": [[list(JOINTS[:7])] * 16], "fingers": [[list(JOINTS[7:])] * 16]}, {}),
    ):
        with pytest.raises(ContractError):
            physical_actions(invalid)


def test_n17_rejects_wrong_generation_checkpoint_and_unreviewed_processor():
    from learning.gr00t.n17 import validate_generation

    expected = {
        "model_type": "Gr00tN1d7",
        "architectures": ["Gr00tN1d7"],
        "action_horizon": 40,
        "max_action_dim": 132,
        "max_state_dim": 132,
        "model_name": "nvidia/Cosmos-Reason2-2B",
    }
    validate_generation(expected, {"processor_class": "Gr00tN1d7Processor"})
    for bad in (
        {**expected, "model_type": "gr00t_n1_5"},
        {**expected, "action_horizon": 16},
        {**expected, "auto_map": {"AutoModel": "external.Evil"}},
    ):
        with pytest.raises(ContractError):
            validate_generation(bad, {"processor_class": "Gr00tN1d7Processor"})


def test_n17_weight_use_is_explicitly_blocked_while_license_sources_conflict():
    from learning.gr00t.n17 import authorize_weight_use

    with pytest.raises(ContractError, match="license"):
        authorize_weight_use()
