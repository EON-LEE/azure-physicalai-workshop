"""CPU authority/serialization checks for an explicitly selected TRAIN-only mask recipe."""

from copy import deepcopy
from dataclasses import asdict

import pytest

from learning.common import ContractError, canonical, digest
from learning.contract import Scope
from learning.smolvla import checkpoint_runner as runner
from tests.learning.test_checkpoint_job_plan import checkpoint_config
from tests.learning.test_training_checkpoints import binding, native_bundle, origin, seal

RECIPE = {
    "schema": "physicalai.smolvla-training-recipe/v1",
    "id": "native-temporal-padding-alias-v1",
    "boundary": "post-preprocessor-update-policy",
    "canonical_mask": "action_is_pad",
    "native_mask": "actions_id_pad",
}
POLICY = "physicalai.smolvla-checkpointing/v2"
CONTEXT = "physicalai.smolvla-checkpoint-run/v3"


def recipe_config(*, resume=False):
    value = checkpoint_config(resume=resume)
    value["checkpointing"].update(schema=POLICY, training_recipe=deepcopy(RECIPE))
    return value


def test_corrected_recipe_requires_explicit_versioned_checkpoint_selection():
    value = recipe_config()
    runner.validate_policy(value["checkpointing"], parameters=value["parameters"])
    assert runner.checkpoint_recipe(value["checkpointing"]) == RECIPE
    assert runner.checkpoint_recipe(checkpoint_config()["checkpointing"]) is None


@pytest.mark.parametrize(
    "change", ["v1-with-recipe", "v2-missing", "unknown", "extra", "renamed-mask"]
)
def test_unknown_or_implicit_recipe_cannot_change_training(change):
    value = recipe_config()
    if change == "v1-with-recipe":
        value["checkpointing"]["schema"] = runner.POLICY_SCHEMA
    elif change == "v2-missing":
        value["checkpointing"].pop("training_recipe")
    elif change == "unknown":
        value["checkpointing"]["training_recipe"]["id"] = "different-loss"
    elif change == "extra":
        value["checkpointing"]["training_recipe"]["normalize_valid_steps"] = True
    else:
        value["checkpointing"]["training_recipe"]["native_mask"] = "action_is_pad"
    with pytest.raises(ContractError):
        runner.validate_policy(value["checkpointing"], parameters=value["parameters"])


def make_binding(config, monkeypatch):
    monkeypatch.setattr(runner, "training_code_sha256", lambda value: "4" * 64)
    return runner.make_binding(
        config,
        {"scope": binding()["scope"], "weights_sha256": "d" * 64},
        {"raw_manifest_sha256": "a" * 64},
        "b" * 64,
        {"cpu_test_only": True},
    )


def test_recipe_changes_bound_training_config_not_numeric_candidate_options(monkeypatch):
    legacy_config, corrected_config = checkpoint_config(), recipe_config()
    legacy, corrected = (
        make_binding(value, monkeypatch) for value in (legacy_config, corrected_config)
    )
    assert legacy["training_config_sha256"] != corrected["training_config_sha256"]
    assert {**legacy, "training_config_sha256": corrected["training_config_sha256"]} == corrected
    from learning.smolvla.train import TrainOptions

    assert digest(canonical(asdict(TrainOptions(**corrected_config["parameters"])))) == digest(
        canonical(asdict(TrainOptions(**legacy_config["parameters"])))
    )


def test_new_same_dataset_p0_weights_only_plan_does_not_require_additional_p1():
    from learning.smolvla.azure import validate_config
    from tests.learning.test_direct_command_jobs import command_config

    value = command_config()
    value["run_id"] = "p0-refinement-new"
    value["parameters"]["resume_mode"] = "weights_only"
    value["checkpointing"].update(schema=POLICY, training_recipe=RECIPE)
    assert value["checkpointing"]["resume"] is None
    assert "training_cohort" not in value
    validate_config(value)


def test_new_recipe_cannot_resume_old_full_state_but_allows_explicit_weights_only(tmp_path):
    old = binding()
    root = native_bundle(tmp_path / "old")
    checksum = seal(root)
    config = recipe_config(resume=True)
    config["checkpointing"]["resume"].update(
        checkpoint_sha256=checksum,
        source_azure_job_id=origin()["azure_job_id"],
        source_azure_pipeline_job_id=origin()["azure_pipeline_job_id"],
    )
    current = {
        **origin(),
        "azure_job_id": origin()["azure_job_id"] + "-new",
        "azure_pipeline_job_id": origin()["azure_pipeline_job_id"] + "-new",
        "specification_sha256": "9" * 64,
    }
    changed = {
        **old,
        "training_code_sha256": "6" * 64,
        "training_config_sha256": "7" * 64,
        "runtime_sha256": "8" * 64,
    }
    value = runner.validate_resume_checkpoint(
        root, config=config, expected_binding=changed, current_origin=current
    )
    assert value["binding"] == old
    config["parameters"]["resume_mode"] = "full_state"
    with pytest.raises(ContractError, match="binding"):
        runner.validate_resume_checkpoint(
            root, config=config, expected_binding=changed, current_origin=current
        )


def test_recipe_context_is_closed_and_legacy_context_does_not_implicitly_enable_it():
    assert runner.context_recipe({"schema": runner.CONTEXT_SCHEMA}) is None
    assert runner.context_recipe({"schema": CONTEXT, "training_recipe": RECIPE}) == RECIPE
    for value in (
        {"schema": runner.CONTEXT_SCHEMA, "training_recipe": RECIPE},
        {"schema": CONTEXT},
        {"schema": CONTEXT, "training_recipe": {**RECIPE, "id": "unknown"}},
    ):
        with pytest.raises(ContractError):
            runner.context_recipe(value)


def test_selected_recipe_context_is_explicit_without_candidate_parameter_changes():
    from learning.smolvla import train

    fields = runner.recipe_context_fields(RECIPE)
    assert fields == {"schema": CONTEXT, "training_recipe": RECIPE}
    assert runner.recipe_context_fields(None) == {"schema": runner.CONTEXT_SCHEMA}
    assert not {"training_recipe", "recipe"} & asdict(train.TrainOptions()).keys()


@pytest.mark.parametrize("value", [None, [], "v2", True])
def test_nonobject_recipe_policy_has_a_contract_error(value):
    with pytest.raises(ContractError):
        runner.checkpoint_recipe(value)


def test_actual_training_wrapper_routes_recipe_into_hashed_child_context(tmp_path, monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace

    from learning.common import file_digest, read_json
    from learning.gr00t import azure as jobs
    from learning.paused.contract import PausedControlProfile
    from learning.smolvla import azure, train

    value = recipe_config()
    value["parameters"]["resume_mode"] = "weights_only"
    data, parent_root, backbone = (tmp_path / name for name in ("data", "parent", "backbone"))
    data.mkdir()
    (data / "conversion.json").write_bytes(b'{"cpu_fixture":true}')
    profile = PausedControlProfile("f" * 64)
    task = {"task_id": "place", "instruction": "Place the part.", "goal_id": "rejected"}
    parent = {
        "role": "candidate",
        "scope": binding()["scope"],
        "control_profile": asdict(profile),
        "task": task,
        "backbone_manifest_sha256": "a" * 64,
        "weights_sha256": "d" * 64,
        "training": {"cumulative_optimizer_steps": 1000},
    }
    converted = {
        "test_only": False,
        "control_profile": asdict(profile),
        "raw_manifest_sha256": "a" * 64,
        "episodes": [{"demonstration": task}],
    }
    runtime_calls, native_calls, seal_calls = [], [], []
    runtime = {"cpu_fixture": True}
    monkeypatch.setattr(azure, "job_deadline", lambda config: SimpleNamespace(check=lambda: 60))
    monkeypatch.setattr(train, "require_lerobot", lambda: None)
    monkeypatch.setattr(train, "validate_backbone", lambda *args, **kwargs: backbone)
    monkeypatch.setattr(train, "prepare_seed", lambda *args: args[-1].mkdir())
    monkeypatch.setattr(train, "parameter_fingerprint", lambda path: "a" * 64)
    monkeypatch.setattr(runner, "training_code_sha256", lambda config: "6" * 64)

    def native_runtime(**kwargs):
        runtime_calls.append(kwargs)
        return runtime

    monkeypatch.setattr(runner, "native_runtime", native_runtime)
    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: True, get_device_name=lambda: "CPU double")
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(
        jobs,
        "running_job_binding",
        lambda *args, **kwargs: {
            "azure_job_id": "new-pipeline",
            "azure_component_job_id": "new-component",
            "specification_sha256": "8" * 64,
        },
    )

    def subprocess_run(command, **kwargs):
        native_calls.append((command, kwargs))
        return SimpleNamespace(returncode=0)

    def seal_checkpoint(*args, **kwargs):
        seal_calls.append(kwargs)
        return parent

    monkeypatch.setattr(train.subprocess, "run", subprocess_run)
    monkeypatch.setattr(train, "_seal_checkpoint", seal_checkpoint)
    output = tmp_path / "out"
    train._run_training(
        data,
        parent_root,
        backbone,
        output,
        scope=Scope(**binding()["scope"]),
        parent_model_sha256="c" * 64,
        conversion_sha256=file_digest(data / "conversion.json"),
        code_snapshot_sha256="9" * 64,
        config=value,
        client=object(),
        options=train.TrainOptions(**value["parameters"]),
        parent_validator=lambda *a, **k: parent,
        conversion_validator=lambda *a: converted,
        profile_type=PausedControlProfile,
    )
    assert runtime_calls == [
        {"environment_image": value["environment_image"], "training_recipe": RECIPE}
    ]
    child = read_json(output / "checkpoint-run.json")
    assert child["schema"] == CONTEXT and child["training_recipe"] == RECIPE
    assert child["binding"]["training_code_sha256"] == "6" * 64
    command = native_calls[0][0]
    assert command[2] == "learning.smolvla.checkpoint_runner"
    assert f"--checkpoint-context-sha256={file_digest(output / 'checkpoint-run.json')}" in command
    original_fields = read_json(output / "training-context.json")
    assert original_fields["code_snapshot_sha256"] == "9" * 64
    assert "training_recipe" not in original_fields
    assert "training_recipe" not in seal_calls[0]
