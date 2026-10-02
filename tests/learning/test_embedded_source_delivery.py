import base64
import copy
import json
import os
import shlex
import sys
from pathlib import Path

import pytest

from learning.common import ContractError, canonical, digest, read_json
from learning.smolvla import azure
from tests.learning.test_checkpoint_job_plan import checkpoint_config


def embedded_config(static_sha256="a" * 64):
    value = checkpoint_config()
    value["source_delivery"] = {
        "schema": "physicalai.smolvla-source-delivery/v1",
        "mode": "image_embedded",
        "static_sha256": static_sha256,
    }
    return value


@pytest.fixture
def package(tmp_path):
    from learning.smolvla.embedded_source import prepare_context

    root = tmp_path / "image-context"
    receipt = prepare_context(root)
    return root, receipt


def materialization_arguments(package, tmp_path):
    root, receipt = package
    config = embedded_config(receipt["static_sha256"])
    payload = canonical(config) + b"\n"
    files = read_json(root / "static-manifest.json")["files"]
    snapshot_sha = digest(canonical({**files, "run-config.json": digest(payload)}))
    return {
        "source_root": root / "source",
        "manifest_path": root / "static-manifest.json",
        "config_base64": base64.b64encode(payload).decode(),
        "config_sha256": digest(payload),
        "snapshot_sha256": snapshot_sha,
        "static_sha256": receipt["static_sha256"],
        "destination": tmp_path / "materialized",
    }, config


def test_static_context_has_no_runtime_config_or_self_referenced_image(package):
    from learning.smolvla.embedded_source import static_code_files

    root, receipt = package
    manifest = read_json(root / "static-manifest.json")
    assert set(manifest["files"]) == set(static_code_files())
    assert manifest["sha256"] == receipt["static_sha256"] == digest(canonical(manifest["files"]))
    assert not (root / "source" / "run-config.json").exists()
    assert not (root / "source" / "snapshot.json").exists()
    assert (root / "source-delivery.json").is_file()
    assert read_json(root / "source-delivery.json")["static_sha256"] == receipt["static_sha256"]
    assert receipt["runtime_config_embedded"] is False
    assert receipt["image_built"] is False
    dockerfile = (root / "Dockerfile").read_text()
    assert "sha256:441f2a33a8bb0c534a10ad7d56c4f7be611dccde39e8ee0b99610b4a7a75e8f9" in dockerfile
    assert "pip " not in dockerfile and "uv sync" not in dockerfile
    assert "ENTRYPOINT" not in dockerfile
    assert "COPY source/ /opt/physicalai/source/" in dockerfile


def test_explicit_embedded_commands_omit_code_without_altering_native_job_controls(
    package, tmp_path
):
    from learning.smolvla.image_bootstrap import BOOTSTRAP_PATH, NATIVE_PYTHON

    _, receipt = package
    config = embedded_config(receipt["static_sha256"])
    plan_dir = tmp_path / "plan"
    checksum = azure.create_plan(config, plan_dir, deterministic_job_name="embedded-p0")
    plan = azure.read_plan(plan_dir)
    assert plan["plan_sha256"] == checksum
    job = read_json(plan_dir / "job.json")
    assert set(job["jobs"]) == {"convert", "train"}
    assert "code" not in job
    for name, command in job["jobs"].items():
        assert "code" not in command
        tokens = shlex.split(command["command"])
        assert tokens[:4] == [NATIVE_PYTHON, "-I", "-B", BOOTSTRAP_PATH]
        assert tokens[4:6] == ["run", "export" if name == "convert" else "train"]
        payload = base64.b64decode(tokens[tokens.index("--config-base64") + 1], validate=True)
        assert payload == (plan_dir / "code" / "run-config.json").read_bytes()
        assert digest(payload) == tokens[tokens.index("--config-sha256") + 1]
        assert tokens[tokens.index("--snapshot-sha256") + 1] == plan["snapshot_sha256"]
        assert command["environment"]["image"] == config["environment_image"]
        assert command["compute"] == "azureml:" + config["compute"]
        assert command["resources"]["instance_count"] == 1
        assert command["identity"]["client_id"] == config["managed_identity_client_id"]
        assert command["tags"]["job_deadline_utc"] == config["job_deadline_utc"]
        assert command["tags"]["static_source_sha256"] == receipt["static_sha256"]
        assert command["tags"]["source_delivery"] == "image_embedded"
        assert len(command["command"].encode()) <= 32768
    assert (
        sum(node["limits"]["timeout"] for node in job["jobs"].values())
        == config["parameters"]["timeout_seconds"]
    )


def test_legacy_delivery_still_uses_exact_existing_local_code_shape():
    config = checkpoint_config()
    job = azure.build_job(config, "a" * 64, "legacy-job")
    assert all(step["code"] == "./code" for step in job["jobs"].values())
    assert all(
        "python -m learning.paused.components" in step["command"] for step in job["jobs"].values()
    )
    assert "source_delivery" not in job["tags"] and "static_source_sha256" not in job["tags"]


@pytest.mark.parametrize(
    "change",
    [
        "unknown-mode",
        "missing-sha",
        "extra-key",
        "legacy-schema",
        "realtime",
        "compare",
    ],
)
def test_partial_unknown_or_wrong_mode_delivery_is_rejected(change):
    config = embedded_config()
    if change == "unknown-mode":
        config["source_delivery"]["mode"] = "download_arbitrary_url"
    elif change == "missing-sha":
        config["source_delivery"].pop("static_sha256")
    elif change == "extra-key":
        config["source_delivery"]["bootstrap_path"] = "/tmp/unreviewed.py"
    elif change == "legacy-schema":
        config["schema"] = "physicalai.smolvla-azure/v1"
    elif change == "realtime":
        for key in azure.PAUSED_FIELDS:
            config.pop(key)
    else:
        config["kind"] = "compare"
    with pytest.raises(ContractError):
        azure.validate_config(config)


def test_runtime_config_can_change_after_image_packaging_without_static_hash_cycle(
    package, tmp_path
):
    _, receipt = package
    config = embedded_config(receipt["static_sha256"])
    first = tmp_path / "first"
    second = tmp_path / "second"
    azure.create_plan(config, first, deterministic_job_name="first")
    changed = copy.deepcopy(config)
    changed["environment_image"] = (
        config["environment_image"].split("@sha256:")[0] + "@sha256:" + "9" * 64
    )
    changed["job_deadline_utc"] = "2030-01-01T01:00:00Z"
    azure.create_plan(changed, second, deterministic_job_name="second")
    old, new = azure.read_plan(first), azure.read_plan(second)
    assert old["snapshot_sha256"] != new["snapshot_sha256"]
    assert old["config"]["source_delivery"] == new["config"]["source_delivery"]
    assert old["plan_sha256"] != new["plan_sha256"]


def test_stale_image_source_binding_is_rejected_before_plan_directory_creation(tmp_path):
    with pytest.raises((ContractError, ValueError), match="static|source"):
        azure.create_plan(
            embedded_config("0" * 64), tmp_path / "stale", deterministic_job_name="stale"
        )
    assert not (tmp_path / "stale").exists()


def test_materialized_tree_matches_native_snapshot_even_when_aml_overrides_cwd(
    package, tmp_path, monkeypatch
):
    from learning.smolvla.image_bootstrap import materialize_source

    arguments, expected = materialization_arguments(package, tmp_path)
    unrelated = tmp_path / "azureml-working-directory"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)
    actual = materialize_source(**arguments)
    assert actual == expected
    root = arguments["destination"]
    assert (root / "run-config.json").read_bytes() == canonical(expected) + b"\n"
    azure.verify_code(root, arguments["snapshot_sha256"], config=expected)
    assert not (root / "static-manifest.json").exists()
    assert not (unrelated / "run-config.json").exists()


@pytest.mark.parametrize(
    "change", ["config", "config-sha", "snapshot", "static", "expired", "oversized", "noncanonical"]
)
def test_config_or_authority_tampering_rejects_before_source_materialization(
    package, tmp_path, change
):
    from learning.smolvla.image_bootstrap import ImageSourceError, materialize_source

    arguments, config = materialization_arguments(package, tmp_path)
    if change == "config":
        config["parameters"]["batch_size"] += 1
        arguments["config_base64"] = base64.b64encode(canonical(config) + b"\n").decode()
    elif change == "config-sha":
        arguments["config_sha256"] = "0" * 64
    elif change == "snapshot":
        arguments["snapshot_sha256"] = "0" * 64
    elif change == "static":
        arguments["static_sha256"] = "0" * 64
    elif change == "expired":
        config["job_deadline_utc"] = "2020-01-01T00:00:00Z"
        payload = canonical(config) + b"\n"
        arguments["config_base64"] = base64.b64encode(payload).decode()
        arguments["config_sha256"] = digest(payload)
    elif change == "oversized":
        arguments["config_base64"] = "A" * 100000
    else:
        payload = json.dumps(config, indent=2).encode()
        arguments["config_base64"] = base64.b64encode(payload).decode()
        arguments["config_sha256"] = digest(payload)
    with pytest.raises(ImageSourceError):
        materialize_source(**arguments)
    assert not arguments["destination"].exists()


@pytest.mark.parametrize(
    "change", ["missing", "modified", "extra", "symlink", "manifest-extra", "unsafe-path"]
)
def test_static_payload_is_bounded_exact_and_symlink_free(package, tmp_path, change):
    from learning.smolvla.image_bootstrap import ImageSourceError, materialize_source

    arguments, _ = materialization_arguments(package, tmp_path)
    source = arguments["source_root"]
    selected = source / "learning" / "common.py"
    if change == "missing":
        selected.unlink()
    elif change == "modified":
        selected.write_bytes(selected.read_bytes() + b"\n# not reviewed\n")
    elif change == "extra":
        (source / "unreviewed.py").write_text("raise SystemExit(0)")
    elif change == "symlink":
        if os.name != "posix":
            pytest.skip("Symlink production contract runs in Linux images")
        selected.unlink()
        selected.symlink_to(Path(__file__).resolve())
    else:
        manifest = read_json(arguments["manifest_path"])
        manifest["files"]["../outside.py" if change == "unsafe-path" else "learning/extra.py"] = (
            "a" * 64
        )
        arguments["manifest_path"].write_bytes(canonical(manifest) + b"\n")
    with pytest.raises(ImageSourceError):
        materialize_source(**arguments)
    assert not arguments["destination"].exists()


def test_exec_only_runs_the_same_native_component_from_materialized_root(
    package, tmp_path, monkeypatch
):
    from learning.smolvla.image_bootstrap import execute_component, materialize_source

    arguments, _ = materialization_arguments(package, tmp_path)
    config = materialize_source(**arguments)
    root = arguments["destination"]
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHONHOME", "/wrong-isaac-python")
    monkeypatch.setenv("PYTHONPATH", "/wrong/unreviewed")
    observed = []

    def execute(executable, args, env):
        observed.append((executable, args, env, Path.cwd()))
        assert (Path.cwd() / "run-config.json").is_file()

    monkeypatch.setattr(os, "execve", execute)
    execute_component(
        root,
        config,
        command="train",
        snapshot_sha256=arguments["snapshot_sha256"],
        input_path=tmp_path / "data",
        output_path=tmp_path / "output",
        parent_path=tmp_path / "parent",
        backbone_path=tmp_path / "backbone",
    )
    executable, args, env, cwd = observed[0]
    assert executable == sys.executable
    assert args[1:4] == ["-B", "-m", "learning.paused.components"]
    assert args[4] == "train"
    assert "--run-component" not in args
    assert cwd == root
    assert env["PYTHONPATH"] == str(root)
    assert "PYTHONHOME" not in env
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert "--snapshot-sha256" in args


@pytest.mark.parametrize("command", ["compare", "bootstrap", "python", "exec"])
def test_arbitrary_component_fallback_is_forbidden(package, tmp_path, monkeypatch, command):
    from learning.smolvla.image_bootstrap import (
        ImageSourceError,
        execute_component,
        materialize_source,
    )

    arguments, _ = materialization_arguments(package, tmp_path)
    config = materialize_source(**arguments)
    monkeypatch.setattr(os, "execve", lambda *args: pytest.fail("Unapproved command executed"))
    with pytest.raises(ImageSourceError):
        execute_component(
            arguments["destination"],
            config,
            command=command,
            snapshot_sha256=arguments["snapshot_sha256"],
            input_path=tmp_path / "data",
            output_path=tmp_path / "output",
        )


def test_embedded_full_state_resume_keeps_checkpoint_inputs_and_omits_code(package, tmp_path):
    _, receipt = package
    config = checkpoint_config(resume=True)
    config["parameters"]["resume_mode"] = "full_state"
    config["source_delivery"] = embedded_config(receipt["static_sha256"])["source_delivery"]
    directory = tmp_path / "resume"
    azure.create_plan(config, directory, deterministic_job_name="new-authorized-embedded-resume")
    job = read_json(directory / "job.json")
    assert set(job["jobs"]) == {"train"}
    assert "code" not in job["jobs"]["train"]
    assert "--resume-checkpoint" in job["jobs"]["train"]["command"]
    assert job["jobs"]["train"]["inputs"]["dataset"] == "${{parent.inputs.converted_dataset}}"


@pytest.mark.parametrize("field", ["source_delivery", "static_source_sha256"])
def test_owned_job_status_rejects_different_source_delivery_without_mutation(package, field):
    from types import SimpleNamespace

    from learning.gr00t.azure import job_tags, workspace_id

    _, receipt = package
    config = embedded_config(receipt["static_sha256"])
    job = SimpleNamespace(
        name="embedded-bound",
        id=workspace_id(config) + "/jobs/embedded-bound",
        status="Queued",
        tags=job_tags(config, "b" * 64, policy_type="smolvla"),
    )
    jobs = azure.PolicyJobs(
        SimpleNamespace(jobs=SimpleNamespace(get=lambda name: job)), config, storage_client=None
    )
    assert jobs.status(job.name)["azure_status"] == "Queued"
    job.tags[field] = "unreviewed"
    with pytest.raises(ContractError, match="source delivery"):
        jobs.status(job.name)


@pytest.mark.parametrize("target", ["parent", "component"])
def test_actual_component_binding_requires_static_source_tags_on_both_jobs(
    package, monkeypatch, target
):
    from types import SimpleNamespace

    from learning.gr00t import azure as shared

    _, receipt = package
    config = embedded_config(receipt["static_sha256"])
    tags = shared.job_tags(config, "b" * 64, policy_type="smolvla")
    parent = SimpleNamespace(
        name="pipeline",
        id=shared.workspace_id(config) + "/jobs/pipeline",
        tags=dict(tags),
        status="Running",
    )
    component = SimpleNamespace(
        name="component",
        id=shared.workspace_id(config) + "/jobs/component",
        tags=dict(tags),
        status="Running",
        parent_job_name="pipeline",
    )
    client = SimpleNamespace(
        jobs=SimpleNamespace(get=lambda name: parent if name == "pipeline" else component)
    )
    monkeypatch.setattr(shared, "azure_job_identity", lambda value: component.id)
    assert shared.running_job_binding(client, config)["azure_job_id"] == parent.id
    (parent if target == "parent" else component).tags["static_source_sha256"] = "0" * 64
    with pytest.raises(ContractError):
        shared.running_job_binding(client, config)


def test_expiry_between_materialization_and_exec_is_not_renewed(package, tmp_path, monkeypatch):
    from datetime import UTC, datetime

    from learning.smolvla import image_bootstrap as bootstrap

    arguments, _ = materialization_arguments(package, tmp_path)
    config = bootstrap.materialize_source(**arguments)

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2031, 1, 1, tzinfo=UTC)

    monkeypatch.setattr(bootstrap, "datetime", Clock)
    monkeypatch.setattr(os, "execve", lambda *args: pytest.fail("Expired component executed"))
    with pytest.raises(bootstrap.ImageSourceError, match="deadline"):
        bootstrap.execute_component(
            arguments["destination"],
            config,
            command="export",
            snapshot_sha256=arguments["snapshot_sha256"],
            input_path=tmp_path / "input",
            output_path=tmp_path / "output",
        )


def test_source_modified_after_materialization_is_rejected_before_exec(
    package, tmp_path, monkeypatch
):
    from learning.smolvla.image_bootstrap import (
        ImageSourceError,
        execute_component,
        materialize_source,
    )

    arguments, _ = materialization_arguments(package, tmp_path)
    config = materialize_source(**arguments)
    (arguments["destination"] / "learning" / "common.py").write_bytes(b"not-reviewed")
    monkeypatch.setattr(os, "execve", lambda *args: pytest.fail("Modified source executed"))
    with pytest.raises(ImageSourceError, match="checksum"):
        execute_component(
            arguments["destination"],
            config,
            command="export",
            snapshot_sha256=arguments["snapshot_sha256"],
            input_path=tmp_path / "input",
            output_path=tmp_path / "output",
        )


@pytest.mark.parametrize("change", ["manifest-limit", "file-limit", "destination-link"])
def test_static_bounds_and_fresh_directory_rules_cannot_be_bypassed(package, tmp_path, change):
    from learning.smolvla.image_bootstrap import (
        MAX_FILE_BYTES,
        MAX_MANIFEST_BYTES,
        ImageSourceError,
        materialize_source,
    )

    arguments, _ = materialization_arguments(package, tmp_path)
    if change == "manifest-limit":
        with arguments["manifest_path"].open("wb") as stream:
            stream.truncate(MAX_MANIFEST_BYTES + 1)
    elif change == "file-limit":
        with (arguments["source_root"] / "learning" / "common.py").open("wb") as stream:
            stream.truncate(MAX_FILE_BYTES + 1)
    else:
        if os.name != "posix":
            pytest.skip("Deployment uses Linux symlinks")
        directory = tmp_path / "outside"
        directory.mkdir()
        arguments["destination"].symlink_to(directory, target_is_directory=True)
    with pytest.raises(ImageSourceError):
        materialize_source(**arguments)
