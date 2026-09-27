import base64
import os
import sys
from pathlib import Path

import pytest

from learning.common import ContractError, canonical, digest, file_digest, read_json, write_json
from learning.smolvla import azure
from tests.learning.test_direct_command_jobs import command_config

LEGACY_FILES = (
    "learning/common.py",
    "learning/contract.py",
    "learning/gr00t/ipc.py",
    "learning/gr00t/artifacts.py",
    "learning/smolvla/__init__.py",
    "learning/smolvla/artifacts.py",
    "learning/smolvla/inference.py",
    "learning/smolvla/adaptation.py",
    "learning/paused/__init__.py",
    "learning/paused/contract.py",
    "learning/paused/capture.py",
    "learning/paused/inference.py",
    "learning/paused/artifacts.py",
    "learning/paused/ipc.py",
    "learning/paused/model.py",
)
LEGACY_SHA256 = "383aa98c35e8d9d7a0868ae5fdc2e3b3c7045cda583906588591d33d23ece4e6"


def test_all_fifteen_learning_files_in_frozen_851df_servo_bundle_are_byte_exact():
    root = Path(__file__).resolve().parents[2]
    assert (
        digest(canonical({name: file_digest(root / name) for name in LEGACY_FILES}))
        == LEGACY_SHA256
    )


def test_new_explicit_bootstrap_executes_only_materialized_command_with_all_legacy_bytes(
    tmp_path,
    monkeypatch,
):
    from learning.smolvla.embedded_source import prepare_context
    from learning.smolvla.image_bootstrap import execute_component, materialize_source

    context = tmp_path / "static"
    receipt = prepare_context(context, direct=True)
    config = command_config()
    config["source_delivery"] = read_json(context / "source-delivery.json")
    assert read_json(context / "job-execution.json") == config["job_execution"]
    assert receipt["file_count"] == 74
    assert not (context / "source" / "run-config.json").exists()
    assert (
        digest(canonical({name: file_digest(context / "source" / name) for name in LEGACY_FILES}))
        == LEGACY_SHA256
    )
    plan_dir = tmp_path / "plan"
    azure.create_plan(config, plan_dir, deterministic_job_name=config["run_id"])
    plan = azure.read_plan(plan_dir)
    job = read_json(plan_dir / "job.json")
    assert not {"code", "inputs", "outputs", "jobs", "settings"} & job.keys()
    payload = canonical(config) + b"\n"
    root = tmp_path / "materialized"
    materialize_source(
        context / "source",
        context / "static-manifest.json",
        config_base64=base64.b64encode(payload).decode(),
        config_sha256=digest(payload),
        static_sha256=receipt["static_sha256"],
        snapshot_sha256=plan["snapshot_sha256"],
        destination=root,
    )
    azure.verify_code(root, plan["snapshot_sha256"], config=config)
    monkeypatch.chdir(tmp_path)
    calls = []
    monkeypatch.setattr(
        os, "execve", lambda exe, args, env: calls.append((exe, args, env, Path.cwd()))
    )
    execute_component(
        root, config, command="command-train", snapshot_sha256=plan["snapshot_sha256"]
    )
    _, args, env, actual_cwd = calls[0]
    assert args[1:4] == ["-B", "-m", "learning.paused.command"]
    assert "--input" not in args and "--output" not in args and "--run-command" not in args
    assert actual_cwd == root and env["PYTHONPATH"] == str(root)


def test_direct_native_entry_rejects_expired_authority_before_clients_or_child_process(
    tmp_path,
    monkeypatch,
):
    from learning.paused.command import main

    config = command_config()
    config["job_deadline_utc"] = "2020-01-01T00:00:00Z"
    path = tmp_path / "expired.json"
    write_json(path, config)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "command",
            "--runtime-config",
            str(path),
            "--snapshot-sha256",
            "a" * 64,
        ],
    )
    monkeypatch.setattr(
        azure,
        "clients_for_managed_identity",
        lambda *args, **kwargs: pytest.fail("Expired request acquired a client"),
    )
    monkeypatch.setattr(
        azure,
        "verify_code",
        lambda *args, **kwargs: pytest.fail("Expired request began source admission"),
    )
    with pytest.raises(ContractError, match="deadline"):
        main()
