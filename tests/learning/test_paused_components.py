import sys

import pytest

from learning.common import ContractError, write_json
from tests.learning.test_paused_azure import config
from tests.learning.test_paused_contract import profile


def setup_component(monkeypatch, tmp_path, value, command, extra=()):
    from learning.paused import components

    path = tmp_path / "config.json"
    write_json(path, value)
    monkeypatch.setattr(components, "verify_code", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        components, "clients_for_managed_identity", lambda *args, **kwargs: (None, None)
    )
    monkeypatch.setattr(
        components, "running_job_binding", lambda *args: {"azure_job_id": "unit-only"}
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "components",
            command,
            "--runtime-config",
            str(path),
            "--snapshot-sha256",
            "a" * 64,
            "--input",
            str(tmp_path / "input"),
            "--output",
            str(tmp_path / "output"),
            "--run-component",
            *extra,
        ],
    )
    return components


def test_train_dispatch_never_falls_through_to_the_evaluation_plan(monkeypatch, tmp_path):
    value = config()
    value["control_profile_sha256"] = profile().sha256
    components = setup_component(
        monkeypatch,
        tmp_path,
        value,
        "train",
        ("--parent", str(tmp_path / "parent"), "--backbone", str(tmp_path / "backbone")),
    )
    monkeypatch.setattr(
        components,
        "validate_model",
        lambda *args, **kwargs: {"backbone_manifest_sha256": value["inputs"]["backbone"]["sha256"]},
    )
    from dataclasses import asdict

    monkeypatch.setattr(
        components,
        "validate_conversion",
        lambda *args: {
            "raw_manifest_sha256": value["inputs"]["demonstrations"]["sha256"],
            "control_profile": asdict(profile()),
        },
    )
    monkeypatch.setattr(components, "file_digest", lambda *args: "c" * 64)
    calls = []
    monkeypatch.setattr(components, "run_training", lambda *args, **kwargs: calls.append(kwargs))
    components.main()
    assert len(calls) == 1
    assert calls[0]["config"] == value


def test_failed_evaluation_writes_false_report_then_exits_nonzero(monkeypatch, tmp_path):
    from learning.common import read_json
    from learning.paused import evaluation
    from tests.learning.test_paused_evaluation import plan

    value = config()
    native_plan = plan()
    value["kind"] = "compare"
    value["parameters"] = {"timeout_seconds": 60}
    value["control_profile_sha256"] = profile().sha256
    asset = value["inputs"]["demonstrations"]
    value["inputs"] = {
        name: {**asset, "name": name, "type": "uri_file" if name == "plan" else "uri_folder"}
        for name in ("policy_before", "policy_after", "evidence", "plan")
    }
    for role in ("before", "after"):
        value["inputs"][f"policy_{role}"]["sha256"] = native_plan[f"policy_{role}_sha256"]
    native_path = tmp_path / "plan.json"
    write_json(native_path, native_plan)
    components = setup_component(
        monkeypatch,
        tmp_path,
        value,
        "compare",
        (
            "--parent",
            str(tmp_path / "before"),
            "--after",
            str(tmp_path / "after"),
            "--plan",
            str(native_path),
        ),
    )
    monkeypatch.setattr(
        evaluation,
        "evaluate_pair",
        lambda *args, **kwargs: {
            "quality_gate_passed": False,
            "real_time_admission": False,
        },
    )
    with pytest.raises(ContractError, match="quality"):
        components.main()
    report = read_json(tmp_path / "output" / "report.json")
    assert report["quality_gate_passed"] is False
    assert report["real_time_admission"] is False
