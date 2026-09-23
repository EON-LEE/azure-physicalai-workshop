import pytest

from learning.common import ContractError


def test_shared_fixture_budget_is_shape_derived_and_not_the_log_limit():
    from learning.checks.smolvla_vendor_profile import artifact_byte_limit

    tensor_payload = 2 * ((3 * 3 * 256 * 256) + 6 + (50 * 32)) * 4
    assert tensor_payload == 4_731_440
    assert artifact_byte_limit("explicit-inputs-and-noise.safetensors") == tensor_payload + 65536
    assert 4_732_400 <= artifact_byte_limit("explicit-inputs-and-noise.safetensors")
    assert artifact_byte_limit("baseline.log") == 2 * 1024 * 1024
    assert artifact_byte_limit("compile.log") == 2 * 1024 * 1024
    assert artifact_byte_limit("baseline-trace.json.gz") == 8 * 1024 * 1024
    assert artifact_byte_limit("baseline-report.json") == 256 * 1024
    assert artifact_byte_limit("explicit-inputs.json") == 256 * 1024


def test_an_unapproved_weight_or_arbitrary_file_cannot_use_the_fixture_allowance():
    from learning.checks.smolvla_vendor_profile import artifact_byte_limit

    for name in ("model.safetensors", "customer-data.bin", "../compile.log"):
        with pytest.raises(ContractError):
            artifact_byte_limit(name)


def test_secondary_upload_failure_never_replaces_the_actual_compile_error():
    from learning.checks.smolvla_vendor_profile import record_artifact_failure

    report = {
        "passed": False,
        "artifacts": {},
        "failure_type": "CompileTimeout",
        "failure": "compile process exceeded 240 seconds",
    }
    record_artifact_failure(report, "baseline.log", 3 * 1024 * 1024, 2 * 1024 * 1024)
    assert report["failure_type"] == "CompileTimeout"
    assert report["failure"] == "compile process exceeded 240 seconds"
    assert report["passed"] is False
    assert report["artifacts"]["baseline.log"]["uploaded"] is False
    assert report["artifact_failures"][0]["limit_bytes"] == 2 * 1024 * 1024


def test_missing_required_proof_cannot_leave_a_successful_overall_result():
    from learning.checks.smolvla_vendor_profile import record_artifact_failure

    report = {"passed": True, "artifacts": {}}
    record_artifact_failure(report, "baseline-report.json", 300000, 262144)
    assert report["passed"] is False
    assert report["failure_type"] == "ArtifactBudgetExceeded"
