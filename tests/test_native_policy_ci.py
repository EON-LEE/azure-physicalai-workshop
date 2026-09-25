from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def native_job() -> str:
    workflow = (ROOT / ".github" / "workflows" / "integration.yml").read_text()
    marker = "\n  native-policy-contracts:"
    assert marker in workflow, "Installed policy and ML SDK contracts must run in CI."
    return workflow.split(marker, 1)[1]


def test_native_policy_ci_uses_isolated_frozen_cpu_dependencies():
    job = native_job()
    assert 'python-version: "3.11"' in job
    assert "UV_PROJECT_ENVIRONMENT: ${{ runner.temp }}/physicalai-native-policy-venv" in job
    assert "uv sync --project learning/smolvla --frozen --extra cpu --extra azure" in job
    assert "timeout-minutes: 20" in job
    assert "PYTHONPATH: ${{ github.workspace }}" in job


def test_native_policy_ci_checks_real_package_interfaces_without_cloud_or_weights():
    job = native_job()
    for module in (
        "learning.checks.smolvla_api_check",
        "learning.checks.smolvla_export_check",
        "learning.checks.smolvla_aml_check",
        "learning.checks.paused_conversion_check",
    ):
        assert f"-m {module}" in job
    assert 'HF_HUB_OFFLINE: "1"' in job
    assert 'HF_DATASETS_OFFLINE: "1"' in job
    assert 'HF_HUB_DISABLE_IMPLICIT_TOKEN: "1"' in job
    assert "native-policy-test-only" in job
    assert "if: ${{ always() }}" in job
    assert "retention-days: 7" in job
    for prohibited in ("azure/login", "az login", "jobs create", "smolvla_vendor_diagnostic"):
        assert prohibited not in job
