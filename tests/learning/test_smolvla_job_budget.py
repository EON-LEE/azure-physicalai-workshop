from learning.checks.smolvla_aml_check import example_config
from learning.smolvla.azure import build_job


def test_converter_and_training_share_one_api_authorized_time_budget():
    config = example_config()
    config["parameters"]["timeout_seconds"] = 300
    job = build_job(config, "a" * 64, "smol-bounded")
    timeouts = [step["limits"]["timeout"] for step in job["jobs"].values()]
    assert all(value > 0 for value in timeouts)
    assert sum(timeouts) == 300
