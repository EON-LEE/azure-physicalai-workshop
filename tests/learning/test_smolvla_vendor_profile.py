from copy import deepcopy

import pytest

from learning.common import ContractError


def test_profile_configuration_is_closed_and_preserves_the_authorized_workload():
    from learning.checks.smolvla_vendor_profile import PROFILE_SETTINGS, validate_settings

    validate_settings(deepcopy(PROFILE_SETTINGS))
    assert PROFILE_SETTINGS["measured_calls"] == 20
    assert PROFILE_SETTINGS["compile_process_seconds"] == 240
    assert PROFILE_SETTINGS["num_steps"] == 10
    assert PROFILE_SETTINGS["chunk_size"] == 50
    assert PROFILE_SETTINGS["state_dim"] == 6 and PROFILE_SETTINGS["camera_count"] == 3
    assert PROFILE_SETTINGS["compile_mode"] == "reduce-overhead"
    assert PROFILE_SETTINGS["rtol"] == 0.001 and PROFILE_SETTINGS["atol"] == 0.0001


@pytest.mark.parametrize(
    "change",
    [
        {"num_steps": 4},
        {"camera_count": 2},
        {"chunk_size": 16},
        {"state_dim": 9},
        {"compile_process_seconds": 241},
        {"compile_mode": "max-autotune"},
        {"change_precision": True},
        {"rtol": 0.01},
        {"atol": 0.1},
        {"use_rtc": True},
    ],
)
def test_approved_parameters_and_numeric_tolerances_cannot_silently_change(change):
    from learning.checks.smolvla_vendor_profile import PROFILE_SETTINGS, validate_settings

    with pytest.raises(ContractError):
        validate_settings({**PROFILE_SETTINGS, **change})


def test_explicit_noise_comparison_rejects_mismatch_nonfinite_and_shape_changes():
    from learning.checks.smolvla_vendor_profile import numerical_comparison

    result = numerical_comparison([1.0, 0.0, -2.0], [1.0001, 0.00001, -2.0001])
    assert result["passed"] is True
    assert result["rtol"] == 0.001 and result["atol"] == 0.0001
    assert result["max_absolute_error"] > 0
    for actual in ([1.1, 0.0, -2.0], [1.0, 0.0], [float("nan"), 0.0, -2.0]):
        with pytest.raises(ContractError):
            numerical_comparison([1.0, 0.0, -2.0], actual)


def test_profiler_summary_distinguishes_device_busy_span_from_host_launches():
    from learning.checks.smolvla_vendor_profile import trace_summary

    summary = trace_summary(
        {
            "traceEvents": [
                {"cat": "kernel", "name": "k1", "ts": 100.0, "dur": 20.0},
                {"cat": "kernel", "name": "k2", "ts": 110.0, "dur": 20.0},
                {"cat": "kernel", "name": "k3", "ts": 150.0, "dur": 10.0},
                {"cat": "cuda_runtime", "name": "cudaLaunchKernel", "ts": 90.0, "dur": 4.0},
                {"cat": "gpu_memcpy", "name": "HtoD", "ts": 80.0, "dur": 5.0},
            ]
        }
    )
    assert summary["kernel_count"] == 3
    assert summary["kernel_duration_sum_us"] == 50.0
    assert summary["kernel_busy_union_us"] == 40.0
    assert summary["inter_kernel_idle_us"] == 20.0
    assert summary["host_kernel_launch_count"] == 1
    assert summary["host_kernel_launch_duration_us"] == 4.0


def test_missing_cuda_trace_is_not_a_fake_stage_profile():
    from learning.checks.smolvla_vendor_profile import trace_summary

    with pytest.raises(ContractError, match="CUDA"):
        trace_summary({"traceEvents": [{"cat": "cpu_op", "name": "test"}]})


def test_spawn_selection_is_explicit_and_not_fork_after_cuda(monkeypatch):
    from learning.checks import smolvla_vendor_profile as profile

    calls = []
    expected = object()

    def context(method):
        calls.append(method)
        return expected

    monkeypatch.setattr(profile.multiprocessing, "get_context", context)
    assert profile.spawn_context() is expected
    assert calls == ["spawn"]
