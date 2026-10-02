"""Approved one-shot vendor profiling experiment; never training, actuation or admission."""

from __future__ import annotations

import base64
import copy
import gzip
import importlib.util
import json
import math
import multiprocessing
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from learning import deadlines
from learning.common import (
    ContractError,
    canonical,
    digest,
    file_digest,
    finite,
    keys,
    parse_json,
    read_json,
    require,
    sha256,
)
from learning.deadlines import JobDeadline, validate_deadline
from learning.offline import enforce_offline

PROFILE_SCHEMA = "physicalai.smolvla-vendor-profile/v2"
LEGACY_PROFILE_SCHEMA = "physicalai.smolvla-vendor-profile/v1"
PROFILE_SETTINGS = {
    "warmup_calls": 3,
    "measured_calls": 20,
    "profiled_calls": 1,
    "numerical_fixtures": 2,
    "baseline_process_seconds": 150,
    "compile_process_seconds": 240,
    "compile_backend": "inductor",
    "compile_mode": "reduce-overhead",
    "dynamic": False,
    "fullgraph": False,
    "change_precision": False,
    "use_rtc": False,
    "num_steps": 10,
    "chunk_size": 50,
    "state_dim": 6,
    "camera_count": 3,
    "internal_image_size": 512,
    "rtol": 0.001,
    "atol": 0.0001,
    "maximum_log_bytes": 2 * 1024 * 1024,
    "maximum_trace_bytes": 128 * 1024 * 1024,
    "maximum_compressed_trace_bytes": 8 * 1024 * 1024,
}


def validate_settings(value: dict) -> None:
    require(
        canonical(value) == canonical(PROFILE_SETTINGS),
        "Profiling settings differ from the exact approved workload or tolerances",
    )


def artifact_byte_limit(name: str) -> int:
    if name in ("baseline.log", "compile.log"):
        return PROFILE_SETTINGS["maximum_log_bytes"]
    if name == "baseline-trace.json.gz":
        return PROFILE_SETTINGS["maximum_compressed_trace_bytes"]
    if name in ("explicit-inputs.json", "baseline-report.json", "compile-report.json"):
        return 256 * 1024
    if name == "explicit-inputs-and-noise.safetensors":
        # Two float32 fixtures: three 3x256x256 images, six states, and 1x50x32 explicit noise.
        payload = (
            PROFILE_SETTINGS["numerical_fixtures"]
            * (
                PROFILE_SETTINGS["camera_count"] * 3 * 256 * 256
                + PROFILE_SETTINGS["state_dim"]
                + PROFILE_SETTINGS["chunk_size"] * 32
            )
            * 4
        )
        return payload + 65536
    if name == "eager-reference-outputs.safetensors":
        return (
            PROFILE_SETTINGS["numerical_fixtures"]
            * PROFILE_SETTINGS["chunk_size"]
            * PROFILE_SETTINGS["state_dim"]
            * 4
            + 65536
        )
    raise ContractError(f"Unapproved profiling artifact class: {name}")


def record_artifact_failure(report: dict, name: str, size: int, limit: int) -> None:
    report["artifacts"][name] = {
        "bytes": size,
        "uploaded": False,
        "limit_bytes": limit,
        "reason": "artifact_exceeded_approved_byte_limit",
    }
    report.setdefault("artifact_failures", []).append(
        {
            "name": name,
            "bytes": size,
            "limit_bytes": limit,
        }
    )
    report["passed"] = False
    report.setdefault("failure_type", "ArtifactBudgetExceeded")
    report.setdefault("failure", "Required profiling artifact exceeded its upload limit")


def spawn_context():
    return multiprocessing.get_context("spawn")


def comparison_statistics(reference: list, actual: list) -> dict:
    require(len(reference) == len(actual) and len(reference) > 0, "Numeric output shape mismatch")
    expected = [finite(value, "eager reference") for value in reference]
    observed = [finite(value, "compiled output") for value in actual]
    absolute = [abs(a - b) for a, b in zip(expected, observed, strict=True)]
    relative = [
        delta / max(abs(value), 1e-12) for delta, value in zip(absolute, expected, strict=True)
    ]
    return {
        "rtol": PROFILE_SETTINGS["rtol"],
        "atol": PROFILE_SETTINGS["atol"],
        "max_absolute_error": max(absolute),
        "mean_absolute_error": sum(absolute) / len(absolute),
        "max_relative_error": max(relative),
        "mean_relative_error": sum(relative) / len(relative),
        "passed": all(
            delta <= PROFILE_SETTINGS["atol"] + PROFILE_SETTINGS["rtol"] * abs(value)
            for delta, value in zip(absolute, expected, strict=True)
        ),
    }


def numerical_comparison(reference: list, actual: list) -> dict:
    result = comparison_statistics(reference, actual)
    require(result["passed"], "Compiled outputs exceed the preapproved fixed numeric tolerances")
    return result


def trace_summary(value: dict) -> dict:
    events = value.get("traceEvents")
    require(isinstance(events, list), "Profiler did not emit trace events")
    kernels = [
        event
        for event in events
        if event.get("cat") == "kernel" and "ts" in event and "dur" in event
    ]
    require(bool(kernels), "Profiler did not capture actual CUDA kernels")
    spans = sorted(
        (float(event["ts"]), float(event["ts"]) + float(event["dur"])) for event in kernels
    )
    start, end = spans[0]
    busy = 0.0
    for low, high in spans[1:]:
        if low <= end:
            end = max(end, high)
        else:
            busy += end - start
            start, end = low, high
    busy += end - start
    launches = [
        event
        for event in events
        if event.get("cat") == "cuda_runtime" and "LaunchKernel" in str(event.get("name", ""))
    ]
    return {
        "kernel_count": len(kernels),
        "kernel_duration_sum_us": sum(float(event["dur"]) for event in kernels),
        "kernel_busy_union_us": busy,
        "inter_kernel_idle_us": max(0.0, max(end for _, end in spans) - spans[0][0] - busy),
        "host_kernel_launch_count": len(launches),
        "host_kernel_launch_duration_us": sum(float(event.get("dur", 0)) for event in launches),
        "gpu_memcpy_count": sum(event.get("cat") == "gpu_memcpy" for event in events),
        "note": "Kernel sum overlaps; union/idle spans first-to-last kernel, not the full call.",
    }


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".new")
    with temporary.open("wb") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _helper(spec: dict):
    path = Path(__file__).with_name("smolvla_vendor_diagnostic.py")
    require(
        file_digest(path) == spec["vendor_specification"]["diagnostic_code_sha256"],
        "Reviewed vendor helper code changed",
    )
    module_spec = importlib.util.spec_from_file_location("_approved_smol_vendor_helper", path)
    require(
        module_spec is not None and module_spec.loader is not None, "Vendor helper is unavailable"
    )
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def validate_specification(spec: dict) -> None:
    fields = {"schema", "vendor_specification", "profile_code_sha256", "settings"}
    if spec.get("schema") == PROFILE_SCHEMA:
        fields |= {"start_deadline_utc", "entry_code_sha256", "deadline_code_sha256"}
    keys(
        spec,
        fields,
        "profile specification",
    )
    require(
        spec["schema"] in (PROFILE_SCHEMA, LEGACY_PROFILE_SCHEMA), "Wrong vendor profiling schema"
    )
    if spec["schema"] == PROFILE_SCHEMA:
        validate_deadline(spec["start_deadline_utc"])
        sha256(spec["entry_code_sha256"])
        sha256(spec["deadline_code_sha256"])
    sha256(spec["profile_code_sha256"])
    validate_settings(spec["settings"])
    helper = _helper(spec)
    helper.validate_specification(spec["vendor_specification"])
    require(
        spec["vendor_specification"]["execution_timeout_seconds"] == 600,
        "Approved profiling execution budget is exactly 600 seconds",
    )


def admit_start(spec: dict) -> dict:
    require(
        spec.get("schema") == PROFILE_SCHEMA,
        "New diagnostic execution requires v2 with an explicit immutable start deadline",
    )
    deadline = JobDeadline(spec.get("start_deadline_utc"))
    deadline.check()
    validate_specification(spec)
    for path, key in (
        (Path(__file__), "profile_code_sha256"),
        (Path(__file__).with_name("smolvla_profile_entry.py"), "entry_code_sha256"),
        (Path(deadlines.__file__), "deadline_code_sha256"),
    ):
        require(file_digest(path) == spec[key], f"Approved {key} code checksum mismatch")
    deadline.check()
    return {
        "start_deadline_utc": deadline.value,
        "start_deadline_sha256": digest(canonical({"start_deadline_utc": deadline.value})),
        "checked_at_utc": deadlines.utcnow().isoformat().replace("+00:00", "Z"),
        "admitted": True,
    }


def precision_flags() -> dict:
    import torch

    return {
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cuda_autocast_enabled": torch.is_autocast_enabled("cuda"),
        "cuda_autocast_dtype": str(torch.get_autocast_dtype("cuda")),
        "default_dtype": str(torch.get_default_dtype()),
    }


def _parameter_dtypes(policy) -> dict:
    groups = {
        "vlm": policy.model.vlm_with_expert.vlm,
        "expert": policy.model.vlm_with_expert.lm_expert,
        "state_projection": policy.model.state_proj,
        "action_in_projection": policy.model.action_in_proj,
        "action_out_projection": policy.model.action_out_proj,
        "action_time_in": policy.model.action_time_mlp_in,
        "action_time_out": policy.model.action_time_mlp_out,
    }
    result = {}
    for name, module in groups.items():
        dtypes = {}
        for parameter in module.parameters():
            key = str(parameter.dtype)
            item = dtypes.setdefault(key, {"tensors": 0, "elements": 0, "bytes": 0})
            item["tensors"] += 1
            item["elements"] += parameter.numel()
            item["bytes"] += parameter.numel() * parameter.element_size()
        result[name] = dtypes
    return result


def _build_policy(root: Path, spec: dict, report: dict):
    report["weight_load_admission"] = admit_start(spec)
    enforce_offline()
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    __import__("lerobot.policies.smolvla.processor_smolvla")
    helper = _helper(spec)
    require(
        torch.cuda.is_available() and torch.cuda.device_count() == 1, "Expected one real CUDA GPU"
    )
    helper._source_license()
    original = read_json(root / "assets" / "model" / "config.json")
    helper.validate_vendor_configuration(original)
    require(
        original["num_steps"] == 10
        and original["resize_imgs_with_padding"] == [512, 512]
        and original["use_cache"] is True
        and original["use_amp"] is False,
        "Actual vendor denoising, resize, caching or precision configuration changed",
    )
    config = PreTrainedConfig.from_pretrained(root / "assets" / "model", local_files_only=True)
    require(
        config.compile_model is False and config.rtc_config is None,
        "No implicit native compile/RTC",
    )
    config.vlm_model_name = str((root / "assets" / "backbone").resolve())
    config.device, config.push_to_hub = "cuda", False
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    report["precision_flags_before_load"] = precision_flags()
    report["weight_load_admission"] = admit_start(spec)
    started = time.monotonic()
    policy = SmolVLAPolicy.from_pretrained(
        root / "assets" / "model",
        config=config,
        local_files_only=True,
        strict=True,
    )
    policy.eval()
    policy.requires_grad_(False)
    torch.cuda.synchronize()
    report["load_seconds"] = time.monotonic() - started
    report["precision_flags"] = precision_flags()
    report["parameter_dtypes"] = _parameter_dtypes(policy)
    report["parameters"] = sum(parameter.numel() for parameter in policy.parameters())
    report["vendor_weights_loaded"] = True
    gpu = torch.cuda.get_device_properties(0)
    report["gpu"] = {
        "name": gpu.name,
        "total_memory_bytes": gpu.total_memory,
        "nvidia_smi": subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=uuid,name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout.strip(),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    report["effective_configuration"] = {
        "num_steps": config.num_steps,
        "chunk_size": config.chunk_size,
        "n_action_steps": config.n_action_steps,
        "n_obs_steps": config.n_obs_steps,
        "use_cache": config.use_cache,
        "use_amp": config.use_amp,
        "compile_model": config.compile_model,
        "rtc_config": config.rtc_config,
        "resize_imgs_with_padding": list(config.resize_imgs_with_padding),
        "input_features": original["input_features"],
        "output_features": original["output_features"],
    }
    pre, post = make_pre_post_processors(
        config,
        pretrained_path=str(root / "assets" / "model"),
        preprocessor_overrides={
            "device_processor": {"device": "cuda"},
            "tokenizer_processor": {"tokenizer_name": str(root / "assets" / "backbone")},
        },
    )
    _atomic_json(root / f"{report['phase']}-report.json", report)
    return policy, pre, post


def _save_fixtures(root: Path, spec: dict) -> dict:
    import torch
    from safetensors.torch import save_file

    base = _helper(spec).fixture_observation()
    values = {}
    for index in range(2):
        for name, tensor in base.items():
            if name == "task":
                continue
            value = tensor.clone()
            if index == 1:
                value = value + 0.01 if name == "observation.state" else torch.flip(value, dims=[2])
            values[f"case{index}.{name}"] = value.contiguous()
        generator = torch.Generator(device="cpu").manual_seed(4200 + index)
        values[f"case{index}.noise"] = torch.randn(
            (1, 50, 32), generator=generator, dtype=torch.float32
        )
    target = root / "explicit-inputs-and-noise.safetensors"
    save_file(values, str(target))
    metadata = {
        "tensor_file": target.name,
        "sha256": file_digest(target),
        "task": base["task"],
        "cases": 2,
        "observation_source": "fixture",
        "noise_comparison": "Explicit identical tensor bytes shared between spawned processes",
        "shapes": {name: list(value.shape) for name, value in values.items()},
    }
    _atomic_json(root / "explicit-inputs.json", metadata)
    return metadata


def _load_fixtures(root: Path):
    from safetensors.torch import load_file

    metadata = read_json(root / "explicit-inputs.json")
    require(
        file_digest(root / metadata["tensor_file"]) == metadata["sha256"],
        "Shared input/noise changed",
    )
    tensors = load_file(str(root / metadata["tensor_file"]), device="cpu")
    fixtures = []
    for index in range(2):
        observation = {
            name.removeprefix(f"case{index}."): value
            for name, value in tensors.items()
            if name.startswith(f"case{index}.observation.")
        }
        observation["task"] = metadata["task"]
        noise = tensors[f"case{index}.noise"]
        require(tuple(noise.shape) == (1, 50, 32), "Wrong explicit shared diffusion-noise shape")
        fixtures.append((observation, noise.to("cuda")))
    return fixtures, metadata


def _single(policy, pre, post, fixture, *, marker=None):
    import torch

    observation, noise = fixture
    policy.reset()
    pre.reset()
    post.reset()
    torch.cuda.synchronize()
    started = time.monotonic_ns()
    with torch.inference_mode():
        if marker is None:
            batch = pre(copy.deepcopy(observation))
            output = post(policy.predict_action_chunk(batch, noise=noise))
        else:
            with marker("pipeline.preprocessor"):
                batch = pre(copy.deepcopy(observation))
            with marker("pipeline.native_forward"):
                native = policy.predict_action_chunk(batch, noise=noise)
            with marker("pipeline.postprocessor"):
                output = post(native)
    torch.cuda.synchronize()
    elapsed = (time.monotonic_ns() - started) / 1e6
    require(
        tuple(output.shape) == (1, 50, 6) and bool(output.isfinite().all()),
        "Actual native forward changed shape or produced nonfinite values",
    )
    return output.detach().float().cpu().clone(), elapsed


def _latency_summary(values: list[float]) -> dict:
    ordered = sorted(values)
    require(len(values) == PROFILE_SETTINGS["measured_calls"], "Incomplete twenty-call measurement")
    return {
        "all_ms": values,
        "calls": len(values),
        "p50_ms": ordered[math.ceil(len(values) * 0.5) - 1],
        "p95_ms": ordered[math.ceil(len(values) * 0.95) - 1],
        "max_ms": ordered[-1],
        "min_ms": ordered[0],
        "mean_ms": sum(values) / len(values),
        "instrumented": False,
    }


def _activation_description(value, depth=0):
    import torch

    if isinstance(value, torch.Tensor):
        return {"dtype": str(value.dtype), "device": str(value.device), "shape": list(value.shape)}
    if depth >= 2:
        return {"container_type": type(value).__name__}
    if isinstance(value, dict):
        return {
            str(name): _activation_description(item, depth + 1)
            for name, item in list(value.items())[:4]
        }
    if isinstance(value, (list, tuple)):
        return [_activation_description(item, depth + 1) for item in value[:4]]
    return {"type": type(value).__name__}


def _profile_native(policy, pre, post, fixture, root: Path) -> dict:
    from contextlib import ExitStack

    import torch

    spans, activations = [], {}

    @contextmanager
    def marker(name):
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        start = time.monotonic_ns()
        with torch.profiler.record_function(name):
            yield
        elapsed = (time.monotonic_ns() - start) / 1e6
        end.record()
        spans.append((name, begin, end, elapsed))

    def instrument(original, name):
        def wrapped(*args, **kwargs):
            with marker(name):
                return original(*args, **kwargs)

        return wrapped

    expert = policy.model.vlm_with_expert
    vision_calls, denoise_calls = [0], [0]
    original_image, original_denoise, original_forward = (
        expert.embed_image,
        policy.model.denoise_step,
        expert.forward,
    )

    def image_call(*args, **kwargs):
        vision_calls[0] += 1
        return instrument(original_image, f"vision.camera_{vision_calls[0]}")(*args, **kwargs)

    def denoise_call(*args, **kwargs):
        denoise_calls[0] += 1
        return instrument(original_denoise, f"expert.denoise_{denoise_calls[0]:02d}")(
            *args, **kwargs
        )

    def forward_call(*args, **kwargs):
        region = "prefix.kv_cache" if kwargs.get("fill_kv_cache") else "expert.transformer"
        return instrument(original_forward, region)(*args, **kwargs)

    handles = []
    modules = {
        "vision_model": expert.get_vlm_model().vision_model,
        "vision_connector": expert.get_vlm_model().connector,
        "state_projection": policy.model.state_proj,
        "action_in_projection": policy.model.action_in_proj,
        "action_out_projection": policy.model.action_out_proj,
    }
    for name, module in modules.items():

        def observe(current, inputs, output, label=name):
            descriptor = {
                "inputs": _activation_description(inputs),
                "output": _activation_description(output),
            }
            collection = activations.setdefault(label, [])
            if descriptor not in collection:
                require(len(collection) < 8, "Unbounded activation-shape variants")
                collection.append(descriptor)

        handles.append(module.register_forward_hook(observe))
    try:
        with ExitStack() as stack:
            stack.enter_context(
                patch.object(
                    policy, "prepare_images", instrument(policy.prepare_images, "preprocess.images")
                )
            )
            stack.enter_context(
                patch.object(
                    policy, "prepare_state", instrument(policy.prepare_state, "preprocess.state")
                )
            )
            stack.enter_context(
                patch.object(
                    policy.model,
                    "embed_prefix",
                    instrument(policy.model.embed_prefix, "prefix.embeddings"),
                )
            )
            stack.enter_context(patch.object(expert, "embed_image", image_call))
            stack.enter_context(patch.object(expert, "forward", forward_call))
            stack.enter_context(patch.object(policy.model, "denoise_step", denoise_call))
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
            ) as profiler:
                output, wall = _single(policy, pre, post, fixture, marker=marker)
        require(
            vision_calls[0] == 3 and denoise_calls[0] == 10,
            "Instrumented native workload does not match the published camera/denoising count",
        )
        raw = root / "baseline-trace.json"
        profiler.export_chrome_trace(str(raw))
        require(
            raw.stat().st_size <= PROFILE_SETTINGS["maximum_trace_bytes"],
            "Profiler trace exceeded byte limit",
        )
        trace = json.loads(raw.read_bytes())
        summary = trace_summary(trace)
        packed = root / "baseline-trace.json.gz"
        with raw.open("rb") as source, packed.open("wb") as destination:
            with gzip.GzipFile(fileobj=destination, mode="wb", mtime=0) as compressed:
                while chunk := source.read(1024 * 1024):
                    compressed.write(chunk)
        require(
            packed.stat().st_size <= PROFILE_SETTINGS["maximum_compressed_trace_bytes"],
            "Compressed profiler trace exceeded the approved upload budget",
        )
        operators = []
        for event in profiler.key_averages():
            operators.append(
                {
                    "name": event.key,
                    "count": event.count,
                    "cpu_total_us": event.cpu_time_total,
                    "cpu_self_us": event.self_cpu_time_total,
                    "cuda_total_us": event.device_time_total,
                    "cuda_self_us": event.self_device_time_total,
                }
            )
        operators.sort(key=lambda item: item["cuda_self_us"], reverse=True)
        return {
            "profiled_calls": 1,
            "instrumented_wall_ms": wall,
            "activation_dtypes": activations,
            "stages": [
                {"name": name, "cpu_wall_ms": cpu, "cuda_elapsed_ms": begin.elapsed_time(end)}
                for name, begin, end, cpu in spans
            ],
            "stage_accounting": (
                "Nested spans and CUDA elapsed include overlap/host gaps; "
                "do not sum as exclusive kernel time."
            ),
            "operator_summary_top40": operators[:40],
            "trace_summary": summary,
            "trace_file": packed.name,
            "trace_sha256": file_digest(packed),
            "trace_bytes": packed.stat().st_size,
            "profile_output_sha256": digest(output.numpy().tobytes()),
        }
    finally:
        for handle in handles:
            handle.remove()


def _baseline(root: Path, spec: dict, report: dict) -> None:
    import torch
    from safetensors.torch import save_file

    policy, pre, post = _build_policy(root, spec, report)
    report["inputs"] = _save_fixtures(root, spec)
    fixtures, _ = _load_fixtures(root)
    warm, timed, references = [], [], {}
    report["warmup_ms"] = warm
    report["measured_ms"] = timed
    for collection, count in ((warm, 3), (timed, 20)):
        for _ in range(count):
            collection.append(_single(policy, pre, post, fixtures[0])[1])
            report["actual_forward_calls"] += 1
    for index, fixture in enumerate(fixtures):
        references[f"case{index}"] = _single(policy, pre, post, fixture)[0]
        report["actual_forward_calls"] += 1
    reference_path = root / "eager-reference-outputs.safetensors"
    save_file(references, str(reference_path))
    report["reference_output_sha256"] = file_digest(reference_path)
    report["steady_latency"] = _latency_summary(timed)
    _atomic_json(root / "baseline-report.json", report)
    report["profile"] = _profile_native(policy, pre, post, fixtures[0], root)
    report["actual_forward_calls"] += 1
    report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    report["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    require(precision_flags() == report["precision_flags"], "Baseline precision flags changed")
    report["precision_flags_after"] = precision_flags()
    report["passed"] = True


def _counter_snapshot() -> dict:
    from torch._dynamo.utils import counters

    return {
        str(group): {str(name): int(value) for name, value in values.items()}
        for group, values in counters.items()
        if values
    }


def _compiled(root: Path, spec: dict, report: dict) -> None:
    import torch
    from safetensors.torch import load_file
    from torch._dynamo.utils import counters

    baseline = read_json(root / "baseline-report.json", max_bytes=1024 * 1024)
    require(
        baseline.get("passed") is True, "Compile comparison needs the actual completed baseline"
    )
    policy, pre, post = _build_policy(root, spec, report)
    require(
        report["gpu"] == baseline["gpu"]
        and report["precision_flags"] == baseline["precision_flags"]
        and report["parameter_dtypes"] == baseline["parameter_dtypes"]
        and report["effective_configuration"] == baseline["effective_configuration"],
        "Compiled process changed device, precision, dtypes or native configuration",
    )
    fixtures, metadata = _load_fixtures(root)
    require(
        metadata == baseline["inputs"],
        "Compiled process did not use identical explicit inputs/noise",
    )
    reference_path = root / "eager-reference-outputs.safetensors"
    require(
        file_digest(reference_path) == baseline["reference_output_sha256"],
        "Eager references changed",
    )
    references = load_file(str(reference_path), device="cpu")
    require(
        torch._dynamo.config.suppress_errors is False,
        "Silent compiler-error suppression is forbidden",
    )
    import torch._inductor.config as compiler_config

    require(
        compiler_config.worker_start_method in ("subprocess", "spawn"),
        "Compiler helpers must not fork an initialized CUDA process",
    )
    report["compiler_worker_configuration"] = {
        "start_method": compiler_config.worker_start_method,
        "compile_threads": compiler_config.compile_threads,
    }
    counters.clear()
    setup = time.monotonic()
    policy.model.sample_actions = torch.compile(
        policy.model.sample_actions,
        backend="inductor",
        mode="reduce-overhead",
        dynamic=False,
        fullgraph=False,
    )
    report["compile_wrapper_setup_seconds"] = time.monotonic() - setup
    report["compilation_status"] = "wrapper_constructed_first_call_pending"
    _atomic_json(root / "compile-report.json", report)
    warm = []
    report["warmup_ms"] = warm
    report["numeric_comparisons"] = []
    for _ in range(3):
        torch.compiler.cudagraph_mark_step_begin()
        _, elapsed = _single(policy, pre, post, fixtures[0])
        warm.append(elapsed)
        report["actual_forward_calls"] += 1
        report["compiler_counters"] = _counter_snapshot()
        report["compilation_status"] = "native_compiled_call_returned"
        _atomic_json(root / "compile-report.json", report)
    report["first_compile_and_call_seconds"] = warm[0] / 1000
    for index, fixture in enumerate(fixtures):
        torch.compiler.cudagraph_mark_step_begin()
        output, elapsed = _single(policy, pre, post, fixture)
        report["actual_forward_calls"] += 1
        expected = references[f"case{index}"]
        summary = comparison_statistics(expected.flatten().tolist(), output.flatten().tolist())
        report["numeric_comparisons"].append({"fixture": index, "call_ms": elapsed, **summary})
        _atomic_json(root / "compile-report.json", report)
        require(
            summary["passed"],
            "Fixed-noise numerical comparison failed; tolerances will not be relaxed",
        )
        torch.testing.assert_close(output, expected, rtol=0.001, atol=0.0001)
    timed = []
    report["measured_ms"] = timed
    for _ in range(20):
        torch.compiler.cudagraph_mark_step_begin()
        _, elapsed = _single(policy, pre, post, fixtures[0])
        timed.append(elapsed)
        report["actual_forward_calls"] += 1
    report["steady_latency"] = _latency_summary(timed)
    report["compiler_counters"] = _counter_snapshot()
    stats = report["compiler_counters"].get("stats", {})
    require(
        stats.get("unique_graphs", 0) > 0 and stats.get("calls_captured", 0) > 0,
        "No actual compiled graph captured; eager fallback is not optimized success",
    )
    report["graph_breaks"] = report["compiler_counters"].get("graph_break", {})
    report["fallback_evidence"] = {
        "suppress_errors": torch._dynamo.config.suppress_errors,
        "graph_break_eager_regions_present": bool(report["graph_breaks"]),
        "unimplemented": report["compiler_counters"].get("unimplemented", {}),
        "fully_compiled_verified": False,
        "note": "fullgraph=False permits eager regions; counters and compiler logs are retained.",
    }
    report["precision_flags_after"] = precision_flags()
    require(
        report["precision_flags_after"] == baseline["precision_flags"],
        "Compile changed precision flags",
    )
    report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    report["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    report["inputs_sha256"] = metadata["sha256"]
    report["passed"] = True


def _bounded_error(error: Exception) -> str:
    message = str(error)
    message = re.sub(r"Bearer\s+\S+", "Bearer [REDACTED]", message, flags=re.IGNORECASE)
    message = re.sub(r"([?&](?:sig|token)=)[^&\s]+", r"\1[REDACTED]", message, flags=re.IGNORECASE)
    return message[:3000]


def _phase_worker(phase: str, root_string: str, spec: dict) -> None:
    root = Path(root_string)
    with (root / f"{phase}.log").open("wb", buffering=0) as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        report = {
            "phase": phase,
            "passed": False,
            "pid": os.getpid(),
            "process_start_method": multiprocessing.get_start_method(),
            "torch_imported_at_process_entry": "torch" in sys.modules,
            "vendor_weights_loaded": False,
            "actual_forward_calls": 0,
        }
        started = time.monotonic()
        try:
            require(
                report["process_start_method"] == "spawn"
                and not report["torch_imported_at_process_entry"],
                "Worker was not freshly spawned before CUDA imports",
            )
            if phase in ("selftest", "selftest_timeout"):
                if phase == "selftest_timeout":
                    time.sleep(10)
                report["passed"] = True
            else:
                report["start_admission"] = admit_start(spec)
                enforce_offline()
                os.environ["TORCH_LOGS"] = "graph_breaks,recompiles,perf_hints"
                if phase == "baseline":
                    _baseline(root, spec, report)
                else:
                    require(phase == "compile", "Unknown isolated profiling phase")
                    _compiled(root, spec, report)
        except Exception as exc:
            report["failure_type"], report["failure"] = type(exc).__name__, _bounded_error(exc)
            traceback.print_exc()
            raise
        finally:
            report["elapsed_seconds"] = time.monotonic() - started
            _atomic_json(root / f"{phase}-report.json", report)
            sys.stdout.flush()
            sys.stderr.flush()


def _terminate_tree(process) -> None:
    import psutil

    try:
        descendants = psutil.Process(process.pid).children(recursive=True)
    except psutil.NoSuchProcess:
        descendants = []
    for child in descendants:
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    if process.is_alive():
        process.terminate()
    process.join(timeout=3)
    _, alive = psutil.wait_procs(descendants, timeout=3)
    for child in alive:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    if process.is_alive():
        process.kill()
    process.join(timeout=3)
    require(not process.is_alive(), "Bounded profiling subprocess could not be stopped")


def run_phase(phase: str, root: Path, spec: dict, *, timeout_seconds: float) -> dict:
    require(
        "torch" not in sys.modules, "Supervisor must not initialize or import CUDA before spawn"
    )
    process = spawn_context().Process(
        target=_phase_worker, args=(phase, str(root), spec), daemon=False
    )
    process.start()
    deadline = time.monotonic() + timeout_seconds
    try:
        while process.is_alive():
            process.join(timeout=0.25)
            log = root / f"{phase}.log"
            require(
                not log.exists() or log.stat().st_size <= PROFILE_SETTINGS["maximum_log_bytes"],
                "Profiling process log exceeded its bounded byte budget",
            )
            require(time.monotonic() < deadline, f"{phase} process exceeded its approved timeout")
        path = root / f"{phase}-report.json"
        report = read_json(path, max_bytes=1024 * 1024) if path.exists() else {}
        require(
            process.exitcode == 0 and report.get("passed") is True,
            f"{phase} process failed: {report.get('failure_type')}: {report.get('failure')}",
        )
        return report
    finally:
        if process.is_alive():
            _terminate_tree(process)


def self_test_spawn() -> dict:
    with tempfile.TemporaryDirectory(prefix="smol-profile-spawn-test-") as temporary:
        report = run_phase("selftest", Path(temporary), {}, timeout_seconds=20)
        require(report["pid"] != os.getpid(), "Spawn check ran in the parent process")
        return report


def self_test_timeout() -> dict:
    from learning.common import ContractError

    with tempfile.TemporaryDirectory(prefix="smol-profile-timeout-test-") as temporary:
        started = time.monotonic()
        try:
            run_phase("selftest_timeout", Path(temporary), {}, timeout_seconds=0.2)
        except ContractError as exc:
            require("timeout" in str(exc), "Unexpected process self-test failure")
            return {
                "bounded_spawn_termination_verified": True,
                "elapsed_seconds": time.monotonic() - started,
                "cuda_used": False,
                "weights_loaded": False,
            }
        raise AssertionError("A ten-second child escaped its 0.2-second supervision budget")


def run_profile(spec: dict) -> dict:
    admission = admit_start(spec)
    require(
        file_digest(Path(__file__)) == spec["profile_code_sha256"],
        "Profile harness checksum mismatch",
    )
    require("torch" not in sys.modules, "Profile supervisor already imported torch")
    helper = _helper(spec)
    vendor_spec = spec["vendor_specification"]
    helper._verify_code(vendor_spec)
    enforce_offline()
    require(
        os.environ.get("AZUREML_RUN_ID") == vendor_spec["job_name"],
        "Not the approved actual AML run",
    )
    report = helper.initial_report(vendor_spec)
    report.update(
        {
            "schema": "physicalai.smolvla-vendor-profile-proof/v2",
            "purpose": "vendor_stage_profile_same_precision_compile_only",
            "profile_code_sha256": spec["profile_code_sha256"],
            "configuration_sha256": digest(canonical(spec)),
            "entry_code_sha256": spec["entry_code_sha256"],
            "deadline_code_sha256": spec["deadline_code_sha256"],
            "start_admission": admission,
            "settings": PROFILE_SETTINGS,
            "started_at_utc": datetime.now(UTC).isoformat(),
            "phases": {},
            "optimization_verified": False,
            "compile_execution_verified": False,
        }
    )
    from azure.identity import ManagedIdentityCredential
    from azure.storage.blob import BlobServiceClient, ContentSettings

    started = time.monotonic()
    deadline = started + 570

    def within_budget():
        require(time.monotonic() < deadline, "Approved profiling execution budget expired")

    with (
        ManagedIdentityCredential(
            client_id=vendor_spec["managed_identity_client_id"]
        ) as credential,
        BlobServiceClient(
            vendor_spec["storage_account_url"],
            credential=credential,
            connection_timeout=10,
            read_timeout=30,
            retry_total=0,
        ) as client,
        tempfile.TemporaryDirectory(prefix="smol-vendor-profile-only-") as temporary,
    ):
        root = Path(temporary)
        container = client.get_container_client("artifacts")
        require(
            next(
                iter(container.list_blobs(name_starts_with=vendor_spec["output_prefix"] + "/")),
                None,
            )
            is None,
            "Profiling output exists; never overwrite or repeat the paid experiment",
        )
        try:
            assets = root / "assets"
            assets.mkdir()
            report["phase"] = "verified_private_asset_read"
            report["verified_assets"] = helper._download_private_assets(
                container, vendor_spec, assets, within_budget
            )
            report["asset_read_seconds"] = time.monotonic() - started
            for phase, ceiling in (("baseline", 150), ("compile", 240)):
                within_budget()
                report["phase"] = phase
                limit = min(ceiling, deadline - time.monotonic() - 10)
                require(limit > 0, "Insufficient time for the next bounded profiling phase")
                report["phases"][phase] = run_phase(phase, root, spec, timeout_seconds=limit)
            baseline, compiled = report["phases"]["baseline"], report["phases"]["compile"]
            report["vendor_weights_loaded"] = True
            report["compile_execution_verified"] = True
            report["optimization_verified"] = (
                compiled["steady_latency"]["p50_ms"] < baseline["steady_latency"]["p50_ms"]
                and compiled["steady_latency"]["p95_ms"] < baseline["steady_latency"]["p95_ms"]
            )
            report["steady_p50_speed_ratio"] = (
                baseline["steady_latency"]["p50_ms"] / compiled["steady_latency"]["p50_ms"]
            )
            report["phase"], report["passed"] = "completed", True
        except Exception as exc:
            report["failure_type"], report["failure"] = type(exc).__name__, _bounded_error(exc)
            report["passed"] = False
            raise
        finally:
            for phase in ("baseline", "compile"):
                phase_path = root / f"{phase}-report.json"
                if phase_path.exists():
                    report["phases"][phase] = read_json(phase_path, max_bytes=1024 * 1024)
            report["vendor_weights_loaded"] = any(
                phase.get("vendor_weights_loaded") is True for phase in report["phases"].values()
            )
            report["actual_forward_calls"] = sum(
                phase.get("actual_forward_calls", 0) for phase in report["phases"].values()
            )
            report["artifacts"] = {}
            names = [
                "explicit-inputs-and-noise.safetensors",
                "explicit-inputs.json",
                "eager-reference-outputs.safetensors",
                "baseline-trace.json.gz",
                "baseline-report.json",
                "compile-report.json",
                "baseline.log",
                "compile.log",
            ]
            for name in names:
                path = root / name
                if not path.exists():
                    continue
                limit = artifact_byte_limit(name)
                if path.stat().st_size > limit:
                    record_artifact_failure(report, name, path.stat().st_size, limit)
                    continue
                with path.open("rb") as stream:
                    container.upload_blob(
                        vendor_spec["output_prefix"] + "/" + name,
                        stream,
                        overwrite=False,
                        metadata={"sha256": file_digest(path)},
                    )
                report["artifacts"][name] = {
                    "sha256": file_digest(path),
                    "bytes": path.stat().st_size,
                }
            report["elapsed_seconds"] = time.monotonic() - started
            report["finished_at_utc"] = datetime.now(UTC).isoformat()
            report["supervisor_torch_imported"] = "torch" in sys.modules
            body = canonical(report) + b"\n"
            require(len(body) <= 256 * 1024, "Profile summary exceeds its bounded size")
            container.upload_blob(
                vendor_spec["output_prefix"] + "/vendor-profile.json",
                body,
                overwrite=False,
                content_settings=ContentSettings(content_type="application/json"),
            )
            print(
                "PHYSICALAI_SMOL_VENDOR_PROFILE "
                + canonical(
                    {
                        "proof_blob": vendor_spec["output_prefix"] + "/vendor-profile.json",
                        "proof_sha256": digest(body),
                        "passed": report["passed"],
                        "phase": report["phase"],
                        "failure_type": report.get("failure_type"),
                        "test_only": True,
                        "optimizer_steps": 0,
                        "actuator_calls": 0,
                        "learning_quality_verified": False,
                    }
                ).decode(),
                flush=True,
            )
    require(report["passed"], "Profiling result is incomplete; no optimized-success fallback")
    return report


if __name__ == "__main__":
    if "--self-test-spawn" in sys.argv:
        print(json.dumps(self_test_spawn(), indent=2))
    elif "--self-test-timeout" in sys.argv:
        print(json.dumps(self_test_timeout(), indent=2))
    else:
        encoded = os.environ["PHYSICALAI_VENDOR_PROFILE_SPEC"]
        require(len(encoded) <= 65536, "Oversized profiling specification")
        run_profile(parse_json(base64.b64decode(encoded, validate=True)))
