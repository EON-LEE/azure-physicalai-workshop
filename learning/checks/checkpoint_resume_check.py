"""Actual native CPU checkpoint tests. No model downloads, Azure calls or quality claims."""

from __future__ import annotations

import argparse
import io
import json
import random
from contextlib import redirect_stdout
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from learning.checks.fixtures import SCOPE
from learning.common import digest, file_digest, require, write_json
from learning.smolvla.checkpoint_runner import NativeCheckpointPublisher, native_runtime
from learning.smolvla.checkpoints import DEFAULT_LIMITS, restore_checkpoint


class MemoryBlobStore:
    """Test-only Blob-shaped byte store; never constructs a credential or makes a network call."""

    def __init__(self):
        self.blobs = {}

    def upload_blob(self, name, data, **kwargs):
        from azure.core.exceptions import ResourceExistsError

        require(kwargs["overwrite"] is False, "Checkpoint test must not overwrite storage")
        if name in self.blobs:
            raise ResourceExistsError("existing test blob")
        self.blobs[name] = data.read()

    def get_blob_client(self, name):
        parent = self

        class Blob:
            def get_blob_properties(self, **kwargs):
                return SimpleNamespace(
                    size=len(parent.blobs[name]), etag=digest(parent.blobs[name])
                )

            def download_blob(self, **kwargs):
                require(kwargs["etag"] == digest(parent.blobs[name]), "Readback generation changed")
                return SimpleNamespace(chunks=lambda: iter((parent.blobs[name],)))

        return Blob()


def fixture_context(runtime: dict) -> dict:
    from learning.common import canonical

    source = (
        "/subscriptions/22222222-2222-4222-8222-222222222222/resourceGroups/cpu-fixture/"
        "providers/Microsoft.MachineLearningServices/workspaces/cpu-fixture/jobs/"
    )
    binding = {
        name: "a" * 64
        for name in (
            "raw_manifest_sha256",
            "conversion_sha256",
            "parent_model_sha256",
            "parent_weights_sha256",
            "control_profile_sha256",
            "task_sha256",
            "criteria_sha256",
            "frozen_plan_sha256",
            "training_config_sha256",
            "training_code_sha256",
        )
    }
    binding.update(scope=asdict(SCOPE), runtime_sha256=digest(canonical(runtime)))
    return {
        "binding": binding,
        "origin": {
            "azure_job_id": source + "test-only-component",
            "azure_pipeline_job_id": source + "test-only-pipeline",
            "specification_sha256": "b" * 64,
            "code_snapshot_sha256": "c" * 64,
            "job_deadline_utc": "2026-09-27T00:00:00Z",
            "test_only": True,
        },
        "limits": asdict(DEFAULT_LIMITS),
        "blob_prefix": (
            f"tenants/{SCOPE.tenant_id}/owners/{SCOPE.owner_id}/learning/outputs/"
            "test-only/checkpoints"
        ),
        "resume_from_checkpoint_sha256": None,
        "prior_optimizer_steps": 0,
    }


def run(output: Path) -> dict:
    from learning.offline import enforce_offline

    enforce_offline()
    require(not output.exists(), "Choose a fresh native CPU checkpoint check directory")
    output.mkdir(parents=True)
    runtime = native_runtime()
    require(runtime["device"] == "cpu", "This check is CPU/test-only, never an implicit GPU job")
    import torch
    from lerobot.utils.train_utils import save_checkpoint
    from safetensors.torch import load_file, save_file

    torch.manual_seed(42)

    class TinyPolicy(torch.nn.Linear):
        def save_pretrained(self, path):
            path.mkdir(parents=True)
            save_file(self.state_dict(), str(path / "model.safetensors"))
            write_json(path / "config.json", {"test_only": True, "type": "tiny-cpu-linear"})

    class Configuration:
        peft = None

        def save_pretrained(self, path):
            write_json(path / "train_config.json", {"test_only": True, "steps": 4})

    class Processor:
        def __init__(self, name):
            self.name = name

        def save_pretrained(self, path):
            write_json(path / self.name, {"steps": [], "test_only": True})

    policy = TinyPolicy(3, 2)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    store, context = MemoryBlobStore(), fixture_context(runtime)
    publisher = NativeCheckpointPublisher(
        original_save=save_checkpoint,
        context=context,
        container=store,
        check_deadline=lambda: 60.0,
    )
    for step in range(1, 5):
        loss = policy(torch.randn(2, 3)).square().mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        if step % 2 == 0:
            with redirect_stdout(io.StringIO()):
                publisher(
                    checkpoint_dir=output / f"native-{step}",
                    step=step,
                    cfg=Configuration(),
                    policy=policy,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    preprocessor=Processor("policy_preprocessor.json"),
                    postprocessor=Processor("policy_postprocessor.json"),
                )
    receipt = publisher.receipts[-1]
    restored = output / "new-managed-job-restored"
    manifest = restore_checkpoint(
        store,
        checkpoint_prefix=receipt["manifest_blob"].rsplit("/", 1)[0],
        expected_sha256=receipt["checkpoint_sha256"],
        expected_scope=SCOPE,
        destination=restored,
        check_deadline=lambda: 60.0,
    )
    reloaded = TinyPolicy(3, 2)
    reloaded.load_state_dict(
        load_file(str(restored / "pretrained_model" / "model.safetensors")), strict=True
    )
    require(
        all(
            torch.equal(a, b)
            for a, b in zip(policy.parameters(), reloaded.parameters(), strict=True)
        ),
        "Actual saved CPU model weights did not round-trip",
    )
    fresh_optimizer = torch.optim.AdamW(reloaded.parameters(), lr=0.001)
    require(not fresh_optimizer.state, "Weights-only restart must not claim optimizer continuation")
    native_steps = {int(state["step"]) for state in optimizer.state.values()}
    require(native_steps == {4}, "Actual native optimizer steps were not persisted")
    report = {
        "check": "native-lerobot-0.4.4-safe-checkpoint-cpu",
        "test_only": True,
        "optimizer_steps": 4,
        "intermediate_steps": [2, 4],
        "native_optimizer_safetensors_sha256": file_digest(
            restored / "training_state" / "optimizer_state.safetensors"
        ),
        "checkpoint_sha256": receipt["checkpoint_sha256"],
        "resume_capability": manifest["state_kind"],
        "actual_weight_roundtrip": True,
        "optimizer_state_restored": False,
        "bitwise_continuation_claimed": False,
        "blob_readback": "in-memory transport test, not Azure proof",
        "cloud_calls": 0,
        "learning_quality_verified": False,
    }
    write_json(output / "report.json", report)
    return report


def run_full_state(output: Path) -> dict:
    """Compare actual native update_policy N+K with save/rehydrate/load and continue."""
    from learning.offline import enforce_offline

    enforce_offline()
    require(not output.exists(), "Choose a new deterministic CPU checkpoint check directory")
    output.mkdir(parents=True)
    runtime = native_runtime()
    require(runtime["device"] == "cpu", "Full-state equivalence fixture is CPU-only")
    from unittest.mock import patch

    import numpy as np
    import torch
    from accelerate import Accelerator
    from lerobot.scripts.lerobot_train import update_policy
    from lerobot.utils.train_utils import load_training_state, save_checkpoint
    from safetensors.torch import load_file, save_file
    from torch.utils.data import DataLoader

    from learning.smolvla.checkpoint_state import ContinuationState, capture_rng

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    class TinyPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 2)
            self.dropout = torch.nn.Dropout(0.2)

        def forward(self, batch):
            noise = random.gauss(0, 1) + float(np.random.normal())
            target = batch["target"] + torch.randn_like(batch["target"]) * 0.01 + noise * 0.001
            return (self.dropout(self.linear(batch["input"])) - target).square().mean(), {}

        def save_pretrained(self, root):
            root.mkdir(parents=True)
            save_file(self.state_dict(), str(root / "model.safetensors"))
            write_json(root / "config.json", {"test_only": True, "type": "tiny-cpu-linear"})

        def get_optim_params(self):
            return self.parameters()

    class Configuration:
        peft = None

        def save_pretrained(self, root):
            write_json(root / "train_config.json", {"test_only": True, "steps": 9})

    class Processor:
        def __init__(self, name):
            self.name = name

        def save_pretrained(self, root):
            write_json(root / self.name, {"test_only": True, "steps": []})

        def __call__(self, value):
            return value

    def initialize(checkpoint=None, resume_step=0):
        random.seed(718)
        np.random.seed(718)
        torch.manual_seed(718)
        random.gauss(0, 1)  # Ensure a nonempty Python Gaussian cache at some save boundaries.
        np.random.normal()  # Ensure NumPy's float64 cache is not always zero.
        policy = TinyPolicy()
        if checkpoint is not None:
            policy.load_state_dict(
                load_file(str(checkpoint / "pretrained_model" / "model.safetensors")), strict=True
            )
        optimizer = torch.optim.AdamW(policy.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)
        dataset = [
            {
                "input": torch.tensor([index / 10, 0.2, -0.3]),
                "target": torch.tensor([0.1, index / 20]),
                "index": index,
            }
            for index in range(10)
        ]
        accelerator = Accelerator(
            cpu=True, mixed_precision="no", step_scheduler_with_optimizer=False
        )
        policy, optimizer, loader, scheduler = accelerator.prepare(
            policy,
            optimizer,
            DataLoader(dataset, batch_size=3, shuffle=True, num_workers=0),
            scheduler,
        )
        if checkpoint is not None:
            step, optimizer, scheduler = load_training_state(checkpoint, optimizer, scheduler)
            require(step == resume_step, "Actual native optimizer step failed to restore")
        state = ContinuationState(seed=42, resume_root=checkpoint, resume_step=resume_step)
        iterator = state.cycle(loader)
        return policy, optimizer, scheduler, accelerator, state, iterator

    def advance(parts, steps):
        policy, optimizer, scheduler, accelerator, _, iterator = parts
        indices = []
        metrics = SimpleNamespace()
        for _ in range(steps):
            batch = next(iterator)
            indices.append(batch["index"].tolist())
            update_policy(
                metrics, policy, batch, optimizer, 1.0, accelerator, lr_scheduler=scheduler
            )
        return indices

    def equal(left, right):
        if isinstance(left, torch.Tensor):
            return isinstance(right, torch.Tensor) and torch.equal(left, right)
        if isinstance(left, dict):
            return (
                isinstance(right, dict)
                and left.keys() == right.keys()
                and all(equal(left[k], right[k]) for k in left)
            )
        if isinstance(left, (list, tuple)):
            return (
                type(left) is type(right)
                and len(left) == len(right)
                and all(equal(a, b) for a, b in zip(left, right, strict=True))
            )
        return left == right

    checks = []
    negative_checks = []
    # Mid-epoch and epoch-boundary stops cover short last batches and the next permutation.
    for stop in (3, 4):
        continuous = initialize()
        all_indices = advance(continuous, 9)
        expected_model = {key: value.clone() for key, value in continuous[0].state_dict().items()}
        expected_optimizer = continuous[1].state_dict()
        expected_scheduler = continuous[2].state_dict()
        expected_rng = capture_rng()
        next_expected = next(continuous[-1])["index"].tolist()

        interrupted = initialize()
        seen = advance(interrupted, stop)
        store, context = MemoryBlobStore(), fixture_context(runtime)
        publisher = NativeCheckpointPublisher(
            original_save=save_checkpoint,
            context=context,
            container=store,
            check_deadline=lambda: 60.0,
            state_provider=interrupted[4],
        )
        source = output / f"source-{stop}"
        with patch.object(torch, "load", side_effect=AssertionError("Pickle loading is forbidden")):
            with redirect_stdout(io.StringIO()):
                publisher(
                    checkpoint_dir=source,
                    step=stop,
                    cfg=Configuration(),
                    policy=interrupted[3].unwrap_model(interrupted[0]),
                    optimizer=interrupted[1],
                    scheduler=interrupted[2],
                    preprocessor=Processor("policy_preprocessor.json"),
                    postprocessor=Processor("policy_postprocessor.json"),
                )
            receipt = publisher.receipts[-1]
            require(
                receipt["resume_capability"] == "full_state",
                "Missing complete continuation capability",
            )
            restored = output / f"restored-{stop}"
            restore_checkpoint(
                store,
                checkpoint_prefix=receipt["manifest_blob"].rsplit("/", 1)[0],
                expected_sha256=receipt["checkpoint_sha256"],
                expected_scope=SCOPE,
                destination=restored,
                check_deadline=lambda: 60.0,
            )
            resumed = initialize(restored, stop)
            resumed_indices = advance(resumed, 9 - stop)
        require(seen + resumed_indices == all_indices, "Actual resumed data order differs")
        require(equal(resumed[0].state_dict(), expected_model), "Resumed CPU model weights differ")
        require(
            equal(resumed[1].state_dict(), expected_optimizer), "Resumed actual AdamW state differs"
        )
        require(
            equal(resumed[2].state_dict(), expected_scheduler), "Resumed LR scheduler state differs"
        )
        require(equal(capture_rng(), expected_rng), "Full Python/NumPy/Torch RNG state differs")
        require(next(resumed[-1])["index"].tolist() == next_expected, "Next resumed batch differs")
        checks.append(
            {
                "stop_step": stop,
                "total_steps": 9,
                "native_optimizer_scheduler_equal": True,
                "model_weights_bitwise_equal_on_cpu_fixture": True,
                "exact_rng_equal": True,
                "all_batch_indices_and_next_batch_equal": True,
                "checkpoint_sha256": receipt["checkpoint_sha256"],
            }
        )
        from copy import deepcopy

        from learning.common import ContractError
        from learning.smolvla.checkpoint_state import restore_rng

        data_metadata, data_tensors = resumed[4].stream.state()
        for bad in ("partial_batch", "wrong_step", "wrong_data", "duplicate_indices"):
            changed = deepcopy(data_metadata)
            tensors = {name: value.clone() for name, value in data_tensors.items()}
            if bad == "partial_batch":
                changed["cursor"] = 1
            elif bad == "wrong_step":
                changed["batches"] = 0
            elif bad == "wrong_data":
                changed["indices_sha256"] = "0" * 64
            else:
                tensors["sampler_order"][0] = tensors["sampler_order"][1]
            try:
                resumed[4].stream.load(changed, tensors, expected_step=resumed[4].stream.batches)
            except ContractError:
                negative_checks.append(bad)
            else:
                raise AssertionError(f"Invalid consumed sampler state admitted: {bad}")
        rng_meta, rng_tensors = capture_rng()
        try:
            restore_rng(({**rng_meta, "cuda_device_count": 1}, rng_tensors))
        except ContractError:
            negative_checks.append("cross_device_rng")
        else:
            raise AssertionError("CPU fixture accepted CUDA RNG state")

    # Exercise the actual decorated train entry, CLI config-path resume, Accelerate preparation
    # and installed optimizer factory as well; only the tiny policy/dataset are test substitutes.
    import sys
    from contextlib import ExitStack, redirect_stderr

    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.scripts import lerobot_train as native

    from learning.smolvla.checkpoint_runner import (
        POLICY_SCHEMA,
        train_native,
        validate_resume_checkpoint,
    )

    class Dataset:
        num_frames, num_episodes = 10, 1
        meta = SimpleNamespace(stats={})

        def __len__(self):
            return self.num_frames

        def __getitem__(self, index):
            return {
                "input": torch.tensor([index / 10, 0.2, -0.3]),
                "target": torch.tensor([0.1, index / 20]),
                "index": index,
            }

    class InterruptedFixture(RuntimeError):
        pass

    def run_entry(name, *, source=None, source_sha=None, interrupt=False):
        directory = output / name
        initial = output / f"{name}-configuration"
        initial.mkdir()
        cfg = TrainPipelineConfig(
            dataset=DatasetConfig(
                repo_id="azure-local/checkpoint-test", root=output / "immutable-fixture"
            ),
            policy=SmolVLAConfig(device="cpu", push_to_hub=False, use_amp=False),
            output_dir=directory,
            steps=9,
            batch_size=3,
            num_workers=0,
            seed=718,
            save_freq=3,
            eval_freq=0,
            log_freq=0,
        )
        cfg.dataset.image_transforms.enable = False
        cfg.save_pretrained(initial)
        context = fixture_context(runtime)
        context.update(
            training_parameters={"seed": 718, "resume_mode": "full_state" if source else "new"},
            resume_checkpoint_root=str(source) if source else None,
            resume_from_checkpoint_sha256=source_sha,
        )
        if source:
            from learning.common import read_json

            previous = read_json(source / "checkpoint.json")
            context["origin"] = {
                **context["origin"],
                "azure_job_id": context["origin"]["azure_job_id"] + "-resumed",
                "azure_pipeline_job_id": context["origin"]["azure_pipeline_job_id"] + "-resumed",
                "specification_sha256": "9" * 64,
            }
            approval = {
                "parameters": {
                    "max_steps": 9,
                    "checkpoint_steps": 3,
                    "batch_size": 3,
                    "seed": 718,
                    "resume_mode": "full_state",
                },
                "checkpointing": {
                    "schema": POLICY_SCHEMA,
                    "limits": asdict(DEFAULT_LIMITS),
                    "resume": {
                        "checkpoint_sha256": source_sha,
                        "step": 3,
                        "source_azure_job_id": previous["origin"]["azure_job_id"],
                        "source_azure_pipeline_job_id": previous["origin"]["azure_pipeline_job_id"],
                    },
                },
            }
            validate_resume_checkpoint(
                source,
                config=approval,
                expected_binding=context["binding"],
                current_origin=context["origin"],
            )
        config_path = (source / "pretrained_model" if source else initial) / "train_config.json"
        args = [
            "fixture-native-entry",
            f"--config_path={config_path}",
            f"--output_dir={directory}",
        ]
        if source:
            args.append("--resume=true")

        def policy_factory(*, cfg, **kwargs):
            instance = TinyPolicy()
            cfg.input_features = {
                "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(3,))
            }
            cfg.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(2,))}
            instance.config = cfg
            if cfg.pretrained_path is not None:
                instance.load_state_dict(
                    load_file(str(cfg.pretrained_path / "model.safetensors")), strict=True
                )
            return instance

        original_last = native.update_last_checkpoint

        def last(path):
            original_last(path)
            if interrupt and path.name == "000003":
                raise InterruptedFixture("Test interruption after completed remote checkpoint")

        store = MemoryBlobStore()
        with ExitStack() as stack:
            stack.enter_context(patch.object(sys, "argv", args))
            stack.enter_context(patch.object(native, "make_dataset", lambda cfg: Dataset()))
            stack.enter_context(patch.object(native, "make_policy", policy_factory))
            stack.enter_context(
                patch.object(
                    native,
                    "make_pre_post_processors",
                    lambda **kwargs: (
                        Processor("policy_preprocessor.json"),
                        Processor("policy_postprocessor.json"),
                    ),
                )
            )
            stack.enter_context(patch.object(native, "update_last_checkpoint", last))
            stack.enter_context(
                patch.object(torch, "load", side_effect=AssertionError("No pickle path"))
            )
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(redirect_stderr(io.StringIO()))
            if interrupt:
                try:
                    train_native(context, store, lambda: 60.0)
                except InterruptedFixture:
                    pass
                else:
                    raise AssertionError(
                        "Native fixture did not stop at the exact checkpoint boundary"
                    )
            else:
                train_native(context, store, lambda: 60.0)
        return directory, store, context

    baseline_dir, _, _ = run_entry("native-uninterrupted")
    stopped_dir, stopped_store, _ = run_entry("native-interrupted", interrupt=True)
    checkpoint = stopped_dir / "checkpoints" / "000003"
    checkpoint_sha = file_digest(checkpoint / "checkpoint.json")
    rehydrated = output / "native-rehydrated"
    restore_checkpoint(
        stopped_store,
        checkpoint_prefix=fixture_context(runtime)["blob_prefix"] + "/step-000003",
        expected_sha256=checkpoint_sha,
        expected_scope=SCOPE,
        destination=rehydrated,
        check_deadline=lambda: 60.0,
    )
    resumed_dir, _, _ = run_entry("native-resumed", source=rehydrated, source_sha=checkpoint_sha)
    baseline_final = baseline_dir / "checkpoints" / "000009"
    resumed_final = resumed_dir / "checkpoints" / "000009"
    for relative in (
        "pretrained_model/model.safetensors",
        "training_state/optimizer_state.safetensors",
        "training_state/continuation.safetensors",
    ):
        require(
            equal(
                load_file(str(baseline_final / relative)), load_file(str(resumed_final / relative))
            ),
            f"Actual native CLI uninterrupted/resumed tensors differ: {relative}",
        )
    from learning.common import read_json

    for relative in ("training_state/scheduler_state.json", "training_state/continuation.json"):
        require(
            read_json(baseline_final / relative) == read_json(resumed_final / relative),
            "Actual native CLI resumed scheduler/RNG/data metadata differs",
        )
    require(
        json.loads((baseline_final / "training_state" / "optimizer_param_groups.json").read_bytes())
        == json.loads(
            (resumed_final / "training_state" / "optimizer_param_groups.json").read_bytes()
        ),
        "Actual native CLI optimizer parameter groups differ",
    )
    report = {
        "check": "native-lerobot-update-save-load-exact-cpu-continuation",
        "test_only": True,
        "checks": checks,
        "resume_capability": "full_state",
        "cloud_calls": 0,
        "gpu_execution": False,
        "cross_device_bitwise_claim": False,
        "real_model_quality_verified": False,
        "pickle_load_calls": 0,
        "actual_native_train_cli_resume_equivalence": True,
        "invalid_state_rejections": negative_checks,
    }
    write_json(output / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full-state", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(run_full_state(args.output) if args.full_state else run(args.output), indent=2)
    )
