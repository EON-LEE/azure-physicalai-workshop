"""Actual native CPU checkpoint tests. No model downloads, Azure calls or quality claims."""

from __future__ import annotations

import argparse
import io
import json
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().output), indent=2))
