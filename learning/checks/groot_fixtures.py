from dataclasses import replace
from pathlib import Path

from learning.capture import EpisodeWriter
from learning.checks.fixtures import JOINTS, PROVENANCE, SCOPE, frame
from learning.contract import AppliedControl, ControlProfile, DemonstrationSource, EpisodeSpec

PROFILE = ControlProfile(servo_profile_sha256="d" * 64)
TASK = DemonstrationSource("human_teleop", "kit", "Place the part in the kitting tray.", "tray")


def teaching(root: Path, *, episode_id="human-1", seed=17, split="train", count=18) -> Path:
    writer = EpisodeWriter(
        root,
        dataset_id="explicit-synthetic-test",
        scope=SCOPE,
        episode=EpisodeSpec(episode_id, "customer-line", "f" * 64, seed, split),
        provenance=replace(PROVENANCE, simulator_version="6.0.0"),
        control_profile=PROFILE,
        demonstration=TASK,
    )
    for index in range(count):
        value = frame(index, terminal=index == count - 1)
        joints = (index * 0.001, *JOINTS[1:])
        controls = tuple(
            AppliedControl(
                value.physics_step + step,
                value.monotonic_ns + step * 10_000_000,
                joints,
                (0.0,) * 9,
                (1.0,) * 7 + (0.0, 0.0),
            )
            for step in range(1, 7)
        )
        writer.append(
            replace(
                value,
                joint_positions=joints,
                commanded_joint_targets=joints,
                applied_controls=controls,
            )
        )
    writer.finalize()
    return root
