"""Owned, paused-only finger stiffness calibration; never a force measurement."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from math import isclose

from learning.common import require, vector
from learning.contract import JOINT_NAMES
from simulation.reference_targets import GRASP_ASSET_SHA256
from simulation.runtime_contracts import CommandBinding

CALIBRATION_ID = "franka-paused-finger1-stiffness/v1"
ORIGINAL_STIFFNESS = 400.0
PAUSED_STIFFNESS = 2000.0
DAMPING = 80.0
FORCE_CAP = 7.2


@dataclass(frozen=True)
class DriveReadback:
    stiffness: tuple[float, ...]
    damping: tuple[float, ...]
    max_effort: tuple[float, ...]

    def validate(self) -> None:
        for name in ("stiffness", "damping", "max_effort"):
            values = vector(getattr(self, name), 9, f"actual {name}")
            require(all(value >= 0 for value in values), f"Actual {name} cannot be negative")

    def require_original_fingers(self) -> None:
        self.validate()
        require(
            self.stiffness[7:] == (ORIGINAL_STIFFNESS, 0.0)
            and self.damping[7:] == (DAMPING, 0.0)
            and isclose(self.max_effort[7], FORCE_CAP, rel_tol=0, abs_tol=1e-6)
            and self.max_effort[8] == 0,
            "Unexpected original live finger gains or caps; no repair is permitted",
        )


class PausedGripperServo:
    def __init__(
        self, *, read: Callable[[], DriveReadback], write_stiffness: Callable[[float], None]
    ) -> None:
        self.read, self.write_stiffness = read, write_stiffness
        self.binding: CommandBinding | None = None
        self.before: DriveReadback | None = None
        self.expected: DriveReadback | None = None
        self.record = {
            "calibration_id": CALIBRATION_ID,
            "status": "inactive",
            "contact_forces_measured": False,
        }

    def _read(self) -> DriveReadback:
        observed = self.read()
        require(isinstance(observed, DriveReadback), "An actual typed drive readback is required")
        observed.validate()
        return observed

    @staticmethod
    def _check_asset(value: dict) -> None:
        require(
            value.get("archive_sha256") == GRASP_ASSET_SHA256
            and tuple(value.get("joint_names", ())) == JOINT_NAMES
            and value.get("driven_joint_index") == 7
            and value.get("driven_joint_type") == "PhysicsPrismaticJoint"
            and value.get("drive_type") == "force"
            and value.get("authored_stiffness") == ORIGINAL_STIFFNESS
            and value.get("authored_damping") == DAMPING
            and isclose(value.get("authored_max_force", -1), FORCE_CAP, rel_tol=0, abs_tol=1e-6)
            and value.get("passive_joint_index") == 8
            and value.get("passive_has_drive") is False
            and value.get("mimic_axis") == "rotX"
            and value.get("mimic_gearing") == -1
            and value.get("mimic_reference") == "panda_finger_joint1",
            "The loaded asset/force drive/passive mimic differs from the approved calibration",
        )

    def apply(self, binding: CommandBinding, asset_evidence: dict) -> None:
        self._check_asset(asset_evidence)
        require(isinstance(binding, CommandBinding), "A concrete paused command owner is required")
        if self.binding is not None:
            require(binding == self.binding, "A different owner cannot reuse a gain override")
            self.verify(binding)
            require(asset_evidence == self.record["authored"], "Owned servo asset metadata changed")
            return
        before = self._read()
        before.require_original_fingers()
        self.before, self.binding = before, binding
        self.expected = replace(
            before, stiffness=before.stiffness[:7] + (PAUSED_STIFFNESS, before.stiffness[8])
        )
        self.record = {
            "calibration_id": CALIBRATION_ID,
            "status": "applying",
            "owner": {key: str(value) for key, value in asdict(binding).items()},
            "authored": deepcopy(asset_evidence),
            "before": asdict(before),
            "contact_forces_measured": False,
        }
        try:
            self.write_stiffness(PAUSED_STIFFNESS)
            after = self._read()
            self.record["after"] = asdict(after)
            if after != self.expected:
                raise RuntimeError(
                    "Native gain readback did not confirm the exact finger-only calibration."
                )
            self.record["status"] = "applied"
        except (ValueError, RuntimeError, TypeError, OSError) as exc:
            self.record.update(status="failed", error=str(exc))
            raise

    def verify(self, binding: CommandBinding) -> None:
        require(
            binding == self.binding and self.expected is not None,
            "No matching owned servo calibration",
        )
        if self.record["status"] != "applied" or self._read() != self.expected:
            raise RuntimeError("The owned paused stiffness or an unchanged drive value changed.")

    def restore(self) -> None:
        if self.binding is None:
            return
        try:
            current = self._read()
            if current == self.before and self.record["status"] == "failed":
                pass
            else:
                if current != self.expected:
                    raise RuntimeError(
                        "Cannot restore foreign or unexpectedly changed drive parameters."
                    )
                self.write_stiffness(self.before.stiffness[7])
            restored = self._read()
            if restored != self.before:
                raise RuntimeError("Native restore readback differs from the owned baseline.")
            self.record.update(status="restored", restored=asdict(restored))
            self.binding = None
        except (ValueError, RuntimeError, TypeError, OSError) as exc:
            self.record["restore_error"] = str(exc)
            raise

    def evidence(self) -> dict:
        return deepcopy(self.record)


def validate_servo_receipt(
    value: dict, *, command_id: str, environment_id: str, revision: str
) -> None:
    from uuid import UUID

    from simulation.paused_control import FrozenPhysicsState

    require(
        isinstance(value, dict)
        and value.get("calibration_id") == CALIBRATION_ID
        and value.get("status") == "restored"
        and not value.get("restore_error")
        and not value.get("error")
        and value.get("contact_forces_measured") is False,
        "Missing, failed, or unrestored actual gripper servo calibration",
    )
    PausedGripperServo._check_asset(value["authored"])
    snapshots = {}
    for name in ("before", "after", "restored"):
        raw = value[name]
        snapshots[name] = DriveReadback(
            **{
                field: vector(raw.get(field), 9, f"gripper servo {name} {field}")
                for field in ("stiffness", "damping", "max_effort")
            }
        )
        snapshots[name].validate()
    before = snapshots["before"]
    before.require_original_fingers()
    require(
        snapshots["after"]
        == replace(before, stiffness=before.stiffness[:7] + (PAUSED_STIFFNESS, before.stiffness[8]))
        and snapshots["restored"] == before,
        "Actual servo gain readback changed an arm, damping, force cap, or passive joint",
    )
    owner = value.get("owner", {})
    require(
        owner.get("command_id") == command_id
        and owner.get("environment_id") == environment_id
        and owner.get("revision") == revision,
        "Gripper servo calibration belongs to another command or scene",
    )
    physical = value.get("frozen_state", {})
    require(
        isinstance(physical.get("before"), dict) and physical["before"] == physical.get("after"),
        "Gripper servo calibration altered the frozen physical state",
    )
    raw = physical["before"]
    frozen = FrozenPhysicsState(
        **{
            key: UUID(item) if key == "epoch" else tuple(item) if isinstance(item, list) else item
            for key, item in raw.items()
        }
    )
    frozen.validate()
    require(str(frozen.epoch) == owner.get("epoch"), "Gripper servo epoch changed")
