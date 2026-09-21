from dataclasses import replace

import pytest

from simulation.extensions import SceneSpec, Station, can_reset_in_place
from simulation.health import check_heartbeat


def spec():
    return SceneSpec(
        stations=(Station("source", "source", (0.35, 0.25, 0.2)),),
        source_id="source",
        inspection_id="inspection",
        accepted_id="accepted",
        rejected_id="rejected",
        seed=42,
        defective=False,
        robot_profile="reference-arm",
        requested_speed=0.2,
        requested_payload=1,
        platform_color=(0.15, 0.3, 0.45),
    )


def test_only_visual_case_changes_reuse_the_reference_render_resources():
    original = spec()
    assert can_reset_in_place(original, replace(original, seed=43, defective=True))
    assert not can_reset_in_place(original, replace(original, requested_speed=0.1))
    assert not can_reset_in_place(original, replace(original, platform_color=(1, 0, 0)))
    assert not can_reset_in_place(
        original, replace(original, stations=(Station("source", "source", (0.4, 0.25, 0.2)),))
    )
    assert not can_reset_in_place(original, replace(original, record_demonstration=True))
    assert not can_reset_in_place(replace(original, record_demonstration=True), original)


def test_health_requires_a_fresh_actual_main_thread_heartbeat(tmp_path):
    path = tmp_path / "heartbeat"
    path.write_text("100")
    check_heartbeat(path, now=130)
    for value in ("0", "200", "nan", "not-a-heartbeat"):
        path.write_text(value)
        with pytest.raises(RuntimeError):
            check_heartbeat(path, now=140)
    path.unlink()
    with pytest.raises(RuntimeError):
        check_heartbeat(path, now=140)
