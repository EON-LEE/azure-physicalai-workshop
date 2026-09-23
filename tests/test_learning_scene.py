"""Reviewed initial layouts are CPU contracts, not proof of settled GPU poses."""

import json
from dataclasses import replace
from math import dist

import pytest
from runtime_support import ACTOR, document, service

from apps.api.models import SaveEnvironment
from simulation.extensions import SceneRegistry, can_reset_in_place


def environment(seed, template="inspection-cell-learning-v1"):
    doc = document()
    doc["scene"].update(template_id=template, seed=seed)
    return service().save_environment(ACTOR, SaveEnvironment(document_json=json.dumps(doc)))


def test_learning_seeds_produce_real_bounded_part_placements_without_defect_confound():
    registry = SceneRegistry(load_installed=False)
    specs = [registry.build(environment(seed)) for seed in range(20)]
    positions = [spec.part_position for spec in specs]
    assert len(set(positions)) == 20
    assert max(dist(left, right) for left in positions for right in positions) >= 0.02
    for spec in specs:
        source = spec.station(spec.source_id).position
        assert abs(spec.part_position[0] - source[0]) <= 0.02
        assert abs(spec.part_position[1] - source[1]) <= 0.02
        assert spec.part_position[2] == source[2]
        assert spec.defective is False
        assert spec.initial_part_position is not None


def test_identical_frozen_case_recreates_identical_before_after_start_and_builder_digest():
    registry = SceneRegistry(load_installed=False)
    case = environment(37)
    before = registry.build(case)
    after = registry.build(case.model_copy(deep=True))
    assert before.part_position == after.part_position
    assert before == after
    assert before.builder_id == "inspection-cell-learning-v1"
    digest = registry.builder_digest(before.builder_id)
    assert len(digest) == 64
    assert registry.builder_digest(after.builder_id) == digest
    changed = registry.build(environment(38))
    assert changed.part_position != before.part_position
    assert environment(38).revision != case.revision


def test_public_reference_builder_keeps_original_source_and_defect_seed_semantics():
    registry = SceneRegistry(load_installed=False)
    for seed in (0, 1):
        spec = registry.build(environment(seed, "inspection-cell-v1"))
        assert spec.part_position == spec.station(spec.source_id).position
        assert spec.initial_part_position is None
        assert spec.defective is bool(seed % 2)


def test_learning_seed_reset_can_reuse_geometry_but_never_changes_a_scene_mid_command():
    registry = SceneRegistry(load_installed=False)
    previous = registry.build(environment(4))
    current = registry.build(environment(5))
    assert can_reset_in_place(previous, current)
    assert not can_reset_in_place(previous, replace(current, requested_speed=0.1))


@pytest.mark.parametrize("seed", [0, 3])
def test_learning_source_requires_room_for_its_approved_variation(seed):
    from apps.api.errors import Problem

    doc = document()
    doc["scene"].update(template_id="inspection-cell-learning-v1", seed=seed)
    source_id = doc["workflow"]["source_station"]
    for station in doc["stations"]:
        if station["id"] == source_id:
            station["position_m"] = [0.15, 0, 0.2]
    record = service().save_environment(ACTOR, SaveEnvironment(document_json=json.dumps(doc)))
    with pytest.raises(Problem, match="margin"):
        SceneRegistry(load_installed=False).build(record)
