"""CPU first-publication proof tests, not native-camera/GPU evidence."""

from dataclasses import replace
from uuid import uuid4

import pytest
from test_paused_control import state

from learning.paused import InitialFrozenPublication
from simulation.paused_observation import PausedPublication, PausedPublicationCache
from tests.learning.test_paused_contract import observation


def publication():
    sample = observation()
    return PausedPublication(
        publication_id=uuid4(),
        scope=sample.scope,
        environment_id=sample.environment_id,
        revision=sample.revision,
        profile_sha256=sample.control_profile_sha256,
        state_revision=sample.state_revision,
        frozen_state=replace(state(), physics_step=sample.physics_step, world_time=1.0),
        freeze_established_ns=1_800_000_000,
        joint_sample_ns=1_850_000_000,
        published_ns=sample.monotonic_ns,
        captured_at_utc=sample.captured_at_utc,
        images=tuple(sample.images.items()),
    )


def take(cache, value, *, now_ns=2_100_000_000, **changes):
    args = {
        "scope": value.scope,
        "environment_id": value.environment_id,
        "revision": value.revision,
        "profile_sha256": value.profile_sha256,
        "state_revision": value.state_revision,
        "state": value.frozen_state,
        "now_ns": now_ns,
    } | changes
    return cache.take(**args)


def test_first_frozen_publication_keeps_original_camera_and_joint_timestamps():
    value = publication()
    cache = PausedPublicationCache()
    cache.record(value)
    actual = take(cache, value)
    assert actual == value
    assert actual.published_ns == 2_000_000_000
    assert dict(actual.images)["inspection"].monotonic_ns == 1_900_000_000
    assert actual.joint_sample_ns == 1_850_000_000
    with pytest.raises(RuntimeError, match="consumed"):
        take(cache, value)


@pytest.mark.parametrize("failure", ["expired", "future", "epoch", "object", "revision", "owner"])
def test_old_changed_or_unbound_publications_never_authorize_a_new_observation(failure):
    value = publication()
    cache = PausedPublicationCache()
    cache.record(value)
    changes = {
        "expired": {"now_ns": 4_000_000_001},
        "future": {"now_ns": 1_999_999_999},
        "epoch": {"state": replace(value.frozen_state, epoch=uuid4())},
        "object": {"state": replace(value.frozen_state, object_position=(0.5, 0.2, 0.2))},
        "revision": {"revision": "a" * 64},
        "owner": {"scope": replace(value.scope, owner_id="e" * 64)},
    }
    with pytest.raises(RuntimeError):
        take(cache, value, **changes[failure])


def test_a_freeze_inferred_after_publication_is_not_contemporaneous_evidence():
    cache = PausedPublicationCache()
    value = publication()
    with pytest.raises(ValueError, match="publication"):
        cache.record(replace(value, freeze_established_ns=value.published_ns + 1))


def test_relabelling_consumed_pixels_with_a_new_uuid_cannot_create_another_first_frame():
    cache = PausedPublicationCache()
    value = publication()
    cache.record(value)
    take(cache, value)
    with pytest.raises(ValueError, match="old frozen publication"):
        cache.record(replace(value, publication_id=uuid4()))


def test_worker_verifies_known_first_publication_without_consuming_or_restamping_it():
    cache = PausedPublicationCache()
    value = publication()
    cache.record(value)
    observed = replace(
        observation(),
        epoch=str(value.frozen_state.epoch),
        joint_positions=value.frozen_state.joint_positions,
        joint_sample_ns=value.joint_sample_ns,
        observation_started_ns=2_100_000_000,
        observation_completed_ns=2_110_000_000,
    )
    proof = InitialFrozenPublication(
        publication_record_id=str(value.publication_id),
        publication_record_sha256=value.record_sha256,
        capture_sha256=observed.capture_sha256,
        freeze_established_ns=value.freeze_established_ns,
        published_at_utc=value.captured_at_utc,
        published_monotonic_ns=value.published_ns,
        age_at_observation_start_ns=100_000_000,
    )
    observed = replace(observed, initial_publication=proof)
    assert not cache.verifies(proof, observed)
    take(cache, value)
    assert cache.verifies(proof, observed)
    assert cache.verifies(proof, observed)
    for invalid in (
        replace(proof, publication_record_id=str(uuid4())),
        replace(proof, publication_record_sha256="a" * 64),
        replace(proof, capture_sha256="a" * 64),
        replace(proof, published_monotonic_ns=value.published_ns + 1),
    ):
        assert not cache.verifies(invalid, replace(observed, initial_publication=invalid))
    assert not cache.verifies(proof, replace(observed, state_revision=observed.state_revision + 1))
    assert cache.publication == value and cache.consumed
