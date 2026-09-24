from dataclasses import asdict, replace

import pytest

from learning.common import ContractError, canonical, parse_json
from tests.learning.test_paused_contract import context, observation, profile


def first_observation():
    from learning.paused import InitialFrozenPublication

    original = observation()
    publication = InitialFrozenPublication(
        publication_record_id="actual-publication-1",
        publication_record_sha256="d" * 64,
        capture_sha256=original.capture_sha256,
        freeze_established_ns=1_050_000_000,
        published_at_utc=original.captured_at_utc,
        published_monotonic_ns=original.monotonic_ns,
        age_at_observation_start_ns=1_000_000_000,
    )
    return replace(
        original,
        observation_started_ns=3_000_000_000,
        observation_completed_ns=3_100_000_000,
        initial_publication=publication,
    )


def first_context():
    value = first_observation()
    return replace(
        context(),
        episode_started_ns=3_000_000_000,
        wall_deadline_ns=603_000_000_000,
        interval_started_ns=3_000_000_000,
        interval_deadline_ns=8_000_000_000,
        operation_started_ns=3_100_000_000,
        operation_deadline_ns=5_100_000_000,
        observation_sha256=value.sha256,
    )


def test_first_original_publication_preserves_timestamps_and_counts_new_observation_work():
    value = first_observation()
    value.validate(profile(), now_ns=3_100_000_000)
    first_context().validate(profile(), value, now_ns=3_100_000_000)
    assert value.monotonic_ns == 2_000_000_000
    assert value.joint_sample_ns == 1_100_000_000
    assert value.images["inspection"].monotonic_ns == 1_900_000_000
    assert value.ready_ns == 3_100_000_000
    assert value.capture_sha256 == observation().capture_sha256
    assert not {"frozen_state_sha256", "object_pose", "qvel"} & set(value.metadata())
    assert not {"frozen_state_sha256", "object_pose", "qvel"} & set(
        asdict(value.initial_publication)
    )


@pytest.mark.parametrize(
    "change",
    [
        "not-first",
        "no-proof",
        "backdated-freeze",
        "old-publication",
        "wrong-age",
        "changed-rgb",
        "restamped",
        "wrong-epoch",
        "work-too-long",
        "missing-record",
    ],
)
def test_initial_publication_is_not_a_global_chronology_waiver(change):
    value = first_observation()
    if change == "not-first":
        value = replace(value, control_tick=1)
    elif change == "no-proof":
        value = replace(value, initial_publication=None)
    elif change == "backdated-freeze":
        value = replace(
            value,
            initial_publication=replace(
                value.initial_publication, freeze_established_ns=1_100_000_001
            ),
        )
    elif change == "old-publication":
        value = replace(
            value,
            observation_started_ns=4_000_000_001,
            observation_completed_ns=4_100_000_000,
            initial_publication=replace(
                value.initial_publication, age_at_observation_start_ns=2_000_000_001
            ),
        )
    elif change == "wrong-age":
        value = replace(
            value,
            initial_publication=replace(value.initial_publication, age_at_observation_start_ns=1),
        )
    elif change == "changed-rgb":
        from learning.checks.fixtures import png

        value = replace(
            value,
            images={
                **value.images,
                "inspection": replace(value.images["inspection"], png=png(color=90)),
            },
        )
    elif change == "restamped":
        value = replace(value, monotonic_ns=3_000_000_000)
    elif change == "wrong-epoch":
        value = replace(value, epoch="other-epoch")
    elif change == "work-too-long":
        value = replace(value, observation_completed_ns=5_000_000_001)
    else:
        value = replace(
            value,
            initial_publication=replace(value.initial_publication, publication_record_id=""),
        )
    with pytest.raises(ContractError):
        value.validate(profile(), now_ns=6_000_000_000)


def test_later_capture_keeps_original_strict_new_publication_chronology():
    value = observation()
    with pytest.raises(ContractError):
        replace(value, observation_started_ns=2_000_000_001).validate(
            profile(), now_ns=3_000_000_000
        )


def test_valid_hashes_alone_do_not_authorize_initial_model_inference():
    from types import SimpleNamespace

    from learning.paused.inference import PausedGuardedPolicyAdapter
    from tests.learning.test_paused_inference import Policy

    clock = SimpleNamespace(now=3_100_000_000)
    policy = Policy(clock)
    adapter = PausedGuardedPolicyAdapter(policy, profile=profile(), clock_ns=lambda: clock.now)
    adapter.reset(first_context())
    with pytest.raises(ContractError, match="publication"):
        adapter.step(first_observation(), first_context())
    assert policy.calls == 0


@pytest.mark.parametrize("authorized", [True, False])
def test_runtime_resolver_is_required_before_and_after_prediction(authorized):
    from types import SimpleNamespace

    from learning.paused.inference import PausedGuardedPolicyAdapter
    from tests.learning.test_paused_inference import Policy

    clock = SimpleNamespace(now=3_100_000_000)
    policy = Policy(clock)
    calls = []

    def guard(publication, value, ctx):
        calls.append(publication.publication_record_id)
        return authorized

    adapter = PausedGuardedPolicyAdapter(
        policy, profile=profile(), clock_ns=lambda: clock.now, publication_guard=guard
    )
    adapter.reset(first_context())
    if authorized:
        command = adapter.step(first_observation(), first_context())
        assert command.observation_sha256 == first_observation().sha256
        assert calls == ["actual-publication-1", "actual-publication-1"]
    else:
        with pytest.raises(ContractError, match="publication"):
            adapter.step(first_observation(), first_context())
        assert policy.calls == 0


def test_changed_private_state_after_prediction_cannot_return_a_command():
    from types import SimpleNamespace

    from learning.paused.inference import PausedGuardedPolicyAdapter
    from tests.learning.test_paused_inference import Policy

    clock = SimpleNamespace(now=3_100_000_000)
    policy = Policy(clock)
    allowed = iter((True, False))
    adapter = PausedGuardedPolicyAdapter(
        policy,
        profile=profile(),
        clock_ns=lambda: clock.now,
        publication_guard=lambda *args: next(allowed),
    )
    adapter.reset(first_context())
    with pytest.raises(ContractError, match="publication"):
        adapter.step(first_observation(), first_context())
    assert policy.calls == 1


def test_same_record_hash_cannot_authorize_a_forged_epoch_or_unknown_record():
    from types import SimpleNamespace

    from learning.paused.inference import PausedGuardedPolicyAdapter
    from tests.learning.test_paused_inference import Policy

    original = first_observation()
    wrong = replace(original, epoch="wrong-epoch")
    wrong = replace(
        wrong,
        initial_publication=replace(wrong.initial_publication, capture_sha256=wrong.capture_sha256),
    )
    clock = SimpleNamespace(now=3_100_000_000)
    policy = Policy(clock)
    registered = {
        original.initial_publication.publication_record_id: (
            original.initial_publication.publication_record_sha256,
            original.capture_sha256,
            original.epoch,
        )
    }

    def guard(publication, value, ctx):
        return registered.get(publication.publication_record_id) == (
            publication.publication_record_sha256,
            value.capture_sha256,
            ctx.epoch,
        )

    adapter = PausedGuardedPolicyAdapter(
        policy, profile=profile(), clock_ns=lambda: clock.now, publication_guard=guard
    )
    ctx = replace(first_context(), epoch=wrong.epoch, observation_sha256=wrong.sha256)
    adapter.reset(ctx)
    with pytest.raises(ContractError, match="publication"):
        adapter.step(wrong, ctx)
    assert policy.calls == 0


def test_raw_frame_roundtrip_preserves_original_publication_and_new_work_times():
    from learning.paused.capture import _decode_frame, _frame_record
    from tests.learning.test_paused_contract import sample

    value = replace(
        sample(),
        observation=first_observation(),
        policy_started_ns=3_100_000_000,
        policy_finished_ns=3_350_000_000,
        hold_started_ns=3_400_000_000,
        hold_deadline_ns=5_400_000_000,
        interval_deadline_ns=8_000_000_000,
        applied_controls=tuple(
            replace(control, monotonic_ns=control.monotonic_ns + 1_100_000_000)
            for control in sample().applied_controls
        ),
    )
    value.validate(profile())
    record = parse_json(canonical(_frame_record(value, value.observation.episode_id, 0)))
    decoded = _decode_frame(
        record, {name: image.png for name, image in value.observation.images.items()}, 0
    )
    decoded.validate(profile())
    assert decoded.observation.monotonic_ns == 2_000_000_000
    assert decoded.observation.ready_ns == 3_100_000_000
    assert decoded.observation.initial_publication == value.observation.initial_publication
