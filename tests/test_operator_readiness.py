"""CPU readiness polling tests. No HTTP, VM operation or motion is performed."""

import pytest

from simulation.operator_readiness import wait_for_known_idle


def snapshot(*, message="No environment is active.", status=200, owner="a" * 64, motion=None):
    return {
        "owner": owner,
        "status_code": status,
        "runtime": {
            "simulation": {
                "status": "unavailable",
                "environment_id": None,
                "epoch": None,
                "motion": motion,
                "message": message,
            }
        },
        "presentation_status": "stopped",
    }


def run(fetch, *, deadline=20):
    now = [0.0]
    sleeps = []

    def sleep(delay):
        sleeps.append(delay)
        now[0] += delay

    result = wait_for_known_idle(
        fetch,
        expected_owner="a" * 64,
        deadline=deadline,
        clock=lambda: now[0],
        sleep=sleep,
    )
    return result, now[0], sleeps


def test_startup_unavailable_can_be_polled_until_known_idle_without_grant_or_window_renewal():
    values = iter(
        [
            snapshot(message="Isaac Sim bridge is unavailable; no fallback was used."),
            snapshot(message="Isaac Sim bridge is unavailable; no fallback was used."),
            snapshot(),
        ]
    )
    result, elapsed, sleeps = run(lambda: next(values))
    assert result == snapshot()
    assert elapsed > 0 and elapsed <= 20
    assert all(delay > 0 for delay in sleeps)


def test_persistent_unavailability_stops_at_the_original_deadline():
    calls = []

    def fetch():
        calls.append(True)
        return snapshot(message="Isaac Sim bridge is unavailable; no fallback was used.")

    with pytest.raises(RuntimeError, match="deadline"):
        run(fetch, deadline=11)
    assert len(calls) <= 4


@pytest.mark.parametrize("status", [401, 403])
def test_authentication_or_authorization_error_is_fatal_without_retry(status):
    calls = []

    def fetch():
        calls.append(True)
        return snapshot(status=status)

    with pytest.raises(RuntimeError, match="authorization"):
        run(fetch)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "value",
    [
        snapshot(owner="b" * 64),
        {"status_code": 200},
        snapshot(motion={"phase": "transporting", "command_id": "active"}),
        snapshot() | {"presentation_status": "moving"},
    ],
)
def test_foreign_malformed_or_running_state_cannot_become_idle(value):
    with pytest.raises(RuntimeError):
        run(lambda: value)


def test_a_slow_fetch_cannot_cross_the_deadline_then_claim_its_idle_result():
    now = [0.0]

    def fetch():
        now[0] = 301
        return snapshot()

    with pytest.raises(RuntimeError, match="deadline"):
        wait_for_known_idle(
            fetch,
            expected_owner="a" * 64,
            deadline=300,
            clock=lambda: now[0],
            sleep=lambda delay: None,
        )
