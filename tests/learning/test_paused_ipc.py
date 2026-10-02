from copy import deepcopy

import pytest

from learning.checks.fixtures import JOINTS, SCOPE
from learning.common import ContractError
from tests.learning.test_paused_contract import context, observation, profile
from tests.learning.test_paused_inference import Policy


def request():
    from learning.paused.ipc import make_request

    return make_request(
        observation(),
        context(),
        model_sha256="c" * 64,
        profile=profile(),
        task=Policy.task,
        request_id="paused-request",
        sequence=0,
        now_ns=2_000_000_000,
    )


def test_paused_wire_v2_preserves_sensor_input_and_original_absolute_deadline():
    from learning.paused.ipc import validate_request

    value = request()
    decoded, ctx = validate_request(
        value,
        model_sha256="c" * 64,
        scope=SCOPE,
        profile=profile(),
        task=Policy.task,
        now_ns=2_100_000_000,
    )
    assert value["schema"] == "physicalai.smolvla-request/v2"
    assert value["real_time_admission"] is False
    assert decoded.sha256 == observation().sha256
    assert decoded.images["inspection"].png == observation().images["inspection"].png
    assert ctx.operation_deadline_ns == 4_000_000_000
    assert not {"object_pose", "seed", "success", "frozen_state_sha256"} & set(value["observation"])


@pytest.mark.parametrize(
    "change",
    ["old-wire", "scope", "image", "deadline", "freeze", "model", "private-label"],
)
def test_paused_request_rejects_rebinding_and_unreviewed_features(change):
    from learning.paused.ipc import validate_request

    value = deepcopy(request())
    if change == "old-wire":
        value["schema"] = "physicalai.smolvla-request/v1"
    elif change == "scope":
        value["context"]["scope"]["owner_id"] = "0" * 64
    elif change == "image":
        value["observation"]["images"]["inspection"]["png_base64"] = "not-an-image"
    elif change == "deadline":
        value["context"]["operation_deadline_ns"] += 1
    elif change == "freeze":
        value["freeze_id"] = "another-freeze"
    elif change == "model":
        value["model_sha256"] = "0" * 64
    else:
        value["observation"]["object_pose"] = [1, 2, 3]
    with pytest.raises(ContractError):
        validate_request(
            value,
            model_sha256="c" * 64,
            scope=SCOPE,
            profile=profile(),
            task=Policy.task,
            now_ns=2_100_000_000,
        )


def test_paused_response_has_real_wall_latency_not_rt_admission():
    from learning.paused.ipc import make_response, validate_response

    value = request()
    response = make_response(value, (JOINTS,) * 50, inference_latency_ms=250)
    assert response["schema"] == "physicalai.smolvla-response/v2"
    assert response["inference_latency_ms"] == 250
    assert response["real_time_admission"] is False
    assert validate_response(response, value) == (JOINTS,) * 50
    with pytest.raises(ContractError):
        make_response(value, (JOINTS,) * 50, inference_latency_ms=2001)
    with pytest.raises(ContractError):
        make_response(value, (JOINTS[:6],) * 50, inference_latency_ms=250)


def test_first_proof_wire_contains_no_private_physical_state():
    from learning.paused.ipc import make_request, validate_request
    from tests.learning.test_initial_frozen_publication import first_context, first_observation

    value = make_request(
        first_observation(),
        first_context(),
        model_sha256="c" * 64,
        profile=profile(),
        task=Policy.task,
        request_id="first-request",
        sequence=0,
        now_ns=3_100_000_000,
    )
    decoded, _ = validate_request(
        value,
        model_sha256="c" * 64,
        scope=SCOPE,
        profile=profile(),
        task=Policy.task,
        now_ns=3_200_000_000,
    )
    assert decoded.monotonic_ns == 2_000_000_000
    assert decoded.ready_ns == 3_100_000_000
    assert not {"frozen_state_sha256", "object_pose", "qvel"} & set(
        value["observation"]["initial_publication"]
    )


def test_legacy_smol_ipc_does_not_accept_new_paused_response():
    from learning.paused.ipc import make_response
    from learning.smolvla.ipc import validate_response

    with pytest.raises(ContractError):
        validate_response(
            make_response(request(), (JOINTS,) * 50, inference_latency_ms=50), request()
        )


def test_real_linux_socket_v2_round_trip_keeps_absolute_request_budget():
    import os
    import socket
    import tempfile
    import time
    from dataclasses import replace
    from pathlib import Path
    from threading import Thread

    from learning.gr00t.ipc import check_peer, receive_packet, send_packet
    from learning.paused.ipc import SocketChunkPolicy, make_response, validate_request

    if os.name != "posix":
        pytest.skip("Deployment IPC uses Linux SO_PEERCRED")
    now = time.monotonic_ns()
    delta = now - 2_000_000_000
    base = observation()
    obs = replace(
        base,
        monotonic_ns=base.monotonic_ns + delta,
        observation_started_ns=base.observation_started_ns + delta,
        joint_sample_ns=base.joint_sample_ns + delta,
        images={
            name: replace(image, monotonic_ns=image.monotonic_ns + delta)
            for name, image in base.images.items()
        },
    )
    base_ctx = context()
    ctx = replace(
        base_ctx,
        observation_sha256=obs.sha256,
        **{
            name: getattr(base_ctx, name) + delta
            for name in (
                "episode_started_ns",
                "wall_deadline_ns",
                "interval_started_ns",
                "interval_deadline_ns",
                "operation_started_ns",
                "operation_deadline_ns",
            )
        },
    )
    errors = []
    with tempfile.TemporaryDirectory(prefix="paused-ipc-") as temporary:
        path = Path(temporary) / "policy.sock"
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
            listener.bind(str(path))
            os.chmod(path, 0o600)
            listener.listen(1)
            listener.settimeout(2)

            def respond():
                try:
                    connection, _ = listener.accept()
                    with connection:
                        check_peer(connection, expected_uid=os.geteuid())
                        value = receive_packet(connection, deadline_ns=ctx.operation_deadline_ns)
                        received, active = validate_request(
                            value,
                            model_sha256="c" * 64,
                            scope=SCOPE,
                            profile=profile(),
                            task=Policy.task,
                        )
                        assert received.sha256 == obs.sha256
                        assert active.operation_deadline_ns == ctx.operation_deadline_ns
                        send_packet(
                            connection,
                            make_response(value, (JOINTS,) * 50, inference_latency_ms=1),
                            deadline_ns=ctx.operation_deadline_ns,
                        )
                except (AssertionError, ContractError, OSError) as exc:
                    errors.append(exc)

            worker = Thread(target=respond, daemon=True)
            worker.start()
            client = SocketChunkPolicy(
                path, scope=SCOPE, model_sha256="c" * 64, profile=profile(), task=Policy.task
            )
            assert client.predict_chunk(obs, ctx) == (JOINTS,) * 50
            worker.join(2)
            assert not worker.is_alive() and not errors
            assert client.predict_calls == 1
