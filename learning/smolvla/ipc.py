from learning.gr00t.ipc import (
    RemoteGuardedPolicyAdapter,
    make_response,
    validate_response,
)
from learning.gr00t.ipc import SocketChunkPolicy as BaseSocketChunkPolicy
from learning.gr00t.ipc import make_request as base_make_request
from learning.gr00t.ipc import validate_request as base_validate_request
from learning.smolvla import POLICY_TYPE

__all__ = [
    "RemoteGuardedPolicyAdapter",
    "SocketChunkPolicy",
    "make_request",
    "make_response",
    "validate_request",
    "validate_response",
]


class SocketChunkPolicy(BaseSocketChunkPolicy):
    def __init__(self, socket_path, *, scope, model_sha256, profile, task, expected_peer_uid=None):
        super().__init__(
            socket_path,
            scope=scope,
            model_sha256=model_sha256,
            profile=profile,
            task=task,
            expected_peer_uid=expected_peer_uid,
            policy_type=POLICY_TYPE,
        )


def make_request(observation, context, **kwargs):
    return base_make_request(observation, context, policy_type=POLICY_TYPE, **kwargs)


def validate_request(request, **kwargs):
    return base_validate_request(request, policy_type=POLICY_TYPE, **kwargs)
