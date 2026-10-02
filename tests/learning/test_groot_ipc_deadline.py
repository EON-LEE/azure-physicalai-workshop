import socket
import struct

import pytest

from learning.common import ContractError
from learning.gr00t import ipc


def test_dribbling_peer_cannot_restart_the_absolute_receive_budget(monkeypatch):
    now = [1_000_000_000]
    monkeypatch.setattr(ipc.time, "monotonic_ns", lambda: now[0])

    class Dribble:
        calls = 0
        timeouts = []

        def settimeout(self, value):
            self.timeouts.append(value)

        def recv(self, size):
            self.calls += 1
            now[0] += 25_000_000
            return struct.pack("!I", 100) if self.calls == 1 else b" "

    peer = Dribble()
    with pytest.raises(ContractError, match="deadline"):
        ipc.receive_packet(peer, deadline_ns=1_080_000_000)
    assert peer.calls <= 4
    assert peer.timeouts == sorted(peer.timeouts, reverse=True)
    assert peer.timeouts[-1] <= 0.03


def test_expired_send_budget_never_sends_any_bytes(monkeypatch):
    monkeypatch.setattr(ipc.time, "monotonic_ns", lambda: 100)

    class Peer:
        def sendall(self, value):
            pytest.fail("Sent data after the approved deadline")

    with pytest.raises(ContractError, match="deadline"):
        ipc.send_packet(Peer(), {"a": 1}, deadline_ns=100)


def test_peer_uid_is_pinned_before_accepting_policy_data():
    class Peer:
        def getsockopt(self, level, option, size):
            assert level == socket.SOL_SOCKET and option == socket.SO_PEERCRED and size == 12
            return struct.pack("3i", 12, 1001, 1001)

    ipc.check_peer(Peer(), expected_uid=1001)
    with pytest.raises(ContractError, match="peer UID"):
        ipc.check_peer(Peer(), expected_uid=1000)
