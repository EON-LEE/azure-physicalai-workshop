import pytest

from scripts.run_gpu_command import guest_succeeded


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Enable succeeded: [stdout]\nPHYSICALAI_REMOTE_EXIT=0\n\n[stderr]\n", True),
        ("Enable succeeded", False),
        ("PHYSICALAI_REMOTE_EXIT=1\n", False),
        ("PHYSICALAI_REMOTE_EXIT=01\n", False),
        ("PHYSICALAI_REMOTE_EXIT=0\nPHYSICALAI_REMOTE_EXIT=1\n", False),
        ("PHYSICALAI_REMOTE_EXIT=0\nPHYSICALAI_REMOTE_EXIT=0\n", False),
        ("not-the-marker: PHYSICALAI_REMOTE_EXIT=0", False),
    ],
)
def test_azure_ack_never_substitutes_for_an_unambiguous_guest_exit(message, expected):
    assert guest_succeeded({"value": [{"message": message}]}) is expected


def test_missing_guest_output_fails_closed():
    assert guest_succeeded({}) is False
