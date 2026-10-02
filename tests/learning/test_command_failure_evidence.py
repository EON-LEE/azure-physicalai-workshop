import hashlib

import pytest

from learning.common import ContractError, read_json
from learning.paused.command import _preserve_failure


class CapturingTransfer:
    def publish_files(self, directory, prefix, *, marker, files):
        self.prefix = prefix
        self.marker = marker
        self.files = {name: (directory / name).read_bytes() for name in files}
        for name, expected in files.items():
            assert hashlib.sha256(self.files[name]).hexdigest() == expected


@pytest.mark.parametrize("message", ["Immutable checkpoint already exists", "x" * 70000])
def test_failure_traceback_survives_without_training_log(tmp_path, message):
    transfer = CapturingTransfer()
    try:
        raise ContractError(message)
    except ContractError as error:
        _preserve_failure(
            transfer,
            {"output_prefix": "scope/learning/outputs", "run_id": "failed-run"},
            tmp_path,
            {"azure_job_type": "command"},
            error,
        )

    proof = read_json(tmp_path / "failure" / "failure.json")
    traceback = transfer.files["traceback.txt"]
    assert 0 < len(traceback) <= 65536
    assert message.encode()[-60000:] in traceback
    assert proof["failure_type"] == "ContractError"
    assert proof["traceback_truncated"] is (len(message) > 65536)
    assert proof["candidate_complete"] is False
    assert proof["learning_quality_verified"] is False
    assert transfer.marker == "failure.json"
    assert transfer.prefix == "scope/learning/outputs/failed-run/transfer/failure"
