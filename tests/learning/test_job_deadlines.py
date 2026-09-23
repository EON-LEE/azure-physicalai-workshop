import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from learning.common import ContractError


def test_clock_rollback_cannot_renew_a_live_deadline():
    from learning.deadlines import JobDeadline

    wall = datetime(2026, 9, 23, 12, tzinfo=UTC)
    clocks = {"wall": wall, "mono": 1.0}
    deadline = JobDeadline(
        "2026-09-23T12:01:00Z",
        clock=lambda: clocks["wall"],
        monotonic=lambda: clocks["mono"],
    )
    assert deadline.remaining_seconds() == 60
    clocks.update(wall=wall - timedelta(hours=1), mono=61.0)
    with pytest.raises(ContractError, match="deadline"):
        deadline.check()


def test_forward_clock_jump_expires_even_before_monotonic_budget_elapses():
    from learning.deadlines import JobDeadline

    wall = [datetime(2026, 9, 23, 12, tzinfo=UTC)]
    deadline = JobDeadline("2026-09-23T12:01:00Z", clock=lambda: wall[0], monotonic=lambda: 0.0)
    wall[0] += timedelta(seconds=60)
    with pytest.raises(ContractError, match="deadline"):
        deadline.check()


@pytest.mark.skipif(os.name != "posix", reason="Production component images use Linux processes")
def test_real_bounded_child_runs_without_changing_environment(monkeypatch):
    from learning.deadlines import JobDeadline, run_bounded

    monkeypatch.setenv("PHYSICALAI_DEADLINE_TEST", "inherited")
    value = (datetime.now(UTC) + timedelta(seconds=10)).isoformat().replace("+00:00", "Z")
    run_bounded(
        [
            sys.executable,
            "-c",
            "import os; assert os.environ['PHYSICALAI_DEADLINE_TEST'] == 'inherited'",
        ],
        JobDeadline(value),
    )


@pytest.mark.skipif(os.name != "posix", reason="Production component images use Linux processes")
def test_real_expired_process_group_is_stopped_without_success():
    from learning.deadlines import JobDeadline, run_bounded

    value = (datetime.now(UTC) + timedelta(seconds=0.25)).isoformat().replace("+00:00", "Z")
    with pytest.raises(ContractError, match="deadline"):
        run_bounded([sys.executable, "-c", "import time; time.sleep(30)"], JobDeadline(value))


@pytest.mark.skipif(os.name != "posix", reason="Production component images use Linux processes")
def test_child_failure_remains_nonzero():
    import subprocess

    from learning.deadlines import JobDeadline, run_bounded

    value = (datetime.now(UTC) + timedelta(seconds=10)).isoformat().replace("+00:00", "Z")
    with pytest.raises(subprocess.CalledProcessError) as error:
        run_bounded([sys.executable, "-c", "raise SystemExit(7)"], JobDeadline(value))
    assert error.value.returncode == 7


@pytest.mark.skipif(os.name != "posix", reason="Production component images use Linux processes")
def test_timeout_kills_the_actual_training_descendant_as_well(tmp_path):
    from learning.deadlines import JobDeadline, run_bounded

    pid_file = tmp_path / "descendant.pid"
    script = (
        "import subprocess,sys,time;from pathlib import Path;"
        "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']);"
        f"Path({str(pid_file)!r}).write_text(str(child.pid));time.sleep(30)"
    )
    value = (datetime.now(UTC) + timedelta(seconds=1)).isoformat().replace("+00:00", "Z")
    with pytest.raises(ContractError, match="deadline"):
        run_bounded([sys.executable, "-c", script], JobDeadline(value))
    pid = int(pid_file.read_text())
    state = Path(f"/proc/{pid}/stat")
    assert not state.exists() or state.read_text().split()[2] == "Z"
