"""CPU bootstrap sequencing tests for the actual missing-asset-environment GPU failure."""

import os
from pathlib import Path

import pytest
from test_probe_shutdown import probe as probe

from simulation import assets, probe_control


def test_verified_asset_bootstrap_sets_both_paths_from_the_pinned_bundle(monkeypatch):
    checksum = "a" * 64
    expected = Path("/data/assets") / checksum / "Franka" / "franka.usd"
    calls = []
    credential = object()
    monkeypatch.setenv("FRANKA_ASSET_SHA256", checksum)
    monkeypatch.setenv("FRANKA_ASSET_ROOT", "/unverified/old/root")
    monkeypatch.setenv("FRANKA_USD_PATH", "/unverified/old/robot.usd")

    def prepare(actual):
        assert actual is credential
        calls.append(True)
        return expected

    monkeypatch.setattr(assets, "prepare_assets", prepare)
    actual = assets.configure_asset_environment(credential)
    assert calls == [True]
    assert actual == expected
    assert os.environ["FRANKA_USD_PATH"] == str(expected)
    assert os.environ["FRANKA_ASSET_ROOT"] == str(Path("/data/assets") / checksum)


def test_direct_probe_prepares_verified_assets_before_creating_isaac(probe, monkeypatch):
    args, events, at_close, _ = probe
    monkeypatch.delenv("FRANKA_ASSET_ROOT", raising=False)
    monkeypatch.delenv("FRANKA_USD_PATH", raising=False)
    factory = probe_control.create_simulation_app

    def initialize():
        events.append("assets_prepared")
        monkeypatch.setenv("FRANKA_ASSET_ROOT", "/data/assets/" + "a" * 64)
        monkeypatch.setenv("FRANKA_USD_PATH", "/data/assets/" + "a" * 64 + "/franka.usd")

    def application(**kwargs):
        assert os.environ.get("FRANKA_ASSET_ROOT") is not None, (
            "Verified asset root was not initialized"
        )
        assert os.environ.get("FRANKA_USD_PATH") is not None
        return factory(**kwargs)

    class Runtime:
        def __init__(self, core, hardware, **kwargs):
            pass

        def tick(self):
            raise RuntimeError("CPU fixture reached scene activation after verified assets")

        def close(self):
            pass

    monkeypatch.setattr(probe_control, "initialize_probe_assets", initialize)
    monkeypatch.setattr(probe_control, "create_simulation_app", application)
    monkeypatch.setattr(probe_control, "SimulatorRuntime", Runtime)
    with pytest.raises(RuntimeError, match="after verified assets"):
        probe_control.run_probe(args)
    assert events[0] == "assets_prepared"
    assert at_close[0]["phase"] == "scene_activation"
