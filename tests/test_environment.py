"""Device records retain the original hardware and every later observation."""

import sys
from types import SimpleNamespace

import pytest

from src import environment
from src.config import ProjectConfig
from src.io import read_json


def test_hardware_changes_append_observations_without_erasing_initial(
    tmp_path, monkeypatch
):
    state = {"available": True, "name": "A40", "uuid": "fixture-gpu-a"}
    fake_torch = SimpleNamespace(
        __version__="fixture",
        version=SimpleNamespace(cuda="fixture"),
        cuda=SimpleNamespace(
            is_available=lambda: state["available"],
            current_device=lambda: 0,
            get_device_properties=lambda _: SimpleNamespace(
                name=state["name"], total_memory=48 * 1024**3, uuid=state["uuid"]
            ),
            get_device_capability=lambda _: (8, 6),
            is_bf16_supported=lambda: True,
        ),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(
        environment,
        "verify_upstream",
        lambda _: {"sam2_revision": ProjectConfig().model.source_revision},
    )
    monkeypatch.setattr(
        environment,
        "environment_manifest",
        lambda: {"python": "fixture", "packages": {}},
    )
    run = SimpleNamespace(root=tmp_path, path=tmp_path / "run", config=ProjectConfig())
    first = environment.inspect_environment(run, require_experiment_device=True)
    initial_path = run.path / "provenance" / "device.json"
    original = initial_path.read_bytes()
    assert first["hardware_changed_from_initial"] is False
    state.update(name="A100", uuid="fixture-gpu-b")
    second = environment.inspect_environment(run, require_experiment_device=True)
    assert second["hardware_changed_from_initial"] is True
    assert initial_path.read_bytes() == original
    assert read_json(second["artifact"])["device_info"]["name"] == "A100"
    state["available"] = False
    with pytest.raises(RuntimeError, match="CUDA/BF16"):
        environment.inspect_environment(run, require_experiment_device=True)
    observations = sorted(
        (run.path / "provenance" / "device_observations").glob("*.json")
    )
    assert len(observations) == 3
    assert read_json(observations[-1])["device"] == "cpu"
    assert initial_path.read_bytes() == original
