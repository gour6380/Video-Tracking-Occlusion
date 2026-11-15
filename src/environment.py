"""Environment readiness is distinct from the owner's CUDA training pilot."""

from __future__ import annotations

import fcntl
import shutil
import uuid
from pathlib import Path

from .io import atomic_json, read_json, sha256_file
from .runs import environment_manifest, now


def verify_upstream(root: Path) -> dict:
    path = root / "artifacts" / "upstream.json"
    if not path.exists():
        raise RuntimeError(
            "Run scripts/fetch_upstream.py to restore pinned upstream sources"
        )
    manifest = read_json(path)
    missing = [
        p
        for p, h in manifest["files"].items()
        if not (root / p).is_file() or sha256_file(root / p) != h
    ]
    if missing:
        raise ValueError(f"Upstream source identity mismatch: {missing[:5]}")
    return manifest


def inspect_environment(run, *, require_experiment_device=False) -> dict:
    import torch

    manifest = verify_upstream(run.root)
    if manifest["sam2_revision"] != run.config.model.source_revision:
        raise ValueError(
            "Installed upstream source and configured SAM 2 revision differ"
        )
    if torch.cuda.is_available():
        device = "cuda"
        index = torch.cuda.current_device()
        gpu = torch.cuda.get_device_properties(index)
        info = {
            "name": gpu.name,
            "total_memory": gpu.total_memory,
            "capability": list(torch.cuda.get_device_capability(index)),
            "device_index": index,
            "uuid": str(gpu.uuid) if getattr(gpu, "uuid", None) is not None else None,
            "bf16_supported": torch.cuda.is_bf16_supported(),
        }
    elif torch.backends.mps.is_available():
        device, info = "mps", {"name": "Apple Metal", "role": "local development only"}
    else:
        device, info = "cpu", {"name": "CPU", "role": "local development only"}
    result = {
        "environment": environment_manifest(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": device,
        "device_info": info,
        "cuda_experiment_ready": device == "cuda" and info.get("bf16_supported", False),
        "pilot_passed": False,
        "scope": "package/device metadata; no numerical pilot performed",
        "disk_free_bytes": shutil.disk_usage(run.root).free,
    }
    provenance = run.path / "provenance"
    provenance.mkdir(parents=True, exist_ok=True)
    path = provenance / "device.json"
    observed_at = now()
    observation_id = uuid.uuid4().hex
    observation = (
        provenance
        / "device_observations"
        / f"{observed_at.replace(':', '-')}-{observation_id}.json"
    )
    result.update(observed_at=observed_at, observation_id=observation_id)
    # Preserve the first observation for historical compatibility. Later checks
    # append their own records, including failed accelerator readiness checks.
    with (provenance / ".device.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        initial = read_json(path) if path.exists() else result
        changed = (initial["device"], initial["device_info"]) != (
            result["device"],
            result["device_info"],
        )
        result["hardware_changed_from_initial"] = changed
        if not path.exists():
            atomic_json(path, result)
        atomic_json(observation, result)
    if require_experiment_device and not result["cuda_experiment_ready"]:
        raise RuntimeError(
            "Run this experiment notebook on the configured CUDA/BF16 RunPod; local checks are separate"
        )
    return {
        "device": device,
        "details": info,
        "cuda_experiment_ready": result["cuda_experiment_ready"],
        "artifact": str(observation),
        "initial_artifact": str(path),
        "hardware_changed_from_initial": changed,
    }
