"""Transactional update-boundary snapshots, with model/state stored separately."""

from __future__ import annotations

import copy
import os
import random
import shutil
import tempfile
import uuid
from pathlib import Path

from .io import atomic_json, canonical_hash, read_json, sha256_file


def identity(run):
    partitions = read_json(run.path / "data" / "partitions.json")
    source = {
        str(path.relative_to(run.root)): sha256_file(path)
        for path in sorted((run.root / "src").rglob("*.py"))
    }
    return {
        "config": canonical_hash(run.config.to_dict()),
        "data": partitions["identity"],
        "project_source": canonical_hash(source),
        "sam2_revision": run.config.model.source_revision,
    }


def cpu_tree(value):
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return copy.deepcopy(value)


def capture_rng(model):
    import torch

    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "python": random.getstate(),
        "prompt": copy.deepcopy(model.rng.bit_generator.state),
    }


def restore_rng(model, state):
    import torch

    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])
    random.setstate(state["python"])
    model.rng.bit_generator.state = state["prompt"]


def scheduler_record(config, completed_updates):
    """The learning-rate schedule has no mutable object beyond its update position."""
    return {
        "kind": "stateless_linear_warmup_cosine",
        "formula_version": 1,
        "completed_updates": completed_updates,
        "next_update_one_based": completed_updates + 1
        if completed_updates < config.updates
        else None,
        "warmup_updates": config.warmup_updates,
        "total_updates": config.updates,
        "peak_learning_rate": config.learning_rate,
        "final_learning_rate": config.final_learning_rate,
        "formula": "step=completed+1; peak*step/warmup if step<=warmup; otherwise final+0.5*(peak-final)*(1+cos(pi*(step-warmup)/(total-warmup)))",
    }


def save_checkpoint(
    run,
    model,
    optimizer,
    *,
    update,
    cursor,
    best_score=None,
    best_update=None,
    pending_validation=False,
    directory=None,
):
    import torch

    parent = Path(directory) if directory else run.path / "checkpoints"
    parent.mkdir(parents=True, exist_ok=True)
    name = f"update_{update:06d}_{uuid.uuid4().hex[:8]}"
    temporary = Path(tempfile.mkdtemp(prefix=".pending-", dir=parent))
    destination = parent / name
    try:
        for filename, payload in (
            ("model.pt", {"model": cpu_tree(model.state_dict())}),
            (
                "state.pt",
                {
                    "optimizer": cpu_tree(optimizer.state_dict()),
                    "rng": capture_rng(model),
                    "update": update,
                    "cursor": cursor,
                    "best_score": best_score,
                    "best_update": best_update,
                    "pending_validation": pending_validation,
                    "scheduler": scheduler_record(run.config.training, update),
                    "identity": identity(run),
                },
            ),
        ):
            with (temporary / filename).open("wb") as stream:
                torch.save(payload, stream)
                stream.flush()
                os.fsync(stream.fileno())
        record = {
            "directory": name,
            "update": update,
            "cursor": cursor,
            "model_sha256": sha256_file(temporary / "model.pt"),
            "state_sha256": sha256_file(temporary / "state.pt"),
        }
        atomic_json(temporary / "manifest.json", record)
        os.replace(temporary, destination)
        atomic_json(parent / "latest.json", record)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination / "model.pt"


def promote_best(run, model_path):
    record = read_json(Path(model_path).parent / "manifest.json")
    atomic_json(run.path / "checkpoints" / "best.json", record)


def restore_checkpoint(run, model, optimizer, *, directory=None):
    import torch

    parent = Path(directory) if directory else run.path / "checkpoints"
    record = read_json(parent / "latest.json")
    source = parent / record["directory"]
    for filename, key in [("model.pt", "model_sha256"), ("state.pt", "state_sha256")]:
        if sha256_file(source / filename) != record[key]:
            raise ValueError(f"Checkpoint checksum mismatch: {filename}")
    state = torch.load(source / "state.pt", map_location="cpu", weights_only=True)
    if state["identity"] != identity(run):
        raise ValueError(
            "Cannot resume: configuration, data, or source identity changed"
        )
    if state.get("scheduler") != scheduler_record(run.config.training, state["update"]):
        raise ValueError(
            "Cannot resume: scheduler formula, configuration, or update position changed"
        )
    model.load_state_dict(
        torch.load(source / "model.pt", map_location="cpu", weights_only=True)["model"],
        strict=True,
    )
    optimizer.load_state_dict(state["optimizer"])
    restore_rng(model, state["rng"])
    optimizer.zero_grad(set_to_none=True)
    return state


def is_better(score, previous):
    import math

    if not math.isfinite(score):
        raise ValueError("Development score must be finite")
    return previous is None or score > previous
