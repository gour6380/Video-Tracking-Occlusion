"""Atomic run numbering, provenance, strict resume and stage receipts."""

from __future__ import annotations

import ast
import fcntl
import importlib.metadata
import platform
import re
import shutil
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .config import ProjectConfig, config_from_dict
from .io import atomic_json, canonical_hash, read_json, sha256_file


def now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class RunContext:
    root: Path
    path: Path
    config: ProjectConfig

    @property
    def name(self) -> str:
        return self.path.name


def _cell_identity(cell: dict) -> dict:
    source = (
        "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
    )
    if "operational-input" not in cell.get("metadata", {}).get("tags", []):
        return {"cell_type": cell["cell_type"], "source": source}
    allowed = {
        "run_name",
        "initial_receipt_path",
        "best_receipt_path",
        "initial_submission_id",
        "best_submission_id",
        "initial_reported_metrics",
        "best_reported_metrics",
    }
    names = []
    for statement in ast.parse(source).body:
        if (
            not isinstance(statement, ast.Assign)
            or len(statement.targets) != 1
            or not isinstance(statement.targets[0], ast.Name)
        ):
            raise ValueError(
                "Operational notebook cells may contain only literal input assignments"
            )
        name = statement.targets[0].id
        if name not in allowed:
            raise ValueError(
                f"Experimental setting cannot be excluded from source identity: {name}"
            )
        ast.literal_eval(statement.value)
        names.append(name)
    return {"cell_type": "code", "operational_input_names": names}


def source_manifest(root: Path) -> dict[str, str]:
    files = []
    for folder in ("src", "notebooks", "scripts"):
        base = root / folder
        if base.exists():
            files.extend(
                p
                for p in base.rglob("*")
                if p.is_file()
                and "__pycache__" not in p.parts
                and ".ipynb_checkpoints" not in p.parts
            )
    files.extend(p for p in root.glob("requirements.*") if p.is_file())
    files.extend(
        p for p in (root / "environments").rglob("requirements.*") if p.is_file()
    )
    for p in (
        root / "artifacts" / "upstream.json",
        root / ".python-version",
        root / "ruff.toml",
        root / "pytest.ini",
    ):
        if p.exists():
            files.append(p)
    result = {}
    for p in sorted(set(files)):
        if p.suffix == ".ipynb":
            book = read_json(p)
            # Execution counts, displayed results and transient widget state do not change code.
            content = [_cell_identity(c) for c in book["cells"]]
            result[str(p.relative_to(root))] = canonical_hash(content)
        else:
            result[str(p.relative_to(root))] = sha256_file(p)
    return result


def environment_manifest() -> dict:
    packages = sorted(
        (d.metadata.get("Name", "").lower().replace("_", "-"), d.version)
        for d in importlib.metadata.distributions()
    )
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "executable": sys.executable,
        "packages": dict(packages),
    }


def start_run(
    config: ProjectConfig | None = None, *, root: Path | str | None = None
) -> RunContext:
    config = config or ProjectConfig()
    config.validate()
    root = Path(root or Path(__file__).resolve().parents[1]).resolve()
    parent = root / "runs"
    parent.mkdir(parents=True, exist_ok=True)
    with (parent / ".allocation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        numbers = [
            int(p.name[4:])
            for p in parent.iterdir()
            if re.fullmatch(r"run_\d+", p.name)
        ]
        number = max(numbers, default=0) + 1
        while True:
            path = parent / f"run_{number:03d}"
            try:
                path.mkdir()
                break
            except FileExistsError:
                number += 1
        for name in (
            "provenance",
            "data",
            "checkpoints",
            "logs",
            "metrics",
            "evaluation",
            "pilot",
            "benchmarks",
            "figures",
            "reports",
            "submissions",
            "demo",
        ):
            (path / name).mkdir()
        atomic_json(path / "config.json", config.to_dict())
        sources = source_manifest(root)
        for relative in sources:
            target = path / "provenance" / "source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / relative, target)
        atomic_json(path / "provenance" / "source_manifest.json", sources)
        atomic_json(path / "provenance" / "environment.json", environment_manifest())
        atomic_json(
            path / "identity.json",
            {
                "config": canonical_hash(config.to_dict()),
                "source": canonical_hash(sources),
            },
        )
        atomic_json(
            path / "status.json",
            {"created": now(), "updated": now(), "state": "prepared", "stages": {}},
        )
    return RunContext(root, path, config)


def resume_run(run_name: str, *, root: Path | str | None = None) -> RunContext:
    if not isinstance(run_name, str) or not re.fullmatch(r"run_\d+", run_name):
        raise ValueError("Resume requires an explicit run_XXX name")
    root = Path(root or Path(__file__).resolve().parents[1]).resolve()
    path = root / "runs" / run_name
    config = config_from_dict(read_json(path / "config.json"))
    run = RunContext(root, path, config)
    # The same lock guards normal stages and the recovery state transition.
    # A live process must never be mistaken for an interrupted predecessor.
    with (path / ".stage.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "Cannot resume while another operation is writing this run"
            ) from exc
        validate_run(run)
        status = read_json(path / "status.json")
        for entry in status["stages"].values():
            if entry["state"] == "running":
                entry.update(
                    state="interrupted",
                    ended=now(),
                    reason="Previous process did not complete its stage",
                )
        status.update(updated=now(), state="resumed")
        atomic_json(path / "status.json", status)
    return run


def validate_run(run: RunContext) -> None:
    saved = read_json(run.path / "identity.json")
    if saved["config"] != canonical_hash(run.config.to_dict()):
        raise ValueError("Run configuration changed; create a new run")
    if saved["source"] != canonical_hash(source_manifest(run.root)):
        raise ValueError(
            "Source/notebook/dependency specification changed; create a new run"
        )
    env = read_json(run.path / "provenance" / "environment.json")
    current = environment_manifest()
    if (env["python"], env["packages"], env["machine"]) != (
        current["python"],
        current["packages"],
        current["machine"],
    ):
        raise ValueError(
            "Installed environment changed; resume with the recorded environment or create a new run"
        )


_active_stages = threading.local()


@contextmanager
def stage(run: RunContext, name: str) -> Iterator[None]:
    validate_run(run)
    status_path = run.path / "status.json"
    held = getattr(_active_stages, "paths", set())
    nested = str(run.path) in held
    lock = None
    if not nested:
        lock = (run.path / ".stage.lock").open("a")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            lock.close()
            raise RuntimeError("Another operation is writing this run") from exc
        _active_stages.paths = held | {str(run.path)}
    try:
        status = read_json(status_path)
        previous = status["stages"].get(name, {})
        status["stages"][name] = {
            "state": "running",
            "started": now(),
            "attempt": previous.get("attempt", 0) + 1,
        }
        status.update(updated=now(), state="running")
        atomic_json(status_path, status)
        try:
            yield
        except BaseException as exc:
            status = read_json(status_path)
            status["stages"][name].update(
                state="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                ended=now(),
                error=f"{type(exc).__name__}: {exc}",
            )
            status.update(state=status["stages"][name]["state"], updated=now())
            atomic_json(status_path, status)
            raise
        else:
            status = read_json(status_path)
            status["stages"][name].update(state="completed", ended=now())
            status.update(state="running" if nested else "ready", updated=now())
            atomic_json(status_path, status)
    finally:
        if lock is not None:
            _active_stages.paths = held
            lock.close()
