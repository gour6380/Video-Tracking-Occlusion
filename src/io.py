"""Content identities and durable atomic artifact writes."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str
        ).encode()
    ).hexdigest()


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_write(path: Path | str, data: bytes) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as out:
            out.write(data)
            out.flush()
            os.fsync(out.fileno())
        os.replace(name, path)
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        Path(name).unlink(missing_ok=True)
    return path


def atomic_json(path: Path | str, value: Any) -> Path:
    return atomic_write(
        path,
        (
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str)
            + "\n"
        ).encode(),
    )


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text())


def verified_json(path: Path | str, value: Any) -> Path:
    path = Path(path)
    if path.exists() and read_json(path) != value:
        raise ValueError(f"Incompatible evidence already exists: {path}")
    if not path.exists():
        atomic_json(path, value)
    return path
