"""Restore pinned source code only; never model weights/data and never invoke Git.

All network payloads are collected and checked before publishing anything. An
existing pinned file is never overwritten, even when other source files are
missing from a fresh project copy.
"""

import hashlib
import io
import json
import os
import re
import tarfile
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
SAM_REV = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
MOSE_REV = "a74b23aa3a31597f487eec10f452e0b9f61347e1"
# The pinned archive includes these legacy convenience aliases. Their canonical
# YAML files are restored and checked like all other source files; we never
# create a symlink or read archive contents through one.
_LEGACY_CONFIG_ALIASES = {
    f"sam2/sam2_hiera_{size}.yaml": f"configs/sam2/sam2_hiera_{size}.yaml"
    for size in ("b+", "l", "s", "t")
}


def fetch(url):
    with urllib.request.urlopen(
        urllib.request.Request(
            url, headers={"User-Agent": "tracking-occlusion-setup/1"}
        ),
        timeout=120,
    ) as response:
        return response.read()


def _digest(body):
    return hashlib.sha256(body).hexdigest()


def _source_target(relative):
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or ".." in pure.parts
        or "\\" in relative
        or "\x00" in relative
        or len(pure.parts) < 3
        or pure.parts[:2] not in (("vendor", "sam2"), ("vendor", "mosev2"))
    ):
        raise RuntimeError(f"Unsafe upstream file path: {relative!r}")
    target = ROOT.joinpath(*pure.parts)
    _reject_redirected_path(target)
    return target


def _reject_redirected_path(target):
    if not target.resolve().is_relative_to(ROOT.resolve()):
        raise RuntimeError(f"Upstream path escapes the project: {target}")
    for component in (target, *target.parents):
        if component == ROOT:
            break
        if component.is_symlink():
            raise RuntimeError(f"Symlink in pinned upstream path: {component}")


def _check_existing(files):
    """Reject changes; return the names absent from the current checkout."""
    missing = []
    for relative, expected in files.items():
        target = _source_target(relative)
        if not target.exists():
            missing.append(relative)
        elif not target.is_file() or _digest(target.read_bytes()) != expected:
            raise RuntimeError(f"Existing pinned upstream source changed: {relative}")
    return missing


def _source_payload(archive):
    payload = {}
    allowed = (
        "sam2",
        "training",
        "LICENSE",
        "LICENSE_cctorch",
        "setup.py",
        "pyproject.toml",
        "README.md",
    )
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        for item in tar:
            pure = PurePosixPath(item.name)
            if (
                pure.is_absolute()
                or ".." in pure.parts
                or "\\" in item.name
                or "\x00" in item.name
            ):
                raise RuntimeError(f"Unsafe upstream archive member: {item.name!r}")
            if not pure.parts or pure.parts[0] != f"sam2-{SAM_REV}":
                raise RuntimeError(f"Unexpected upstream archive root: {item.name!r}")
            parts = pure.parts[1:]
            if not parts or parts[0] not in allowed or item.isdir():
                continue
            alias_target = _LEGACY_CONFIG_ALIASES.get("/".join(parts))
            if alias_target is not None:
                if not item.issym() or item.linkname != alias_target:
                    raise RuntimeError(f"Unexpected legacy config alias: {item.name}")
                continue
            if not item.isfile():
                raise RuntimeError(
                    f"Upstream source must be a regular file: {item.name}"
                )
            relative = "vendor/sam2/" + "/".join(parts)
            _source_target(relative)
            if relative in payload:
                raise RuntimeError(f"Duplicate upstream archive member: {relative}")
            source = tar.extractfile(item)
            if source is None:
                raise RuntimeError(f"Unreadable upstream archive member: {relative}")
            with source:
                body = source.read()
            if len(body) != item.size:
                raise RuntimeError(f"Truncated upstream archive member: {relative}")
            payload[relative] = body
    if not payload:
        raise RuntimeError("SAM2 source archive did not contain any selected files")
    return payload


def _publish_missing(target, body):
    """Atomic create without overwriting a file that appeared concurrently."""
    _reject_redirected_path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".upstream-", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            _reject_redirected_path(target)
            if not target.is_file() or _digest(target.read_bytes()) != _digest(body):
                raise RuntimeError(
                    f"Upstream file changed during restoration: {target}"
                )
    finally:
        Path(temporary).unlink(missing_ok=True)


def main():
    manifest_path = ROOT / "artifacts" / "upstream.json"
    _reject_redirected_path(manifest_path)
    url = f"https://codeload.github.com/facebookresearch/sam2/tar.gz/{SAM_REV}"
    expected = None
    if manifest_path.exists():
        expected = json.loads(manifest_path.read_text())
        if (
            expected.get("sam2_revision") != SAM_REV
            or expected.get("mose_scorer_revision") != MOSE_REV
            or expected.get("sam2_archive_url") != url
        ):
            raise RuntimeError(
                "Pinned upstream revision/URL differs from this setup script"
            )
        files = expected.get("files")
        if (
            not isinstance(files, dict)
            or not files
            or any(
                not isinstance(h, str) or not re.fullmatch("[0-9a-f]{64}", h)
                for h in files.values()
            )
        ):
            raise RuntimeError("Invalid pinned upstream file manifest")
        if not _check_existing(files):
            print("Pinned upstream sources already verified")
            return

    archive = fetch(url)
    archive_digest = _digest(archive)
    if expected is not None and archive_digest != expected.get("sam2_archive_sha256"):
        raise RuntimeError("Pinned SAM2 archive checksum changed")
    payload = _source_payload(archive)
    for name in ("metricsv2.py", "score.py", "LICENSE"):
        source_path = (
            "LICENSE" if name == "LICENSE" else f"MOSEv2/sam2_rcms/tools/{name}"
        )
        payload[f"vendor/mosev2/{name}"] = fetch(
            f"https://raw.githubusercontent.com/henghuiding/MOSE-api/{MOSE_REV}/{source_path}"
        )
    files = {relative: _digest(body) for relative, body in payload.items()}
    # Validate the complete fetched payload before creating any source files.
    # A late MOSE mismatch cannot leave an earlier SAM2 payload published.
    if expected is not None and files != expected["files"]:
        raise RuntimeError(
            "Retrieved upstream files differ from the pinned source manifest"
        )
    missing = _check_existing(files)
    manifest = {
        "sam2_revision": SAM_REV,
        "sam2_archive_url": url,
        "sam2_archive_sha256": archive_digest,
        "mose_scorer_revision": MOSE_REV,
        "files": files,
    }
    for relative in missing:
        _publish_missing(_source_target(relative), payload[relative])
    if expected is None:
        body = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
        _publish_missing(manifest_path, body)
    print(
        f"Restored {len(missing)} of {len(files)} verified upstream source files; no weights or datasets"
    )


if __name__ == "__main__":
    main()
