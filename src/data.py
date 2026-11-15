"""Pinned MOSEv2 acquisition, conservative duplicate audit and frozen partitions.

No download occurs on import. Acquisition is deliberately a notebook operation.
The cache contains the complete author archives; selecting 400 clips does not
reduce the size of the initial download. Missing annotation files are errors,
not observations of an absent object.
"""

from __future__ import annotations

import hashlib
import io
import os
import random
import shutil
import tarfile
import urllib.request
import uuid
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from functools import wraps
from pathlib import Path, PurePosixPath

from .io import atomic_json, canonical_hash, read_json, sha256_file

REVISION = "24bd8fc5f5ab824a482b5f6dcf472e3a3ef5c704"
REPO_ID = "FudanCVL/MOSEv2"
# Read from the author's SHA256SUMS at this revision, not inferred from filenames.
ARCHIVE_HASHES = {
    "train.tar.gz.aa": "d1bdbd3e7a665b63aaf90db87f1a4e1133430e1c23b5bf86e110e8807688bfe5",
    "train.tar.gz.ab": "a9baa1c92ec4be37efdf8a054e93fc8107e28223b8ff313d24a59b8c3fe4a87c",
    "train.tar.gz.ac": "ef8abb89a35267a74c1a9f3c6b00079e2ee9e344cd9dad98d062e5a072c97899",
    "valid.tar.gz": "0677ff50022ac704aa40efa8aac5a1fe75565911835c702e7d40785ac495a38c",
}
METADATA_HASHES = {
    "SHA256SUMS": "9b8d73807ce34aa9497ada28f8ac318b204b5611616979fad9306463c677a1dc",
    "meta_train.json": "f8ac45f8563f22588ca0e275cbd04c307346db5a46d7a11fdd2b3ff1eaebbec2",
    "meta_valid.json": "22ee710d461a6e9b276c619360a0c758cef0fd95b505ce974dd86f21f552843e",
}
LICENSE_URL = "https://creativecommons.org/licenses/by-nc-sa/4.0/"


def _tracked(function):
    @wraps(function)
    def wrapped(run, *args, **kwargs):
        from .runs import stage

        with stage(run, function.__name__):
            return function(run, *args, **kwargs)

    return wrapped


@dataclass(frozen=True)
class VideoRecord:
    video_id: str
    split: str
    frames: tuple[Path, ...]
    annotations: tuple[Path | None, ...]
    object_ids: tuple[int, ...]
    width: int
    height: int


def _cache(run) -> Path:
    if (run.config.data.repo_id, run.config.data.revision) != (REPO_ID, REVISION):
        raise ValueError(
            "Dataset source changed: review and pin its metadata/checksums before acquisition"
        )
    return run.root / "data" / "mosev2" / REVISION


def _evidence(run, name: str) -> Path:
    return run.path / "data" / name


def _immutable_json(path: Path, value: dict) -> None:
    if path.exists():
        if read_json(path) != value:
            raise ValueError(
                f"Existing data evidence is incompatible; create a new run: {path}"
            )
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(path, value)


def _download(url: str, path: Path, expected: str) -> None:
    """Resume a transport interruption, but only publish fully hashed content."""
    from tqdm.auto import tqdm

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if sha256_file(path) != expected:
            raise ValueError(f"Cached file checksum mismatch (not overwritten): {path}")
        return
    partial = path.with_name(path.name + ".partial")
    # A process can stop after the final write but before atomic publication.
    # Verify/promote that complete file before issuing a Range at EOF (HTTP 416).
    if partial.exists() and sha256_file(partial) == expected:
        os.replace(partial, path)
        return
    offset = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url, headers={"User-Agent": "MOSEv2-research/1.0"})
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    with urllib.request.urlopen(request, timeout=120) as response:
        append = offset > 0 and response.status == 206
        if append and not response.headers.get("Content-Range", "").startswith(
            f"bytes {offset}-"
        ):
            raise ValueError("Server returned an incompatible resume range")
        if not append:
            offset = 0
        remaining = response.headers.get("Content-Length")
        total = offset + int(remaining) if remaining else None
        with (
            partial.open("ab" if append else "wb") as out,
            tqdm(
                total=total, initial=offset, unit="B", unit_scale=True, desc=path.name
            ) as progress,
        ):
            while chunk := response.read(8 * 1024 * 1024):
                out.write(chunk)
                progress.update(len(chunk))
            out.flush()
            os.fsync(out.fileno())
    if sha256_file(partial) != expected:
        raise ValueError(
            f"Downloaded checksum mismatch; inspect/remove {partial} before retrying"
        )
    os.replace(partial, path)


class _ConcatenatedReader(io.RawIOBase):
    """Sequential archive parts without allocating another 58 GB joined file."""

    def __init__(self, paths: Iterable[Path]):
        super().__init__()
        self.paths = iter(paths)
        self.current = None

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:
        if self.current is None:
            try:
                self.current = next(self.paths).open("rb")
            except StopIteration:
                return 0
        count = self.current.readinto(buffer)
        if count:
            return count
        self.current.close()
        self.current = None
        return self.readinto(buffer)

    def close(self) -> None:
        if self.current is not None:
            self.current.close()
        super().close()


def _safe_member(name: str) -> Path:
    pure = PurePosixPath(name)
    if pure.is_absolute() or ".." in pure.parts or "\\" in name or "\x00" in name:
        raise ValueError(f"Unsafe archive path: {name!r}")
    if not pure.parts or pure == PurePosixPath("."):
        return Path(".")
    return Path(*pure.parts)


def _extract_archives(paths: list[Path], destination: Path, identities: dict) -> dict:
    marker = destination / "extraction.json"
    if marker.exists():
        previous = read_json(marker)
        if previous.get("archives") != identities:
            raise ValueError(f"Incompatible extraction cache: {destination}")
        return previous
    if destination.exists():
        raise ValueError(
            f"Unverified extraction directory already exists: {destination}"
        )
    staging = destination.with_name(
        destination.name + ".extracting-" + uuid.uuid4().hex
    )
    staging.mkdir(parents=True)
    entries = {}
    try:
        with (
            _ConcatenatedReader(paths) as joined,
            io.BufferedReader(joined) as stream,
            tarfile.open(fileobj=stream, mode="r|gz") as archive,
        ):
            for member in archive:
                relative = _safe_member(member.name)
                if member.isdir():
                    (staging / relative).mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile() or relative == Path("."):
                    raise ValueError(
                        f"Only regular files/directories allowed: {member.name}"
                    )
                key = relative.as_posix()
                if key in entries or key == "extraction.json":
                    raise ValueError(f"Duplicate/reserved archive member: {key}")
                target = staging / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                copied = 0
                with archive.extractfile(member) as source, target.open("xb") as out:
                    while chunk := source.read(1024 * 1024):
                        out.write(chunk)
                        digest.update(chunk)
                        copied += len(chunk)
                if copied != member.size:
                    raise ValueError(f"Truncated archive member: {key}")
                entries[key] = {
                    "sha256": digest.hexdigest(),
                    "size": copied,
                    "mtime_ns": target.stat().st_mtime_ns,
                }
        result = {
            "archives": identities,
            "files": entries,
            "content_identity": canonical_hash(
                {
                    k: {"sha256": v["sha256"], "size": v["size"]}
                    for k, v in entries.items()
                }
            ),
        }
        atomic_json(staging / "extraction.json", result)
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, destination)
        return result
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _split_root(cache: Path, split: str) -> Path:
    extracted = cache / "extracted" / split
    candidates = [extracted, extracted / split]
    matches = [
        p
        for p in candidates
        if (p / "JPEGImages").is_dir() and (p / "Annotations").is_dir()
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one JPEGImages/Annotations root inside {extracted}"
        )
    return matches[0]


def _validate_metadata(value: dict) -> dict:
    videos = value.get("videos")
    if not isinstance(videos, dict) or not videos:
        raise ValueError("Metadata must contain a nonempty videos mapping")
    for video_id, info in videos.items():
        if _safe_member(video_id).parts != (video_id,):
            raise ValueError("Video IDs must be single directory names")
        frames = info["frames"]
        if (
            not frames
            or len(frames) != info["length"]
            or len(set(frames)) != len(frames)
            or info["width"] < 1
            or info["height"] < 1
        ):
            raise ValueError(f"Invalid frame metadata: {video_id}")
        if any(
            _safe_member(n).parts != (n,) or Path(n).suffix.lower() != ".jpg"
            for n in frames
        ):
            raise ValueError(f"Invalid frame filename: {video_id}")
        ids = info["objects"]
        if (
            not ids
            or len(set(ids)) != len(ids)
            or any(type(i) is not int or not 0 < i < 256 for i in ids)
        ):
            raise ValueError(f"Invalid object IDs: {video_id}")
    return videos


def _source_records(run, split: str) -> list[VideoRecord]:
    cache = _cache(run)
    metadata_path = cache / f"meta_{split}.json"
    if sha256_file(metadata_path) != METADATA_HASHES[metadata_path.name]:
        raise ValueError(f"Pinned source metadata changed: {metadata_path}")
    meta = _validate_metadata(read_json(metadata_path))
    root = _split_root(cache, split)
    result = []
    for video_id, info in sorted(meta.items()):
        frames = tuple(root / "JPEGImages" / video_id / name for name in info["frames"])
        masks = tuple(
            root / "Annotations" / video_id / (Path(name).stem + ".png")
            if split == "train" or i == 0
            else None
            for i, name in enumerate(info["frames"])
        )
        for path in (*frames, *(p for p in masks if p is not None)):
            if not path.is_file():
                raise FileNotFoundError(
                    f"Incomplete MOSEv2 {split} data: {path}; missing is not absence"
                )
        result.append(
            VideoRecord(
                video_id,
                split,
                frames,
                masks,
                tuple(info["objects"]),
                info["width"],
                info["height"],
            )
        )
    return result


@_tracked
def acquire_dataset(run) -> dict:
    """Download full official train/valid archives, verify and extract safely."""
    cache = _cache(run)
    base = f"https://huggingface.co/datasets/{REPO_ID}/resolve/{REVISION}/"
    for name, digest in METADATA_HASHES.items():
        _download(base + name, cache / name, digest)
    declared = {}
    for line in (cache / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split()
        declared[name.lstrip("*")] = digest
    if declared != ARCHIVE_HASHES:
        raise ValueError(
            "Author checksum manifest differs from the reviewed pinned manifest"
        )
    for name, digest in ARCHIVE_HASHES.items():
        _download(base + name, cache / "archives" / name, digest)
    extractions = {}
    for split in ("train", "valid"):
        parts = {n: h for n, h in ARCHIVE_HASHES.items() if n.startswith(split + ".")}
        extracted = _extract_archives(
            [cache / "archives" / n for n in parts], cache / "extracted" / split, parts
        )
        extractions[split] = extracted["content_identity"]
    counts = {split: len(_source_records(run, split)) for split in ("train", "valid")}
    if counts != {"train": 3666, "valid": run.config.data.official_valid_count}:
        raise ValueError(f"Pinned dataset count mismatch: {counts}")
    evidence = {
        "repo_id": REPO_ID,
        "revision": REVISION,
        "license": "CC-BY-NC-SA-4.0",
        "license_url": LICENSE_URL,
        "source": "https://github.com/henghuiding/MOSE-api/tree/master/MOSEv2",
        "archive_sha256": ARCHIVE_HASHES,
        "metadata_sha256": METADATA_HASHES,
        "extractions": extractions,
        "counts": counts,
        "cache": str(cache.relative_to(run.root)),
        "annotation_policy": "train: every frame; valid: first frame only; missing is not absence",
    }
    _immutable_json(_evidence(run, "acquisition.json"), evidence)
    return {
        "train_clips": counts["train"],
        "official_valid_clips": counts["valid"],
        "artifact": str(_evidence(run, "acquisition.json")),
    }


class _UnionFind:
    def __init__(self, keys: Iterable[str]):
        self.parents = {key: key for key in keys}

    def find(self, key: str) -> str:
        while self.parents[key] != key:
            self.parents[key] = self.parents[self.parents[key]]
            key = self.parents[key]
        return key

    def join(self, a: str, b: str) -> None:
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parents[max(a, b)] = min(a, b)


class _HammingTree:
    """BK tree for the sampled 64-bit difference hashes."""

    def __init__(self):
        self.root = None

    def add(self, value: int, payload: tuple) -> None:
        if self.root is None:
            self.root = [value, [payload], {}]
            return
        node = self.root
        while True:
            distance = (value ^ node[0]).bit_count()
            if distance == 0:
                node[1].append(payload)
                return
            if distance not in node[2]:
                node[2][distance] = [value, [payload], {}]
                return
            node = node[2][distance]

    def query(self, value: int, radius: int) -> list[tuple]:
        result, stack = [], [self.root] if self.root else []
        while stack:
            node = stack.pop()
            distance = (value ^ node[0]).bit_count()
            if distance <= radius:
                result.extend(node[1])
            stack.extend(
                child
                for edge, child in node[2].items()
                if distance - radius <= edge <= distance + radius
            )
        return result


def _sample_indices(length: int, count: int) -> list[int]:
    if count < 2:
        raise ValueError("Duplicate audit requires at least two sampled frames")
    return (
        sorted(
            {
                round(i * (length - 1) / (min(length, count) - 1))
                for i in range(min(length, count))
            }
        )
        if length > 1
        else [0]
    )


def _dhash(path: Path) -> int | None:
    import numpy as np
    from PIL import Image

    with Image.open(path) as image:
        gray = np.asarray(
            image.convert("L").resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16
        )
    if float(gray.std()) < 2.0:
        return None  # Blank/near-uniform frames are not reliable scene evidence.
    value = 0
    for bit in (gray[:, 1:] > gray[:, :-1]).flat:
        value = (value << 1) | int(bit)
    return value


def _verified_frame_hash(path: Path, extracted: Path, manifest: dict) -> str:
    key = path.relative_to(extracted).as_posix()
    item = manifest["files"].get(key)
    if item is None:
        raise ValueError(f"Frame absent from extraction manifest: {path}")
    stat = path.stat()
    if stat.st_size != item["size"]:
        raise ValueError(f"Extracted frame changed: {path}")
    if stat.st_mtime_ns != item["mtime_ns"] and sha256_file(path) != item["sha256"]:
        raise ValueError(f"Extracted frame checksum changed: {path}")
    return item["sha256"]


def _extraction_manifests(cache: Path, acquisition: dict) -> dict:
    manifests = {}
    for split in ("train", "valid"):
        manifest = read_json(cache / "extracted" / split / "extraction.json")
        identity = canonical_hash(
            {
                k: {"sha256": v["sha256"], "size": v["size"]}
                for k, v in manifest["files"].items()
            }
        )
        if (
            identity != manifest["content_identity"]
            or identity != acquisition["extractions"][split]
        ):
            raise ValueError(f"Extracted file manifest changed: {split}")
        manifests[split] = manifest
    return manifests


@_tracked
def audit_duplicates(run) -> dict:
    """Audit all image hashes plus sampled visual near-duplicates across splits.

    Groups are an approximation, not proof of common-source independence. Two
    distinct sampled-frame matches are needed for a near-duplicate edge; a byte
    identical frame alone creates a deliberately conservative exact edge.
    """
    from tqdm.auto import tqdm

    acquisition = read_json(_evidence(run, "acquisition.json"))
    cache = _cache(run)
    records = _source_records(run, "train") + _source_records(run, "valid")
    ids = {f"{r.split}/{r.video_id}": r for r in records}
    union = _UnionFind(ids)
    manifests = _extraction_manifests(cache, acquisition)
    seen = {}
    edges = {}
    tree = _HammingTree()
    near_matches = defaultdict(set)
    radius = run.config.data.near_duplicate_hamming
    if not 0 <= radius <= 8:
        raise ValueError("Near-duplicate Hamming radius must be between 0 and 8")
    for key, record in tqdm(ids.items(), desc="Audit video duplicates", unit="clip"):
        for path in record.frames:
            digest = _verified_frame_hash(
                path, cache / "extracted" / record.split, manifests[record.split]
            )
            other = seen.setdefault(digest, key)
            if other != key:
                a, b = sorted((other, key))
                union.join(a, b)
                edges[(a, b)] = "exact_frame_sha256"
        for index in _sample_indices(
            len(record.frames), run.config.data.duplicate_sample_frames
        ):
            value = _dhash(record.frames[index])
            if value is None:
                continue
            aspect = record.width / record.height
            for other, other_index, other_aspect in tree.query(value, radius):
                if other == key or abs(aspect / other_aspect - 1) > 0.02:
                    continue
                a, b = sorted((other, key))
                pair = (other_index, index) if a == other else (index, other_index)
                near_matches[(a, b)].add(pair)
            tree.add(value, (key, index, aspect))
    for (a, b), matches in near_matches.items():
        # Distinct time indices on both clips avoid a single shared title frame.
        if len({x for x, _ in matches}) >= 2 and len({y for _, y in matches}) >= 2:
            union.join(a, b)
            edges.setdefault((a, b), "two_sampled_dhash_matches")
    groups = defaultdict(list)
    for key in ids:
        groups[union.find(key)].append(key)
    ordered_groups = [sorted(members) for _, members in sorted(groups.items())]
    result = {
        "acquisition_identity": canonical_hash(acquisition),
        "method": "exact JPEG SHA-256 or >=2 distinct sampled dHash matches; connected components",
        "limitations": "No original source IDs; sampled visual audit cannot establish source independence",
        "settings": {
            "hamming": radius,
            "sample_frames": run.config.data.duplicate_sample_frames,
            "minimum_gray_std": 2.0,
            "aspect_relative_tolerance": 0.02,
        },
        "groups": ordered_groups,
        "edges": [
            {"a": a, "b": b, "reason": reason}
            for (a, b), reason in sorted(edges.items())
        ],
    }
    result["identity"] = canonical_hash(result)
    _immutable_json(_evidence(run, "duplicates.json"), result)
    cross = sum(
        any(x.startswith("train/") for x in g)
        and any(x.startswith("valid/") for x in g)
        for g in ordered_groups
    )
    return {
        "groups": len(groups),
        "cross_official_split_groups": cross,
        "artifact": str(_evidence(run, "duplicates.json")),
    }


def _choose_groups(
    groups: list[list[str]], count: int, max_groups: int | None = None
) -> list[int]:
    """Nearest whole-group quota; prefer the larger total on equally close ties.

    Targets are approximate. Seeded input order breaks equal-size alternatives;
    no model outcomes influence membership. Reserve groups for later partitions
    with max_groups rather than consuming every available group prematurely.
    """
    if not groups or (max_groups is not None and max_groups <= 0):
        raise ValueError(
            "Not enough independent groups to form all nonempty local partitions"
        )
    upper = count + max(map(len, groups))
    solutions = {0: ()}
    for i, members in enumerate(groups):
        for total, chosen in sorted(solutions.items(), reverse=True):
            target = total + len(members)
            if target > upper or (max_groups is not None and len(chosen) >= max_groups):
                continue
            candidate = (*chosen, i)
            if target not in solutions or len(candidate) < len(solutions[target]):
                solutions[target] = candidate
        if count in solutions:
            return list(solutions[count])
    chosen_total = min(
        (total for total in solutions if total > 0),
        key=lambda total: (abs(total - count), total < count, total),
    )
    return list(solutions[chosen_total])


def _allocate_partitions(
    groups: list[list[str]], lengths: dict[str, int], config
) -> dict[str, list[str]]:
    available = [g for g in groups if all(k.startswith("train/") for k in g)]
    random.Random(config.data.seed).shuffle(available)
    result = {}
    # Reserve the larger training set first, excluding groups containing short clips.
    requests = (
        ("train", config.data.train_clips),
        ("dev", config.data.dev_clips),
        ("test", config.data.test_clips),
    )
    for index, (split, quota) in enumerate(requests):
        eligible = [
            g
            for g in available
            if split != "train" or all(lengths[k] >= config.training.frames for k in g)
        ]
        maximum_groups = len(available) - (len(requests) - index - 1)
        selected = [
            eligible[i]
            for i in _choose_groups(eligible, quota, max_groups=maximum_groups)
        ]
        result[split] = sorted(k for group in selected for k in group)
        taken = {k for group in selected for k in group}
        available = [g for g in available if not taken.intersection(g)]
    return result


def _check_initial_mask(record: VideoRecord) -> None:
    import numpy as np
    from PIL import Image

    path = record.annotations[0]
    if path is None:
        raise ValueError(f"Missing initial annotation: {record.video_id}")
    with Image.open(path) as image:
        mask = np.asarray(image)
    if mask.shape != (record.height, record.width) or mask.ndim != 2:
        raise ValueError(f"Initial mask must be indexed at native resolution: {path}")
    present = {int(x) for x in np.unique(mask)} - {0}
    if present != set(record.object_ids):
        raise ValueError(
            f"Initial mask object IDs differ from metadata for {record.video_id}: {present}"
        )


@_tracked
def freeze_splits(run) -> dict:
    acquisition = read_json(_evidence(run, "acquisition.json"))
    audit = read_json(_evidence(run, "duplicates.json"))
    if audit.get("identity") != canonical_hash(
        {k: v for k, v in audit.items() if k != "identity"}
    ):
        raise ValueError("Duplicate audit identity is invalid")
    if audit["acquisition_identity"] != canonical_hash(acquisition):
        raise ValueError("Duplicate audit belongs to a different acquisition")
    all_records = _source_records(run, "train") + _source_records(run, "valid")
    records = {f"{r.split}/{r.video_id}": r for r in all_records}
    members = [k for g in audit["groups"] for k in g]
    if len(members) != len(set(members)) or set(members) != set(records):
        raise ValueError(
            "Duplicate audit does not partition the complete source dataset"
        )
    selected = _allocate_partitions(
        audit["groups"], {k: len(r.frames) for k, r in records.items()}, run.config
    )
    selected["official_valid"] = sorted(k for k in records if k.startswith("valid/"))
    cache = _cache(run)
    extraction = _extraction_manifests(cache, acquisition)
    serialized = {}
    for split, keys in selected.items():
        serialized[split] = []
        for key in keys:
            r = records[key]
            _check_initial_mask(r)
            files = {}
            for path in (*r.frames, *(p for p in r.annotations if p is not None)):
                _verified_frame_hash(
                    path, cache / "extracted" / r.split, extraction[r.split]
                )
                relative = path.relative_to(cache / "extracted" / r.split).as_posix()
                files[str(path.relative_to(run.root))] = extraction[r.split]["files"][
                    relative
                ]
            serialized[split].append(
                {
                    "video_id": r.video_id,
                    "source_split": r.split,
                    "frames": [str(p.relative_to(run.root)) for p in r.frames],
                    "annotations": [
                        str(p.relative_to(run.root)) if p is not None else None
                        for p in r.annotations
                    ],
                    "object_ids": list(r.object_ids),
                    "width": r.width,
                    "height": r.height,
                    "file_identities": files,
                }
            )
    result = {
        "revision": REVISION,
        "acquisition_identity": canonical_hash(acquisition),
        "duplicate_audit_identity": audit["identity"],
        "seed": run.config.data.seed,
        "clip_targets": {
            "train": run.config.data.train_clips,
            "dev": run.config.data.dev_clips,
            "test": run.config.data.test_clips,
        },
        "allocation_policy": "Nearest feasible whole-group quota, larger total on equal-distance ties; seeded ordering",
        "partition_description": "Held out from this fine-tuning; duplicate-audited clips, not proven source-independent",
        "excluded_official_overlap": sorted(
            k
            for g in audit["groups"]
            if any(x.startswith("valid/") for x in g)
            for k in g
            if k.startswith("train/")
        ),
        "ineligible_for_training_short_clips": sorted(
            k
            for k, r in records.items()
            if k.startswith("train/") and len(r.frames) < run.config.training.frames
        ),
        "partitions": serialized,
    }
    result["identity"] = canonical_hash(result)
    _immutable_json(_evidence(run, "partitions.json"), result)
    return {
        "counts": {k: len(v) for k, v in serialized.items()},
        "identity": result["identity"],
        "artifact": str(_evidence(run, "partitions.json")),
    }


def load_partition(run, split: str = "train") -> list[VideoRecord]:
    if split not in {"train", "dev", "test", "official_valid"}:
        raise ValueError(f"Unknown partition: {split}")
    manifest = read_json(_evidence(run, "partitions.json"))
    identity = manifest.get("identity")
    if identity != canonical_hash(
        {k: v for k, v in manifest.items() if k != "identity"}
    ):
        raise ValueError("Partition manifest identity is invalid")

    def resolve(value, identities):
        if value is None:
            return None
        path = (run.root / value).resolve()
        if not path.is_relative_to(run.root.resolve()) or not path.is_file():
            raise FileNotFoundError(
                f"Missing/unsafe data file: {value}; missing is not absence"
            )
        item = identities.get(value)
        if item is None:
            raise ValueError(f"Data file lacks a frozen identity: {value}")
        stat = path.stat()
        if stat.st_size != item["size"]:
            raise ValueError(f"Frozen data file size changed: {value}")
        # Immutable-cache fast path. A replaced/touched file is checked by content,
        # not trusted merely because the dataset-manifest identity stayed fixed.
        if stat.st_mtime_ns != item["mtime_ns"] and sha256_file(path) != item["sha256"]:
            raise ValueError(f"Frozen data file checksum changed: {value}")
        return path

    result = []
    for item in manifest["partitions"][split]:
        identities = item.get("file_identities", {})
        frames = tuple(resolve(v, identities) for v in item["frames"])
        annotations = tuple(resolve(v, identities) for v in item["annotations"])
        if (
            not frames
            or len(frames) != len(annotations)
            or any(p is None for p in frames)
        ):
            raise ValueError("Frames and annotations must align")
        if split != "official_valid" and any(p is None for p in annotations):
            raise ValueError(
                "Local supervision must be complete; None is reserved for hidden validation labels"
            )
        if split == "official_valid" and (
            annotations[0] is None or any(p is not None for p in annotations[1:])
        ):
            raise ValueError("Official validation exposes only initialization masks")
        result.append(
            VideoRecord(
                item["video_id"],
                split,
                frames,
                annotations,
                tuple(item["object_ids"]),
                item["width"],
                item["height"],
            )
        )
    return result


@_tracked
def inspect_data(run) -> dict:
    """Summarize the frozen manifest without inventing category/presence labels."""
    summary = {}
    for split in ("train", "dev", "test", "official_valid"):
        records = load_partition(run, split)
        summary[split] = {
            "clips": len(records),
            "frames": sum(len(r.frames) for r in records),
            "objects": sum(len(r.object_ids) for r in records),
            "multi_object_clips": sum(len(r.object_ids) > 1 for r in records),
            "min_frames": min(len(r.frames) for r in records),
            "max_frames": max(len(r.frames) for r in records),
            "supervision": "first frame only"
            if split == "official_valid"
            else "all frames",
        }
    _immutable_json(_evidence(run, "statistics.json"), summary)
    return {"partitions": summary, "artifact": str(_evidence(run, "statistics.json"))}
