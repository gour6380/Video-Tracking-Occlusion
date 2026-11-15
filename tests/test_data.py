"""Synthetic-only tests: no network, dataset, model or GPU is required."""

import functools
import hashlib
import io
import operator
import tarfile
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from src.config import ProjectConfig
from src.data import (
    _allocate_partitions,
    _choose_groups,
    _download,
    _extract_archives,
    _HammingTree,
    _sample_indices,
    _validate_metadata,
    load_partition,
)
from src.io import atomic_json, canonical_hash, sha256_file


def _archive(tmp_path, members):
    path = tmp_path / "source.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        for name, payload, kind in members:
            info = tarfile.TarInfo(name)
            if kind == "symlink":
                info.type = tarfile.SYMTYPE
                info.linkname = "../../outside"
                archive.addfile(info)
            else:
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
    return path


def test_split_archive_stream_and_extraction_manifest(tmp_path):
    archive = _archive(
        tmp_path,
        [
            ("train/JPEGImages/a/00000.jpg", b"jpeg fixture", "file"),
            ("train/Annotations/a/00000.png", b"mask fixture", "file"),
        ],
    )
    raw = archive.read_bytes()
    parts = [tmp_path / "aa", tmp_path / "ab"]
    parts[0].write_bytes(raw[:19])
    parts[1].write_bytes(raw[19:])
    output = tmp_path / "extracted"
    identities = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in parts}
    manifest = _extract_archives(parts, output, identities)
    key = "train/JPEGImages/a/00000.jpg"
    assert (output / key).read_bytes() == b"jpeg fixture"
    assert (
        manifest["files"][key]["sha256"] == hashlib.sha256(b"jpeg fixture").hexdigest()
    )
    assert _extract_archives(parts, output, identities) == manifest
    with pytest.raises(ValueError, match="Incompatible"):
        _extract_archives(parts, output, {"different": "identity"})


@pytest.mark.parametrize(
    "name,kind",
    [
        ("../escape", "file"),
        ("/absolute", "file"),
        ("a/../../escape", "file"),
        ("link", "symlink"),
    ],
)
def test_archive_rejects_unsafe_members_without_publishing(tmp_path, name, kind):
    path = _archive(tmp_path, [(name, b"bad", kind)])
    destination = tmp_path / "output"
    with pytest.raises(ValueError):
        _extract_archives([path], destination, {"source": "fixture"})
    assert not destination.exists()
    assert not list(tmp_path.glob("output.extracting-*"))


def test_archive_rejects_duplicate_files(tmp_path):
    path = _archive(tmp_path, [("same", b"a", "file"), ("same", b"b", "file")])
    with pytest.raises(ValueError, match="Duplicate"):
        _extract_archives([path], tmp_path / "out", {})


def test_download_only_publishes_matching_checksum(tmp_path, monkeypatch):
    class Response(io.BytesIO):
        status = 200

        def __init__(self, content):
            super().__init__(content)
            self.headers = {"Content-Length": str(len(content))}

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: Response(b"fixture"))
    path = tmp_path / "asset"
    _download(
        "https://example.invalid/fixture", path, hashlib.sha256(b"fixture").hexdigest()
    )
    assert path.read_bytes() == b"fixture"
    wrong = tmp_path / "wrong"
    with pytest.raises(ValueError, match="checksum mismatch"):
        _download("https://example.invalid/fixture", wrong, "0" * 64)
    assert not wrong.exists()


def test_complete_partial_is_published_without_range_request(tmp_path, monkeypatch):
    path = tmp_path / "asset"
    partial = tmp_path / "asset.partial"
    partial.write_bytes(b"completed before interruption")
    expected = hashlib.sha256(partial.read_bytes()).hexdigest()

    def forbidden_request(*args, **kwargs):
        raise AssertionError(
            "A verified complete partial must not issue a request at EOF"
        )

    monkeypatch.setattr("urllib.request.urlopen", forbidden_request)
    _download("https://example.invalid/fixture", path, expected)
    assert path.read_bytes() == b"completed before interruption"
    assert not partial.exists()


def test_hamming_tree_matches_brute_force():
    values = [0, 1, 3, 7, 32, 255, 1024, 0]
    tree = _HammingTree()
    for i, value in enumerate(values):
        tree.add(value, (i,))
    for query in (0, 9, 255):
        assert sorted(tree.query(query, 2)) == [
            (i,) for i, value in enumerate(values) if (query ^ value).bit_count() <= 2
        ]


def test_sampling_covers_endpoints_and_does_not_repeat():
    assert _sample_indices(3, 8) == [0, 1, 2]
    sampled = _sample_indices(50, 8)
    assert sampled[0] == 0 and sampled[-1] == 49 and len(sampled) == 8


def test_split_allocation_preserves_groups_and_excludes_official_overlap():
    config = ProjectConfig()
    config = replace(
        config, data=replace(config.data, train_clips=3, dev_clips=2, test_clips=1)
    )
    groups = [
        ["train/a", "train/b"],
        ["train/c"],
        ["train/d"],
        ["train/e"],
        ["train/f"],
        ["train/short"],
        ["train/leak", "valid/x"],
    ]
    lengths = {key: 8 for group in groups for key in group}
    lengths["train/short"] = 3
    selected = _allocate_partitions(groups, lengths, config)
    assert len(selected["train"]) == 3 and len(selected["dev"]) == 2
    assert len(selected["test"]) in (
        1,
        2,
    )  # The final indivisible group may exceed its target.
    assert "train/short" not in selected["train"]
    assert "train/leak" not in functools.reduce(operator.iadd, selected.values(), [])
    assert len(set(functools.reduce(operator.iadd, selected.values(), []))) == sum(
        map(len, selected.values())
    )
    owners = {key: split for split, keys in selected.items() for key in keys}
    assert owners.get("train/a") == owners.get("train/b")
    assert selected == _allocate_partitions(groups, lengths, config)


def test_indivisible_group_quota_uses_nearest_count_and_larger_tie():
    groups = [["a", "b"], ["c", "d"]]
    assert sum(len(groups[i]) for i in _choose_groups(groups, 3)) == 4
    groups = [[f"{g}/{i}" for i in range(3)] for g in range(20)]
    assert sum(len(groups[i]) for i in _choose_groups(groups, 40)) == 39


def test_approximate_allocation_with_only_three_clip_groups():
    groups = [[f"train/{g}-{i}" for i in range(3)] for g in range(150)]
    lengths = {key: 10 for group in groups for key in group}
    selected = _allocate_partitions(groups, lengths, ProjectConfig())
    assert {key: len(value) for key, value in selected.items()} == {
        "train": 300,
        "dev": 39,
        "test": 60,
    }
    assert len(set(functools.reduce(operator.iadd, selected.values(), []))) == 399


def test_acquisition_failure_has_a_stage_receipt(tmp_path, monkeypatch):
    from src.data import acquire_dataset
    from src.io import read_json
    from src.runs import start_run

    run = start_run(root=tmp_path)

    def fail_download(*args):
        raise ValueError("synthetic transport failure")

    monkeypatch.setattr("src.data._download", fail_download)
    with pytest.raises(ValueError, match="synthetic"):
        acquire_dataset(run)
    receipt = read_json(run.path / "status.json")["stages"]["acquire_dataset"]
    assert receipt["state"] == "failed" and "synthetic" in receipt["error"]


def _manifest_run(tmp_path, split="train", missing=False):
    root = tmp_path / "project"
    root.mkdir()
    frames, annotations, identities = [], [], {}
    for i in range(3):
        frame = root / f"{i}.jpg"
        mask = root / f"{i}.png"
        Image.fromarray(np.zeros((4, 5, 3), dtype=np.uint8)).save(frame)
        Image.fromarray(np.ones((4, 5), dtype=np.uint8)).save(mask)
        frames.append(frame.name)
        annotations.append(mask.name if split != "official_valid" or i == 0 else None)
        for path in (frame, mask):
            identities[path.name] = {
                "sha256": sha256_file(path),
                "size": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
    if missing:
        (root / "1.png").unlink()
    manifest = {
        "partitions": {
            split: [
                {
                    "video_id": "fixture",
                    "frames": frames,
                    "annotations": annotations,
                    "object_ids": [1],
                    "width": 5,
                    "height": 4,
                    "file_identities": identities,
                }
            ]
        }
    }
    manifest["identity"] = canonical_hash(manifest)
    run = SimpleNamespace(root=root, path=root / "runs/run_001", config=ProjectConfig())
    atomic_json(run.path / "data/partitions.json", manifest)
    return run


def test_missing_annotation_never_becomes_absent(tmp_path):
    run = _manifest_run(tmp_path, missing=True)
    with pytest.raises(FileNotFoundError, match="missing is not absence"):
        load_partition(run)


def test_hidden_validation_annotation_is_explicit_none(tmp_path):
    run = _manifest_run(tmp_path, split="official_valid")
    record = load_partition(run, "official_valid")[0]
    assert len(record.frames) == len(record.annotations) == 3
    assert record.annotations[0].is_file()
    assert record.annotations[1:] == (None, None)


def test_modified_data_is_rejected_even_if_manifest_unchanged(tmp_path):
    run = _manifest_run(tmp_path)
    path = run.root / "1.png"
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 1
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="checksum changed"):
        load_partition(run)


def test_metadata_rejects_traversal_and_misaligned_length():
    info = {
        "frames": ["00000.jpg"],
        "length": 2,
        "width": 10,
        "height": 10,
        "objects": [1],
    }
    with pytest.raises(ValueError, match="frame metadata"):
        _validate_metadata({"videos": {"clip": info}})
    info["length"] = 1
    with pytest.raises(ValueError, match="Unsafe"):
        _validate_metadata({"videos": {"../outside": info}})
