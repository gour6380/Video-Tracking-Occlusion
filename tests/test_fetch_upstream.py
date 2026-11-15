"""Offline source-restore tests: tiny in-memory archives and mocked fetches."""

import hashlib
import importlib.util
import io
import json
import tarfile
from pathlib import Path

import pytest


@pytest.fixture
def script(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/fetch_upstream.py"
    spec = importlib.util.spec_from_file_location("_test_fetch_upstream", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    return module


def _fixture(script, tmp_path, *, complete=True):
    contents = {
        "vendor/sam2/sam2/model.py": b"# model fixture\n",
        "vendor/sam2/LICENSE": b"fixture license\n",
        "vendor/mosev2/metricsv2.py": b"# metrics fixture\n",
        "vendor/mosev2/score.py": b"# score fixture\n",
        "vendor/mosev2/LICENSE": b"fixture license\n",
    }
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for relative, body in contents.items():
            if not relative.startswith("vendor/sam2/"):
                continue
            name = f"sam2-{script.SAM_REV}/" + relative.removeprefix("vendor/sam2/")
            info = tarfile.TarInfo(name)
            info.size = len(body)
            archive.addfile(info, io.BytesIO(body))
    data = buffer.getvalue()
    url = f"https://codeload.github.com/facebookresearch/sam2/tar.gz/{script.SAM_REV}"
    manifest = {
        "sam2_revision": script.SAM_REV,
        "sam2_archive_url": url,
        "sam2_archive_sha256": hashlib.sha256(data).hexdigest(),
        "mose_scorer_revision": script.MOSE_REV,
        "files": {
            name: hashlib.sha256(body).hexdigest() for name, body in contents.items()
        },
    }
    manifest_path = tmp_path / "artifacts/upstream.json"
    manifest_path.parent.mkdir()
    manifest_path.write_text(json.dumps(manifest, indent=4) + "\n")
    if complete:
        for name, body in contents.items():
            target = tmp_path / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(body)
    calls = []

    def fetch(requested):
        calls.append(requested)
        if requested == url:
            return data
        return contents["vendor/mosev2/" + requested.rsplit("/", 1)[-1]]

    return contents, manifest_path, fetch, calls


def test_complete_verified_checkout_does_not_fetch_or_rewrite(
    script, tmp_path, monkeypatch
):
    contents, manifest, fetch, calls = _fixture(script, tmp_path)
    before = {name: (tmp_path / name).stat().st_mtime_ns for name in contents}
    manifest_bytes = manifest.read_bytes()
    monkeypatch.setattr(script, "fetch", fetch)
    script.main()
    assert calls == []
    assert before == {name: (tmp_path / name).stat().st_mtime_ns for name in contents}
    assert manifest.read_bytes() == manifest_bytes


@pytest.mark.parametrize("entire_vendor_missing", [False, True])
def test_missing_sources_are_restored_without_rewriting_verified_files(
    script, tmp_path, monkeypatch, entire_vendor_missing
):
    contents, manifest, fetch, calls = _fixture(
        script, tmp_path, complete=not entire_vendor_missing
    )
    preserved = tmp_path / "vendor/sam2/LICENSE"
    before = preserved.stat().st_mtime_ns if preserved.exists() else None
    if not entire_vendor_missing:
        (tmp_path / "vendor/mosev2/score.py").unlink()
    manifest_bytes = manifest.read_bytes()
    monkeypatch.setattr(script, "fetch", fetch)
    script.main()
    assert len(calls) == 4
    assert all(
        (tmp_path / name).read_bytes() == body for name, body in contents.items()
    )
    assert manifest.read_bytes() == manifest_bytes
    if before is not None:
        assert preserved.stat().st_mtime_ns == before


def test_altered_existing_source_rejected_before_fetch(script, tmp_path, monkeypatch):
    _contents, _manifest, fetch, calls = _fixture(script, tmp_path)
    altered = tmp_path / "vendor/sam2/sam2/model.py"
    altered.write_bytes(b"user alteration")
    missing = tmp_path / "vendor/mosev2/score.py"
    missing.unlink()
    monkeypatch.setattr(script, "fetch", fetch)
    with pytest.raises(RuntimeError, match="Existing pinned upstream source changed"):
        script.main()
    assert calls == [] and altered.read_bytes() == b"user alteration"
    assert not missing.exists()


@pytest.mark.parametrize("changed_payload", ["archive", "mose", "file_inventory"])
def test_fetched_mismatch_publishes_nothing_and_preserves_all_existing_files(
    script, tmp_path, monkeypatch, changed_payload
):
    contents, manifest, fetch, _calls = _fixture(script, tmp_path)
    missing = tmp_path / "vendor/sam2/sam2/model.py"
    missing.unlink()
    preserved = {
        name: (tmp_path / name).read_bytes()
        for name in contents
        if (tmp_path / name).exists()
    }
    manifest_before = manifest.read_bytes()
    if changed_payload == "file_inventory":
        modified = json.loads(manifest_before)
        modified["files"]["vendor/mosev2/additional.py"] = "0" * 64
        manifest.write_text(json.dumps(modified))
        manifest_before = manifest.read_bytes()

    def changed_fetch(url):
        result = fetch(url)
        if changed_payload == "archive" and "codeload" in url:
            return result + b"changed"
        if changed_payload == "mose" and url.endswith("/score.py"):
            return b"# changed fetched score\n"
        return result

    monkeypatch.setattr(script, "fetch", changed_fetch)
    with pytest.raises(RuntimeError, match="checksum changed|differ from the pinned"):
        script.main()
    assert not missing.exists()
    assert manifest.read_bytes() == manifest_before
    assert all(
        (tmp_path / name).read_bytes() == body for name, body in preserved.items()
    )


def test_changed_existing_source_during_fetch_is_rejected(
    script, tmp_path, monkeypatch
):
    _contents, _manifest, fetch, _calls = _fixture(script, tmp_path)
    missing = tmp_path / "vendor/sam2/sam2/model.py"
    missing.unlink()
    edited = tmp_path / "vendor/sam2/LICENSE"

    def editing_fetch(url):
        result = fetch(url)
        edited.write_bytes(b"concurrent local edit")
        return result

    monkeypatch.setattr(script, "fetch", editing_fetch)
    with pytest.raises(RuntimeError, match="Existing pinned upstream source changed"):
        script.main()
    assert not missing.exists() and edited.read_bytes() == b"concurrent local edit"


def _alias_archive(script, links):
    root = f"sam2-{script.SAM_REV}"
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        info = tarfile.TarInfo(f"{root}/sam2/configs/sam2/sam2_hiera_t.yaml")
        body = b"model: test\n"
        info.size = len(body)
        archive.addfile(info, io.BytesIO(body))
        for name, kind, target in links:
            alias = tarfile.TarInfo(f"{root}/{name}")
            alias.type = kind
            alias.linkname = target
            archive.addfile(alias)
    return buffer.getvalue()


def test_legacy_aliases_are_omitted_without_reading_through_links(script):
    links = [
        (
            f"sam2/sam2_hiera_{size}.yaml",
            tarfile.SYMTYPE,
            f"configs/sam2/sam2_hiera_{size}.yaml",
        )
        for size in ("b+", "l", "s", "t")
    ]
    payload = script._source_payload(_alias_archive(script, links))
    assert payload == {
        "vendor/sam2/sam2/configs/sam2/sam2_hiera_t.yaml": b"model: test\n"
    }


@pytest.mark.parametrize(
    ("name", "kind", "target"),
    [
        ("sam2/unexpected.yaml", tarfile.SYMTYPE, "configs/sam2/sam2_hiera_t.yaml"),
        ("sam2/sam2_hiera_t.yaml", tarfile.SYMTYPE, "configs/sam2/sam2_hiera_s.yaml"),
        ("sam2/sam2_hiera_t.yaml", tarfile.SYMTYPE, "../../outside"),
        ("sam2/sam2_hiera_t.yaml", tarfile.LNKTYPE, "configs/sam2/sam2_hiera_t.yaml"),
    ],
)
def test_unexpected_config_links_are_rejected(script, name, kind, target):
    with pytest.raises(
        RuntimeError, match="regular file|Unexpected legacy config alias"
    ):
        script._source_payload(_alias_archive(script, [(name, kind, target)]))
