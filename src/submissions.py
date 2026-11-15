"""Validate and export MOSEv2 predictions; receipts are imported manually only."""

from __future__ import annotations

import shutil
import uuid
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from .io import atomic_json, read_json, sha256_file


def _safe_component(value: str) -> str:
    if (
        not value
        or value in {".", ".."}
        or Path(value).name != value
        or "/" in value
        or "\\" in value
    ):
        raise ValueError("Unsafe submission path component")
    return value


def validate_predictions(
    records, predictions_root: Path, *, expected_count: int = 433
) -> dict:
    """Require every official clip/frame, native indexed masks, and original IDs."""
    records = tuple(records)
    predictions_root = Path(predictions_root)
    if len(records) != expected_count:
        raise ValueError(
            f"Expected {expected_count} official clips, found {len(records)}"
        )
    expected_videos = {record.video_id for record in records}
    if len(expected_videos) != len(records):
        raise ValueError("Duplicate video IDs in official manifest")
    actual_videos = {p.name for p in predictions_root.iterdir() if p.is_dir()}
    if expected_videos != actual_videos:
        raise ValueError(
            f"Submission video set mismatch: missing={sorted(expected_videos - actual_videos)}, "
            f"extra={sorted(actual_videos - expected_videos)}"
        )
    if any(p.is_file() for p in predictions_root.iterdir()):
        raise ValueError("Prediction root must contain only video directories")
    files = []
    for record in records:
        _safe_component(record.video_id)
        directory = predictions_root / record.video_id
        expected_frames = {frame.stem + ".png" for frame in record.frames}
        if len(expected_frames) != len(record.frames):
            raise ValueError("Duplicate output frame names")
        actual_frames = {p.name for p in directory.iterdir()}
        if actual_frames != expected_frames:
            raise ValueError(f"Frame completeness mismatch for {record.video_id}")
        for name in sorted(expected_frames):
            _safe_component(name)
            path = directory / name
            if path.is_symlink() or not path.is_file():
                raise ValueError("Submission predictions must be regular PNG files")
            with Image.open(path) as image:
                if image.format != "PNG" or image.mode not in {"P", "L"}:
                    raise ValueError(f"Expected indexed/8-bit PNG: {path}")
                mask = np.asarray(image)
            if mask.shape != (record.height, record.width):
                raise ValueError(f"Non-native prediction dimensions: {path}")
            unknown = set(map(int, np.unique(mask))) - {0, *record.object_ids}
            if unknown:
                raise ValueError(f"Unknown object IDs {unknown} in {path}")
            files.append(
                {
                    "path": f"{record.video_id}/{name}",
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
            )
    return {
        "video_count": len(records),
        "frame_count": len(files),
        "files": files,
        "layout": "video_id/original_frame_stem.png",
        "initialization": "official first-frame masks",
        "corrective_prompts": 0,
    }


def package_predictions(
    records, predictions_root: Path, zip_path: Path, *, expected_count: int = 433
) -> dict:
    manifest = validate_predictions(
        records, predictions_root, expected_count=expected_count
    )
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        raise FileExistsError(f"Refusing to overwrite a submission: {zip_path}")
    temporary = zip_path.with_name(zip_path.name + ".partial-" + uuid.uuid4().hex)
    try:
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=4
        ) as archive:
            for entry in manifest["files"]:
                path = Path(predictions_root) / entry["path"]
                archive.write(path, arcname=entry["path"])
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None or set(archive.namelist()) != {
                entry["path"] for entry in manifest["files"]
            }:
                raise RuntimeError("Submission ZIP integrity verification failed")
        temporary.replace(zip_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    manifest["zip_path"] = str(zip_path)
    manifest["zip_sha256"] = sha256_file(zip_path)
    atomic_json(zip_path.with_suffix(".manifest.json"), manifest)
    return {
        "zip_path": str(zip_path),
        "manifest_path": str(zip_path.with_suffix(".manifest.json")),
        "video_count": manifest["video_count"],
        "frame_count": manifest["frame_count"],
        "zip_sha256": manifest["zip_sha256"],
    }


def export_official(
    run, checkpoint="initial", *, predictor=None, device: str = "cuda"
) -> dict:
    from .data import load_partition
    from .evaluate import checkpoint_tag
    from .inference import track_record
    from .runs import stage
    from .sam2_adapter import build_predictor, checkpoint_path

    records = load_partition(run, "official_valid")
    if len(records) != run.config.data.official_valid_count:
        raise ValueError(
            "The complete official validation manifest is required before export"
        )
    tag = checkpoint_tag(checkpoint)
    output = run.path / "submissions" / tag
    archive = output / f"mosev2-{tag}.zip"
    model_identity = sha256_file(checkpoint_path(run, checkpoint))
    identity_path = output / "identity.json"
    identity = {
        "checkpoint_sha256": model_identity,
        "checkpoint": str(checkpoint),
        "video_ids": [record.video_id for record in records],
        "protocol": "official first-frame mask initialization; no corrections",
    }
    if archive.exists() and not identity_path.exists():
        raise ValueError("Submission archive is missing its checkpoint/data identity")
    if identity_path.exists() and read_json(identity_path) != identity:
        raise ValueError("Submission checkpoint/data identity changed")
    if archive.exists():
        manifest = read_json(archive.with_suffix(".manifest.json"))
        if manifest["zip_sha256"] != sha256_file(archive):
            raise ValueError("Submission ZIP checksum changed")
        return {
            "zip_path": str(archive),
            "manifest_path": str(archive.with_suffix(".manifest.json")),
            "reused": True,
        }
    atomic_json(identity_path, identity)
    owned = predictor is None
    with stage(run, f"official-export-{tag}"):
        if owned:
            predictor = build_predictor(run, checkpoint=checkpoint, device=device)
        predictions = output / "predictions"
        predictions.mkdir(parents=True, exist_ok=True)
        try:
            for record in tqdm(records, desc=f"Official validation: {tag}"):
                tracked = track_record(
                    run,
                    record,
                    checkpoint,
                    "mask",
                    output_dir=output / "tracking" / record.video_id,
                    predictor=predictor,
                    device=device,
                )
                destination = predictions / record.video_id
                destination.mkdir(exist_ok=True)
                for frame in record.frames:
                    source = Path(tracked["labels_dir"]) / (frame.stem + ".png")
                    shutil.copy2(source, destination / source.name)
        finally:
            if owned:
                del predictor
        result = package_predictions(
            records,
            predictions,
            archive,
            expected_count=run.config.data.official_valid_count,
        )
        guide = output / "SUBMISSION.md"
        guide.write_text(
            "# Manual MOSEv2 submission\n\n"
            "Submit the adjacent ZIP through https://www.codabench.org/competitions/10062/ using your account. "
            "Confirm the current official phase and file-layout instructions before submitting. "
            "This package contains all official validation clips, initialized from supplied first-frame masks, "
            "with no corrective prompts. The ZIP contains only video directories and indexed PNGs.\n\n"
            "Save the unmodified server receipt or downloaded results, then call "
            "`import_submission_receipt(run, checkpoint, receipt_path, submission_id=..., reported_metrics=...)`. "
            "Reported metrics are manually transcribed and kept separate from local scoring. "
            "Do not use server results to select or tune a checkpoint.\n",
            encoding="utf-8",
        )
    return {**result, "guide": str(guide), "submitted": False}


def import_submission_receipt(
    run,
    checkpoint,
    receipt_path: Path,
    *,
    submission_id: str,
    reported_metrics: dict[str, float] | None = None,
    result_url: str | None = None,
) -> dict:
    """Preserve an actual receipt; never parse uncertain text into invented scores."""
    from .evaluate import checkpoint_tag

    source = Path(receipt_path)
    if not source.is_file() or source.stat().st_size == 0:
        raise ValueError("Provide the actual nonempty server receipt file")
    if not submission_id.strip():
        raise ValueError("An official submission ID is required")
    metrics = dict(reported_metrics or {})
    if any(
        not isinstance(v, (int, float)) or not np.isfinite(v) for v in metrics.values()
    ):
        raise ValueError("Manually reported metric values must be finite numbers")
    tag = checkpoint_tag(checkpoint)
    submission = run.path / "submissions" / tag
    archive = submission / f"mosev2-{tag}.zip"
    if not archive.exists():
        raise ValueError(
            "Create the local submission package before attaching its receipt"
        )
    checksum = sha256_file(source)
    output = submission / "receipts" / checksum[:16]
    output.mkdir(parents=True, exist_ok=True)
    destination = output / ("receipt" + source.suffix)
    shutil.copy2(source, destination)
    result = {
        "submission_id": submission_id,
        "result_url": result_url,
        "receipt": str(destination),
        "receipt_sha256": checksum,
        "zip_sha256": sha256_file(archive),
        "checkpoint": str(checkpoint),
        "reported_metrics": metrics,
        "metrics_source": "user transcription from attached official receipt",
        "local_and_official_scores_are_distinct": True,
    }
    artifact = output / "receipt.json"
    if artifact.exists() and read_json(artifact) != result:
        raise ValueError("Receipt metadata already exists with different transcription")
    atomic_json(artifact, result)
    return {"artifact": str(artifact), "submission_id": submission_id}
