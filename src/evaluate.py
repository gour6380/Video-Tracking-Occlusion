"""Full-clip local evaluation with untouched and paired correction metrics."""

from __future__ import annotations

import time
from pathlib import Path

from tqdm.auto import tqdm

from .io import atomic_json, read_json, sha256_file


def checkpoint_tag(checkpoint) -> str:
    value = str(checkpoint)
    if value in {"initial", "best", "latest"}:
        return value
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(path)
    return f"{path.stem}-{sha256_file(path)[:12]}"


def evaluate_local(
    run,
    checkpoint="initial",
    split: str = "test",
    protocol: str = "point",
    *,
    predictor=None,
    device: str = "cuda",
) -> dict:
    """Return primary_score (J&F_dot in 0..1 units) and complete saved diagnostics.

    point: primary metrics exclude only first/last frames; paired metrics also
    exclude the predeclared midpoint. point_correction uses those same paired
    frame exclusions. Thus assisted and unassisted comparisons cannot gain from
    scoring the annotated correction frame.
    """
    from .data import load_partition
    from .inference import track_record
    from .metrics import aggregate_scores, score_video
    from .runs import stage
    from .sam2_adapter import build_predictor, checkpoint_path

    if split not in {"train", "dev", "test"}:
        raise ValueError(
            "Official validation has no complete labels; use export_official instead"
        )
    if protocol not in {"point", "point_correction", "mask"}:
        raise ValueError("Unknown evaluation protocol")
    records = load_partition(run, split)
    if not records:
        raise ValueError(f"No videos in {split}")
    model_path = checkpoint_path(run, checkpoint)
    identity = {
        "checkpoint_sha256": sha256_file(model_path),
        "split": split,
        "video_ids": [record.video_id for record in records],
        "protocol": protocol,
        "scorer_revision": run.config.evaluation.scorer_revision,
        "partition_manifest_sha256": sha256_file(run.path / "data" / "partitions.json"),
    }
    directory = run.path / "evaluation" / split / checkpoint_tag(checkpoint) / protocol
    artifact = directory / "scores.json"
    if artifact.exists():
        previous = read_json(artifact)
        if previous["identity"] != identity:
            raise ValueError(
                "Evaluation identity changed; refusing to overwrite prior evidence"
            )
        if [item["video_id"] for item in previous["videos"]] != identity["video_ids"]:
            raise ValueError("Evaluation video inventory changed")
        manifest = previous.get("prediction_manifest", [])
        if len(manifest) == sum(len(record.frames) for record in records) and all(
            Path(item["path"]).is_file() and sha256_file(item["path"]) == item["sha256"]
            for item in manifest
        ):
            return {
                "checkpoint": str(checkpoint),
                "split": split,
                "protocol": protocol,
                "primary_score": previous["aggregate"]["J&F_dot"],
                "score_artifact": str(artifact),
                "reused": True,
            }
        raise ValueError(
            "Evaluation evidence is incomplete; inspect it before retrying"
        )
    owned = predictor is None
    with stage(run, f"evaluate-{split}-{checkpoint_tag(checkpoint)}-{protocol}"):
        if owned:
            predictor = build_predictor(run, checkpoint=checkpoint, device=device)
        scores, paired_scores, video_results, prediction_manifest = [], [], [], []
        started = time.perf_counter()
        try:
            for record in tqdm(
                records, desc=f"{split}: {checkpoint_tag(checkpoint)} / {protocol}"
            ):
                if len(record.annotations) != len(record.frames) or any(
                    p is None for p in record.annotations
                ):
                    raise ValueError(
                        f"Missing supervision in {record.video_id}; absence cannot be inferred"
                    )
                output_dir = directory / "videos" / record.video_id
                tracking = track_record(
                    run,
                    record,
                    checkpoint,
                    protocol,
                    output_dir=output_dir,
                    predictor=predictor,
                    device=device,
                )
                predictions = tuple(
                    Path(tracking["labels_dir"]) / (frame.stem + ".png")
                    for frame in record.frames
                )
                prediction_manifest.extend(
                    {"path": str(path), "sha256": sha256_file(path)}
                    for path in predictions
                )
                midpoint = len(record.frames) // 2
                score = score_video(
                    record.annotations,
                    predictions,
                    record.object_ids,
                    exclude_frames=(midpoint,)
                    if protocol == "point_correction"
                    else (),
                )
                paired = (
                    score_video(
                        record.annotations,
                        predictions,
                        record.object_ids,
                        exclude_frames=(midpoint,),
                    )
                    if protocol == "point"
                    else score
                )
                score["video_id"] = record.video_id
                paired["video_id"] = record.video_id
                atomic_json(
                    output_dir / "scores.json", {"primary": score, "paired": paired}
                )
                scores.append(score)
                paired_scores.append(paired)
                video_results.append(
                    {
                        "video_id": record.video_id,
                        "tracking": tracking,
                        "score": score,
                        "paired_score": paired,
                    }
                )
        finally:
            if owned:
                del predictor
        aggregate = aggregate_scores(scores)
        result = {
            "identity": identity,
            "aggregate": aggregate,
            "paired_aggregate": aggregate_scores(paired_scores),
            "videos": video_results,
            "prediction_manifest": prediction_manifest,
            "elapsed_seconds": time.perf_counter() - started,
            "correction_note": "Simulated annotation assistance; no automatic failure detection",
            "frame_policy": {
                "always_exclude": "first and last",
                "primary_point": "other frames untouched",
                "paired_point_and_correction": "also exclude each clip's predeclared midpoint",
            },
            "initial_prompt_rule": "one frozen EDT-interior point per object or full initialization mask",
            "corrective_clicks": sum(
                item["tracking"]["corrective_clicks"] for item in video_results
            ),
            "scale": "raw 0..1; display percentages only when explicitly labelled",
        }
        atomic_json(artifact, result)
    return {
        "checkpoint": str(checkpoint),
        "split": split,
        "protocol": protocol,
        "primary_score": aggregate["J&F_dot"],
        "score_artifact": str(artifact),
        "reused": False,
    }


def render_comparison_examples(
    run, *, split: str = "test", protocol: str = "point", fps: float = 24.0
) -> dict:
    """Render largest paired improvement/regression and lowest adapted-score clip.

    FPS is a declared playback choice for released frame sequences; it is not a
    measurement of original capture timing. Example selection never tunes weights.
    """
    from .data import load_partition
    from .video import render_video

    baseline_path = (
        run.path / "evaluation" / split / "initial" / protocol / "scores.json"
    )
    adapted_path = run.path / "evaluation" / split / "best" / protocol / "scores.json"
    baseline, adapted = read_json(baseline_path), read_json(adapted_path)
    base_videos = {v["video_id"]: v for v in baseline["videos"]}
    tuned_videos = {v["video_id"]: v for v in adapted["videos"]}
    if set(base_videos) != set(tuned_videos):
        raise ValueError("Cannot compare examples from different clip partitions")
    records = {r.video_id: r for r in load_partition(run, split)}
    value = lambda item: item["score"]["summary"]["J&F_dot"]
    ids = sorted(
        key
        for key in base_videos
        if value(base_videos[key]) is not None and value(tuned_videos[key]) is not None
    )
    if not ids:
        raise ValueError("No clips have eligible scored frames in both checkpoints")
    delta = {key: value(tuned_videos[key]) - value(base_videos[key]) for key in ids}
    selections = [
        ("largest_increase", max(ids, key=lambda k: delta[k])),
        ("largest_decrease", min(ids, key=lambda k: delta[k])),
        ("lowest_adapted", min(ids, key=lambda k: value(tuned_videos[k]))),
    ]
    output = run.path / "examples" / f"{split}-{protocol}"
    examples = []
    for reason, video_id in selections:
        record = records[video_id]
        videos = {}
        for tag, entry in (
            ("initial", base_videos[video_id]),
            ("best", tuned_videos[video_id]),
        ):
            masks = [
                Path(entry["tracking"]["labels_dir"]) / f"{frame.stem}.png"
                for frame in record.frames
            ]
            videos[tag] = str(
                render_video(
                    record.frames,
                    masks,
                    output / f"{reason}-{video_id}-{tag}.mp4",
                    fps=fps,
                    title=f"{tag}: {video_id}",
                )
            )
        examples.append(
            {
                "reason": reason,
                "video_id": video_id,
                "J&F_dot_delta": delta[video_id],
                "videos": videos,
                "prompts": {
                    tag: entries[video_id]["tracking"]["prompt_history"]
                    for tag, entries in (
                        ("initial", base_videos),
                        ("best", tuned_videos),
                    )
                },
            }
        )
    artifact = output / "examples.json"
    atomic_json(
        artifact,
        {
            "examples": examples,
            "playback_fps": fps,
            "selection": "descriptive ranking, not an accuracy threshold",
            "excluded_no_scored_frames": sorted(set(base_videos) - set(ids)),
        },
    )
    return {"artifact": str(artifact), "examples": examples}
