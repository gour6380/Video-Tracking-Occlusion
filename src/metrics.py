"""MOSEv2 public-scorer metrics and explicitly labelled local click exclusions.

Pixel metrics are executed from the checksum-verified vendored implementation.
The official server's private scoring bundle is not claimed to be identical.
Values here stay in [0, 1]; reports may display 100 * value without prematurely
rounding the observations used for aggregation.
"""

from __future__ import annotations

import importlib.util
import math
import operator
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

from .io import read_json, sha256_file

SCORER_REVISION = "a74b23aa3a31597f487eec10f452e0b9f61347e1"
METRIC_KEYS = (
    "J",
    "F",
    "F_dot",
    "J&F",
    "J&F_dot",
    "disappear_J&F_dot",
    "reappear_J&F_dot",
)
DIAGNOSTIC_COUNTS = (
    "absent_scored_frames",
    "absent_frames_with_prediction",
    "absent_prediction_pixels",
    "absent_total_pixels",
)


def _diagnostic_rates(counts: dict) -> dict:
    result = dict(counts)
    result["absent_frame_fp_rate"] = (
        counts["absent_frames_with_prediction"] / counts["absent_scored_frames"]
        if counts["absent_scored_frames"]
        else None
    )
    result["absent_pixel_fp_rate"] = (
        counts["absent_prediction_pixels"] / counts["absent_total_pixels"]
        if counts["absent_total_pixels"]
        else None
    )
    return result


@lru_cache(maxsize=1)
def _official_metrics():
    root = Path(__file__).resolve().parents[1]
    provenance = read_json(root / "artifacts" / "upstream.json")
    if provenance["mose_scorer_revision"] != SCORER_REVISION:
        raise ValueError(
            "MOSEv2 scoring revision differs from this wrapper's audited revision"
        )
    for relative in ("vendor/mosev2/metricsv2.py", "vendor/mosev2/score.py"):
        if sha256_file(root / relative) != provenance["files"][relative]:
            raise ValueError(f"Vendored scorer checksum changed: {relative}")
    spec = importlib.util.spec_from_file_location(
        "_pinned_mosev2_pixel_metrics", root / "vendor/mosev2/metricsv2.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _mean(values: Iterable[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    if not present:
        return None
    if not all(math.isfinite(value) for value in present):
        raise ValueError("Nonfinite metric; do not publish an aggregate")
    return math.fsum(present) / len(present)


def _read_mask(value, role: str, index: int):
    import numpy as np
    from PIL import Image

    if value is None:
        raise ValueError(
            f"Missing {role} mask at frame {index}; missing is not absence"
        )
    if isinstance(value, (str, Path)):
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(f"Missing {role} mask at frame {index}: {path}")
        with Image.open(path) as image:
            array = np.asarray(image).copy()
    else:
        array = np.asarray(value)
    if array.ndim != 2 or array.size == 0:
        raise ValueError(f"{role} mask {index} must be a nonempty indexed 2D image")
    if not (np.issubdtype(array.dtype, np.integer) or array.dtype == np.bool_):
        raise ValueError(f"{role} mask {index} must contain integer object labels")
    if np.min(array) < 0 or np.max(array) > 255:
        raise ValueError(f"{role} mask {index} has IDs outside indexed-PNG range")
    return array


def _event_segments(empty: list[bool]) -> dict[str, list[tuple[int, int]]]:
    """Match public score.py on its original, endpoint-trimmed timeline.

    Exclusions are applied *after* these intervals are identified: deleting a
    midpoint from the timeline must not join two absence or presence episodes.
    The public scorer omits an absence that already started at index zero.
    """
    if not empty:
        return {"disappear": [], "reappear": []}
    disappears = [i for i in range(1, len(empty)) if not empty[i - 1] and empty[i]]
    if empty[0]:
        disappears.insert(0, 0)
    reappears = [i for i in range(1, len(empty)) if empty[i - 1] and not empty[i]]
    disappearance = [
        (start, reappears[i] if i < len(reappears) else len(empty))
        for i, start in enumerate(disappears)
        if start > 0
    ]
    reappearance = [
        (start, disappears[i + 1] if i + 1 < len(disappears) else len(empty))
        for i, start in enumerate(reappears)
    ]
    return {"disappear": disappearance, "reappear": reappearance}


def score_video(gt_masks, pred_masks, object_ids, exclude_frames=()) -> dict:
    """Score complete aligned indexed-mask sequences at their native resolution.

    Always excludes original first/last frames, matching the public scorer.
    ``exclude_frames`` contains ORIGINAL zero-based frame indices and supports
    paired click experiments. Supply the same indices to both arms. Event
    boundaries are computed before exclusion. An event with no remaining scored
    frames is N/A, never zero; wholly unscored objects are also N/A.
    """
    import numpy as np

    if len(gt_masks) != len(pred_masks) or len(gt_masks) < 3:
        raise ValueError(
            "Complete GT/prediction sequences must align and contain at least three frames"
        )
    ids = tuple(operator.index(value) for value in object_ids)
    if (
        not ids
        or len(set(ids)) != len(ids)
        or any(not 0 < value < 256 for value in ids)
    ):
        raise ValueError("Unique positive indexed-PNG object IDs are required")
    exclusions = {operator.index(value) for value in exclude_frames}
    length = len(gt_masks)
    if any(index < 0 or index >= length for index in exclusions):
        raise ValueError("Excluded frame index is outside the original video")
    scored = [i for i in range(1, length - 1) if i not in exclusions]
    allowed_predictions = {0, *ids}
    pixel_metrics = _official_metrics()
    observations = {str(object_id): [] for object_id in ids}
    expected_shape = None
    for index in range(length):
        truth = _read_mask(gt_masks[index], "ground-truth", index)
        prediction = _read_mask(pred_masks[index], "prediction", index)
        if expected_shape is None:
            expected_shape = truth.shape
        if truth.shape != expected_shape or prediction.shape != expected_shape:
            raise ValueError(f"Native mask dimensions differ at frame {index}")
        if not {int(x) for x in np.unique(prediction)}.issubset(allowed_predictions):
            raise ValueError(
                f"Prediction at frame {index} contains undeclared object IDs"
            )
        if index in (0, length - 1):
            continue
        for object_id in ids:
            binary_gt = (truth == object_id).astype("uint8")
            binary_pred = (prediction == object_id).astype("uint8")
            j, absent = pixel_metrics.db_eval_iou(binary_gt, binary_pred)
            f, f_dot = pixel_metrics.db_eval_boundary(
                binary_gt, binary_pred, bound_th=0.1, bound_mode="area"
            )
            observations[str(object_id)].append(
                {
                    "frame": index,
                    "J": float(j),
                    "F": float(f),
                    "F_dot": float(f_dot),
                    "absent": bool(absent),
                    "scored": index not in exclusions,
                    "predicted_target_pixels": int(np.count_nonzero(binary_pred)),
                    "total_pixels": int(truth.size),
                }
            )
    objects = {}
    for object_id, rows in observations.items():
        included = [row for row in rows if row["scored"]]
        result = {
            key: _mean(row[key] for row in included) for key in ("J", "F", "F_dot")
        }
        result["J&F"] = _mean((result["J"], result["F"]))
        result["J&F_dot"] = _mean((result["J"], result["F_dot"]))
        result["scored_frame_count"] = len(included)
        # These transparent false-positive diagnostics are separate from MOSEv2
        # event metrics. Denominators count eligible *object*-frames/pixels, so
        # absence at the first interior frame also contributes here even though
        # the public disappearance-event reducer omits an already-started event.
        absent = [row for row in included if row["absent"]]
        result["local_diagnostics"] = _diagnostic_rates(
            {
                "absent_scored_frames": len(absent),
                "absent_frames_with_prediction": sum(
                    row["predicted_target_pixels"] > 0 for row in absent
                ),
                "absent_prediction_pixels": sum(
                    row["predicted_target_pixels"] for row in absent
                ),
                "absent_total_pixels": sum(row["total_pixels"] for row in absent),
            }
        )
        result["segments"] = {}
        for kind, intervals in _event_segments([row["absent"] for row in rows]).items():
            segments = []
            for start, end in intervals:
                eligible = [row for row in rows[start:end] if row["scored"]]
                j = _mean(row["J"] for row in eligible)
                f = _mean(row["F_dot"] for row in eligible)
                segments.append(
                    {
                        "start_frame": start + 1,
                        "end_frame_exclusive": end + 1,
                        "scored_frames": [row["frame"] for row in eligible],
                        "J&F_dot": _mean((j, f)),
                    }
                )
            result["segments"][kind] = segments
            result[f"{kind}_J&F_dot"] = _mean(
                segment["J&F_dot"] for segment in segments
            )
        objects[object_id] = result
    output = {
        "scorer_revision": SCORER_REVISION,
        "units": "fraction",
        "protocol": "public MOSEv2 scorer"
        if not exclusions
        else "local paired-frame-exclusion adaptation",
        "frame_count": length,
        "scored_frames": scored,
        "excluded_frames": sorted({0, length - 1, *exclusions}),
        "objects": objects,
        "per_frame": observations,
    }
    output["summary"] = aggregate_scores([output])
    return output


def aggregate_scores(video_scores: Iterable[dict]) -> dict:
    """Macro-average official scores; pool local FP counts with explicit denominators.

    Reappearance recovery remains the existing event J&F_dot and saved original
    segment intervals. No custom success threshold or recovery latency is added.
    """
    videos = list(video_scores)
    objects = [item for video in videos for item in video["objects"].values()]
    result = {key: _mean(item[key] for item in objects) for key in METRIC_KEYS}
    result.update(
        {
            "units": "fraction",
            "video_count": len(videos),
            "object_count": len(objects),
            "eligible_objects": {
                key: sum(item[key] is not None for item in objects)
                for key in METRIC_KEYS
            },
        }
    )
    result["local_diagnostics"] = _diagnostic_rates(
        {
            key: sum(item["local_diagnostics"][key] for item in objects)
            for key in DIAGNOSTIC_COUNTS
        }
    )
    return result
