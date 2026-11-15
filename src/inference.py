"""Chronological prompt replay and native-resolution, ID-preserving SAM 2 tracking.

The ledger is the source of truth: a later prompt is never inserted before an
older frame is predicted. Editing an earlier prompt resets all model memory.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .io import atomic_json, canonical_hash, read_json, sha256_file


@dataclass(frozen=True)
class Prompt:
    frame_index: int
    object_id: int
    kind: str = "point"
    points: tuple[tuple[float, float], ...] = ()
    labels: tuple[int, ...] = ()
    box: tuple[float, float, float, float] | None = None
    mask_path: str | None = None
    source: str = "human"

    @classmethod
    def from_dict(cls, value: dict) -> Prompt:
        fields = dict(value)
        fields["points"] = tuple(tuple(p) for p in fields.get("points", ()))
        fields["labels"] = tuple(fields.get("labels", ()))
        if fields.get("box") is not None:
            fields["box"] = tuple(fields["box"])
        return cls(**fields)

    def validate(self, frames: int, width: int, height: int) -> None:
        if (
            not isinstance(self.frame_index, Integral)
            or isinstance(self.frame_index, bool)
            or not isinstance(self.object_id, Integral)
            or isinstance(self.object_id, bool)
            or not 0 <= self.frame_index < frames
            or not 1 <= self.object_id <= 255
        ):
            raise ValueError("Frame index or object ID invalid (IDs must be 1..255)")
        if self.kind not in {"point", "box", "mask"}:
            raise ValueError(f"Unsupported prompt kind: {self.kind}")
        if self.kind == "point":
            if not self.points or len(self.points) != len(self.labels):
                raise ValueError("Every point requires a positive/negative label")
            if any(label not in (0, 1) for label in self.labels):
                raise ValueError("Point labels must be 0 or 1")
            if any(
                not np.isfinite([x, y]).all()
                or not (0 <= x < width and 0 <= y < height)
                for x, y in self.points
            ):
                raise ValueError("Point outside native image coordinates")
        elif self.kind == "box":
            if self.box is None or len(self.box) != 4:
                raise ValueError("Box requires x0,y0,x1,y1")
            x0, y0, x1, y1 = self.box
            if not np.isfinite(self.box).all() or not (
                0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height
            ):
                raise ValueError("Box outside native image coordinates or empty")
        elif self.mask_path is None or not Path(self.mask_path).is_file():
            raise ValueError("Mask prompt must reference an existing indexed mask")


def interior_point(mask: np.ndarray) -> tuple[float, float]:
    """Farthest-from-boundary positive point; row-major deterministic tie breaking."""
    from scipy.ndimage import distance_transform_edt

    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or not mask.any():
        raise ValueError("Cannot prompt an absent object")
    distances = distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]
    y, x = np.unravel_index(int(distances.argmax()), mask.shape)
    return float(x), float(y)


def correction_prompt(
    gt: np.ndarray, prediction: np.ndarray, object_id: int, frame_index: int
) -> Prompt | None:
    """One point in the largest connected FN or FP component; FN wins ties."""
    from scipy.ndimage import label

    gt, prediction = np.asarray(gt), np.asarray(prediction)
    if gt.shape != prediction.shape or gt.ndim != 2:
        raise ValueError("Correction masks must have the same two-dimensional shape")
    target, predicted = gt == object_id, prediction == object_id
    best: tuple[int, int, np.ndarray] | None = None
    for sign, error in ((1, target & ~predicted), (0, predicted & ~target)):
        regions, count = label(error)  # Explicit default 4-connected components.
        if count:
            areas = np.bincount(regions.ravel())[1:]
            index = int(areas.argmax()) + 1
            size = int(areas[index - 1])
            if best is None or size > best[0]:
                best = size, sign, regions == index
    if best is None:
        return None
    return Prompt(
        frame_index,
        object_id,
        points=(interior_point(best[2]),),
        labels=(best[1],),
        source="simulated_midpoint_ground_truth_assistance",
    )


def initial_prompts(
    run, record, kind: str = "point", object_ids=None
) -> tuple[Prompt, ...]:
    """Freeze prompts once and verify the annotation identity on every reuse."""
    if Path(record.video_id).name != record.video_id or record.video_id in {".", ".."}:
        raise ValueError("Unsafe video identity")
    if kind not in {"point", "mask", "box"}:
        raise ValueError("Initial prompt kind must be point, box, or mask")
    ids = tuple(sorted(record.object_ids if object_ids is None else object_ids))
    if not ids or not set(ids).issubset(record.object_ids):
        raise ValueError("No valid target IDs selected")
    first = record.annotations[0]
    if first is None or not Path(first).is_file():
        raise ValueError(f"Missing initialization annotation: {record.video_id}")
    annotation = np.asarray(Image.open(first))
    if annotation.shape != (record.height, record.width):
        raise ValueError("Initialization mask shape does not match the video")
    identity = {
        "video_id": record.video_id,
        "annotation_sha256": sha256_file(first),
        "first_frame_sha256": sha256_file(record.frames[0]),
        "object_ids": list(ids),
        "kind": kind,
        "coordinates": "native pixel x,y",
        "point_rule": "EDT_max_row_major",
    }
    path = (
        run.path
        / "prompts"
        / "initial"
        / f"{record.video_id}-{kind}-{canonical_hash(ids)[:10]}.json"
    )
    if path.exists():
        saved = read_json(path)
        if saved["identity"] != identity:
            raise ValueError(
                "Frozen prompt identity changed; do not mix data within a run"
            )
        return tuple(Prompt.from_dict(item) for item in saved["prompts"])
    prompts = []
    for object_id in ids:
        mask = annotation == object_id
        if not mask.any():
            raise ValueError(
                f"Object {object_id} in {record.video_id} is absent from its initialization frame"
            )
        if kind == "point":
            prompt = Prompt(
                0,
                object_id,
                points=(interior_point(mask),),
                labels=(1,),
                source="frozen_initial",
            )
        elif kind == "mask":
            prompt = Prompt(
                0, object_id, kind="mask", mask_path=str(first), source="frozen_initial"
            )
        else:
            y, x = np.nonzero(mask)
            prompt = Prompt(
                0,
                object_id,
                kind="box",
                box=(
                    float(x.min()),
                    float(y.min()),
                    float(x.max() + 1),
                    float(y.max() + 1),
                ),
                source="frozen_initial",
            )
        prompts.append(prompt)
    atomic_json(path, {"identity": identity, "prompts": [asdict(p) for p in prompts]})
    return tuple(prompts)


def labels_from_logits(object_ids, logits, *, threshold: float = 0.0) -> np.ndarray:
    """Background=0, maximum positive logit wins; lowest object ID wins ties."""
    if hasattr(logits, "detach"):
        logits = logits.detach().float().cpu().numpy()
    values = np.asarray(logits, dtype=np.float32)
    ids = np.asarray(object_ids, dtype=np.int64)
    if values.ndim == 4 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim != 3 or values.shape[0] != len(ids) or not len(ids):
        raise ValueError("Logits must have shape objects x [1 x] height x width")
    if (
        not np.isfinite(values).all()
        or len(np.unique(ids)) != len(ids)
        or np.any((ids < 1) | (ids > 255))
    ):
        raise ValueError("Invalid/nonfinite logits or object identities")
    order = np.argsort(ids, kind="stable")
    ids, values = ids[order], values[order]
    winner = values.argmax(axis=0)
    maximum = values.max(axis=0)
    return np.where(maximum > threshold, ids[winner], 0).astype(np.uint8)


@contextmanager
def prediction_context(predictor, device: str = "cuda") -> Iterator[None]:
    """Lazy torch import; fake predictors and CPU data tests require no torch."""
    if not hasattr(predictor, "parameters"):
        yield
        return
    import torch

    with torch.inference_mode():
        if str(device).startswith("cuda"):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                yield
        else:
            yield


class TrackingSession:
    """Session-local predictor state with deterministic chronological ledger replay.

    Construction decodes/transfers frames and warms the image backbone via SAM 2's
    init_state. Benchmark this separately. replay_raw does not transfer outputs to
    host; replay additionally materializes native-resolution indexed labels.
    """

    def __init__(
        self,
        predictor,
        frame_dir: Path,
        frame_paths: tuple[Path, ...],
        *,
        offload_video_to_cpu: bool = True,
        device: str = "cuda",
    ):
        self.predictor = predictor
        self.frame_dir = Path(frame_dir)
        self.frame_paths = tuple(Path(p) for p in frame_paths)
        if not self.frame_paths:
            raise ValueError("A tracking session requires frames")
        with Image.open(self.frame_paths[0]) as im:
            self.width, self.height = im.size
        self.device = device
        self.prompts: list[Prompt] = []
        self.outputs: dict[int, np.ndarray] = {}
        with prediction_context(predictor, device):
            self.state = predictor.init_state(
                video_path=str(frame_dir),
                offload_video_to_cpu=offload_video_to_cpu,
                offload_state_to_cpu=False,
                async_loading_frames=False,
            )
        if self.state.get("num_frames", len(self.frame_paths)) != len(self.frame_paths):
            raise ValueError(
                "Predictor frame ordering/count does not match the manifest"
            )

    def add_prompt(self, prompt: Prompt) -> None:
        prompt.validate(len(self.frame_paths), self.width, self.height)
        if prompt.kind == "mask":
            mask = np.asarray(Image.open(prompt.mask_path))
            if mask.shape != (self.height, self.width):
                raise ValueError("Prompt mask must have native image dimensions")
        self.prompts.append(prompt)
        self.outputs = {i: v for i, v in self.outputs.items() if i < prompt.frame_index}

    def replace_prompts(self, prompts: list[Prompt]) -> None:
        for prompt in prompts:
            prompt.validate(len(self.frame_paths), self.width, self.height)
            if prompt.kind == "mask" and np.asarray(
                Image.open(prompt.mask_path)
            ).shape != (self.height, self.width):
                raise ValueError("Prompt mask must have native image dimensions")
        self.prompts = list(prompts)
        self.outputs.clear()

    def reset(self) -> None:
        self.predictor.reset_state(self.state)
        self.outputs.clear()

    def _apply(self, prompt: Prompt):
        if prompt.kind == "mask":
            mask = np.asarray(Image.open(prompt.mask_path)) == prompt.object_id
            return self.predictor.add_new_mask(
                inference_state=self.state,
                frame_idx=prompt.frame_index,
                obj_id=prompt.object_id,
                mask=mask,
            )
        kwargs = {
            "inference_state": self.state,
            "frame_idx": prompt.frame_index,
            "obj_id": prompt.object_id,
            "normalize_coords": True,
        }
        if prompt.kind == "box":
            return self.predictor.add_new_points_or_box(
                **kwargs,
                box=np.asarray(prompt.box, dtype=np.float32),
                clear_old_points=True,
            )
        return self.predictor.add_new_points_or_box(
            **kwargs,
            points=np.asarray(prompt.points, dtype=np.float32),
            labels=np.asarray(prompt.labels, dtype=np.int32),
            clear_old_points=False,
        )

    def _propagate(self, start: int, end: int):
        if end < start:
            return
        # Upstream max_frame_num_to_track is an inclusive offset, not a count.
        for frame, ids, logits in self.predictor.propagate_in_video(
            self.state,
            start_frame_idx=start,
            max_frame_num_to_track=end - start,
            reverse=False,
        ):
            if not start <= frame <= end:
                raise RuntimeError(
                    "Predictor emitted a frame beyond the chronological segment"
                )
            yield int(frame), ids, logits

    def replay_raw(self, stop_frame: int | None = None):
        """Yield model outputs in chronological order (a corrected frame may repeat).

        No future event is inserted in the state before the earlier segment is
        predicted. Future prompts after stop_frame are never applied. No host
        conversion is performed here, allowing benchmark boundaries to be exact.
        """
        stop = len(self.frame_paths) - 1 if stop_frame is None else int(stop_frame)
        if not 0 <= stop < len(self.frame_paths):
            raise ValueError("Invalid stop frame")
        events = sorted(
            enumerate(self.prompts), key=lambda item: (item[1].frame_index, item[0])
        )
        events = [p for _, p in events if p.frame_index <= stop]
        if not events:
            raise ValueError("Add at least one prompt before tracking")
        with prediction_context(self.predictor, self.device):
            self.reset()
            grouped: dict[int, list[Prompt]] = {}
            for prompt in events:
                grouped.setdefault(prompt.frame_index, []).append(prompt)
            cursor = None
            for frame, prompts in grouped.items():
                if cursor is not None:
                    # Predict through the correction frame before providing it an annotation.
                    yield from self._propagate(cursor, frame)
                for prompt in prompts:
                    yield self._apply(prompt)
                cursor = frame
            yield from self._propagate(cursor, stop)

    def replay(
        self,
        stop_frame: int | None = None,
        *,
        on_frame: Callable[[int, np.ndarray], Any] | None = None,
    ) -> dict[int, np.ndarray]:
        stop = len(self.frame_paths) - 1 if stop_frame is None else int(stop_frame)
        outputs = {}
        first_prompt = min(
            (p.frame_index for p in self.prompts if p.frame_index <= stop), default=None
        )
        if first_prompt is None:
            raise ValueError("Add a prompt at or before the requested frame")
        for frame in range(first_prompt):
            outputs[frame] = np.zeros((self.height, self.width), dtype=np.uint8)
        for frame, ids, logits in self.replay_raw(stop_frame):
            labels = labels_from_logits(ids, logits)
            if labels.shape != (self.height, self.width):
                raise ValueError("Predictor output is not at native resolution")
            outputs[frame] = labels
            if on_frame is not None:
                on_frame(frame, labels)
        if set(outputs) != set(range(stop + 1)):
            raise RuntimeError(
                "Incomplete trajectory; refusing to manufacture missing masks"
            )
        self.outputs = outputs
        return outputs

    def save_ledger(self, path: Path) -> Path:
        atomic_json(
            path,
            {
                "schema_version": 1,
                "frame_count": len(self.frame_paths),
                "coordinates": "native pixel x,y",
                "replay": "chronological_forward_only",
                "prompts": [asdict(p) for p in self.prompts],
            },
        )
        return path


def save_indexed_mask(path: Path, labels: np.ndarray) -> None:
    from .video import mask_palette

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(labels)
    if labels.ndim != 2 or np.any((labels < 0) | (labels > 255)):
        raise ValueError("Indexed PNG labels must fit object IDs 0..255")
    image = Image.fromarray(labels.astype(np.uint8), mode="P")
    image.putpalette(mask_palette())
    temporary = path.with_name(path.name + ".partial")
    image.save(temporary, format="PNG")
    temporary.replace(path)


def track_record(
    run,
    record,
    checkpoint="initial",
    protocol: str = "point",
    *,
    output_dir: Path | None = None,
    predictor=None,
    device: str = "cuda",
) -> dict:
    """Track one full video; midpoint assistance is calculated from its own prediction."""
    from .sam2_adapter import build_predictor
    from .video import prepare_frame_directory

    if Path(record.video_id).name != record.video_id or record.video_id in {".", ".."}:
        raise ValueError("Unsafe video identity")
    if len({frame.stem for frame in record.frames}) != len(record.frames):
        raise ValueError("Frame output names must be unique within a video")
    if protocol not in {"point", "point_correction", "mask"}:
        raise ValueError("Unknown tracking protocol")
    from .evaluate import checkpoint_tag

    directory = (
        Path(output_dir)
        if output_dir
        else run.path
        / "predictions"
        / checkpoint_tag(checkpoint)
        / protocol
        / record.video_id
    )
    directory.mkdir(parents=True, exist_ok=True)
    frames_dir = prepare_frame_directory(
        record.frames, run.root / "data" / "frame_views" / record.video_id
    )
    owned = predictor is None
    if owned:
        predictor = build_predictor(run, checkpoint=checkpoint, device=device)
    started = time.perf_counter()
    session = TrackingSession(
        predictor,
        frames_dir,
        record.frames,
        offload_video_to_cpu=run.config.evaluation.offload_video_to_cpu,
        device=device,
    )
    setup_seconds = time.perf_counter() - started
    for prompt in initial_prompts(
        run, record, "mask" if protocol == "mask" else "point"
    ):
        session.add_prompt(prompt)
    midpoint = len(record.frames) // 2
    started = time.perf_counter()
    corrective_clicks = 0
    if protocol == "point_correction":
        if record.annotations[midpoint] is None:
            raise ValueError(
                "Missing midpoint supervision; cannot simulate a correction"
            )
        partial = session.replay(stop_frame=midpoint)
        gt = np.asarray(Image.open(record.annotations[midpoint]))
        for object_id in sorted(record.object_ids):
            prompt = correction_prompt(gt, partial[midpoint], object_id, midpoint)
            if prompt is not None:
                session.add_prompt(prompt)
                corrective_clicks += 1
    outputs = session.replay()
    elapsed = time.perf_counter() - started
    labels_dir = directory / "masks"
    for frame, labels in outputs.items():
        save_indexed_mask(labels_dir / f"{Path(record.frames[frame]).stem}.png", labels)
    ledger = session.save_ledger(directory / "prompts.json")
    result = {
        "video_id": record.video_id,
        "protocol": protocol,
        "labels_dir": str(labels_dir),
        "prompt_history": str(ledger),
        "frame_count": len(record.frames),
        "initial_clicks": len(record.object_ids) if protocol != "mask" else 0,
        "corrective_clicks": corrective_clicks,
        "midpoint_frame": midpoint,
        "elapsed_model_seconds": elapsed,
        "frame_setup_seconds": setup_seconds,
        "timing_scope": "end_to_end_tracking_including_materialization; use benchmark for model-only timing",
    }
    atomic_json(directory / "tracking.json", result)
    session.reset()
    del session
    if owned:
        del predictor
    return result
