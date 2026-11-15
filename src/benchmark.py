"""Tracking benchmarks with explicit input-transfer and model boundaries."""

from __future__ import annotations

import gc
import hashlib
import sys
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from .io import atomic_json
from .runs import stage


@contextmanager
def quiet_upstream_progress():
    """Exclude SAM2's generator-level tqdm rendering from timed next() calls."""
    module = sys.modules.get("sam2.sam2_video_predictor")
    if module is None:
        yield
        return
    original = module.tqdm
    module.tqdm = lambda iterable, *args, **kwargs: iterable
    try:
        yield
    finally:
        module.tqdm = original


class StagedFrames:
    """Keep the dataset on CPU and pre-transfer exactly one frame outside timing."""

    def __init__(self, frames):
        self.frames = frames
        self.index = None
        self.value = None

    def __len__(self):
        return len(self.frames)

    def stage(self, index, device, synchronize):
        synchronize()
        start = time.perf_counter()
        self.value = self.frames[index].to(device=device, non_blocking=False).float()
        self.index = index
        synchronize()
        return time.perf_counter() - start

    def __getitem__(self, index):
        if index != self.index:
            raise RuntimeError(f"Unstaged frame {index} requested inside model timing")
        return self.value


def timed_call(function, synchronize):
    synchronize()
    start = time.perf_counter()
    result = function()
    synchronize()
    return result, time.perf_counter() - start


def _kwargs(prompt, state, device):
    import torch

    common = {
        "inference_state": state,
        "frame_idx": prompt.frame_index,
        "obj_id": prompt.object_id,
        "normalize_coords": True,
        "clear_old_points": False,
    }
    if prompt.kind != "point":
        raise ValueError("Performance protocol uses fixed point prompts")
    common["points"] = torch.tensor(prompt.points, dtype=torch.float32, device=device)
    common["labels"] = torch.tensor(prompt.labels, dtype=torch.int32, device=device)
    return common


def _trajectory(session, prompts, synchronize, stop=None):
    from .inference import labels_from_logits, prediction_context

    session.reset()
    if session.state.get("obj_ids", []):
        raise RuntimeError("Reset leaked object identities")
    session.state["cached_features"] = {}
    stop = len(session.frame_paths) - 1 if stop is None else stop
    original = session.state["images"]
    frames = StagedFrames(original)
    session.state["images"] = frames
    samples, prompt_times, outputs, transfer = [], [], {}, 0.0
    output_transfer = 0.0
    try:
        with (
            prediction_context(session.predictor, session.device),
            quiet_upstream_progress(),
        ):
            transfer += frames.stage(0, session.device, synchronize)
            started = time.perf_counter()
            kwargs = [
                _kwargs(prompt, session.state, session.device) for prompt in prompts
            ]
            synchronize()
            transfer += time.perf_counter() - started
            for arguments in kwargs:
                _, seconds = timed_call(
                    lambda arguments=arguments: session.predictor.add_new_points_or_box(
                        **arguments
                    ),
                    synchronize,
                )
                prompt_times.append(seconds)
            iterator = session.predictor.propagate_in_video(
                session.state,
                start_frame_idx=0,
                max_frame_num_to_track=stop,
                reverse=False,
            )
            for expected in range(stop + 1):
                transfer += frames.stage(expected, session.device, synchronize)
                (index, ids, logits), seconds = timed_call(
                    lambda: next(iterator), synchronize
                )
                if int(index) != expected:
                    raise RuntimeError("Non-chronological propagation during benchmark")
                samples.append(seconds)
                start = time.perf_counter()
                outputs[expected] = labels_from_logits(ids, logits)
                output_transfer += time.perf_counter() - start
    finally:
        session.state["images"] = original
    return {
        "frame_seconds": samples,
        "initial_prompt_seconds": prompt_times,
        "model_seconds": sum(samples) + sum(prompt_times),
        "transfer_seconds": transfer,
        "output_materialization_seconds": output_transfer,
        "outputs": outputs,
    }


def _correction(session, record, prompts, synchronize, midpoint_prediction):
    from .inference import correction_prompt, prediction_context

    midpoint = len(record.frames) // 2
    gt = np.asarray(Image.open(record.annotations[midpoint]))
    corrections = [
        p
        for obj in (p.object_id for p in prompts)
        if (p := correction_prompt(gt, midpoint_prediction, obj, midpoint)) is not None
    ]
    if not corrections:
        return {"clicks": 0, "seconds": None, "preparation_seconds": 0.0}
    start = time.perf_counter()
    _trajectory(session, prompts, synchronize, stop=midpoint)
    original = session.state["images"]
    frames = StagedFrames(original)
    session.state["images"] = frames
    try:
        frames.stage(midpoint, session.device, synchronize)
        kwargs = [_kwargs(p, session.state, session.device) for p in corrections]
        synchronize()
        preparation = time.perf_counter() - start
        with prediction_context(session.predictor, session.device):
            seconds = sum(
                timed_call(
                    lambda k=k: session.predictor.add_new_points_or_box(**k),
                    synchronize,
                )[1]
                for k in kwargs
            )
    finally:
        session.state["images"] = original
    return {
        "clicks": len(corrections),
        "seconds": seconds,
        "preparation_seconds": preparation,
    }


def check_session_isolation(
    predictor, frame_dir, frame_paths, prompt, *, device="cuda", synchronize=None
):
    """Interleave A→B→A without resetting A; distinct IDs expose state leakage.

    This is a separate bounded three-frame probe, outside the model timings.
    """
    from .inference import TrackingSession, labels_from_logits, prediction_context

    synchronize = synchronize or (lambda: None)
    sessions = []
    try:
        first = TrackingSession(
            predictor, frame_dir, frame_paths, device=device, offload_video_to_cpu=True
        )
        sessions.append(first)
        second = TrackingSession(
            predictor, frame_dir, frame_paths, device=device, offload_video_to_cpu=True
        )
        sessions.append(second)
        second_id = 1 if prompt.object_id != 1 else 2
        alternative = replace(prompt, object_id=second_id, source="isolation_probe")
        stop = min(2, len(frame_paths) - 1)
        first.add_prompt(prompt)
        before = first.replay(stop_frame=stop)
        first_ids = tuple(first.state["obj_ids"])
        second.add_prompt(alternative)
        second.replay(stop_frame=stop)
        second_ids = tuple(second.state["obj_ids"])
        independent = (
            first.state is not second.state
            and first.state["output_dict_per_obj"]
            is not second.state["output_dict_per_obj"]
        )
        unchanged_ids = tuple(first.state["obj_ids"]) == first_ids == (
            prompt.object_id,
        ) and second_ids == (second_id,)
        after = {}
        # Keep A's memory intact when returning from B; resetting A would hide leaks.
        with prediction_context(predictor, device), quiet_upstream_progress():
            for index, ids, logits in first._propagate(0, stop):
                if tuple(ids) != first_ids:
                    raise RuntimeError(
                        "A second session changed the first session object IDs"
                    )
                after[index] = labels_from_logits(ids, logits)
        synchronize()
        equal = before.keys() == after.keys() and all(
            np.array_equal(before[index], after[index]) for index in before
        )
        result = {
            "passed": independent and unchanged_ids and equal,
            "scope": "one object per independent state; A-B-A interleaving without resetting A",
            "probe_frames": stop + 1,
            "first_ids": list(first_ids),
            "second_ids": list(second_ids),
            "independent_state_containers": independent,
            "unchanged_ids": unchanged_ids,
            "unchanged_predictions": equal,
            "before": {
                str(i): hashlib.sha256(mask.tobytes()).hexdigest()
                for i, mask in before.items()
            },
            "after": {
                str(i): hashlib.sha256(mask.tobytes()).hexdigest()
                for i, mask in after.items()
            },
        }
        if not result["passed"]:
            raise RuntimeError(
                f"Independent tracking session isolation failed: {result}"
            )
        return result
    finally:
        for session in sessions:
            session.reset()


def benchmark_tracking(
    run, checkpoints=("initial", "best"), output_dir=None, pilot=False
) -> dict:
    import torch

    from .data import load_partition
    from .evaluate import checkpoint_tag
    from .inference import TrackingSession, initial_prompts
    from .sam2_adapter import build_predictor, checkpoint_path
    from .video import prepare_frame_directory

    if not torch.cuda.is_available():
        raise RuntimeError(
            "This benchmark requires CUDA; Mac/CPU timings are not substituted"
        )
    destination = (
        Path(output_dir) if output_dir else run.path / "benchmarks" / "tracking"
    )
    destination.mkdir(parents=True, exist_ok=True)
    records = load_partition(run, "dev")
    cases, unsupported, isolation_probes = [], [], []
    whole_start = time.perf_counter()
    with stage(run, "pilot_tracking" if pilot else "benchmark_tracking"):
        for checkpoint in checkpoints:
            tag = checkpoint_tag(checkpoint)
            isolation = None
            loading = time.perf_counter()
            predictor = build_predictor(run, checkpoint=checkpoint, device="cuda")
            torch.cuda.synchronize()
            loading = time.perf_counter() - loading
            for count in run.config.benchmark.object_counts:
                eligible = sorted(
                    (r for r in records if len(r.object_ids) >= count),
                    key=lambda r: r.video_id,
                )
                eligible = eligible[
                    : (1 if pilot else run.config.benchmark.clips_per_condition)
                ]
                if not eligible:
                    unsupported.append(
                        {
                            "checkpoint": str(checkpoint),
                            "objects": count,
                            "reason": "No eligible development video",
                        }
                    )
                    continue
                for record in eligible:
                    count_start = time.perf_counter()
                    frame_dir = prepare_frame_directory(
                        record.frames,
                        run.root / "data" / "frame_views" / record.video_id,
                    )
                    setup_start = time.perf_counter()
                    session = TrackingSession(
                        predictor,
                        frame_dir,
                        record.frames,
                        offload_video_to_cpu=True,
                        device="cuda",
                    )
                    torch.cuda.synchronize()
                    setup = time.perf_counter() - setup_start
                    prompts = initial_prompts(
                        run,
                        record,
                        "point",
                        object_ids=tuple(sorted(record.object_ids)[:count]),
                    )
                    first = _trajectory(session, prompts, torch.cuda.synchronize)
                    reference = {
                        i: hashlib.sha256(x.tobytes()).hexdigest()
                        for i, x in first.pop("outputs").items()
                    }
                    for _ in range(run.config.benchmark.warmups):
                        _trajectory(session, prompts, torch.cuda.synchronize)
                    repetitions = []
                    for repeat in range(
                        1 if pilot else run.config.benchmark.repetitions
                    ):
                        torch.cuda.reset_peak_memory_stats()
                        result = _trajectory(session, prompts, torch.cuda.synchronize)
                        peak = torch.cuda.max_memory_allocated()
                        outputs = result.pop("outputs")
                        if outputs.keys() != reference.keys() or any(
                            hashlib.sha256(x.tobytes()).hexdigest() != reference[i]
                            for i, x in outputs.items()
                        ):
                            raise RuntimeError(
                                "Reset/replay changed indexed predictions"
                            )
                        correction = _correction(
                            session,
                            record,
                            prompts,
                            torch.cuda.synchronize,
                            outputs[len(record.frames) // 2],
                        )
                        model_seconds = result["model_seconds"]
                        result.update(
                            repetition=repeat,
                            peak_allocated_bytes=peak,
                            propagation_fps=len(record.frames)
                            / sum(result["frame_seconds"]),
                            model_fps=len(record.frames) / model_seconds,
                            median_frame_seconds=float(
                                np.median(result["frame_seconds"])
                            ),
                            p95_frame_seconds=float(
                                np.percentile(result["frame_seconds"], 95)
                            ),
                            correction=correction,
                        )
                        repetitions.append(result)
                    if isolation is None:
                        isolation = check_session_isolation(
                            predictor,
                            frame_dir,
                            record.frames,
                            prompts[0],
                            device="cuda",
                            synchronize=torch.cuda.synchronize,
                        )
                        isolation_probes.append(
                            {
                                "checkpoint": tag,
                                "video_id": record.video_id,
                                **isolation,
                            }
                        )
                    case = {
                        "checkpoint": str(checkpoint),
                        "checkpoint_path": str(checkpoint_path(run, checkpoint)),
                        "video_id": record.video_id,
                        "frame_count": len(record.frames),
                        "object_count": count,
                        "model_loading_seconds": loading,
                        "session_setup_seconds": setup,
                        "first_trajectory": first,
                        "repetitions": repetitions,
                        "model_seconds": float(
                            np.mean([r["model_seconds"] for r in repetitions])
                        ),
                        "transfer_seconds": float(
                            np.mean([r["transfer_seconds"] for r in repetitions])
                        ),
                        "total_seconds": time.perf_counter() - count_start,
                        "reset_replay_checkpass": True,
                        "session_isolation_checkpass": isolation["passed"],
                    }
                    cases.append(case)
                    atomic_json(
                        destination / f"{tag}-{record.video_id}-{count}.json", case
                    )
                    del session
            del predictor
            gc.collect()
            torch.cuda.empty_cache()
        if not cases:
            raise RuntimeError("No valid benchmark cases")
        workload_counts = {}
        for split in ("dev", "test", "official_valid"):
            videos = load_partition(run, split)
            workload_counts[split] = {
                "videos": len(videos),
                "frames": sum(len(r.frames) for r in videos),
                "frame_objects": sum(len(r.frames) * len(r.object_ids) for r in videos),
            }
        result = {
            "schema_version": 1,
            "cases": cases,
            "unsupported": unsupported,
            "supportedcounts": sorted({c["object_count"] for c in cases}),
            "reset_replay_checkpass": all(c["reset_replay_checkpass"] for c in cases),
            "session_isolation_checkpass": all(
                probe["passed"] for probe in isolation_probes
            ),
            "isolation_checkpass": all(probe["passed"] for probe in isolation_probes),
            "isolation_probes": isolation_probes,
            "per_frame_seconds": sum(c["model_seconds"] for c in cases)
            / sum(c["frame_count"] for c in cases),
            "per_frame_object_seconds": sum(
                c["model_seconds"] + c["transfer_seconds"] for c in cases
            )
            / sum(c["frame_count"] * c["object_count"] for c in cases),
            "stage_duration_seconds": time.perf_counter() - whole_start,
            "workload_counts": workload_counts,
            "boundary": "CUDA synchronized model calls; staged frame/prompt transfers and host output materialization separate",
            "session_setup": "CPU frame decoding plus upstream init_state first image encoding; excluded from model timings",
            "memory_label": "torch.cuda.max_memory_allocated; not process RSS",
            "precision": "BF16 autocast / FP32 parameters",
            "cpu_frame_storage": True,
            "pilot": pilot,
        }
        atomic_json(destination / "summary.json", result)
    return result
