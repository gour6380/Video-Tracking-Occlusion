"""Measured GPU pilot. No success record is emitted without executing every gate."""

from __future__ import annotations

import hashlib
import math
import time

from .checkpoints import (
    capture_rng,
    cpu_tree,
    restore_checkpoint,
    restore_rng,
    save_checkpoint,
)
from .io import atomic_json
from .sam2_adapter import build_loss, build_train_model, require_cuda
from .train import ClipSampler, make_optimizer, optimizer_update, seed_training


def parameter_digest(model, trainable=False):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad == trainable:
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _loss_on_fixed_clips(model, criterion, sampler, cursors, *, device="cuda"):
    import numpy as np
    import torch

    saved_rng = capture_rng(model)
    modes = {module: module.training for module in model.modules()}
    model.eval()
    # Use the training prompt mixture without dropout during this diagnostic.
    model.training = True
    values = []
    try:
        # Upstream chooses the prompt branch with NumPy, but samples point
        # coordinates / box jitter with torch.rand on the mask's device.
        # Reset both generators for paired diagnostics, then restore every RNG
        # below so measuring loss does not perturb subsequent training.
        seed_training(sampler.config.seed)
        model.rng = np.random.default_rng(sampler.config.seed)
        with (
            torch.no_grad(),
            torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=str(device).startswith("cuda")
            ),
        ):
            for cursor in cursors:
                batch, _ = sampler.batch(cursor, device)
                value = float(criterion(model(batch), batch.masks)["core_loss"])
                if not math.isfinite(value):
                    raise FloatingPointError("Nonfinite fixed-clip diagnostic loss")
                values.append(value)
    finally:
        for module, training in modes.items():
            module.training = training
        restore_rng(model, saved_rng)
    return sum(values) / len(values)


def _overfit_selections(sampler, occlusion):
    """Four distinct videos, with the annotated-disappearance fixture first."""
    selected = {0: occlusion}
    seen = {occlusion[0].video_id}
    for cursor in range(len(sampler.records)):
        selection = sampler.selection(cursor)
        if selection[0].video_id in seen:
            continue
        seen.add(selection[0].video_id)
        selected[len(selected)] = selection
        if len(selected) == 4:
            return selected
    raise ValueError(
        "Four distinct training videos are required for the overfit diagnostic"
    )


def _tree_close(first, second, *, atol=0.0, rtol=0.0):
    import torch

    if isinstance(first, torch.Tensor):
        return (
            isinstance(second, torch.Tensor)
            and first.shape == second.shape
            and bool(torch.allclose(first.cpu(), second.cpu(), atol=atol, rtol=rtol))
        )
    if isinstance(first, dict):
        return first.keys() == second.keys() and all(
            _tree_close(first[key], second[key], atol=atol, rtol=rtol) for key in first
        )
    if isinstance(first, (list, tuple)):
        return len(first) == len(second) and all(
            _tree_close(a, b, atol=atol, rtol=rtol) for a, b in zip(first, second)
        )
    return first == second


def find_occlusion_selection(records, config):
    """Select a real annotated visible->absent four-frame clip; never synthesize absence."""
    import numpy as np
    from PIL import Image

    for record in sorted(records, key=lambda item: item.video_id):
        present = []
        for annotation in record.annotations:
            with Image.open(annotation) as image:
                present.append(set(np.unique(np.asarray(image)).tolist()))
        for stride in config.strides:
            for start in range(len(record.frames) - stride * (config.frames - 1)):
                indices = tuple(start + i * stride for i in range(config.frames))
                for object_id in sorted(set(record.object_ids) & present[start]):
                    if any(object_id not in present[index] for index in indices[1:]):
                        return record, indices, object_id, False
    raise ValueError(
        "No visible-to-absent training clip fits the declared four-frame/stride protocol; inspect the active manifest"
    )


def project_cost(
    run,
    update_seconds,
    tracking,
    partitions,
    *,
    measured_pilot_seconds=0.0,
    checkpoint_save_seconds=0.0,
):
    """Advisory projection of declared work, excluding unmeasured external queues."""
    cases = tracking.get("cases", [])
    observations = []
    for case in cases:
        for repetition in case["repetitions"]:
            observations.append(
                (
                    repetition["model_seconds"]
                    + repetition["transfer_seconds"]
                    + repetition["output_materialization_seconds"]
                )
                / (case["frame_count"] * case["object_count"])
            )
    if not observations:
        raise ValueError(
            "Pilot benchmark did not provide measured frame-object throughput"
        )
    # Median observed per-frame-object cost is an explicit approximation; long clips
    # and correction replays may differ and are not promised to scale linearly.
    import statistics

    seconds_per_object_frame = statistics.median(observations)
    units = {
        split: sum(len(item.frames) * len(item.object_ids) for item in records)
        for split, records in partitions.items()
    }
    validation_updates = set(
        range(
            run.config.training.validate_every,
            run.config.training.updates + 1,
            run.config.training.validate_every,
        )
    ) | {run.config.training.updates}
    checks = 1 + len(
        validation_updates
    )  # The notebook also evaluates initial weights on dev.
    correction_multiplier = 2.0  # one initial pass plus at most one full replay
    local_multiplier = 2 * (
        2 + correction_multiplier
    )  # two checkpoints; point, mask, correction
    requested_units = (
        units["dev"] * checks + units["test"] * local_multiplier + units["valid"] * 2
    )
    fit_seconds = update_seconds * run.config.training.updates
    evaluation_seconds = seconds_per_object_frame * requested_units
    full_cases = 2 * sum(
        min(
            run.config.benchmark.clips_per_condition,
            sum(len(record.object_ids) >= count for record in partitions["dev"]),
        )
        for count in run.config.benchmark.object_counts
    )
    benchmark_seconds = (
        tracking["stage_duration_seconds"]
        * full_cases
        / len(cases)
        * (1 + run.config.benchmark.warmups + run.config.benchmark.repetitions)
        / (2 + run.config.benchmark.warmups)
    )
    setup_per_video = statistics.median(case["session_setup_seconds"] for case in cases)
    setup_seconds = setup_per_video * (
        len(partitions["dev"]) * checks
        + len(partitions["test"]) * 6
        + len(partitions["valid"]) * 2
    )
    loading_per_model = statistics.median(
        case["model_loading_seconds"] for case in cases
    )
    model_load_seconds = loading_per_model * (1 + checks + 6 + 2)
    ordinary_saves = sum(
        update not in validation_updates
        for update in range(
            run.config.training.checkpoint_every,
            run.config.training.updates + 1,
            run.config.training.checkpoint_every,
        )
    )
    checkpoint_writes = ordinary_saves + 2 * len(validation_updates)
    checkpoint_write_seconds = checkpoint_save_seconds * checkpoint_writes
    # A visible advisory allowance, not a measured demo-runtime prediction.
    demo_allowance_seconds = 1800.0
    measured_gpu_seconds = (
        fit_seconds
        + evaluation_seconds
        + setup_seconds
        + model_load_seconds
        + checkpoint_write_seconds
        + benchmark_seconds
        + measured_pilot_seconds
        + demo_allowance_seconds
    )
    storage_usd = (
        run.config.pilot.storage_gb
        * run.config.pilot.storage_monthly_usd_per_gb
        * run.config.pilot.storage_days
        / 30
    )
    gpu_usd = measured_gpu_seconds / 3600 * run.config.pilot.gpu_hourly_usd
    return {
        "projected_gpu_seconds": measured_gpu_seconds,
        "fit_seconds": fit_seconds,
        "evaluation_seconds": evaluation_seconds,
        "session_setup_seconds": setup_seconds,
        "model_load_seconds": model_load_seconds,
        "checkpoint_write_seconds": checkpoint_write_seconds,
        "checkpoint_writes": checkpoint_writes,
        "dev_evaluation_passes": checks,
        "benchmark_seconds": benchmark_seconds,
        "measured_pilot_seconds": measured_pilot_seconds,
        "demo_allowance_seconds": demo_allowance_seconds,
        "gpu_usd": gpu_usd,
        "storage_usd": storage_usd,
        "projected_usd": gpu_usd + storage_usd,
        "planning_target_usd": run.config.pilot.planning_target_usd,
        "frame_object_units": units,
        "assumptions": [
            "Median full-clip seconds per frame-object; nonlinear scaling is unverified",
            "Correction protocol allows one replay, bounded by twice the initial pass",
            "Includes measured pilot and an explicit 30-minute demo allowance; this allowance is not measured",
            "Model-load projection uses observed initial checkpoint loading; adapted loading may differ",
            "Checkpoint-write projection uses one measured complete recovery snapshot; retries are additional",
            "Validation server queue, downloads, environment setup, CPU scoring, prediction PNG/ZIP writes and unattended idle are additional",
            "Hourly and storage prices are notebook configuration inputs, not live quoted prices",
        ],
    }


def run_pilot(run):
    import torch
    from tqdm.auto import tqdm

    from .benchmark import benchmark_tracking
    from .data import load_partition
    from .runs import stage

    require_cuda()
    directory = run.path / "pilot"
    directory.mkdir(parents=True, exist_ok=True)
    result = {
        "passed": False,
        "gates": {},
        "scope": "Actual CUDA pilot, discarded weight updates",
    }
    pilot_started = time.perf_counter()
    with stage(run, "pilot"):
        try:
            seed_training(run.config.training.seed)
            model = build_train_model(run)
            criterion = build_loss()
            optimizer = make_optimizer(model, run.config.training)
            records = load_partition(run, "train")
            if len(records) < 4:
                raise ValueError("Pilot requires four distinct training clips")
            sampler = ClipSampler(
                records, run.config.training, run.config.model.image_size
            )
            sampler.run_prefixes = run.config.model.trainable_prefixes
            occlusion = find_occlusion_selection(records, run.config.training)
            representative_sampler = ClipSampler(
                records,
                run.config.training,
                run.config.model.image_size,
                overrides={0: occlusion},
            )
            representative_sampler.run_prefixes = sampler.run_prefixes
            frozen_before = parameter_digest(model)
            trainable_before = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            }
            gradient_checks = {
                prefix: {"finite": True, "nonzero": False}
                for prefix in sampler.run_prefixes
            }
            cursor, samples, durations = 0, [], []
            torch.cuda.reset_peak_memory_stats()
            for update in tqdm(range(run.config.pilot.updates), desc="Pilot updates"):
                torch.cuda.synchronize()
                started = time.perf_counter()
                record = optimizer_update(
                    model,
                    optimizer,
                    criterion,
                    representative_sampler,
                    cursor,
                    run.config.training,
                    inspect_gradients=update == 0,
                )
                torch.cuda.synchronize()
                durations.append(time.perf_counter() - started)
                cursor = record["cursor"]
                samples.extend(record["samples"])
                for prefix, details in (record["gradients"] or {}).items():
                    gradient_checks[prefix]["finite"] &= details["finite"]
                    gradient_checks[prefix]["nonzero"] |= details["nonzero"]
            changed = {
                prefix: any(
                    not torch.equal(trainable_before[name], parameter.detach().cpu())
                    for name, parameter in model.named_parameters()
                    if name.startswith(prefix) and parameter.requires_grad
                )
                for prefix in sampler.run_prefixes
            }
            result["gates"]["gradient_flow"] = all(
                item["finite"] and item["nonzero"] for item in gradient_checks.values()
            ) and all(changed.values())
            result["gates"]["frozen_identity"] = (
                parameter_digest(model) == frozen_before
            )
            result["gates"]["annotated_disappearance"] = samples[0]["contains_absence"]
            result["gradient_details"], result["changed_prefixes"] = (
                gradient_checks,
                changed,
            )
            result["update_seconds"], result["samples"] = durations, samples
            result["peak_training_bytes"] = torch.cuda.max_memory_allocated()
            del trainable_before

            # Recover a complete boundary, then independently replay its next update.
            recovery_dir = directory / "recovery"
            save_started = time.perf_counter()
            save_checkpoint(
                run,
                model,
                optimizer,
                update=run.config.pilot.updates,
                cursor=cursor,
                directory=recovery_dir,
            )
            checkpoint_save_seconds = time.perf_counter() - save_started
            result["checkpoint_save_seconds"] = checkpoint_save_seconds
            expected_model, expected_optimizer = (
                cpu_tree(model.state_dict()),
                cpu_tree(optimizer.state_dict()),
            )
            next_record = optimizer_update(
                model, optimizer, criterion, sampler, cursor, run.config.training
            )
            next_model, next_optimizer = (
                cpu_tree(model.state_dict()),
                cpu_tree(optimizer.state_dict()),
            )
            restored = restore_checkpoint(run, model, optimizer, directory=recovery_dir)
            exact_boundary = _tree_close(
                model.state_dict(), expected_model
            ) and _tree_close(optimizer.state_dict(), expected_optimizer)
            replay_record = optimizer_update(
                model,
                optimizer,
                criterion,
                sampler,
                restored["cursor"],
                run.config.training,
            )
            replay_close = _tree_close(
                model.state_dict(), next_model, atol=1e-6, rtol=1e-5
            ) and _tree_close(
                optimizer.state_dict(), next_optimizer, atol=1e-6, rtol=1e-5
            )
            result["gates"]["checkpoint_recovery"] = (
                exact_boundary
                and replay_close
                and next_record["samples"] == replay_record["samples"]
            )
            result["recovery"] = {
                "exact_boundary": exact_boundary,
                "next_update_close": replay_close,
                "atol": 1e-6,
                "rtol": 1e-5,
            }
            del (
                expected_model,
                expected_optimizer,
                next_model,
                next_optimizer,
                model,
                optimizer,
            )
            torch.cuda.empty_cache()

            seed_training(run.config.training.seed)
            model = build_train_model(run)
            optimizer = make_optimizer(model, run.config.training)
            fixed = (0, 1, 2, 3)
            overfit_sampler = ClipSampler(
                records,
                run.config.training,
                run.config.model.image_size,
                overrides=_overfit_selections(sampler, occlusion),
            )
            overfit_sampler.run_prefixes = sampler.run_prefixes
            before = _loss_on_fixed_clips(model, criterion, overfit_sampler, fixed)
            overfit_samples = []
            for update in tqdm(
                range(run.config.pilot.overfit_updates),
                desc="Four-clip overfit diagnostic",
            ):
                overfit_record = optimizer_update(
                    model,
                    optimizer,
                    criterion,
                    overfit_sampler,
                    0,
                    run.config.training,
                    fixed_cursors=fixed,
                )
                if update == 0:
                    overfit_samples = overfit_record["samples"]
            after = _loss_on_fixed_clips(model, criterion, overfit_sampler, fixed)
            includes_absence = any(
                sample["contains_absence"] for sample in overfit_samples
            )
            distinct_videos = (
                len({sample["video_id"] for sample in overfit_samples}) == 4
            )
            result["overfit"] = {
                "initial_loss": before,
                "final_loss": after,
                "updates": run.config.pilot.overfit_updates,
                "samples": overfit_samples,
                "includes_annotated_disappearance": includes_absence,
                "four_distinct_videos": distinct_videos,
                "criterion": "same fixed clips/prompts without dropout; final loss < initial loss",
            }
            result["gates"]["four_clip_overfit"] = (
                includes_absence
                and distinct_videos
                and math.isfinite(after)
                and after < before
            )
            del model, optimizer, criterion
            torch.cuda.empty_cache()

            tracking = benchmark_tracking(
                run,
                checkpoints=("initial",),
                output_dir=directory / "tracking",
                pilot=True,
            )
            result["tracking"] = tracking
            accounted = set(tracking["supportedcounts"]) | {
                item["objects"]
                for item in tracking["unsupported"]
                if item.get("reason")
            }
            result["gates"]["full_clip_object_counts"] = bool(
                tracking["supportedcounts"]
            ) and set(run.config.benchmark.object_counts).issubset(accounted)
            result["gates"]["reset_replay"] = bool(tracking["reset_replay_checkpass"])
            result["gates"]["session_isolation"] = bool(
                tracking["session_isolation_checkpass"]
            )
            partitions = {
                split: load_partition(run, split) for split in ("dev", "test")
            }
            partitions["valid"] = load_partition(run, "official_valid")
            result["forecast"] = project_cost(
                run,
                sum(durations[1:]) / len(durations[1:]),
                tracking,
                partitions,
                measured_pilot_seconds=time.perf_counter() - pilot_started,
                checkpoint_save_seconds=checkpoint_save_seconds,
            )
            result["passed"] = all(result["gates"].values())
            atomic_json(directory / "pilot.json", result)
            if not result["passed"]:
                failed = [
                    name for name, passed in result["gates"].items() if not passed
                ]
                raise RuntimeError(
                    f"Pilot gates failed: {failed}; inspect {directory / 'pilot.json'}"
                )
        except BaseException as error:
            result["error"] = f"{type(error).__name__}: {error}"
            atomic_json(directory / "pilot.json", result)
            raise
    return {
        "passed": True,
        "artifact": str(directory / "pilot.json"),
        "forecast": result["forecast"],
    }
