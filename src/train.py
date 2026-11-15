"""Deterministic short-clip temporal fine-tuning; notebook owns orchestration only."""

from __future__ import annotations

import json
import logging
import math
import random
import time
import uuid
from dataclasses import dataclass

from .checkpoints import is_better, promote_best, restore_checkpoint, save_checkpoint
from .io import atomic_json, read_json
from .sam2_adapter import build_loss, build_train_model, require_cuda, set_training_mode


@dataclass
class ClipBatch:
    flat_img_batch: object
    masks: object
    flat_obj_to_img_idx: object
    num_frames: int


def learning_rate(config, completed_updates):
    step = completed_updates + 1
    if config.warmup_updates and step <= config.warmup_updates:
        return config.learning_rate * step / config.warmup_updates
    progress = (step - config.warmup_updates) / (config.updates - config.warmup_updates)
    progress = min(1.0, max(0.0, progress))
    return config.final_learning_rate + 0.5 * (
        config.learning_rate - config.final_learning_rate
    ) * (1 + math.cos(math.pi * progress))


def make_optimizer(model, config):
    import torch

    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (
                no_decay if parameter.ndim < 2 or name.endswith(".bias") else decay
            ).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=config.learning_rate,
        betas=config.betas,
        eps=config.epsilon,
    )


def seed_training(seed):
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ClipSampler:
    """Sample identity depends only on (seed,cursor), so interrupted updates replay."""

    def __init__(self, records, config, image_size=1024, overrides=None):
        self.records = tuple(records)
        self.config = config
        self.image_size = image_size
        self.overrides = dict(overrides or {})
        if not self.records:
            raise ValueError("Training partition is empty")
        if any(
            record.annotations is None
            or len(record.frames) != len(record.annotations)
            or len(record.frames) < config.frames
            for record in self.records
        ):
            raise ValueError(
                "Short-clip training requires >=4 aligned, fully annotated frames"
            )

    def selection(self, cursor):
        import numpy as np
        from PIL import Image

        if cursor in self.overrides:
            return self.overrides[cursor]
        epoch, offset = divmod(cursor, len(self.records))
        order = np.random.default_rng(
            np.random.SeedSequence([self.config.seed, epoch, 991])
        ).permutation(len(self.records))
        record = self.records[int(order[offset])]
        rng = np.random.default_rng(np.random.SeedSequence([self.config.seed, cursor]))
        strides = [
            s
            for s in self.config.strides
            if 1 + s * (self.config.frames - 1) <= len(record.frames)
        ]
        rng.shuffle(strides)
        for stride in strides:
            starts = rng.permutation(
                len(record.frames) - stride * (self.config.frames - 1)
            )
            for start in starts:
                with Image.open(record.annotations[int(start)]) as image:
                    mask = np.asarray(image)
                present = sorted(set(np.unique(mask).tolist()) & set(record.object_ids))
                if present:
                    object_id = int(rng.choice(present))
                    return (
                        record,
                        tuple(
                            int(start) + i * stride for i in range(self.config.frames)
                        ),
                        object_id,
                        bool(rng.random() < self.config.horizontal_flip_probability),
                    )
        raise ValueError(
            f"No visible object can initialize four frames in {record.video_id}"
        )

    def batch(self, cursor, device):
        import numpy as np
        import torch
        from PIL import Image

        record, indices, object_id, flip = self.selection(cursor)
        images, masks = [], []
        for index in indices:
            with Image.open(record.frames[index]) as image:
                # Explicit BICUBIC matches PIL's RGB default used by SAM2's loader.
                rgb = image.convert("RGB").resize(
                    (self.image_size, self.image_size), Image.Resampling.BICUBIC
                )
                array = np.array(rgb, dtype=np.float32) / 255.0
            with Image.open(record.annotations[index]) as image:
                binary = (np.asarray(image) == object_id).astype(np.uint8)
            binary = np.asarray(
                Image.fromarray(binary).resize(
                    (self.image_size, self.image_size), Image.Resampling.NEAREST
                )
            )
            if flip:
                array, binary = array[:, ::-1], binary[:, ::-1]
            images.append(torch.from_numpy(array.copy()).permute(2, 0, 1))
            masks.append(torch.from_numpy(binary.copy()).bool())
        if not bool(masks[0].any()):
            raise ValueError(
                f"Object vanished during resizing in {record.video_id}; do not silently resample"
            )
        tensor = torch.stack(images)
        tensor = (
            tensor - tensor.new_tensor([0.485, 0.456, 0.406])[:, None, None]
        ) / tensor.new_tensor([0.229, 0.224, 0.225])[:, None, None]
        batch = ClipBatch(
            tensor.to(device),
            torch.stack(masks)[:, None].to(device),
            torch.arange(len(indices), device=device)[:, None],
            len(indices),
        )
        return batch, {
            "cursor": cursor,
            "video_id": record.video_id,
            "object_id": object_id,
            "frame_indices": indices,
            "horizontal_flip": flip,
            "contains_absence": any(not bool(mask.any()) for mask in masks),
        }


def gradient_report(model, prefixes):
    import torch

    result = {}
    for prefix in prefixes:
        grads = [
            p.grad
            for name, p in model.named_parameters()
            if name.startswith(prefix) and p.requires_grad and p.grad is not None
        ]
        result[prefix] = {
            "parameters_with_gradient": len(grads),
            "finite": bool(grads) and all(bool(torch.isfinite(g).all()) for g in grads),
            "nonzero": any(bool(torch.count_nonzero(g)) for g in grads),
        }
    return result


def optimizer_update(
    model,
    optimizer,
    criterion,
    sampler,
    cursor,
    config,
    *,
    inspect_gradients=False,
    fixed_cursors=None,
    device="cuda",
):
    import torch

    optimizer.zero_grad(set_to_none=True)
    metrics, samples = {}, []
    cursors = (
        tuple(fixed_cursors)
        if fixed_cursors is not None
        else tuple(range(cursor, cursor + config.accumulation))
    )
    for sample_cursor in cursors:
        batch, description = sampler.batch(sample_cursor, device)
        # No no_grad around forward_image or memory: decoder projections and temporal
        # predictions must remain connected even though the image encoder is frozen.
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=str(device).startswith("cuda"),
        ):
            output = model(batch)
            losses = criterion(output, batch.masks)
            loss = losses["core_loss"] / len(cursors)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite short-clip loss")
        loss.backward()
        for key, value in losses.items():
            metrics[key] = metrics.get(key, 0.0) + float(value.detach()) / len(cursors)
        samples.append(description)
        del output, losses, loss, batch
    gradients = (
        gradient_report(model, sampler.run_prefixes) if inspect_gradients else None
    )
    norm = torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad],
        config.clip_norm,
        error_if_nonfinite=True,
    )
    optimizer.step()
    metrics["gradient_norm"] = float(norm)
    return {
        "metrics": metrics,
        "samples": samples,
        "gradients": gradients,
        "cursor": cursor + len(cursors),
    }


def _optimizer_to(optimizer, device):
    import torch

    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor) and key != "step":
                state[key] = value.to(device)


def fine_tune(run):
    import torch
    from torch.utils.tensorboard import SummaryWriter
    from tqdm.auto import tqdm

    from .data import load_partition
    from .evaluate import evaluate_local
    from .runs import stage

    require_cuda()
    pilot = read_json(run.path / "pilot" / "pilot.json")
    if not pilot.get("passed"):
        raise RuntimeError("The pilot has not passed; inspect pilot/pilot.json")
    with stage(run, "fine_tune"):
        config = run.config.training
        seed_training(config.seed)
        model = build_train_model(run)
        optimizer = make_optimizer(model, config)
        criterion = build_loss()
        sampler = ClipSampler(
            load_partition(run, "train"), config, run.config.model.image_size
        )
        sampler.run_prefixes = run.config.model.trainable_prefixes
        state = {
            "update": 0,
            "cursor": 0,
            "best_score": None,
            "best_update": None,
            "pending_validation": False,
        }
        if (run.path / "checkpoints" / "latest.json").exists():
            state = restore_checkpoint(run, model, optimizer)
        writer = SummaryWriter(str(run.path / "tensorboard" / "training"))
        attempt = uuid.uuid4().hex
        resumed_from = state["update"]
        log_path = run.path / "logs" / "training.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger(__name__)
        old_level = logger.level
        handler = logging.FileHandler(run.path / "logs" / "training.log")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        def validate():
            model_path = save_checkpoint(
                run,
                model,
                optimizer,
                **{
                    key: state[key]
                    for key in ("update", "cursor", "best_score", "best_update")
                },
                pending_validation=True,
            )
            model.to("cpu")
            _optimizer_to(optimizer, "cpu")
            torch.cuda.empty_cache()
            try:
                result = evaluate_local(
                    run, checkpoint=model_path, split="dev", protocol="point"
                )
            finally:
                model.to("cuda")
                _optimizer_to(optimizer, "cuda")
                set_training_mode(model)
            score = float(result["primary_score"])
            if is_better(score, state["best_score"]):
                state["best_score"], state["best_update"] = score, state["update"]
                promote_best(run, model_path)
            state["pending_validation"] = False
            writer.add_scalar("dev/JF_dot", score, state["update"])
            save_checkpoint(
                run,
                model,
                optimizer,
                **{
                    key: state[key]
                    for key in (
                        "update",
                        "cursor",
                        "best_score",
                        "best_update",
                        "pending_validation",
                    )
                },
            )

        try:
            if state["pending_validation"]:
                validate()
            for update in tqdm(
                range(state["update"], config.updates),
                desc="Fine-tuning",
                initial=state["update"],
                total=config.updates,
            ):
                started = time.perf_counter()
                rate = learning_rate(config, update)
                for group in optimizer.param_groups:
                    group["lr"] = rate
                result = optimizer_update(
                    model, optimizer, criterion, sampler, state["cursor"], config
                )
                state["update"], state["cursor"] = update + 1, result["cursor"]
                record = {
                    "update": update + 1,
                    "attempt": attempt,
                    "resumed_from_update": resumed_from,
                    "learning_rate": rate,
                    "seconds": time.perf_counter() - started,
                    **result,
                }
                with log_path.open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
                if (update + 1) % config.log_every == 0:
                    for key, value in result["metrics"].items():
                        writer.add_scalar(f"train/{key}", value, update + 1)
                    writer.add_scalar("train/learning_rate", rate, update + 1)
                    logger.info(
                        "attempt=%s update=%s loss=%.5f",
                        attempt,
                        update + 1,
                        result["metrics"]["core_loss"],
                    )
                if (
                    update + 1
                ) % config.validate_every == 0 or update + 1 == config.updates:
                    state["pending_validation"] = True
                    validate()
                elif (update + 1) % config.checkpoint_every == 0:
                    save_checkpoint(
                        run,
                        model,
                        optimizer,
                        **{
                            key: state[key]
                            for key in (
                                "update",
                                "cursor",
                                "best_score",
                                "best_update",
                                "pending_validation",
                            )
                        },
                    )
        finally:
            writer.close()
            logger.removeHandler(handler)
            handler.close()
            logger.setLevel(old_level)
        result = {
            "updates": state["update"],
            "best_update": state["best_update"],
            "best_dev_JF_dot": state["best_score"],
        }
        atomic_json(run.path / "checkpoints" / "training_summary.json", result)
        return result
