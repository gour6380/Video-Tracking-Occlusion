"""Lazy adapter for the pinned, vendored SAM2 source; no framework import at startup."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from types import MethodType
from urllib.request import Request, urlopen

from .io import atomic_json, read_json, sha256_file


def activate_source(run):
    source = run.root / "vendor" / "sam2"
    if not (source / "sam2" / "build_sam.py").is_file():
        raise FileNotFoundError(
            "Pinned SAM2 source is missing; run the documented environment setup first"
        )
    provenance = read_json(run.root / "artifacts" / "upstream.json")
    if provenance["sam2_revision"] != run.config.model.source_revision:
        raise ValueError("Vendored SAM2 revision differs from the run configuration")
    for relative, expected in provenance["files"].items():
        if (
            relative.startswith("vendor/sam2/")
            and sha256_file(run.root / relative) != expected
        ):
            raise ValueError(f"Pinned upstream source changed: {relative}")
    # Never mix another installed SAM2 checkout into this experiment.
    if "sam2" in sys.modules:
        loaded = Path(sys.modules["sam2"].__file__).resolve()
        if not loaded.is_relative_to(source.resolve()):
            raise RuntimeError(
                f"Another SAM2 source is already imported: {loaded}; restart the kernel"
            )
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    return source


def require_cuda():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "This video experiment requires CUDA; local CPU tests do not run the model"
        )
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "The selected CUDA device does not support the declared BF16 protocol"
        )
    return torch.device("cuda")


def ensure_model_assets(run):
    """Download the creator checkpoint at explicit runtime; retain its content identity."""
    activate_source(run)
    target = run.root / "data" / "models" / "sam2.1_hiera_tiny.pt"
    manifest = target.with_suffix(".json")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not manifest.exists():
            raise ValueError(
                f"Unverified checkpoint exists at {target}; preserve it and reacquire explicitly"
            )
        identity = read_json(manifest)
        if identity["url"] != run.config.model.checkpoint_url or identity[
            "sha256"
        ] != sha256_file(target):
            raise ValueError("Cached initial checkpoint identity changed")
    else:
        fd, temporary = tempfile.mkstemp(prefix=".checkpoint-", dir=target.parent)
        try:
            with (
                os.fdopen(fd, "wb") as stream,
                urlopen(
                    Request(
                        run.config.model.checkpoint_url,
                        headers={"User-Agent": "Video-Tracking-Occlusion/1.0"},
                    ),
                    timeout=120,
                ) as response,
            ):
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
            identity = {
                "url": run.config.model.checkpoint_url,
                "sha256": sha256_file(target),
                "checksum_scope": "recorded download identity; no creator SHA-256 supplied",
            }
            atomic_json(manifest, identity)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    artifact = run.path / "provenance" / "initial_checkpoint.json"
    if artifact.exists() and read_json(artifact) != identity:
        raise ValueError("Initial checkpoint identity differs from this run")
    atomic_json(artifact, identity)
    return target


def checkpoint_path(run, checkpoint="initial"):
    if isinstance(checkpoint, Path) or checkpoint not in ("initial", "best", "latest"):
        result = Path(checkpoint)
    elif checkpoint == "initial":
        result = ensure_model_assets(run)
    else:
        pointer = read_json(run.path / "checkpoints" / f"{checkpoint}.json")
        result = run.path / "checkpoints" / pointer["directory"] / "model.pt"
        if sha256_file(result) != pointer["model_sha256"]:
            raise ValueError(f"{checkpoint} checkpoint checksum mismatch")
    if not result.is_file():
        raise FileNotFoundError(result)
    return result


def freeze_for_training(model, prefixes):
    counts = {}
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(any(name.startswith(prefix) for prefix in prefixes))
        if parameter.requires_grad:
            counts[name] = parameter.numel()
    for prefix in prefixes:
        if not any(name.startswith(prefix) for name in counts):
            raise ValueError(f"Trainable prefix matched no parameters: {prefix}")
    set_training_mode(model)
    return counts


def set_training_mode(model):
    model.train()
    model.image_encoder.eval()
    model.sam_prompt_encoder.eval()


def disable_zero_correction_loop(model):
    """Pinned upstream's zero-iteration loop returns an unbound local; skip that loop."""
    original = model.prepare_prompt_inputs

    def prepare(self, *args, **kwargs):
        prepared = original(*args, **kwargs)
        if self.num_correction_pt_per_frame == 0:
            prepared["frames_to_add_correction_pt"] = []
        return prepared

    model.prepare_prompt_inputs = MethodType(prepare, model)
    return model


def build_train_model(run, checkpoint="initial", device="cuda"):
    activate_source(run)
    import numpy as np

    # Importing sam2 initializes its Hydra config module.
    import sam2  # noqa: F401
    import torch
    from hydra import compose
    from hydra.utils import instantiate
    from omegaconf import OmegaConf, open_dict

    if str(device).startswith("cuda"):
        require_cuda()
    config = compose(config_name=run.config.model.config_name)
    model_config = config.model
    model_config._target_ = "training.model.sam2.SAM2Train"
    training = run.config.training
    point_branch = training.point_probability + training.box_probability
    additions = {
        "freeze_image_encoder": True,
        "prob_to_use_pt_input_for_train": point_branch,
        "prob_to_use_box_input_for_train": training.box_probability / point_branch
        if point_branch
        else 0.0,
        "prob_to_use_pt_input_for_eval": 0.0,
        "prob_to_use_box_input_for_eval": 0.0,
        "num_init_cond_frames_for_train": 1,
        "num_init_cond_frames_for_eval": 1,
        "rand_init_cond_frames_for_train": False,
        "rand_init_cond_frames_for_eval": False,
        "num_frames_to_correct_for_train": 1,
        "num_frames_to_correct_for_eval": 1,
        "rand_frames_to_correct_for_train": False,
        "rand_frames_to_correct_for_eval": False,
        "num_correction_pt_per_frame": 0,
        "prob_to_sample_from_gt_for_train": 0.0,
        "add_all_frames_to_correct_as_cond": False,
        "forward_backbone_per_frame_for_eval": True,
    }
    with open_dict(model_config):
        for key, value in additions.items():
            model_config[key] = value
    OmegaConf.resolve(config)
    model = instantiate(model_config, _recursive_=True)
    state = torch.load(
        checkpoint_path(run, checkpoint), map_location="cpu", weights_only=True
    )
    model.load_state_dict(state["model"], strict=True)
    model.rng = np.random.default_rng(training.seed)
    disable_zero_correction_loop(model)
    freeze_for_training(model, run.config.model.trainable_prefixes)
    return model.to(device)


def patch_predictor_memory_clearing(predictor):
    """Supply the per-object helper missing from the pinned upstream revision.

    Both native preflight and propagation call this name, but that revision only
    defines an older all-object variant. Bind an instance-local implementation;
    preserve its inclusive temporal radius without clearing another object's
    memories or any object's conditioning prompts. Vendor files remain intact.
    """
    if hasattr(predictor, "_clear_obj_non_cond_mem_around_input"):
        return predictor

    def clear_for_object(self, inference_state, frame_idx, obj_idx):
        radius = self.memory_temporal_stride_for_eval * self.num_maskmem
        memories = inference_state["output_dict_per_obj"][obj_idx][
            "non_cond_frame_outputs"
        ]
        for index in range(frame_idx - radius, frame_idx + radius + 1):
            memories.pop(index, None)

    predictor._clear_obj_non_cond_mem_around_input = MethodType(
        clear_for_object, predictor
    )
    return predictor


def build_predictor(run, checkpoint="initial", device="cuda"):
    activate_source(run)
    if str(device).startswith("cuda"):
        require_cuda()
    from sam2.build_sam import build_sam2_video_predictor

    predictor = build_sam2_video_predictor(
        run.config.model.config_name,
        ckpt_path=str(checkpoint_path(run, checkpoint)),
        device=device,
        apply_postprocessing=False,
        vos_optimized=False,
        # Resolve overlaps once in labels_from_logits after sorting persistent IDs.
        # Native suppression would irrevocably break ties by insertion order.
        hydra_overrides_extra=[
            "++model.non_overlap_masks=false",
            "++model.add_all_frames_to_correct_as_cond=true",
            "++model.clear_non_cond_mem_around_input=true",
        ],
    )
    return patch_predictor_memory_clearing(predictor)


def build_loss():
    from training.loss_fns import MultiStepMultiMasksAndIous

    return MultiStepMultiMasksAndIous(
        weight_dict={"loss_mask": 20, "loss_dice": 1, "loss_iou": 1, "loss_class": 1},
        supervise_all_iou=True,
        iou_use_l1_loss=True,
        pred_obj_scores=True,
        focal_gamma_obj_score=0.0,
        focal_alpha_obj_score=-1.0,
    )


def inspect_model(run):
    """CPU architecture inspection is an explicit notebook operation, never import-time."""
    from .runs import stage

    with stage(run, "inspect_model"):
        model = build_train_model(run, device="cpu")
        names = {n: p.numel() for n, p in model.named_parameters() if p.requires_grad}
        result = {
            "source_revision": run.config.model.source_revision,
            "config_name": run.config.model.config_name,
            "parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(names.values()),
            "trainable_names": names,
            "architecture": str(model),
            "initial_checkpoint": str(checkpoint_path(run)),
            "training_inputs": "4 chronological frames, 1 object, first-frame prompt only",
            "loss_weights": {
                "focal_mask": 20,
                "dice": 1,
                "iou": 1,
                "object_presence": 1,
            },
        }
        atomic_json(run.path / "provenance" / "model.json", result)
    return {
        "parameters": result["parameters"],
        "trainable_parameters": result["trainable_parameters"],
        "artifact": str(run.path / "provenance" / "model.json"),
    }
