"""Immutable, notebook-edited experiment settings (no hidden CLI config)."""

import math
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class DataConfig:
    repo_id: str = "FudanCVL/MOSEv2"
    revision: str = "24bd8fc5f5ab824a482b5f6dcf472e3a3ef5c704"
    train_clips: int = 300
    dev_clips: int = 40
    test_clips: int = 60
    seed: int = 2027
    official_valid_count: int = 433
    near_duplicate_hamming: int = 4
    duplicate_sample_frames: int = 8


@dataclass(frozen=True)
class ModelConfig:
    source_revision: str = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
    config_name: str = "configs/sam2.1/sam2.1_hiera_t.yaml"
    checkpoint_url: str = (
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
    )
    image_size: int = 1024
    trainable_prefixes: tuple[str, ...] = (
        "sam_mask_decoder.",
        "memory_attention.",
        "memory_encoder.",
        "obj_ptr_proj.",
        "obj_ptr_tpos_proj.",
    )
    precision: str = "bfloat16"
    compile: bool = False


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 2027
    frames: int = 4
    strides: tuple[int, ...] = (1, 2, 4, 8)
    accumulation: int = 4
    updates: int = 1000
    learning_rate: float = 1e-5
    final_learning_rate: float = 1e-6
    warmup_updates: int = 50
    betas: tuple[float, float] = (0.9, 0.999)
    epsilon: float = 1e-8
    weight_decay: float = 0.01
    clip_norm: float = 1.0
    horizontal_flip_probability: float = 0.5
    point_probability: float = 0.5
    box_probability: float = 0.25
    mask_probability: float = 0.25
    validate_every: int = 200
    checkpoint_every: int = 100
    log_every: int = 10


@dataclass(frozen=True)
class EvaluationConfig:
    scorer_revision: str = "a74b23aa3a31597f487eec10f452e0b9f61347e1"
    protocols: tuple[str, ...] = ("point", "point_correction", "mask")
    threshold: float = 0.0
    offload_video_to_cpu: bool = True


@dataclass(frozen=True)
class BenchmarkConfig:
    object_counts: tuple[int, ...] = (1, 2, 4)
    warmups: int = 1
    repetitions: int = 3
    clips_per_condition: int = 2


@dataclass(frozen=True)
class PilotConfig:
    updates: int = 20
    overfit_updates: int = 30
    target_gpu: str = "A40 48 GB"
    gpu_hourly_usd: float = 0.49
    storage_gb: int = 300
    storage_monthly_usd_per_gb: float = 0.07
    storage_days: float = 3.0
    planning_target_usd: float = 30.0


@dataclass(frozen=True)
class DemoConfig:
    server_name: str = "127.0.0.1"
    server_port: int = 7860
    share: bool = False


@dataclass(frozen=True)
class ProjectConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    pilot: PilotConfig = field(default_factory=PilotConfig)
    demo: DemoConfig = field(default_factory=DemoConfig)

    def validate(self) -> None:
        if (
            self.model.image_size != 1024
            or self.model.compile
            or self.model.precision != "bfloat16"
        ):
            raise ValueError(
                "This protocol requires native 1024 inputs, BF16 and no compilation"
            )
        t = self.training
        if t.frames != 4 or any(s < 1 for s in t.strides) or not t.strides:
            raise ValueError(
                "Use four chronological frames and positive temporal strides"
            )
        for value in (
            t.accumulation,
            t.updates,
            t.validate_every,
            t.checkpoint_every,
            t.log_every,
        ):
            if value < 1:
                raise ValueError("Update and interval settings must be positive")
        if not 0 <= t.warmup_updates < t.updates:
            raise ValueError("Warm-up must be shorter than training")
        if abs(t.point_probability + t.box_probability + t.mask_probability - 1) > 1e-9:
            raise ValueError("Prompt probabilities must sum to one")
        if not 0 <= t.horizontal_flip_probability <= 1:
            raise ValueError("Invalid flip probability")
        if self.pilot.updates < 20:
            raise ValueError(
                "Pilot requires at least 20 representative optimizer updates"
            )
        if min(self.data.train_clips, self.data.dev_clips, self.data.test_clips) < 1:
            raise ValueError("All local partitions must be nonempty")
        probabilities = (
            t.point_probability,
            t.box_probability,
            t.mask_probability,
            t.horizontal_flip_probability,
        )
        if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
            raise ValueError(
                "Every probability must be finite and between zero and one"
            )
        if not all(
            math.isfinite(x) and x > 0
            for x in (t.learning_rate, t.final_learning_rate, t.epsilon, t.clip_norm)
        ):
            raise ValueError(
                "Learning rates, epsilon and clipping must be finite and positive"
            )
        if (
            t.final_learning_rate > t.learning_rate
            or not math.isfinite(t.weight_decay)
            or t.weight_decay < 0
        ):
            raise ValueError("Invalid decay or final learning rate")
        if len(t.betas) != 2 or any(not 0 <= b < 1 for b in t.betas):
            raise ValueError("AdamW betas must be in [0,1)")
        if (
            self.pilot.overfit_updates < 1
            or self.benchmark.warmups < 1
            or self.benchmark.repetitions < 1
        ):
            raise ValueError(
                "Pilot overfit and benchmark repetition counts must be positive"
            )
        if (
            not self.benchmark.object_counts
            or min(self.benchmark.object_counts) < 1
            or self.benchmark.clips_per_condition < 1
        ):
            raise ValueError("Benchmark object/clip counts must be positive")
        prices = (
            self.pilot.gpu_hourly_usd,
            self.pilot.storage_monthly_usd_per_gb,
            self.pilot.storage_days,
            self.pilot.planning_target_usd,
        )
        if (
            any(not math.isfinite(p) or p < 0 for p in prices)
            or self.pilot.storage_gb < 1
        ):
            raise ValueError("Advisory price and storage settings are invalid")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def config_from_dict(value: dict[str, Any]) -> ProjectConfig:
    classes = {
        "data": DataConfig,
        "model": ModelConfig,
        "training": TrainingConfig,
        "evaluation": EvaluationConfig,
        "benchmark": BenchmarkConfig,
        "pilot": PilotConfig,
        "demo": DemoConfig,
    }
    converted = {}
    for name, cls in classes.items():
        fields = dict(value.get(name, {}))
        for key, item in fields.items():
            if isinstance(item, list):
                fields[key] = tuple(item)
        converted[name] = cls(**fields)
    result = ProjectConfig(**converted)
    result.validate()
    return result
