"""Synthetic CPU-only checks; no checkpoint download, SAM2 execution or real data."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from src.checkpoints import is_better
from src.config import ProjectConfig, TrainingConfig
from src.train import ClipSampler, learning_rate


def test_schedule_endpoints_and_ties():
    config = TrainingConfig()
    assert learning_rate(config, 0) == pytest.approx(
        config.learning_rate / config.warmup_updates
    )
    assert learning_rate(config, config.warmup_updates - 1) == config.learning_rate
    assert learning_rate(config, config.updates - 1) == config.final_learning_rate
    assert is_better(0.4, None)
    assert not is_better(0.4, 0.4)
    with pytest.raises(ValueError):
        is_better(float("nan"), None)


def _record(tmp_path):
    frames, masks = [], []
    for index in range(4):
        frame = tmp_path / f"{index:03}.jpg"
        mask = tmp_path / f"{index:03}.png"
        Image.fromarray(np.full((8, 8, 3), index * 30, dtype=np.uint8)).save(frame)
        pixels = np.zeros((8, 8), dtype=np.uint8)
        if index != 2:
            pixels[2:5, 2:5] = 1
        Image.fromarray(pixels).save(mask)
        frames.append(frame)
        masks.append(mask)
    return SimpleNamespace(
        video_id="synthetic",
        frames=tuple(frames),
        annotations=tuple(masks),
        object_ids=(1,),
    )


def test_sampler_replay_and_absent_frames(tmp_path):
    pytest.importorskip("torch")
    sampler = ClipSampler([_record(tmp_path)], TrainingConfig(), image_size=8)
    first = sampler.selection(17)
    assert first == sampler.selection(17)
    assert first[1] == (0, 1, 2, 3)
    batch, metadata = sampler.batch(17, "cpu")
    assert batch.flat_img_batch.shape == (4, 3, 8, 8)
    assert batch.masks.shape == (4, 1, 8, 8)
    assert batch.masks[0].any() and not batch.masks[2].any()
    assert metadata["contains_absence"]


def test_pilot_selects_annotated_disappearance(tmp_path):
    from src.pilot import find_occlusion_selection

    record = _record(tmp_path)
    selected, indices, object_id, flipped = find_occlusion_selection(
        [record], TrainingConfig()
    )
    assert selected.video_id == record.video_id
    assert indices == (0, 1, 2, 3) and object_id == 1 and not flipped


def test_fixed_diagnostic_prompts_repeat_and_preserve_surrounding_rng():
    import random

    torch = pytest.importorskip("torch")
    from src.pilot import _loss_on_fixed_clips

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.rng = np.random.default_rng(2027)
            self.dropout = torch.nn.Dropout(0.5)

        def forward(self, batch):
            # Exercise both upstream prompt RNG sources, with no weight changes.
            return self.dropout(torch.rand(()) + float(self.rng.random()))

    class Sampler:
        config = SimpleNamespace(seed=2027)

        def batch(self, cursor, device):
            return SimpleNamespace(masks=torch.zeros(())), {}

    model = Model()
    sampler = Sampler()
    criterion = lambda output, target: {"core_loss": output}
    torch.manual_seed(17)
    random.seed(17)
    outer_torch = torch.get_rng_state().clone()
    outer_python = random.getstate()
    outer_prompt = model.rng.bit_generator.state
    before = _loss_on_fixed_clips(model, criterion, sampler, (0, 1, 2, 3), device="cpu")
    assert torch.equal(torch.get_rng_state(), outer_torch)
    assert random.getstate() == outer_python
    assert model.rng.bit_generator.state == outer_prompt
    assert model.training and model.dropout.training
    torch.rand(100)
    model.rng.random(100)
    after = _loss_on_fixed_clips(model, criterion, sampler, (0, 1, 2, 3), device="cpu")
    assert before == after


def test_overfit_selections_force_absence_and_four_distinct_videos():
    from src.pilot import _overfit_selections

    records = [SimpleNamespace(video_id=str(i)) for i in range(5)]

    def selection(index):
        return records[index], (0, 1, 2, 3), 1, False

    sampler = SimpleNamespace(records=records, selection=selection)
    occlusion = (records[2], (1, 3, 5, 7), 2, False)
    fixed = _overfit_selections(sampler, occlusion)
    assert fixed[0] == occlusion
    assert len({item[0].video_id for item in fixed.values()}) == 4
    assert list(fixed) == [0, 1, 2, 3]


def test_accumulation_is_microbatch_average():
    torch = pytest.importorskip("torch")
    from src.train import optimizer_update

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(2.0))

        def forward(self, batch):
            return self.weight * batch.flat_img_batch

    class Sampler:
        def batch(self, cursor, device):
            return SimpleNamespace(
                flat_img_batch=torch.tensor(float(cursor + 1)), masks=torch.tensor(0.0)
            ), {"cursor": cursor}

    model = Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    config = replace(TrainingConfig(), accumulation=4, clip_norm=1e9)
    result = optimizer_update(
        model,
        optimizer,
        lambda output, target: {"core_loss": (output - target).square()},
        Sampler(),
        0,
        config,
        device="cpu",
    )
    # d mean((2*x)^2)/dw = 2*2*mean([1,4,9,16]) = 30.
    assert model.weight.item() == pytest.approx(1.7)
    assert result["cursor"] == 4


def _checkpoint_fixture(tmp_path):
    torch = pytest.importorskip("torch")
    from src.io import atomic_json

    root = tmp_path / "project"
    root.mkdir()
    path = root / "runs" / "run_001"
    atomic_json(path / "data" / "partitions.json", {"identity": "synthetic-data"})
    run = SimpleNamespace(root=root, path=path, config=ProjectConfig())
    model = torch.nn.Linear(2, 1)
    model.rng = np.random.default_rng(2027)
    optimizer = torch.optim.AdamW(model.parameters())
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    return run, model, optimizer


def test_checkpoint_restores_model_optimizer_cursor_and_rng(tmp_path):
    torch = pytest.importorskip("torch")
    from src.checkpoints import (
        cpu_tree,
        restore_checkpoint,
        save_checkpoint,
        scheduler_record,
    )
    from src.pilot import _tree_close

    run, model, optimizer = _checkpoint_fixture(tmp_path)
    save_checkpoint(run, model, optimizer, update=3, cursor=12)
    expected = cpu_tree(model.state_dict())
    expected_optimizer = cpu_tree(optimizer.state_dict())
    expected_rng = (torch.rand(3), model.rng.random())
    with torch.no_grad():
        model.weight.add_(99)
    state = restore_checkpoint(run, model, optimizer)
    assert state["cursor"] == 12 and state["update"] == 3
    assert state["scheduler"] == scheduler_record(run.config.training, 3)
    assert state["scheduler"]["next_update_one_based"] == 4
    assert _tree_close(expected, model.state_dict())
    assert _tree_close(expected_optimizer, optimizer.state_dict())
    assert torch.equal(expected_rng[0], torch.rand(3))
    assert expected_rng[1] == model.rng.random()


def test_failed_checkpoint_preserves_latest(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from src.checkpoints import save_checkpoint

    run, model, optimizer = _checkpoint_fixture(tmp_path)
    save_checkpoint(run, model, optimizer, update=1, cursor=4)
    pointer = run.path / "checkpoints" / "latest.json"
    before = pointer.read_bytes()
    monkeypatch.setattr(
        torch,
        "save",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            OSError("synthetic interruption")
        ),
    )
    with pytest.raises(OSError):
        save_checkpoint(run, model, optimizer, update=2, cursor=8)
    assert pointer.read_bytes() == before
    assert not list(pointer.parent.glob(".pending-*"))


def test_resume_rejects_changed_data(tmp_path):
    from src.checkpoints import restore_checkpoint, save_checkpoint
    from src.io import atomic_json

    run, model, optimizer = _checkpoint_fixture(tmp_path)
    save_checkpoint(run, model, optimizer, update=1, cursor=4)
    atomic_json(run.path / "data" / "partitions.json", {"identity": "changed"})
    with pytest.raises(ValueError, match="identity changed"):
        restore_checkpoint(run, model, optimizer)


def test_resume_rejects_changed_scheduler_position(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from src.checkpoints import restore_checkpoint, save_checkpoint

    run, model, optimizer = _checkpoint_fixture(tmp_path)
    save_checkpoint(run, model, optimizer, update=3, cursor=12)
    load = torch.load

    def altered_load(path, *args, **kwargs):
        state = load(path, *args, **kwargs)
        if Path(path).name == "state.pt":
            state["scheduler"]["next_update_one_based"] = 999
        return state

    monkeypatch.setattr(torch, "load", altered_load)
    with pytest.raises(ValueError, match="scheduler"):
        restore_checkpoint(run, model, optimizer)


def test_forecast_uses_one_trajectory_not_entire_benchmark_case():
    from src.pilot import project_cost

    run = SimpleNamespace(config=ProjectConfig())
    video = SimpleNamespace(frames=(1, 2, 3, 4), object_ids=(1, 2))
    tracking = {
        "stage_duration_seconds": 20,
        "cases": [
            {
                "frame_count": 4,
                "object_count": 2,
                "total_seconds": 1e9,
                "session_setup_seconds": 2,
                "model_loading_seconds": 1,
                "repetitions": [
                    {
                        "model_seconds": 1,
                        "transfer_seconds": 1,
                        "output_materialization_seconds": 1,
                    }
                ],
            }
        ],
    }
    result = project_cost(
        run,
        1,
        tracking,
        {split: [video] for split in ("dev", "test", "valid")},
        measured_pilot_seconds=10,
        checkpoint_save_seconds=2,
    )
    # Initial+5 dev passes, 8 equivalent test trajectories, 2 official passes.
    assert result["evaluation_seconds"] == pytest.approx(48)
    assert result["dev_evaluation_passes"] == 6
    assert result["model_load_seconds"] == 15
    assert result["checkpoint_write_seconds"] == 30
    assert result["measured_pilot_seconds"] == 10
    assert result["demo_allowance_seconds"] == 1800
