from types import SimpleNamespace

import pytest

from src.sam2_adapter import disable_zero_correction_loop, freeze_for_training


def test_zero_corrections_skip_upstream_unbound_loop():
    class Model:
        num_correction_pt_per_frame = 0

        def prepare_prompt_inputs(self, value):
            return {
                "init_cond_frames": [0],
                "frames_to_add_correction_pt": [0],
                "value": value,
            }

    model = disable_zero_correction_loop(Model())
    prepared = model.prepare_prompt_inputs("synthetic")
    assert prepared["init_cond_frames"] == [0]
    assert prepared["frames_to_add_correction_pt"] == []
    assert prepared["value"] == "synthetic"


def test_freeze_allowlist_keeps_decoder_projection_gradients():
    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.image_encoder = torch.nn.Sequential(
                torch.nn.Linear(2, 2), torch.nn.Dropout(0.5)
            )
            self.sam_prompt_encoder = torch.nn.Linear(2, 2)
            self.sam_mask_decoder = torch.nn.Linear(2, 1)

        def forward(self, value):
            return self.sam_mask_decoder(self.image_encoder(value))

    model = Model()
    frozen = model.image_encoder[0].weight.detach().clone()
    names = freeze_for_training(model, ("sam_mask_decoder.",))
    assert set(names) == {"sam_mask_decoder.weight", "sam_mask_decoder.bias"}
    assert not model.image_encoder.training and not model.sam_prompt_encoder.training
    model(torch.ones(1, 2)).sum().backward()
    assert model.sam_mask_decoder.weight.grad is not None
    assert model.image_encoder[0].weight.grad is None
    assert torch.equal(frozen, model.image_encoder[0].weight)


def test_empty_trainable_prefix_rejected():
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="matched no parameters"):
        freeze_for_training(torch.nn.Linear(2, 2), ("does_not_exist.",))


def test_upstream_temporal_driver_preserves_earlier_frame_gradient():
    """Exercise the pinned chronological driver with scalar synthetic memory only."""
    torch = pytest.importorskip("torch")
    from pathlib import Path

    from src.config import ProjectConfig
    from src.sam2_adapter import activate_source

    root = Path(__file__).resolve().parents[1]
    activate_source(SimpleNamespace(root=root, config=ProjectConfig()))
    from training.model.sam2 import SAM2Train

    features = torch.ones(1, 4, 2, requires_grad=True)
    called = []

    class SyntheticDriver:
        add_all_frames_to_correct_as_cond = False

        def _prepare_backbone_features(self, backbone):
            return None, [features], [torch.zeros_like(features)], [(1, 1)]

        def track_step(self, **kwargs):
            index = kwargs["frame_idx"]
            called.append(index)
            value = kwargs["current_vision_feats"][0].sum()
            if index:
                bank = (
                    kwargs["output_dict"]["cond_frame_outputs"]
                    if index == 1
                    else kwargs["output_dict"]["non_cond_frame_outputs"]
                )
                value = value + bank[index - 1]["value"]
            return {"value": value}

    backbone = {
        "backbone_fpn": True,
        "num_frames": 4,
        "init_cond_frames": [0],
        "frames_not_in_init_cond": [1, 2, 3],
        "frames_to_add_correction_pt": [],
        "point_inputs_per_frame": {},
        "mask_inputs_per_frame": {},
        "gt_masks_per_frame": {},
    }
    batch = SimpleNamespace(flat_obj_to_img_idx=torch.arange(4)[:, None])
    outputs = SAM2Train.forward_tracking(SyntheticDriver(), backbone, batch)
    outputs[-1]["value"].square().backward()
    assert called == [0, 1, 2, 3]
    assert features.grad[:, 0].abs().sum() > 0
    assert torch.equal(features.grad[:, 0], features.grad[:, 3])


def test_predictor_preserves_logits_for_sorted_id_tie_resolution(monkeypatch, tmp_path):
    import sys

    import numpy as np

    from src import sam2_adapter
    from src.config import ProjectConfig
    from src.inference import labels_from_logits

    captured = {}

    def factory(config, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(sam2_adapter, "activate_source", lambda run: None)
    monkeypatch.setattr(
        sam2_adapter, "checkpoint_path", lambda run, checkpoint: tmp_path / "unused.pt"
    )
    monkeypatch.setitem(
        sys.modules,
        "sam2.build_sam",
        SimpleNamespace(build_sam2_video_predictor=factory),
    )
    sam2_adapter.build_predictor(SimpleNamespace(config=ProjectConfig()), device="cpu")
    assert "++model.non_overlap_masks=false" in captured["hydra_overrides_extra"]
    assert "++model.non_overlap_masks=true" not in captured["hydra_overrides_extra"]
    assert labels_from_logits([8, 2], np.ones((2, 1, 1, 1), np.float32)).item() == 2


def _native_memory_fixture():
    """Actual pinned driver methods, synthetic 2x2 tensors, no neural model."""
    torch = pytest.importorskip("torch")
    from pathlib import Path

    from src.config import ProjectConfig
    from src.sam2_adapter import activate_source, patch_predictor_memory_clearing

    activate_source(
        SimpleNamespace(
            root=Path(__file__).resolve().parents[1], config=ProjectConfig()
        )
    )
    from sam2.sam2_video_predictor import SAM2VideoPredictor

    class MinimalPredictor(SAM2VideoPredictor):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.memory_temporal_stride_for_eval = 2
            self.num_maskmem = 1
            self.clear_non_cond_mem_around_input = True
            self.non_overlap_masks = False

    def output(value=1):
        return {
            "maskmem_features": torch.tensor([value]),
            "pred_masks": torch.full((1, 1, 2, 2), float(value)),
        }

    state = {
        "obj_ids": [10, 20],
        "obj_idx_to_id": {0: 10, 1: 20},
        "num_frames": 12,
        "device": torch.device("cpu"),
        "video_height": 2,
        "video_width": 2,
        "frames_tracked_per_obj": {0: {}, 1: {}},
        "output_dict_per_obj": {
            index: {
                "cond_frame_outputs": {0: output(index + 1)},
                "non_cond_frame_outputs": {
                    frame: output(index + 1) for frame in range(1, 12)
                },
            }
            for index in (0, 1)
        },
        "temp_output_dict_per_obj": {
            index: {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
            for index in (0, 1)
        },
    }
    return patch_predictor_memory_clearing(MinimalPredictor()), state, output


def test_native_preflight_clears_only_corrected_object_memory():
    from src.sam2_adapter import patch_predictor_memory_clearing

    predictor, state, output = _native_memory_fixture()
    corrected = output(9)
    state["temp_output_dict_per_obj"][0]["cond_frame_outputs"][5] = corrected
    other_before = dict(state["output_dict_per_obj"][1]["non_cond_frame_outputs"])
    helper = predictor._clear_obj_non_cond_mem_around_input
    assert (
        patch_predictor_memory_clearing(predictor)._clear_obj_non_cond_mem_around_input
        is helper
    )
    predictor.propagate_in_video_preflight(state)
    assert set(state["output_dict_per_obj"][0]["non_cond_frame_outputs"]) == {
        1,
        2,
        8,
        9,
        10,
        11,
    }
    assert state["output_dict_per_obj"][0]["cond_frame_outputs"][5] is corrected
    assert set(state["output_dict_per_obj"][0]["cond_frame_outputs"]) == {0, 5}
    assert state["output_dict_per_obj"][1]["non_cond_frame_outputs"] == other_before
    assert not state["temp_output_dict_per_obj"][0]["cond_frame_outputs"]


def test_native_propagation_also_uses_missing_helper_compatibility():
    from src.benchmark import quiet_upstream_progress

    predictor, state, output = _native_memory_fixture()
    for index in (0, 1):
        state["output_dict_per_obj"][index]["cond_frame_outputs"][5] = output(index + 1)
    with quiet_upstream_progress():
        frames = list(
            predictor.propagate_in_video(
                state, start_frame_idx=5, max_frame_num_to_track=0, reverse=False
            )
        )
    assert len(frames) == 1 and frames[0][0] == 5 and frames[0][1] == [10, 20]
    assert frames[0][2].shape == (2, 1, 2, 2)
    for index in (0, 1):
        assert set(state["output_dict_per_obj"][index]["non_cond_frame_outputs"]) == {
            1,
            2,
            8,
            9,
            10,
            11,
        }
        assert state["frames_tracked_per_obj"][index][5] == {"reverse": False}
