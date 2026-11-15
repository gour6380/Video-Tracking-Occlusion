from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from src.inference import (
    Prompt,
    TrackingSession,
    correction_prompt,
    initial_prompts,
    interior_point,
    labels_from_logits,
)


class FakePredictor:
    """State-local fake that intentionally exposes conditioning by future prompts."""

    def __init__(self):
        self.calls = []

    def init_state(self, video_path, **kwargs):
        frames = sorted(Path(video_path).glob("*.jpg"))
        with Image.open(frames[0]) as im:
            width, height = im.size
        return {
            "num_frames": len(frames),
            "width": width,
            "height": height,
            "prompts": [],
            "ids": [],
        }

    def reset_state(self, state):
        state["prompts"] = []
        state["ids"] = []
        self.calls.append(("reset",))

    def output(self, state, frame):
        ids = state["ids"]
        logits = np.full(
            (len(ids), 1, state["height"], state["width"]), -2.0, np.float32
        )
        for i, obj in enumerate(ids):
            events = [event for event in state["prompts"] if event[1] == obj]
            # A future prompt incorrectly inserted early contaminates column zero.
            if any(event[0] > frame for event in events):
                logits[i, 0, :, 0] = 10
            x = int(events[-1][2])
            logits[i, 0, :, x] = 2 + i
        return frame, list(ids), logits

    def add_new_points_or_box(
        self,
        inference_state,
        frame_idx,
        obj_id,
        points=None,
        labels=None,
        box=None,
        **kwargs,
    ):
        state = inference_state
        if obj_id not in state["ids"]:
            state["ids"].append(obj_id)
        x = box[0] if box is not None else points[-1][0]
        state["prompts"].append((frame_idx, obj_id, x))
        self.calls.append(("prompt", frame_idx, obj_id))
        return self.output(state, frame_idx)

    def add_new_mask(self, inference_state, frame_idx, obj_id, mask):
        y, x = np.nonzero(mask)
        return self.add_new_points_or_box(
            inference_state,
            frame_idx,
            obj_id,
            points=np.array([[x[0], y[0]]]),
            labels=np.array([1]),
        )

    def propagate_in_video(
        self, state, start_frame_idx=None, max_frame_num_to_track=None, reverse=False
    ):
        end = min(start_frame_idx + max_frame_num_to_track, state["num_frames"] - 1)
        for frame in range(start_frame_idx, end + 1):
            self.calls.append(
                ("frame", frame, tuple(event[0] for event in state["prompts"]))
            )
            yield self.output(state, frame)


@pytest.fixture
def frames(tmp_path):
    result = []
    for i in range(5):
        path = tmp_path / f"{i:08d}.jpg"
        Image.fromarray(np.zeros((8, 9, 3), np.uint8)).save(path)
        result.append(path)
    return tuple(result)


def test_interior_point_and_largest_error_component():
    target = np.zeros((9, 10), np.uint8)
    target[2:7, 2:7] = 3
    assert interior_point(target == 3) == (4.0, 4.0)
    assert correction_prompt(target, target, 3, 4) is None
    predicted = np.zeros_like(target)
    prompt = correction_prompt(target, predicted, 3, 4)
    assert prompt.labels == (1,) and prompt.points == ((4.0, 4.0),)
    assert correction_prompt(predicted, target, 3, 4).labels == (0,)


def test_overlap_max_logit_preserves_ids_and_background():
    logits = np.array([[[[3, -1, 3]]], [[[4, -2, 3]]]], dtype=np.float32)
    labels = labels_from_logits([8, 2], logits)
    assert labels.tolist() == [[2, 0, 2]]
    with pytest.raises(ValueError, match="nonfinite"):
        labels_from_logits([1], np.array([[[np.nan]]]))


def test_future_prompt_never_conditions_earlier_frames(frames):
    predictor = FakePredictor()
    session = TrackingSession(predictor, frames[0].parent, frames, device="cpu")
    session.add_prompt(Prompt(0, 5, points=((2, 2),), labels=(1,)))
    session.add_prompt(Prompt(3, 5, points=((4, 2),), labels=(1,)))
    outputs = session.replay()
    assert len(outputs) == len(frames)
    assert not outputs[1][:, 0].any()
    assert (outputs[1][:, 2] == 5).all()
    assert (outputs[4][:, 4] == 5).all()
    for call in predictor.calls:
        if call[0] == "frame":
            assert all(prompt_frame <= call[1] for prompt_frame in call[2])


def test_earlier_edit_invalidates_and_replays_without_state_leak(frames):
    predictor = FakePredictor()
    one = TrackingSession(predictor, frames[0].parent, frames, device="cpu")
    two = TrackingSession(predictor, frames[0].parent, frames, device="cpu")
    one.add_prompt(Prompt(0, 7, points=((2, 2),), labels=(1,)))
    two.add_prompt(Prompt(0, 8, points=((3, 2),), labels=(1,)))
    one.replay()
    assert set(np.unique(two.replay()[2])) == {0, 8}
    one.add_prompt(Prompt(1, 7, points=((5, 2),), labels=(1,)))
    assert set(one.outputs) == {0}
    assert (one.replay()[2][:, 5] == 7).all()
    assert two.state["ids"] == [8]


def test_stop_frame_ignores_future_prompts(frames):
    predictor = FakePredictor()
    session = TrackingSession(predictor, frames[0].parent, frames, device="cpu")
    session.add_prompt(Prompt(0, 1, points=((2, 2),), labels=(1,)))
    session.add_prompt(Prompt(4, 1, points=((4, 2),), labels=(1,)))
    result = session.replay(stop_frame=2)
    assert set(result) == {0, 1, 2}
    assert all(event[0] <= 2 for event in session.state["prompts"])


def test_frozen_initial_prompts_are_reused_and_identity_checked(frames, tmp_path):
    mask = np.zeros((8, 9), np.uint8)
    mask[2:7, 1:6] = 3
    annotation = tmp_path / "annotation.png"
    Image.fromarray(mask).save(annotation)
    record = SimpleNamespace(
        video_id="video",
        object_ids=(3,),
        annotations=(annotation,),
        frames=frames,
        height=8,
        width=9,
    )
    run = SimpleNamespace(path=tmp_path / "run")
    assert initial_prompts(run, record) == initial_prompts(run, record)
    mask[0, 0] = 3
    Image.fromarray(mask).save(annotation)
    with pytest.raises(ValueError, match="identity changed"):
        initial_prompts(run, record)


def test_missing_initial_annotation_is_not_absence(frames, tmp_path):
    record = SimpleNamespace(
        video_id="missing",
        object_ids=(1,),
        annotations=(None,),
        frames=frames,
        height=8,
        width=9,
    )
    with pytest.raises(ValueError, match="Missing initialization"):
        initial_prompts(SimpleNamespace(path=tmp_path), record)


def test_full_clip_evaluation_paired_exclusions_and_cache_identity(
    frames, tmp_path, monkeypatch
):
    from contextlib import nullcontext

    from src import data, runs, sam2_adapter
    from src.config import ProjectConfig
    from src.evaluate import evaluate_local
    from src.io import atomic_json, read_json

    annotations = []
    for index in range(len(frames)):
        path = tmp_path / f"annotation-{index}.png"
        labels = np.zeros((8, 9), np.uint8)
        labels[:, 2] = 5
        Image.fromarray(labels).save(path)
        annotations.append(path)
    record = SimpleNamespace(
        video_id="fixture",
        object_ids=(5,),
        annotations=tuple(annotations),
        frames=frames,
        height=8,
        width=9,
    )
    run = SimpleNamespace(root=tmp_path, path=tmp_path / "run", config=ProjectConfig())
    atomic_json(run.path / "data" / "partitions.json", {"identity": "test-fixture"})
    checkpoint = tmp_path / "fixture-model.pt"
    checkpoint.write_bytes(b"fixture weight identity; no model is loaded")
    monkeypatch.setattr(data, "load_partition", lambda *_: [record])
    monkeypatch.setattr(runs, "stage", lambda *_: nullcontext())
    monkeypatch.setattr(sam2_adapter, "checkpoint_path", lambda *_: checkpoint)
    result = evaluate_local(run, "initial", predictor=FakePredictor(), device="cpu")
    saved = read_json(result["score_artifact"])
    assert saved["videos"][0]["score"]["scored_frames"] == [1, 2, 3]
    assert saved["videos"][0]["paired_score"]["scored_frames"] == [1, 3]
    corrected = evaluate_local(
        run,
        "initial",
        protocol="point_correction",
        predictor=FakePredictor(),
        device="cpu",
    )
    paired = read_json(corrected["score_artifact"])
    assert paired["videos"][0]["score"]["scored_frames"] == [1, 3]
    assert paired["corrective_clicks"] == 0
    assert evaluate_local(run, "initial", predictor=FakePredictor(), device="cpu")[
        "reused"
    ]
    path = Path(saved["prediction_manifest"][0]["path"])
    path.write_bytes(b"corrupted cached evidence")
    with pytest.raises(ValueError, match="incomplete"):
        evaluate_local(run, "initial", predictor=FakePredictor(), device="cpu")
