import pytest

from src.benchmark import StagedFrames, timed_call


class Frame:
    def __init__(self, events):
        self.events = events

    def to(self, **kw):
        self.events.append("transfer")
        return self

    def float(self):
        return self


def test_input_is_staged_outside_measured_region():
    events = []
    frames = StagedFrames([Frame(events), Frame(events)])
    sync = lambda: events.append("sync")
    with pytest.raises(RuntimeError, match="Unstaged"):
        frames[0]
    frames.stage(0, "cuda", sync)
    result, seconds = timed_call(lambda: (events.append("model"), frames[0])[1], sync)
    assert result is frames[0] and seconds >= 0
    assert events == ["sync", "transfer", "sync", "sync", "model", "sync"]
    with pytest.raises(RuntimeError):
        frames[1]


def test_upstream_progress_is_disabled_and_restored_on_failure(monkeypatch):
    import sys
    from types import SimpleNamespace

    from src.benchmark import quiet_upstream_progress

    def rendering_progress(*args, **kwargs):
        raise AssertionError("progress renderer entered a timed region")

    module = SimpleNamespace(tqdm=rendering_progress)
    monkeypatch.setitem(sys.modules, "sam2.sam2_video_predictor", module)
    with pytest.raises(RuntimeError, match="synthetic"), quiet_upstream_progress():
        assert list(module.tqdm(range(3), desc="ignored")) == [0, 1, 2]
        raise RuntimeError("synthetic interrupted timing")
    assert module.tqdm is rendering_progress


def test_actual_isolation_probe_interleaves_distinct_states(tmp_path):
    import numpy as np
    from PIL import Image

    from src.benchmark import check_session_isolation
    from src.inference import Prompt

    frames = []
    for index in range(4):
        path = tmp_path / f"{index:06d}.jpg"
        Image.fromarray(np.zeros((8, 9, 3), dtype=np.uint8)).save(path)
        frames.append(path)

    class Predictor:
        def __init__(self, leak=False):
            self.shared = {} if leak else None
            self.events = []

        def init_state(self, **kwargs):
            return {
                "num_frames": 4,
                "obj_ids": [],
                "output_dict_per_obj": {} if self.shared is None else self.shared,
                "point": None,
            }

        def reset_state(self, state):
            self.events.append(("reset", id(state)))
            state["obj_ids"].clear()
            state["output_dict_per_obj"].clear()
            state["point"] = None

        def add_new_points_or_box(
            self, inference_state, frame_idx, obj_id, points, **kwargs
        ):
            inference_state["obj_ids"].append(obj_id)
            inference_state["point"] = int(points[0][0])
            return self.output(inference_state, frame_idx)

        def output(self, state, index):
            self.events.append(("frame", id(state), tuple(state["obj_ids"])))
            values = np.full((1, 1, 8, 9), -1, dtype=np.float32)
            values[:, :, :, state["point"]] = 2
            return index, state["obj_ids"], values

        def propagate_in_video(
            self, state, start_frame_idx, max_frame_num_to_track, **kwargs
        ):
            for index in range(
                start_frame_idx, start_frame_idx + max_frame_num_to_track + 1
            ):
                yield self.output(state, index)

    predictor = Predictor()
    prompt = Prompt(0, 7, points=((2, 2),), labels=(1,))
    result = check_session_isolation(
        predictor, tmp_path, tuple(frames), prompt, device="cpu"
    )
    assert (
        result["passed"] and result["first_ids"] == [7] and result["second_ids"] == [1]
    )
    assert result["before"] == result["after"]
    # Each state is reset once before its first replay, then once during cleanup.
    resets = [event for event in predictor.events if event[0] == "reset"]
    assert len(resets) == 4
    with pytest.raises(RuntimeError, match="isolation failed"):
        check_session_isolation(
            Predictor(leak=True), tmp_path, tuple(frames), prompt, device="cpu"
        )
