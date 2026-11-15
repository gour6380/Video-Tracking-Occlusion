import numpy as np
import pytest
from PIL import Image

from src.demo import DemoSession, preview_frame
from src.inference import Prompt


def make_session(tmp_path, name):
    directory = tmp_path / name
    directory.mkdir()
    frames = []
    for index in range(4):
        path = directory / f"{index:08d}.jpg"
        Image.fromarray(np.zeros((40, 50, 3), np.uint8)).save(path)
        frames.append(path)
    return DemoSession(name, directory, tuple(frames), 50, 40, 24.0, "fixture-sha")


def test_sessions_and_persistent_object_ids_are_isolated(tmp_path):
    one = make_session(tmp_path, "one")
    two = make_session(tmp_path, "two")
    one.add(Prompt(0, 4, points=((5, 7),), labels=(1,)))
    one.add(Prompt(0, 8, kind="box", box=(10, 10, 20, 25)))
    assert one.object_ids == (4, 8)
    assert two.object_ids == ()
    one.remove_object(4)
    assert one.object_ids == (8,)
    assert not two.prompts


def test_earlier_edits_invalidate_future_outputs_and_preserve_ledger(tmp_path):
    session = make_session(tmp_path, "edit")
    session.add(Prompt(2, 3, points=((9, 9),), labels=(1,)))
    session.results = {
        "initial": {"video": "stale.mp4"},
        "best": {"video": "stale2.mp4"},
    }
    session.add(Prompt(0, 3, points=((7, 7),), labels=(1,)))
    assert session.results == {}
    assert [p.frame_index for p in session.prompts] == [2, 0]
    session.undo()
    assert [p.frame_index for p in session.prompts] == [2]
    session.reset()
    assert not session.prompts


def test_native_coordinate_validation_and_prompt_preview(tmp_path):
    session = make_session(tmp_path, "view")
    session.add(Prompt(0, 1, points=((25, 20),), labels=(1,)))
    image = preview_frame(session, 0)
    assert image.shape == (40, 50, 3) and image.any()
    with pytest.raises(ValueError, match="outside"):
        session.add(Prompt(0, 2, points=((50, 20),), labels=(1,)))
    with pytest.raises(ValueError, match="empty"):
        session.add(Prompt(0, 2, kind="box", box=(10, 10, 9, 20)))


def test_demo_build_resolves_select_event_without_missing_argument_warning():
    import warnings
    from types import SimpleNamespace

    from src.demo import create_demo

    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        app = create_demo(SimpleNamespace())
    assert app is not None
    assert not any("Expected 5 arguments" in str(item.message) for item in seen)


def test_h264_render_preserves_all_frames_and_pads_only_odd_edges(tmp_path):
    import cv2

    from src.video import decode_video, render_video

    frames = []
    for index in range(3):
        path = tmp_path / f"image-{index}.png"
        image = np.zeros((41, 51, 3), np.uint8)
        image[:, :, index] = 180
        Image.fromarray(image).save(path)
        frames.append(path)
    masks = [np.zeros((41, 51), np.uint8) for _ in frames]
    video = render_video(frames, masks, tmp_path / "preview.mp4", fps=12)
    decoded = decode_video(video, tmp_path / "decoded")
    assert decoded["frame_count"] == 3
    assert (decoded["height"], decoded["width"]) == (42, 52)
    assert decoded["audio"] == "discarded"
    reader = cv2.VideoCapture(str(video))
    codec = int(reader.get(cv2.CAP_PROP_FOURCC))
    reader.release()
    assert "".join(chr((codec >> (8 * i)) & 255) for i in range(4)) in {
        "avc1",
        "h264",
        "H264",
    }


def test_demo_tracking_uses_independent_checkpoint_states(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from test_inference import FakePredictor

    from src import demo
    from src.demo import run_demo_tracking

    session = make_session(tmp_path, "tracked")
    session.add(Prompt(0, 3, points=((5, 7),), labels=(1,)))
    session.add(Prompt(2, 3, points=((9, 7),), labels=(1,)))
    predictors = []

    def factory(_):
        predictor = FakePredictor()
        predictors.append(predictor)
        return predictor

    # Encoder correctness has its own test. Keep this test focused on model state.
    def rendered(frames, masks, output_path, **kwargs):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"rendered-fixture")
        return output_path

    monkeypatch.setattr(demo, "render_video", rendered)
    result = run_demo_tracking(
        SimpleNamespace(), session, predictor_factory=factory, device="cpu"
    )
    assert set(result) == {"initial", "best"}
    assert len(predictors) == 2 and predictors[0] is not predictors[1]
    assert result["initial"]["prompts"] != result["best"]["prompts"]
    for predictor in predictors:
        for call in predictor.calls:
            if call[0] == "frame":
                assert all(prompt_frame <= call[1] for prompt_frame in call[2])


def test_concurrent_edit_uses_one_snapshot_and_rejects_stale_publication(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from test_inference import FakePredictor

    from src import demo
    from src.io import read_json

    session = make_session(tmp_path, "race")
    session.add(Prompt(0, 3, points=((5, 7),), labels=(1,)))
    original = session.snapshot()

    def factory(checkpoint):
        if checkpoint == "best":
            session.add(Prompt(2, 9, points=((12, 7),), labels=(1,)))
        return FakePredictor()

    def rendered(frames, masks, output_path, **kwargs):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"rendered-fixture")
        return output_path

    monkeypatch.setattr(demo, "render_video", rendered)
    with pytest.raises(RuntimeError, match="stale results were not published"):
        demo.run_demo_tracking(
            SimpleNamespace(), session, predictor_factory=factory, device="cpu"
        )
    assert session.results == {}
    assert session.object_ids == (3, 9)
    metadata = sorted(session.directory.glob("results/*/*/metadata.json"))
    assert len(metadata) == 2
    for path in metadata:
        saved = read_json(path)
        assert saved["object_ids"] == [3]
        assert saved["ledger_identity"] == original.identity
        assert saved["ledger_revision"] == original.revision
        assert len(read_json(path.parent / "prompts.json")["prompts"]) == 1


def test_session_deepcopy_has_independent_lock_and_ledger(tmp_path):
    import copy

    original = make_session(tmp_path, "copy")
    original.add(Prompt(0, 3, points=((5, 7),), labels=(1,)))
    duplicate = copy.deepcopy(original)
    duplicate.undo()
    assert len(original.prompts) == 1 and not duplicate.prompts
    assert original._lock is not duplicate._lock


def test_all_ui_state_changes_share_tracking_serial_queue():
    from types import SimpleNamespace

    from src.demo import create_demo

    app = create_demo(SimpleNamespace())
    assert len(app.fns) >= 8
    assert all(
        fn.concurrency_id == "tracking_gpu" and fn.concurrency_limit == 1
        for fn in app.fns.values()
    )
