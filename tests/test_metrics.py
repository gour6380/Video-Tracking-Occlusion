"""Hand-checked fixtures plus direct comparison with the pinned public reducer."""

import ast
from pathlib import Path

import numpy as np
import pytest

from src.metrics import _event_segments, aggregate_scores, score_video


def _mask(present=True):
    # Tiny 32x32 images trigger the public scorer's fractional-radius disk edge
    # case; use >=128 resolution so this golden perfect-mask fixture is valid.
    result = np.zeros((128, 128), dtype=np.uint8)
    if present:
        result[10:14, 10:14] = 1
    return result


def test_perfect_predictions_and_endpoint_exclusion():
    masks = [_mask(), _mask(), _mask(False), _mask(), _mask()]
    predictions = [x.copy() for x in masks]
    predictions[0][:] = 0
    predictions[-1][:] = 0
    result = score_video(masks, predictions, [1])
    assert result["scored_frames"] == [1, 2, 3]
    for key in ("J", "F", "F_dot", "J&F_dot", "disappear_J&F_dot", "reappear_J&F_dot"):
        assert result["summary"][key] == 1.0


def test_add_one_iou_and_false_positive_absence_penalty():
    truth = [_mask(), _mask(False), _mask()]
    prediction = [x.copy() for x in truth]
    prediction[1][3, 3] = 1
    result = score_video(truth, prediction, [1])["summary"]
    assert result["J"] == 0.5  # (intersection + 1)/(union + 1), not zero.
    assert result["F_dot"] == 0
    assert result["J&F_dot"] == 0.25
    assert result["disappear_J&F_dot"] is None  # Public scorer omits initial absence.


def test_complete_false_negative_uses_object_area():
    truth = [_mask(), _mask(), _mask()]
    result = score_video(truth, [_mask(False)] * 3, [1])["summary"]
    assert result["J"] == pytest.approx(1 / 17)
    assert result["F_dot"] == 0


def test_midpoint_exclusion_does_not_bridge_temporal_events():
    truth = [_mask(), _mask(), _mask(False), _mask(), _mask(False), _mask(), _mask()]
    result = score_video(truth, truth, [1], exclude_frames=[3])
    segments = result["objects"]["1"]["segments"]
    assert [
        (x["start_frame"], x["end_frame_exclusive"]) for x in segments["disappear"]
    ] == [(2, 3), (4, 5)]
    assert segments["reappear"][0]["scored_frames"] == []
    assert segments["reappear"][0]["J&F_dot"] is None
    assert result["summary"]["reappear_J&F_dot"] == 1
    assert result["scored_frames"] == [1, 2, 4, 5]


def test_all_interior_frames_excluded_are_na_not_zero():
    masks = [_mask()] * 3
    result = score_video(masks, masks, [1], exclude_frames=[1])
    assert result["summary"]["J&F_dot"] is None
    assert result["summary"]["eligible_objects"]["J&F_dot"] == 0


def test_object_macro_average_not_frame_weighting():
    long = [_mask()] * 9
    short = [_mask()] * 3
    good = score_video(long, long, [1])
    bad = score_video(short, [_mask(False)] * 3, [1])
    aggregate = aggregate_scores([good, bad])
    assert aggregate["J"] == pytest.approx((1 + 1 / 17) / 2)
    assert aggregate["video_count"] == 2 and aggregate["object_count"] == 2


@pytest.mark.parametrize(
    "issue", ["missing", "wrong_shape", "rgb", "unknown_id", "float"]
)
def test_invalid_prediction_is_rejected_even_at_excluded_endpoint(issue):
    truth = [_mask()] * 3
    prediction = [x.copy() for x in truth]
    prediction[0] = {
        "missing": None,
        "wrong_shape": np.zeros((2, 2), dtype=np.uint8),
        "rgb": np.zeros((32, 32, 3), dtype=np.uint8),
        "unknown_id": np.full((32, 32), 9, dtype=np.uint8),
        "float": np.zeros((32, 32), dtype=np.float32),
    }[issue]
    with pytest.raises(ValueError):
        score_video(truth, prediction, [1])


def test_missing_ground_truth_does_not_become_absence():
    with pytest.raises(ValueError, match="missing is not absence"):
        score_video([_mask(), None, _mask()], [_mask()] * 3, [1])


def test_custom_aggregation_agrees_with_pinned_reducer_without_exclusions():
    # Load only the pure official reduction function, avoiding its CLI imports.
    path = Path(__file__).resolve().parents[1] / "vendor/mosev2/score.py"
    tree = ast.parse(path.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "merge_chunk_results"
    )
    namespace = {"np": np}
    exec(  # noqa: S102 — pinned trusted source fixture only
        compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    merge = namespace["merge_chunk_results"]
    for presence in (
        [True, False, False, True, False, True],
        [False, True, False, True],
        [True] * 4,
    ):
        truth = [_mask()] + [_mask(p) for p in presence] + [_mask()]
        prediction = [_mask()] * len(truth)
        scored = score_video(truth, prediction, [1])
        rows = scored["per_frame"]["1"]
        official = merge(
            [
                {
                    "vid_name": "fixture",
                    "anno_id": 1,
                    "chunk_start": 1,
                    "error": None,
                    "j_list": [r["J"] for r in rows],
                    "f_list": [r["F_dot"] for r in rows],
                    "f_list_": [r["F"] for r in rows],
                    "empty_gt": [r["absent"] for r in rows],
                }
            ],
            "fixture",
            1,
        )
        actual = scored["objects"]["1"]
        assert actual["J"] == pytest.approx(official[0])
        assert actual["F_dot"] == pytest.approx(official[1])
        assert actual["F"] == pytest.approx(official[2])
        for metric, left, right in (
            ("disappear_J&F_dot", official[3], official[4]),
            ("reappear_J&F_dot", official[5], official[6]),
        ):
            if left is None:
                assert actual[metric] is None
            else:
                assert actual[metric] == pytest.approx((left + right) / 2)


def test_absence_segments_at_beginning_follow_public_convention():
    events = _event_segments([True, True, False, True, False])
    assert events == {"disappear": [(3, 4)], "reappear": [(2, 3), (4, 5)]}


def test_absent_false_positive_diagnostics_respect_pair_exclusions():
    truth = [
        _mask(),
        _mask(),
        _mask(False),
        _mask(False),
        _mask(False),
        _mask(),
        _mask(),
    ]
    prediction = [x.copy() for x in truth]
    prediction[2][2:4, 2:4] = 1  # Four false-positive target pixels.
    prediction[3][3, 3] = 1  # This annotated correction frame is excluded.
    result = score_video(truth, prediction, [1], exclude_frames=(3,))
    expected = {
        "absent_scored_frames": 2,
        "absent_frames_with_prediction": 1,
        "absent_prediction_pixels": 4,
        "absent_total_pixels": 2 * 128 * 128,
        "absent_frame_fp_rate": 0.5,
        "absent_pixel_fp_rate": 4 / (2 * 128 * 128),
    }
    assert result["objects"]["1"]["local_diagnostics"] == expected
    assert result["summary"]["local_diagnostics"] == expected
    # Recovery keeps the existing segment metric, without a new success threshold.
    assert result["objects"]["1"]["segments"]["reappear"][0]["scored_frames"] == [5]
    assert result["summary"]["reappear_J&F_dot"] == 1


def test_absence_diagnostic_aggregation_pools_counts_not_per_object_rates():
    first_truth = [_mask(), _mask(False), _mask()]
    first_pred = [_mask(), _mask(), _mask()]
    second_truth = [_mask()] + [_mask(False)] * 3 + [_mask()]
    first = score_video(first_truth, first_pred, [1])
    second = score_video(second_truth, second_truth, [1])
    diagnostic = aggregate_scores([first, second])["local_diagnostics"]
    assert diagnostic["absent_scored_frames"] == 4
    assert diagnostic["absent_frames_with_prediction"] == 1
    assert diagnostic["absent_frame_fp_rate"] == 0.25
    assert diagnostic["absent_prediction_pixels"] == 16
    assert diagnostic["absent_total_pixels"] == 4 * 128 * 128


def test_no_absence_diagnostic_denominator_yields_na_rates():
    result = score_video([_mask()] * 3, [_mask()] * 3, [1])
    diagnostic = result["summary"]["local_diagnostics"]
    assert diagnostic["absent_scored_frames"] == 0
    assert diagnostic["absent_frame_fp_rate"] is None
    assert diagnostic["absent_pixel_fp_rate"] is None
