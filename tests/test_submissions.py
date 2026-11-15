import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from src.inference import save_indexed_mask
from src.submissions import package_predictions, validate_predictions


def fixture_predictions(tmp_path):
    records = []
    root = tmp_path / "predictions"
    for name in ("clip_a", "clip_b"):
        directory = root / name
        directory.mkdir(parents=True)
        frames = tuple(Path(f"{i:05d}.jpg") for i in range(3))
        records.append(
            SimpleNamespace(
                video_id=name, frames=frames, object_ids=(3, 9), width=7, height=6
            )
        )
        for frame in frames:
            labels = np.zeros((6, 7), np.uint8)
            labels[1:3, 1:3] = 3
            labels[4:, 5:] = 9
            save_indexed_mask(directory / (frame.stem + ".png"), labels)
    return records, root


def test_complete_submission_zip_layout_checksums_and_no_overwrite(tmp_path):
    records, root = fixture_predictions(tmp_path)
    result = package_predictions(
        records, root, tmp_path / "submission.zip", expected_count=2
    )
    assert result["frame_count"] == 6 and result["video_count"] == 2
    with zipfile.ZipFile(result["zip_path"]) as archive:
        assert archive.testzip() is None
        assert "clip_a/00000.png" in archive.namelist()
        assert len(archive.namelist()) == 6
    with pytest.raises(FileExistsError):
        package_predictions(
            records, root, tmp_path / "submission.zip", expected_count=2
        )


def test_missing_clip_or_frame_is_rejected(tmp_path):
    records, root = fixture_predictions(tmp_path)
    with pytest.raises(ValueError, match="433"):
        validate_predictions(records, root)
    (root / "clip_a" / "00001.png").unlink()
    with pytest.raises(ValueError, match="completeness"):
        validate_predictions(records, root, expected_count=2)


def test_wrong_ids_and_native_resolution_fail(tmp_path):
    records, root = fixture_predictions(tmp_path)
    target = root / "clip_b" / "00002.png"
    save_indexed_mask(target, np.full((6, 7), 2, np.uint8))
    with pytest.raises(ValueError, match="Unknown object"):
        validate_predictions(records, root, expected_count=2)
    save_indexed_mask(target, np.zeros((6, 8), np.uint8))
    with pytest.raises(ValueError, match="Non-native"):
        validate_predictions(records, root, expected_count=2)


def test_extra_png_and_rgb_mask_are_rejected(tmp_path):
    records, root = fixture_predictions(tmp_path)
    target = root / "clip_a" / "00001.png"
    Image.fromarray(np.zeros((6, 7, 3), np.uint8)).save(target)
    with pytest.raises(ValueError, match="indexed"):
        validate_predictions(records, root, expected_count=2)
