"""Sequential, audio-free decoding, stable SAM 2 frame views, and mask overlays."""

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

from .io import atomic_json, canonical_hash, read_json, sha256_file


def mask_palette() -> list[int]:
    # Pascal-style bit-interleaved palette, deterministic for all persistent IDs.
    palette = []
    for index in range(256):
        red = green = blue = 0
        value = index
        for shift in range(8):
            red |= (value & 1) << (7 - shift)
            green |= ((value >> 1) & 1) << (7 - shift)
            blue |= ((value >> 2) & 1) << (7 - shift)
            value >>= 3
        palette.extend((red, green, blue))
    return palette


def prepare_frame_directory(frames, destination: Path) -> Path:
    """Map manifest order onto the numeric JPEG names required by upstream SAM 2.

    No recompression for existing JPEGs. Links live in a reusable data cache;
    their identity includes source hashes so filenames alone never imply reuse.
    """
    frames = tuple(Path(p).resolve() for p in frames)
    if not frames:
        raise ValueError("Cannot prepare an empty video")
    identity = [{"path": str(p), "sha256": sha256_file(p)} for p in frames]
    fingerprint = canonical_hash(identity)
    destination = Path(destination) / fingerprint[:20]
    manifest = destination / "frames.json"
    if manifest.exists():
        saved = read_json(manifest)
        if saved["fingerprint"] != fingerprint or saved["sources"] != identity:
            raise ValueError("Frame-view cache identity mismatch")
        for i in range(len(frames)):
            if not (destination / f"{i:08d}.jpg").is_file():
                raise ValueError(
                    "Incomplete frame-view cache; remove the incomplete cache explicitly"
                )
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(destination.name + ".partial-" + uuid.uuid4().hex)
    staging.mkdir()
    try:
        size = None
        for i, path in enumerate(frames):
            with Image.open(path) as image:
                if size is None:
                    size = image.size
                if image.size != size:
                    raise ValueError("Video frames have inconsistent native dimensions")
                output = staging / f"{i:08d}.jpg"
                if image.format == "JPEG":
                    output.symlink_to(path)
                else:
                    image.convert("RGB").save(output, quality=100, subsampling=0)
        atomic_json(
            staging / "frames.json",
            {
                "fingerprint": fingerprint,
                "sources": identity,
                "frame_count": len(frames),
                "native_size": size,
                "non_jpeg_conversion": "JPEG quality=100 subsampling=0; only non-JPEG sources",
            },
        )
        try:
            staging.rename(destination)
        except FileExistsError:
            shutil.rmtree(staging)
            return prepare_frame_directory(frames, destination.parent)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return destination


def decode_video(source: Path, output_dir: Path) -> dict:
    """Decode every video frame in presentation order; audio is never read/exported."""
    import cv2

    source, output_dir = Path(source), Path(output_dir)
    if not source.is_file():
        raise FileNotFoundError(source)
    identity = sha256_file(source)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "video.json"
    if manifest_path.exists():
        saved = read_json(manifest_path)
        if saved["source_sha256"] != identity:
            raise ValueError("Output directory already belongs to a different video")
        if all(Path(p).is_file() for p in saved["frames"]):
            return saved
        raise ValueError("Decoded video is incomplete; choose a new session directory")
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    reader = cv2.VideoCapture(str(source))
    if not reader.isOpened():
        raise ValueError("OpenCV could not decode this video")
    fps = float(reader.get(cv2.CAP_PROP_FPS))
    declared_frame_count = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    if not np.isfinite(fps) or fps <= 0:
        reader.release()
        raise ValueError(
            "Video has no usable frame-rate metadata; transcode it explicitly first"
        )
    timestamps, paths = [], []
    shape = None
    try:
        index = 0
        while True:
            success, image = reader.read()
            if not success:
                break
            if shape is None:
                shape = image.shape[:2]
            if image.shape[:2] != shape:
                raise ValueError(
                    "Variable-resolution videos require explicit preprocessing"
                )
            path = frames_dir / f"{index:08d}.jpg"
            if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 100]):
                raise OSError(f"Could not write {path}")
            paths.append(str(path))
            timestamps.append(float(reader.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0)
            index += 1
    finally:
        reader.release()
    if not paths:
        raise ValueError("No video frames were decoded")
    if declared_frame_count > 0 and declared_frame_count != len(paths):
        raise ValueError(
            f"Decoded {len(paths)} frames but the source declares {declared_frame_count}; inspect/trancode the input explicitly"
        )
    result = {
        "source": str(source.resolve()),
        "source_sha256": identity,
        "frames": paths,
        "frame_count": len(paths),
        "width": shape[1],
        "height": shape[0],
        "fps": fps,
        "source_timestamps_seconds": timestamps,
        "audio": "discarded",
        "decoder": f"OpenCV {cv2.__version__}",
        "rendering": "constant source-reported FPS; tracking uses frame indices, not timestamp spacing",
    }
    atomic_json(manifest_path, result)
    return result


def overlay_mask(
    image: np.ndarray, mask: np.ndarray, opacity: float = 0.45
) -> np.ndarray:
    image, mask = np.asarray(image), np.asarray(mask)
    if image.ndim != 3 or image.shape[2] != 3 or image.shape[:2] != mask.shape:
        raise ValueError("RGB frame and indexed mask dimensions differ")
    colors = np.asarray(mask_palette(), dtype=np.uint8).reshape(256, 3)
    if not 0 <= opacity <= 1 or np.any((mask < 0) | (mask > 255)):
        raise ValueError("Invalid overlay opacity or object ID")
    result = image.astype(np.float32)
    selected = mask > 0
    result[selected] = (1 - opacity) * result[selected] + opacity * colors[
        mask[selected].astype(int)
    ]
    return np.clip(result, 0, 255).astype(np.uint8)


def render_video(
    frames, masks, output_path: Path, *, fps: float = 24.0, title: str = ""
) -> Path:
    """Browser-playable H.264 overlays with no audio and no image resizing.

    yuv420p requires even dimensions: an odd right/bottom edge is padded by one
    black pixel for the video only. Saved indexed masks always stay native size.
    RGB is piped directly to ffmpeg, avoiding OpenCV's odd-dimension cropping.
    """
    import subprocess
    import tempfile

    import cv2
    import imageio_ffmpeg

    frames, masks = tuple(frames), tuple(masks)
    if not frames or len(frames) != len(masks) or not np.isfinite(fps) or fps <= 0:
        raise ValueError("Rendering requires aligned frames/masks and positive FPS")
    with Image.open(frames[0]) as im:
        width, height = im.size
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.stem + ".partial.mp4")
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-movflags",
        "+faststart",
        str(temporary),
    ]
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errors
        )
        try:
            for frame_path, mask in zip(frames, masks):
                with Image.open(frame_path) as im:
                    rgb = np.asarray(im.convert("RGB"))
                if rgb.shape != (height, width, 3):
                    raise ValueError(
                        "Rendering encountered inconsistent frame dimensions"
                    )
                labels = (
                    np.asarray(Image.open(mask))
                    if isinstance(mask, (str, Path))
                    else np.asarray(mask)
                )
                rendered = overlay_mask(rgb, labels)
                if title:
                    cv2.putText(
                        rendered,
                        title,
                        (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                process.stdin.write(np.ascontiguousarray(rendered).tobytes())
            process.stdin.close()
            return_code = process.wait()
            if return_code:
                errors.seek(0)
                raise RuntimeError(
                    "H.264 encoding failed: "
                    + errors.read().decode("utf-8", "replace")[-4000:]
                )
        except BaseException:
            if process.poll() is None:
                process.kill()
                process.wait()
            temporary.unlink(missing_ok=True)
            raise
    if not temporary.is_file() or temporary.stat().st_size == 0:
        raise RuntimeError("Video rendering produced no output")
    reader = cv2.VideoCapture(str(temporary))
    try:
        success, decoded = reader.read()
        expected = (height + height % 2, width + width % 2)
        if (
            not success
            or decoded.shape[:2] != expected
            or int(reader.get(cv2.CAP_PROP_FRAME_COUNT)) != len(frames)
        ):
            raise RuntimeError(
                "Rendered video failed dimension/frame-count verification"
            )
    finally:
        reader.release()
    os.replace(temporary, output_path)
    return output_path
