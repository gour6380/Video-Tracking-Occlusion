"""Session-isolated, forward-only multiple-object Gradio tracking demo."""

from __future__ import annotations

import threading
import uuid
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .inference import Prompt, TrackingSession, save_indexed_mask
from .io import atomic_json, canonical_hash, sha256_file
from .video import decode_video, mask_palette, render_video


@dataclass(frozen=True)
class DemoSnapshot:
    session_id: str
    directory: Path
    frames: tuple[Path, ...]
    width: int
    height: int
    fps: float
    source_sha256: str
    prompts: tuple[Prompt, ...]
    revision: int

    @property
    def object_ids(self) -> tuple[int, ...]:
        return tuple(sorted({p.object_id for p in self.prompts}))

    @property
    def identity(self) -> str:
        return canonical_hash(
            {
                "source_sha256": self.source_sha256,
                "frames": [str(path) for path in self.frames],
                "width": self.width,
                "height": self.height,
                "fps": self.fps,
                "prompts": [asdict(p) for p in self.prompts],
            }
        )


@dataclass
class DemoSession:
    session_id: str
    directory: Path
    frames: tuple[Path, ...]
    width: int
    height: int
    fps: float
    source_sha256: str
    prompts: list[Prompt] = field(default_factory=list)
    results: dict[str, dict] = field(default_factory=dict)
    revision: int = 0
    _lock: object = field(default_factory=threading.RLock, repr=False, compare=False)

    def __getstate__(self):
        # Gradio can deepcopy state; each copied session needs its own lock.
        with self._lock:
            return {
                key: value for key, value in self.__dict__.items() if key != "_lock"
            }

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.RLock()

    def snapshot(self) -> DemoSnapshot:
        with self._lock:
            return DemoSnapshot(
                self.session_id,
                self.directory,
                tuple(self.frames),
                self.width,
                self.height,
                self.fps,
                self.source_sha256,
                tuple(self.prompts),
                self.revision,
            )

    def _invalidate(self) -> None:
        self.revision += 1
        self.results.clear()

    def add(self, prompt: Prompt) -> None:
        with self._lock:
            prompt.validate(len(self.frames), self.width, self.height)
            self.prompts.append(prompt)
            self._invalidate()

    def undo(self) -> None:
        with self._lock:
            if self.prompts:
                self.prompts.pop()
            self._invalidate()

    def remove_object(self, object_id: int) -> None:
        with self._lock:
            self.prompts = [p for p in self.prompts if p.object_id != object_id]
            self._invalidate()

    def reset(self) -> None:
        with self._lock:
            self.prompts.clear()
            self._invalidate()

    @property
    def object_ids(self) -> tuple[int, ...]:
        return self.snapshot().object_ids

    def ledger(self) -> list[dict]:
        return [asdict(p) for p in self.snapshot().prompts]

    def publish(self, snapshot: DemoSnapshot, results: dict) -> None:
        with self._lock:
            if self.snapshot() != snapshot:
                raise RuntimeError(
                    "Video or prompt history changed during tracking; stale results were not published. Track the current history again."
                )
            self.results = results


def new_demo_session(run, video_path: Path) -> DemoSession:
    session_id = uuid.uuid4().hex
    directory = run.path / "demo" / session_id
    video = decode_video(video_path, directory / "input")
    return DemoSession(
        session_id,
        directory,
        tuple(Path(p) for p in video["frames"]),
        video["width"],
        video["height"],
        video["fps"],
        video["source_sha256"],
    )


def preview_frame(session: DemoSession, frame_index: int) -> np.ndarray:
    index = int(frame_index)
    if not 0 <= index < len(session.frames):
        raise ValueError("Frame is outside this uploaded video")
    with Image.open(session.frames[index]) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    colors = np.asarray(mask_palette()).reshape(256, 3)
    for prompt in session.prompts:
        if prompt.frame_index != index:
            continue
        color = tuple(map(int, colors[prompt.object_id]))
        if prompt.kind == "point":
            for (x, y), label in zip(prompt.points, prompt.labels):
                draw.ellipse(
                    (x - 5, y - 5, x + 5, y + 5), fill=color, outline="white", width=2
                )
                draw.text(
                    (x + 7, y - 7),
                    f"{prompt.object_id}{'+' if label else '-'}",
                    fill="white",
                )
        elif prompt.kind == "box":
            draw.rectangle(prompt.box, outline=color, width=3)
            draw.text(prompt.box[:2], str(prompt.object_id), fill="white")
    return np.asarray(image)


def run_demo_tracking(
    run,
    session: DemoSession,
    selection: str = "both",
    *,
    predictor_factory=None,
    device: str = "cuda",
) -> dict:
    """Rebuild independent states for both checkpoints from the same prompt ledger."""
    if selection not in {"initial", "best", "both"}:
        raise ValueError("Choose initial, best, or both")
    snapshot = session.snapshot()
    if not snapshot.prompts:
        raise ValueError("Select an object with a positive point or box first")
    # Require a positive initialization before a negative-only correction for each ID.
    for object_id in snapshot.object_ids:
        first = min(
            (p for p in snapshot.prompts if p.object_id == object_id),
            key=lambda p: p.frame_index,
        )
        if first.kind == "point" and not any(first.labels):
            raise ValueError(f"Object {object_id} needs a positive initialization")
    factory_supplied = predictor_factory is not None
    if predictor_factory is None:
        from .sam2_adapter import build_predictor

        predictor_factory = lambda checkpoint: build_predictor(
            run, checkpoint=checkpoint, device=device
        )
    ledger_identity = snapshot.identity
    checkpoints = ("initial", "best") if selection == "both" else (selection,)
    # Returning a new mapping prevents the unselected output from an older ledger
    # or checkpoint run from appearing as part of this comparison.
    results = {}
    for checkpoint in checkpoints:
        if factory_supplied:
            checkpoint_identity = "injected_predictor_for_testing"
        else:
            from .sam2_adapter import checkpoint_path

            checkpoint_identity = sha256_file(checkpoint_path(run, checkpoint))
        predictor = predictor_factory(checkpoint)
        tracking = None
        try:
            tracking = TrackingSession(
                predictor,
                snapshot.frames[0].parent,
                snapshot.frames,
                offload_video_to_cpu=True,
                device=device,
            )
            for prompt in snapshot.prompts:
                tracking.add_prompt(prompt)
            masks = tracking.replay()
            output = (
                snapshot.directory
                / "results"
                / ledger_identity[:20]
                / f"{checkpoint}-{checkpoint_identity[:12]}"
            )
            labels_dir = output / "masks"
            for index, labels in masks.items():
                save_indexed_mask(
                    labels_dir / (snapshot.frames[index].stem + ".png"), labels
                )
            ledger_path = tracking.save_ledger(output / "prompts.json")
            video = render_video(
                snapshot.frames,
                [masks[i] for i in range(len(snapshot.frames))],
                output / "tracking.mp4",
                fps=snapshot.fps,
                title=checkpoint,
            )
            metadata = {
                "checkpoint": checkpoint,
                "checkpoint_sha256": checkpoint_identity,
                "session_id": snapshot.session_id,
                "source_sha256": snapshot.source_sha256,
                "ledger_identity": ledger_identity,
                "object_ids": list(snapshot.object_ids),
                "width": snapshot.width,
                "height": snapshot.height,
                "ledger_revision": snapshot.revision,
                "frame_count": len(snapshot.frames),
                "audio": "not processed or exported",
                "before_first_prompt": "background; no backward propagation",
                "render_fps": snapshot.fps,
                "native_prompt_coordinates": True,
            }
            atomic_json(output / "metadata.json", metadata)
            archive_path = output / "masks-and-prompts.zip"
            with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
                for path in sorted(labels_dir.glob("*.png")):
                    archive.write(path, "masks/" + path.name)
                archive.write(ledger_path, "prompts.json")
                archive.write(output / "metadata.json", "metadata.json")
            results[checkpoint] = {
                "video": str(video),
                "masks_zip": str(archive_path),
                "prompts": str(ledger_path),
                "metadata": str(output / "metadata.json"),
            }
        finally:
            if tracking is not None:
                tracking.reset()
            del tracking, predictor
    session.publish(snapshot, results)
    return results


def create_demo(run):
    """Construct UI without launching, loading checkpoints, or sharing a public URL."""
    import gradio as gr

    def require_session(value):
        if value is None:
            raise gr.Error("Upload a video first")
        return value

    def load_upload(path):
        if path is None:
            return (
                None,
                gr.update(minimum=0, maximum=1, value=0),
                None,
                [],
                "Upload a video",
                None,
                None,
                [],
            )
        value = new_demo_session(run, Path(path))
        return (
            value,
            gr.update(minimum=0, maximum=max(1, len(value.frames) - 1), value=0),
            preview_frame(value, 0),
            [],
            f"{len(value.frames)} frames; {value.width}×{value.height}. Audio excluded.",
            None,
            None,
            [],
        )

    def change_frame(value, index):
        value = require_session(value)
        return preview_frame(value, min(int(index), len(value.frames) - 1))

    def click(value, index, object_id, sign, event: gr.SelectData):
        value = require_session(value)
        x, y = event.index
        value.add(
            Prompt(
                int(index),
                int(object_id),
                points=((float(x), float(y)),),
                labels=(1 if sign == "Positive" else 0,),
            )
        )
        return (
            value,
            preview_frame(value, int(index)),
            value.ledger(),
            "Prompt saved; previous results invalidated.",
            None,
            None,
            [],
        )

    def add_box(value, index, object_id, x0, y0, x1, y1):
        value = require_session(value)
        value.add(
            Prompt(
                int(index),
                int(object_id),
                kind="box",
                box=(float(x0), float(y0), float(x1), float(y1)),
            )
        )
        return (
            value,
            preview_frame(value, int(index)),
            value.ledger(),
            "Box saved; previous results invalidated.",
            None,
            None,
            [],
        )

    def edit(value, index, object_id, action):
        value = require_session(value)
        if action == "undo":
            value.undo()
        elif action == "remove":
            value.remove_object(int(object_id))
        else:
            value.reset()
        return (
            value,
            preview_frame(value, int(index)),
            value.ledger(),
            "Ledger updated; previous results invalidated.",
            None,
            None,
            [],
        )

    def run_tracking(value, selection):
        value = require_session(value)
        result = run_demo_tracking(run, value, selection)
        files = [
            item[k]
            for item in result.values()
            for k in ("masks_zip", "prompts", "video")
        ]
        return (
            value,
            result.get("initial", {}).get("video"),
            result.get("best", {}).get("video"),
            files,
            f"Tracked {len(value.frames)} frames for object IDs {value.object_ids}; independent states, identical prompt ledger.",
        )

    # Resolve the locally imported event type for Gradio's callback inspection.
    click.__annotations__["event"] = gr.SelectData

    queue_options = {"concurrency_limit": 1, "concurrency_id": "tracking_gpu"}

    with gr.Blocks(title="Tracking Through Occlusion", analytics_enabled=False) as app:
        gr.Markdown(
            "# Tracking Through Occlusion\nSelect objects with clicks or boxes. Move to a later frame to correct a track. "
            "Editing an earlier frame replays the prompt history chronologically. Audio is excluded."
        )
        session = gr.State(None)
        upload = gr.Video(label="Upload video", sources=["upload"], format="mp4")
        frame_index = gr.Slider(
            minimum=0, maximum=1, step=1, value=0, label="Frame index"
        )
        image = gr.Image(
            label="Click on this native-resolution frame",
            type="numpy",
            interactive=False,
        )
        with gr.Row():
            object_id = gr.Number(
                value=1,
                minimum=1,
                maximum=255,
                precision=0,
                label="Persistent object ID",
            )
            sign = gr.Radio(
                ["Positive", "Negative"], value="Positive", label="Click type"
            )
        gr.Markdown(
            "Use a different object ID to add another target. Coordinates below are native image pixels."
        )
        with gr.Row():
            x0 = gr.Number(value=0, label="Box left")
            y0 = gr.Number(value=0, label="Box top")
            x1 = gr.Number(value=100, label="Box right")
            y1 = gr.Number(value=100, label="Box bottom")
            box_button = gr.Button("Add box")
        with gr.Row():
            undo = gr.Button("Undo last prompt")
            remove = gr.Button("Remove selected object")
            reset = gr.Button("Clear all prompts")
        ledger = gr.JSON(label="Prompt history")
        selection = gr.Radio(
            ["initial", "best", "both"], value="both", label="Checkpoint comparison"
        )
        run_button = gr.Button("Track from prompt history", variant="primary")
        status = gr.Markdown("Upload a video to begin.")
        with gr.Row():
            initial_video = gr.Video(label="Pretrained SAM 2.1 Tiny")
            best_video = gr.Video(label="Selected adapted checkpoint")
        downloads = gr.Files(label="Masks, videos, and prompt histories")
        edits = [session, image, ledger, status, initial_video, best_video, downloads]
        upload.change(
            load_upload,
            [upload],
            [
                session,
                frame_index,
                image,
                ledger,
                status,
                initial_video,
                best_video,
                downloads,
            ],
            **queue_options,
        )
        frame_index.change(
            change_frame, [session, frame_index], [image], **queue_options
        )
        image.select(
            click, [session, frame_index, object_id, sign], edits, **queue_options
        )
        box_button.click(
            add_box,
            [session, frame_index, object_id, x0, y0, x1, y1],
            edits,
            **queue_options,
        )
        undo.click(
            lambda s, i, o: edit(s, i, o, "undo"),
            [session, frame_index, object_id],
            edits,
            **queue_options,
        )
        remove.click(
            lambda s, i, o: edit(s, i, o, "remove"),
            [session, frame_index, object_id],
            edits,
            **queue_options,
        )
        reset.click(
            lambda s, i, o: edit(s, i, o, "reset"),
            [session, frame_index, object_id],
            edits,
            **queue_options,
        )
        run_button.click(
            run_tracking,
            [session, selection],
            [session, initial_video, best_video, downloads, status],
            **queue_options,
        )
    return app


def launch_demo(run):
    """Launch private by default; RunPod access uses an explicit port/tunnel."""
    import os

    username = os.environ.get("TRACKING_DEMO_USERNAME")
    password = os.environ.get("TRACKING_DEMO_PASSWORD")
    public = (
        run.config.demo.server_name not in {"127.0.0.1", "localhost", "::1"}
        or run.config.demo.share
    )
    if public and (not username or not password):
        raise ValueError(
            "Public binding requires TRACKING_DEMO_USERNAME and TRACKING_DEMO_PASSWORD; otherwise use the default private SSH tunnel"
        )
    auth = (username, password) if username and password else None
    app = create_demo(run)
    app.queue()
    app.launch(
        server_name=run.config.demo.server_name,
        server_port=run.config.demo.server_port,
        share=run.config.demo.share,
        auth=auth,
        allowed_paths=[str(run.path / "demo")],
    )
    return app
