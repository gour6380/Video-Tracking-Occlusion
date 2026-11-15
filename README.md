# Video Tracking Through Occlusion

[Repository](https://github.com/gour6380/Video-Tracking-Occlusion) · [Controller notebook](notebooks/video_tracking_occlusion.ipynb) · [Experiment design](docs/design.md) · [RunPod setup](docs/runpod.md) · [Verification](verification/local-checks.json)

Can four-frame fine-tuning help SAM 2.1 Tiny track an object after it disappears and returns? This PyTorch study uses MOSEv2 to compare the pretrained checkpoint with a partially fine-tuned checkpoint. The primary comparison starts each object with the same positive click and follows it through the entire held-out clip without further guidance.

**Status: implementation and local verification.** On 15 November 2026, **102 tests passed** in a fresh Python 3.13.11 environment installed from the hashed Mac lock. A separate [CPU smoke check](verification/pre-upload-native-smoke.json) exercised the actual model with random weights: it tracked two objects across four synthetic frames, replayed a correction, and took one optimizer step while preserving frozen weights. No CUDA pilot, MOSEv2 training, measured tracking results, or official submission is included. This evidence establishes that those local code paths ran; it does not establish a quality gain from fine-tuning.

## What the experiment measures

| Comparison | How to interpret it |
|---|---|
| Pretrained vs. adapted, one click per object | Primary test: region overlap (J) and adaptive boundary accuracy (Ḟ), including disappearance and reappearance segments. |
| One click vs. one additional correction opportunity | A simulated annotator clicks the largest error region at the midpoint if needed. The paired scores exclude that frame from both conditions. |
| First-frame masks, no corrections | A separate protocol for local evaluation and the two planned official validation packages. Local scores and server scores are reported separately. |
| One, two, or four objects on fixed development clips | CUDA-synchronized propagation throughput, correction latency, and peak allocated GPU memory; loading and transfers are measured separately. |

The target split is approximately **300 training, 40 development, and 60 local test clips** from the official training pool. The pipeline groups duplicate and visually overlapping clips before partitioning. Each training sample follows one object through four chronological frames at native 1024 preprocessing. The image and prompt encoders stay frozen; the mask decoder, memory modules, and object-pointer projections train for 1,000 optimizer updates. The highest development click score selects the checkpoint. An annotated disappearance can also mean the object left the frame; it does not establish physical occlusion.

## Run it

From the repository root, with `uv` installed, choose the setup for your machine:

| Machine | Setup | Intended use |
|---|---|---|
| Apple Silicon Mac | `sh scripts/setup_local.sh` | Local tests, notebook editing, and report inspection. |
| Linux x86-64 RunPod | `sh scripts/setup_runpod.sh` | CUDA/BF16 pilot and the full experiment. The target is one A40 with 48 GB VRAM and about 300 GB persistent storage. |

Both setup scripts create a Python 3.13 environment, install the platform lock, restore upstream **source code** against the [checksum manifest](artifacts/upstream.json), and register the **Video Tracking — Python 3.13** kernel. Dataset and model downloads happen later in explicit notebook cells.

```sh
.venv/bin/python -m pytest -q
.venv/bin/jupyter lab notebooks/video_tracking_occlusion.ipynb
```

On RunPod, use the [server and kernel instructions](docs/runpod.md). Run the notebook's environment check before the bulk data download, and review the pilot's diagnostics and cost forecast before starting the main fit. Leave `run_name = None` for a new `runs/run_XXX/` folder; use a specific name to resume from a saved optimizer update. The [notebook guide](docs/notebook-guide.md) maps each stage to its outputs.

## Repository map

| Path | Contents |
|---|---|
| [`notebooks/`](notebooks/video_tracking_occlusion.ipynb) | Controller for preparation, pilot, fine-tuning, evaluation, reporting, and the demo. |
| [`src/`](src/) | Configuration, model adapter, training loop, metrics, inference, submission export, and multi-object Gradio interface. |
| [`tests/`](tests/) | Data, model, recovery, measurement, and interactive tracking checks. |
| [`scripts/`](scripts/) and [`environments/runpod/`](environments/runpod/) | Environment setup and Linux dependency lock. The Mac lock is at the root. |
| [`docs/`](docs/) | Experimental protocol, notebook and RunPod instructions, and third-party provenance. |
| [`verification/`](verification/local-checks.json) and [`reports/`](reports/README.md) | Portable local test evidence and the location for future reviewed results. |

The demo supports multiple object IDs, points, boxes, and corrections to earlier frames. It replays prompts chronologically and can export native masks and silent overlay videos. Dataset files, model weights, uploaded videos, raw runs, and submission ZIPs are generated locally and excluded from the public source. After a real run, the notebook can export a mask-initialized package for each checkpoint covering all **433 official validation clips**. Only a returned [Codabench](https://www.codabench.org/competitions/10062/) receipt supports an official score.

## Attribution

The [MIT license](LICENSE) covers this project's original code. [SAM 2](https://github.com/facebookresearch/sam2) and the [MOSEv2 dataset](https://huggingface.co/datasets/FudanCVL/MOSEv2) retain their own terms; the dataset is for noncommercial research. See [third-party records](docs/third-party.md) for the pinned revisions, licenses, and papers.
