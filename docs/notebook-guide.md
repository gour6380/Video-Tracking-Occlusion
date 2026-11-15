# Notebook guide

Open [the controller](../notebooks/video_tracking_occlusion.ipynb) from the project root or `notebooks/`. Select **Video Tracking — Python 3.13**. The notebook locates the root and imports normal modules from `src`; no project package installation is required.

Use [the RunPod setup](runpod.md) for the experiment. The local Mac environment supports development and synthetic checks; the notebook's experiment readiness check requires CUDA with BF16 support.

## Responsibilities and outputs

| Notebook responsibility | Implementation | Saved evidence |
|---|---|---|
| Define settings and create/resume a run | `src/config.py`, `src/runs.py` | Run configuration, identities and status |
| Inspect the environment | `src/environment.py` | Initial device record and later observations |
| Acquire/audit/split data | `src/data.py` | Project-local cache and run data manifests |
| Inspect the model and run the pilot | `src/sam2_adapter.py`, `src/pilot.py` | Model manifest, pilot gates and forecast |
| Fine-tune | `src/train.py`, `src/checkpoints.py` | Checkpoints, logs and TensorBoard events |
| Evaluate each protocol/checkpoint | `src/evaluate.py`, `src/metrics.py` | Full-clip predictions, prompts and scores |
| Measure performance | `src/benchmark.py` | Raw synchronized samples and memory measurements |
| Export examples and comparisons | `src/evaluate.py`, `src/reporting.py` | Example videos, figures and tables |
| Package official validation | `src/submissions.py` | Submission ZIPs, manifests and imported receipts |
| Write the report or launch the demo | `src/reporting.py`, `src/demo.py` | Technical report or isolated session exports |

Training and evaluation loops belong to `src`. Notebook cells coordinate one responsibility each; they do not duplicate those implementations.

## Creating and resuming a run

Keep `run_name = None` for a fresh, atomically numbered run. The next cell creates that run once; later cells reuse its context. To resume deliberately, set `run_name` to an existing name such as `"run_001"` before running the context cell.

Resume requires compatible source, configuration, installed environment and data identities. A second process cannot resume a run while another operation holds its write lock. The last saved complete optimizer update is the recovery boundary; subsequent unsaved work is replayed. Do not edit a run's identity records to bypass a mismatch.

The run selector and receipt-input cells are explicitly operational inputs. Other notebook code and experimental settings are part of source identity. Failed pilot gates must be inspected before main training; a failed stage and its diagnostics remain recorded.

## Where files go

Directories that contain generated outputs appear as their stages execute:

```text
runs/run_XXX/
├── config.json, identity.json, status.json
├── provenance/            Source snapshot, dependencies, model and device identities
├── data/                  Partitions, duplicate audit and data references
├── checkpoints/           Latest/selected pointers and resumable update bundles
├── pilot/                 Correctness gates, recovery diagnostic and cost forecast
├── logs/                  Plain logs and JSONL training records
├── tensorboard/           Training events
├── evaluation/            Predictions, prompt histories and per-object/frame scores
├── benchmarks/            Timing cases, raw samples and independent-state checks
├── metrics/               CSV tables and paired checkpoint differences
├── figures/               Learning curves and comparison plots
├── examples/              Illustrative success/failure/correction videos
├── submissions/           Validated packages and actual imported server receipts
├── demo/                  Independent uploaded-video sessions and exports
└── reports/               Generated technical report and evidence manifest
```

Large reusable downloads stay in project-local `data/`. Pinned framework/evaluator source is restored into `vendor/`. Small source identity metadata stays in `artifacts/`. These roles are distinct from a run's scientific results.

The top-level `reports/` is for a later reviewed, compact evidence bundle. It does not replace the full local run. No real results are copied there during implementation preparation.

View recorded training events with `.venv/bin/tensorboard --logdir runs`. Submit official ZIPs manually, then fill the dedicated receipt inputs and regenerate the report. Launching the demo is an explicit final cell; see the RunPod guide for access and shutdown.
