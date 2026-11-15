# RunPod setup and owner execution

The implementation has not rented a pod, provisioned storage, downloaded the dataset/model, run a GPU experiment or submitted predictions.

## Prepare the machine

Choose one A40 48 GB and approximately 300 GB persistent storage. Confirm the actual hourly/storage price in your account. The default forecast uses $0.49/GPU-hour and $0.07/GB-month; edit those advisory inputs before starting a run when the offer differs. Around $30 is a planning target, not a runtime guarantee. Storage can remain billable while the GPU is stopped.

Use a current Linux CUDA-capable image and NVIDIA driver compatible with the runtime selected in `environments/runpod/requirements.txt`. Inspect `nvidia-smi` before installation. The generated lock currently selects PyTorch with CUDA 13 runtime dependencies; an older-driver image must be updated or its compatibility resolved explicitly, not bypassed with CPU fallback. The environment check and CUDA/BF16 requirement reject unsupported accelerator execution.

Place the source of [Video-Tracking-Occlusion](https://github.com/gour6380/Video-Tracking-Occlusion) at `/workspace/Video-Tracking-Occlusion` on the persistent volume (or substitute your actual extracted directory below). Keep `artifacts/upstream.json` and both dependency locks. Setup restores the vendored source; Mac environments, caches, data, and runs do not need to be transferred. Start with `uv` installed and run:

```sh
cd /workspace/Video-Tracking-Occlusion
sh scripts/setup_runpod.sh
.venv/bin/python -m pytest -q
.venv/bin/jupyter lab --ip=0.0.0.0 --port=8888 --ServerApp.port_retries=0 --no-browser --allow-root
```

The source retrieval script checks exact pinned revisions/hashes and never invokes Git. If restoring into a fresh machine, the recorded upstream manifest is used to verify freshly retrieved source files. Keep the Jupyter token enabled; use the pod's authenticated access or an SSH tunnel.

`--allow-root` supports RunPod containers that run as root; it does not disable authentication. The explicit port must match your pod's proxy or SSH tunnel. Disabling port retries prevents Jupyter from silently switching to 8889 while your browser still opens an older server on 8888.

If your image already provides a running Jupyter server, use it instead of starting a second server on the same port. Register this project's kernel for that server's user, refresh its kernel list, and select **Video Tracking — Python 3.13**:

```sh
.venv/bin/python -m ipykernel install --user --name video-tracking-occlusion --display-name 'Video Tracking — Python 3.13'
```

Run that registration as the same user who owns the existing Jupyter server. Alternatively, launch the project server on a different explicitly exposed port and open that matching endpoint.

## Execute the notebook

Open `notebooks/video_tracking_occlusion.ipynb` with the registered kernel. Review immutable configuration first. Leave `run_name=None` for a new run or provide an explicit name to resume.

The CUDA readiness cell runs before data acquisition. Acquisition downloads roughly 85 GB of archives to the project's persistent cache, validates creator checksums and extracts safely. Duplicate auditing and partition freezing precede any fit. The downloaded initial checkpoint is also explicitly recorded.

Run the pilot and read its forecast. It tests gradient flow, frozen weights, a visible-to-absent case, checkpoint recovery, learning on four clips, and full-video tracking. Main fine-tuning requires a passed pilot. There is no silent workload shrinkage if memory allocation fails. The forecast includes measured throughput, planned development/test/official evaluation, measured pilot cost and an explicit demo allowance; actual download/setup/idle/server times remain separate.

Execute the visible baseline/adapted evaluations, benchmarks, examples and official ZIP exports. Those exports are not submissions. Use your own Codabench account, review the current phase, submit each final package once, and save its actual receipt. Enter receipt paths/IDs/metrics in the notebook's operational input cells and regenerate the report; do not use server scores for further model selection.

## Demo and shutdown

Default demo binding is `127.0.0.1:7860`. Tunnel that port from your Mac to the pod over SSH. If you deliberately bind externally, provide `TRACKING_DEMO_USERNAME` and `TRACKING_DEMO_PASSWORD` in the pod environment; never save credentials in the notebook. Public exposure is optional, and Gradio sharing is disabled by default.

The interface accepts videos and exports silent H.264 overlays plus native indexed masks and prompt history. Model computations remain on CUDA. This implementation does not claim local MPS SAM2 performance.

Save/download the run's compact report, metrics, manifests, prompts and selected checkpoint before terminating compute. Keep or remove persistent storage deliberately through your RunPod account after verifying the retained artifacts. This project does not silently stop/delete external resources.

## Local Mac development

`sh scripts/setup_local.sh` creates a separate persistent Python 3.13 `.venv` from the Mac lock. Use it for synthetic tests, notebook editing and report review. The experiment notebook itself requires CUDA before downloads/training. Do not copy a Mac-created experimental run to Linux and silently resume under a different environment; start the run on the intended GPU environment.
