"""Evidence-derived figures and report; absent results remain explicitly pending."""

from __future__ import annotations

import csv
import json
import statistics

from .io import atomic_json, atomic_write, read_json, sha256_file


def training_rows(run):
    path = run.path / "logs" / "training.jsonl"
    if not path.exists():
        return []
    # Raw replay attempts stay in JSONL. Curves show the latest record per update.
    rows = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["update"]] = row
    return [rows[key] for key in sorted(rows)]


def plot_learning_curves(run) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = training_rows(run)
    if not rows:
        raise ValueError("No training measurements exist yet")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(
        [r["update"] for r in rows],
        [r["metrics"]["core_loss"] for r in rows],
        color="#2563eb",
        linewidth=1,
    )
    axes[0].set(
        xlabel="Optimizer update",
        ylabel="Accumulation-mean upstream loss",
        title="Training objective",
    )
    axes[1].plot(
        [r["update"] for r in rows], [r["learning_rate"] for r in rows], color="#0f766e"
    )
    axes[1].set(
        xlabel="Optimizer update", ylabel="Learning rate", title="Executed schedule"
    )
    fig.tight_layout()
    path = run.path / "figures" / "learning-curves.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    table = run.path / "metrics" / "training.csv"
    table.parent.mkdir(parents=True, exist_ok=True)
    with table.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "update",
                "learning_rate",
                "seconds",
                "core_loss",
                "gradient_norm",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {key: row[key] for key in ("update", "learning_rate", "seconds")}
                | {key: row["metrics"][key] for key in ("core_loss", "gradient_norm")}
            )
    return {"figure": str(path), "csv": str(table), "updates": len(rows)}


def _evaluations(run):
    values = {}
    for checkpoint in ("initial", "best"):
        for protocol in ("point", "point_correction", "mask"):
            path = (
                run.path / "evaluation" / "test" / checkpoint / protocol / "scores.json"
            )
            if path.exists():
                values[(checkpoint, protocol)] = (path, read_json(path))
    return values


def _plot_score(value):
    return float("nan") if value is None else value


def _format_score(value):
    return "N/A (no eligible observations)" if value is None else f"{value:.6f}"


def make_comparison_figures(run) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = _evaluations(run)
    required = {
        (c, p)
        for c in ("initial", "best")
        for p in ("point", "point_correction", "mask")
    }
    if set(values) != required:
        raise ValueError(
            f"Complete both checkpoints and all local protocols first: {sorted(required - set(values))}"
        )
    rows = []
    for (checkpoint, protocol), (_, data) in values.items():
        rows.append(
            {
                "checkpoint": checkpoint,
                "protocol": protocol,
                **data["aggregate"],
                "paired_JF_dot": data["paired_aggregate"]["J&F_dot"],
                "corrective_clicks": data["corrective_clicks"],
            }
        )
    keys = sorted(set().union(*(row.keys() for row in rows)))
    csv_path = run.path / "metrics" / "comparison.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for index, checkpoint in enumerate(("initial", "best")):
        primary = values[(checkpoint, "point")][1]
        correction = values[(checkpoint, "point_correction")][1]
        mask = values[(checkpoint, "mask")][1]
        heights = [
            _plot_score(primary["aggregate"]["J&F_dot"]),
            _plot_score(mask["aggregate"]["J&F_dot"]),
        ]
        axes[0].bar(
            [x + index * 0.35 for x in (0, 1)], heights, width=0.35, label=checkpoint
        )
        axes[1].plot(
            [0, 1],
            [
                _plot_score(primary["paired_aggregate"]["J&F_dot"]),
                _plot_score(correction["paired_aggregate"]["J&F_dot"]),
            ],
            marker="o",
            label=checkpoint,
        )
    axes[0].set(
        xticks=[0.175, 1.175],
        xticklabels=["One point", "Full initial mask"],
        ylabel="J & adaptive F (0–1)",
        title="Held-out custom split",
    )
    axes[1].set(
        xticks=[0, 1],
        xticklabels=["No correction", "One opportunity"],
        ylabel="Paired J & adaptive F (0–1)",
        title="Midpoint excluded from both conditions",
    )
    for ax in axes:
        ax.legend()
        ax.set_ylim(0, 1)
    fig.tight_layout()
    figure = run.path / "figures" / "comparison.png"
    fig.savefig(figure, dpi=160)
    plt.close(fig)
    baseline = values[("initial", "point")][1]
    adapted = values[("best", "point")][1]
    base = {v["video_id"]: v["score"]["summary"]["J&F_dot"] for v in baseline["videos"]}
    tuned = {v["video_id"]: v["score"]["summary"]["J&F_dot"] for v in adapted["videos"]}
    if base.keys() != tuned.keys():
        raise ValueError("Different evaluation video identities")
    deltas = [
        {
            "video_id": key,
            "baseline": base[key],
            "adapted": tuned[key],
            "delta": None
            if base[key] is None or tuned[key] is None
            else tuned[key] - base[key],
            "eligible": base[key] is not None and tuned[key] is not None,
        }
        for key in sorted(base)
    ]
    delta_path = run.path / "metrics" / "paired-video-deltas.json"
    atomic_json(delta_path, deltas)
    return {
        "figure": str(figure),
        "csv": str(csv_path),
        "paired_deltas": str(delta_path),
    }


def write_report(run) -> dict:
    values = _evaluations(run)
    lines = [
        "# Tracking Through Occlusion — technical report",
        "",
        f"Run: `{run.name}`. Values below are read from saved artifacts; no inference is performed by report generation.",
        "",
        "## Question and protocol",
        "",
        (
            "Does partial temporal fine-tuning improve click-initialized tracking in crowded videos? "
            "Control: the official SAM 2.1 Tiny checkpoint. Treatment: the development-selected adapted checkpoint. "
            "The fixed experiment uses four-frame clips and one tracked object per training sample; multi-object inference is per-object tracking."
        ),
        "",
        "## Recorded local results",
        "",
        "| Checkpoint | Prompt protocol | J | Adaptive F | J & adaptive F | Corrective clicks |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for (checkpoint, protocol), (_, data) in values.items():
        lines.append(
            f"| {checkpoint} | {protocol} | {_format_score(data['aggregate'].get('J'))} | "
            f"{_format_score(data['aggregate'].get('F_dot'))} | "
            f"{_format_score(data['aggregate']['J&F_dot'])} | {data['corrective_clicks']} |"
        )
    if not values:
        lines.extend(["", "Pending: no local quality evaluations exist."])
    else:
        lines.extend(
            [
                "",
                "### Disappearance and recovery",
                "",
                "| Checkpoint | Protocol | Disappearance J & adaptive F | Reappearance J & adaptive F | Absent frames with prediction / eligible absent frames |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for (checkpoint, protocol), (_, data) in values.items():
            aggregate = data["aggregate"]
            diagnostics = aggregate.get("local_diagnostics", {})
            counts = (
                f"{diagnostics['absent_frames_with_prediction']} / {diagnostics['absent_scored_frames']}"
                if diagnostics
                else "N/A"
            )
            lines.append(
                f"| {checkpoint} | {protocol} | {_format_score(aggregate.get('disappear_J&F_dot'))} | "
                f"{_format_score(aggregate.get('reappear_J&F_dot'))} | {counts} |"
            )
        lines.extend(
            [
                "",
                (
                    "Recovery is described by the public scorer’s reappearance-segment quality; no physical-occlusion or fixed recovery-time claim is implied. "
                    "Absent-frame prediction counts are separate local diagnostics. Complete per-object, per-frame and event records are in `evaluation/test/`."
                ),
            ]
        )
    if ("initial", "point") in values and ("best", "point") in values:
        before = values[("initial", "point")][1]["aggregate"]["J&F_dot"]
        after = values[("best", "point")][1]["aggregate"]["J&F_dot"]
        if before is not None and after is not None:
            delta = after - before
            lines.extend(
                [
                    "",
                    f"**Observation:** adapted minus pretrained point-prompt score is {delta:+.6f} on this held-out subset.",
                ]
            )
    lines.extend(
        [
            "",
            (
                "Correction benefits must use `paired_aggregate` for both point and point_correction, "
                "which excludes the annotated midpoint. Primary point and mask scores exclude first/last frames. "
                "Metrics follow the pinned public scorer, including add-one-smoothed J. N/A means no eligible observations, never a zero score."
            ),
            "",
            "## Official validation",
            "",
        ]
    )
    receipts = sorted((run.path / "submissions").glob("*/receipts/*/receipt.json"))
    if not receipts:
        lines.append(
            "Pending: no official server receipts have been imported. Local scores are not official validation scores."
        )
    for receipt_path in receipts:
        receipt = read_json(receipt_path)
        lines.append(
            f"- {receipt['checkpoint']}, submission {receipt['submission_id']}: "
            f"`{json.dumps(receipt['reported_metrics'], sort_keys=True)}`. User-transcribed from attached receipt; source: `{receipt_path.relative_to(run.path)}`."
        )
    lines.extend(["", "## Compute and operational measurements", ""])
    pilot = run.path / "pilot" / "pilot.json"
    if pilot.exists():
        result = read_json(pilot)
        lines.append(
            f"Pilot passed: **{result.get('passed', False)}**. Forecast is an estimate, not billing: `{json.dumps(result.get('forecast', {}), sort_keys=True)}`."
        )
    else:
        lines.append(
            "GPU feasibility and runtime projection remain pending the RunPod pilot."
        )
    benchmark = run.path / "benchmarks" / "tracking" / "summary.json"
    if benchmark.exists():
        result = read_json(benchmark)
        lines.append(
            f"Completed {len(result['cases'])} timing cases. Raw per-frame samples: `benchmarks/tracking/`. {result['boundary']}."
        )
        lines.append(
            "Allocator memory is separate from process/system memory; no cross-framework or universal speed ranking is made."
        )
        lines.extend(
            [
                "",
                "| Checkpoint | Development clip | Objects | Load (s) | Propagation FPS, median | Correction (s), median | Peak allocated GiB |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for case in result["cases"]:
            repetitions = case["repetitions"]
            corrections = [
                row["correction"]["seconds"]
                for row in repetitions
                if row["correction"]["seconds"] is not None
            ]
            correction = (
                f"{statistics.median(corrections):.6f}"
                if corrections
                else "N/A (no click needed)"
            )
            lines.append(
                f"| {case['checkpoint']} | {case['video_id']} | {case['object_count']} | "
                f"{case['model_loading_seconds']:.3f} | {statistics.median(row['propagation_fps'] for row in repetitions):.3f} | "
                f"{correction} | {max(row['peak_allocated_bytes'] for row in repetitions) / 2**30:.3f} |"
            )
        lines.extend(
            [
                "",
                (
                    "First-trajectory, per-frame p95, input/output transfer, session initialization and actual corrective-click counts remain separate in the raw case records. "
                    "Missing object-count conditions are listed as unsupported rather than replaced with smaller workloads."
                ),
            ]
        )
    else:
        lines.append("Tracking performance measurements remain pending.")
    lines.extend(
        [
            "",
            "## Interpretation and limitations",
            "",
            (
                "A positive local delta supports this specific bounded adaptation, not general tracking superiority. "
                "A negative or neutral result is retained. No automated parameter search or extra training follows the result."
            ),
            "",
            "- One seed and a small development/test subset limit generalization claims.",
            "- Exact/near-duplicate groups reduce overlap; original-source independence and foundation-model pretraining exclusion are not proven.",
            "- Four-frame training is shorter than the complete videos used for evaluation.",
            "- Corrective points use ground truth to simulate annotation assistance, not deployable failure detection.",
            "- An empty annotated target is not proof of physical occlusion; it may be out of view.",
            "- Public local evaluator and private official server equivalence is unverified.",
            "- MOSEv2 is noncommercial research data. Do not redistribute its raw video collection in the project repository.",
            "",
            "## Evidence and reproduction",
            "",
            (
                "Open the controller notebook with the recorded environment, inspect configuration and the pilot forecast, "
                "then execute the visible stages. Configuration, source, environment, data identities, prompts, and checkpoint hashes "
                "are saved under this run. Official submissions remain a manual owner action."
            ),
            "",
        ]
    )
    artifact = run.path / "reports" / "technical-report.md"
    atomic_write(artifact, ("\n".join(lines)).encode())
    evidence = {
        str(path.relative_to(run.path)): sha256_file(path)
        for path, _ in values.values()
    }
    evidence.update({str(p.relative_to(run.path)): sha256_file(p) for p in receipts})
    for supporting in (
        pilot,
        benchmark,
        run.path / "metrics" / "paired-video-deltas.json",
        run.path / "provenance" / "model.json",
        run.path / "checkpoints" / "training_summary.json",
    ):
        if supporting.exists():
            evidence[str(supporting.relative_to(run.path))] = sha256_file(supporting)
    atomic_json(
        run.path / "reports" / "evidence.json",
        {
            "files": evidence,
            "quality_complete": len(values) == 6,
            "official_receipts": len(receipts),
        },
    )
    return {
        "report": str(artifact),
        "local_protocols_completed": len(values),
        "official_receipts": len(receipts),
    }
