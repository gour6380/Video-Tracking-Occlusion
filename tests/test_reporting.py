from src.reporting import write_report
from src.runs import start_run


def test_report_never_fills_missing_results(tmp_path):
    run = start_run(root=tmp_path)
    result = write_report(run)
    text = (run.path / "reports" / "technical-report.md").read_text()
    assert "no local quality evaluations" in text
    assert "no official server receipts" in text
    assert result["local_protocols_completed"] == 0


def test_empty_eligible_frames_are_not_zero_scores(tmp_path):
    import json
    from pathlib import Path

    from src.io import atomic_json
    from src.reporting import make_comparison_figures

    run = start_run(root=tmp_path)
    for checkpoint in ("initial", "best"):
        for protocol in ("point", "point_correction", "mask"):
            atomic_json(
                run.path
                / "evaluation"
                / "test"
                / checkpoint
                / protocol
                / "scores.json",
                {
                    "aggregate": {"J&F_dot": None},
                    "paired_aggregate": {"J&F_dot": None},
                    "corrective_clicks": 0,
                    "videos": [
                        {"video_id": "short", "score": {"summary": {"J&F_dot": None}}}
                    ],
                },
            )
    figure = make_comparison_figures(run)
    rows = json.loads(Path(figure["paired_deltas"]).read_text())
    assert rows[0]["delta"] is None
    assert rows[0]["eligible"] is False
    report = write_report(run)
    text = Path(report["report"]).read_text()
    assert "N/A (no eligible observations)" in text
    assert "**Observation:**" not in text
