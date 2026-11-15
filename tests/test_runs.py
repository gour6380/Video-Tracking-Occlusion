from dataclasses import replace

import pytest

from src.config import ProjectConfig, config_from_dict
from src.io import atomic_json, atomic_write, read_json
from src.runs import resume_run, stage, start_run


def make_root(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "x.py").write_text("x=1\n")
    return tmp_path


def test_run_numbering_and_explicit_resume(tmp_path):
    root = make_root(tmp_path)
    first = start_run(root=root)
    (root / "runs" / "run_009").mkdir()
    second = start_run(root=root)
    assert first.name == "run_001" and second.name == "run_010"
    assert resume_run(first.name, root=root).config == first.config
    with pytest.raises(ValueError):
        resume_run("../run_001", root=root)


def test_source_changes_rejected(tmp_path):
    root = make_root(tmp_path)
    run = start_run(root=root)
    (root / "src" / "x.py").write_text("x=2\n")
    with pytest.raises(ValueError, match="Source"):
        resume_run(run.name, root=root)


def test_nested_stage_and_failure_receipts(tmp_path):
    run = start_run(root=make_root(tmp_path))
    with stage(run, "fit"), stage(run, "validation"):
        pass
    status = read_json(run.path / "status.json")
    assert all(v["state"] == "completed" for v in status["stages"].values())
    with pytest.raises(RuntimeError), stage(run, "fail"):
        raise RuntimeError("deliberate")
    assert read_json(run.path / "status.json")["stages"]["fail"]["state"] == "failed"


def test_atomic_writes_and_config_roundtrip(tmp_path):
    target = tmp_path / "nested" / "result"
    atomic_write(target, b"old")
    atomic_write(target, b"new")
    assert target.read_bytes() == b"new"
    config = ProjectConfig()
    assert config_from_dict(config.to_dict()) == config
    bad = replace(config, training=replace(config.training, accumulation=0))
    with pytest.raises(ValueError):
        bad.validate()


def test_notebook_outputs_do_not_invalidate_resume(tmp_path):
    root = make_root(tmp_path)
    book = root / "notebooks" / "x.ipynb"
    atomic_json(
        book,
        {
            "cells": [
                {
                    "cell_type": "code",
                    "source": ["x=1"],
                    "outputs": [],
                    "execution_count": None,
                }
            ]
        },
    )
    run = start_run(root=root)
    atomic_json(
        book,
        {
            "cells": [
                {
                    "cell_type": "code",
                    "source": ["x=1"],
                    "outputs": [{"text": "result"}],
                    "execution_count": 2,
                }
            ]
        },
    )
    assert resume_run(run.name, root=root).path == run.path


def test_resume_selector_is_operational_but_config_is_protected(tmp_path):
    root = make_root(tmp_path)
    book = root / "notebooks" / "x.ipynb"

    def content(value):
        return {
            "cells": [
                {
                    "cell_type": "code",
                    "source": f"run_name = {value!r}",
                    "metadata": {"tags": ["operational-input"]},
                }
            ]
        }

    atomic_json(book, content(None))
    run = start_run(root=root)
    atomic_json(book, content(run.name))
    assert resume_run(run.name, root=root).name == run.name
    atomic_json(
        book,
        {
            "cells": [
                {
                    "cell_type": "code",
                    "source": "config = 1",
                    "metadata": {"tags": ["operational-input"]},
                }
            ]
        },
    )
    with pytest.raises(ValueError, match="Experimental"):
        resume_run(run.name, root=root)


def test_resume_rejects_active_stage_without_altering_status(tmp_path):
    run = start_run(root=make_root(tmp_path))
    with stage(run, "active_fit"):
        before = (run.path / "status.json").read_bytes()
        with pytest.raises(RuntimeError, match="Cannot resume"):
            resume_run(run.name, root=run.root)
        assert (run.path / "status.json").read_bytes() == before
        assert (
            read_json(run.path / "status.json")["stages"]["active_fit"]["state"]
            == "running"
        )
    assert resume_run(run.name, root=run.root).path == run.path


def test_resume_marks_only_abandoned_unlocked_stages_interrupted(tmp_path):
    run = start_run(root=make_root(tmp_path))
    status = read_json(run.path / "status.json")
    status["stages"] = {
        "abandoned": {"state": "running"},
        "finished": {"state": "completed"},
    }
    atomic_json(run.path / "status.json", status)
    resume_run(run.name, root=run.root)
    recovered = read_json(run.path / "status.json")
    assert recovered["stages"]["abandoned"]["state"] == "interrupted"
    assert recovered["stages"]["finished"]["state"] == "completed"
