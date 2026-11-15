import ast
from pathlib import Path

import nbformat

from src.runs import _cell_identity


def test_notebook_schema_imports_and_unexecuted_handoff():
    root = Path(__file__).resolve().parents[1]
    book = nbformat.read(
        root / "notebooks" / "video_tracking_occlusion.ipynb", as_version=4
    )
    nbformat.validate(book)
    for cell in book.cells:
        if cell.cell_type == "code":
            ast.parse(cell.source)
            assert cell.execution_count is None and not cell.outputs
        _cell_identity(cell)
    code = "\n".join(c.source for c in book.cells if c.cell_type == "code")
    assert "require_experiment_device=True" in code
    assert "checkpoint='best', split='test', protocol='point_correction'" in code
    assert code.index("run_pilot(run)") < code.index("fine_tune(run)")
