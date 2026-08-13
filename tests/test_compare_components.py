import os
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from scripts.compare_components import (
    COMPONENTS,
    compare,
    make_inputs,
    metric_status,
    parity_metrics,
)


@pytest.mark.parametrize("component", COMPONENTS)
def test_parity_inputs_match_component_contract(component, tmp_path) -> None:
    path = tmp_path / f"{component}.npz"

    shapes = make_inputs(component, path)

    arrays = dict(np.load(path))
    assert set(arrays) == set(shapes)
    assert {name: list(value.shape) for name, value in arrays.items()} == shapes


def test_parity_metrics_identical_arrays_pass() -> None:
    reference = np.arange(12, dtype=np.float32).reshape(3, 4)

    metrics = parity_metrics(reference, reference.copy())

    assert metrics["cosine_similarity"] == pytest.approx(1.0)
    assert metrics["relative_rmse"] == 0.0
    assert metric_status("transformer", metrics) == "pass"


def test_parity_metrics_large_error_fails() -> None:
    reference = np.ones((3, 4), dtype=np.float32)
    actual = np.arange(12, dtype=np.float32).reshape(3, 4)

    metrics = parity_metrics(reference, actual)

    assert metric_status("transformer", metrics) == "fail"


def test_parity_metrics_reject_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        parity_metrics(np.zeros((2, 3)), np.zeros((3, 2)))


def test_compare_preserves_reference_virtualenv_entrypoint(tmp_path, monkeypatch) -> None:
    reference = tmp_path / "reference"
    (reference / ".git").mkdir(parents=True)
    venv_python = tmp_path / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(Path(os.sys.executable))
    captured = {}

    def fake_check_output(command, text):
        captured["git"] = command
        return "locked-revision\n"

    monkeypatch.setattr("scripts.compare_components.subprocess.check_output", fake_check_output)
    monkeypatch.setattr("scripts.compare_components.COMPONENTS", ())
    args = Namespace(
        model=tmp_path,
        reference_source=reference,
        reference_python=venv_python,
        reference_revision="locked-revision",
        component="all",
        checkpoint_revision="checkpoint",
        source_repo="source/model",
        source_revision="source-revision",
        reference_device="cpu",
        report=tmp_path / "report.json",
        seed=7,
    )

    report = compare(args)

    assert report["status"] == "in_progress"
    assert captured["git"][2] == str(reference.resolve())
