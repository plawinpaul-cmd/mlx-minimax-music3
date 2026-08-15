import json
import os
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from scripts.compare_components import (
    COMPONENTS,
    _source_shards,
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


def test_source_shards_reads_local_inventory_without_downloading(tmp_path) -> None:
    component = tmp_path / "transformer"
    component.mkdir()
    (component / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "part-2.safetensors", "b": "part-1.safetensors"}})
    )

    assert _source_shards("transformer", "unused", "unused", tmp_path, tmp_path) == [
        "part-1.safetensors",
        "part-2.safetensors",
    ]


def test_compare_preserves_reference_virtualenv_entrypoint(tmp_path, monkeypatch) -> None:
    reference = tmp_path / "reference"
    (reference / ".git").mkdir(parents=True)
    (tmp_path / "config.json").write_text(
        json.dumps(
            {"quantization": {"group_size": 64, "bits": 4, "mode": "affine"}}
        )
    )
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
        checkpoint_id=None,
        source_repo="source/model",
        source_root=None,
        source_revision="source-revision",
        reference_device="cpu",
        report=tmp_path / "report.json",
        seed=7,
    )

    report = compare(args)

    assert report["status"] == "in_progress"
    assert captured["git"][2] == str(reference.resolve())
