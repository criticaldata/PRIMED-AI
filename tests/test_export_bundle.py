"""Tests for the sanitized result bundle (#74 review follow-up)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import export_result_bundle


def _run(tmp_path, out, monkeypatch, results="results"):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_result_bundle.py",
            "--manifest",
            str(tmp_path / "manifest.parquet"),
            "--results",
            str(tmp_path / results),
            "--probes",
            str(tmp_path / "probes"),
            "--out",
            str(out),
        ],
    )
    export_result_bundle.main()


def _sums(out):
    return {
        line.split("  ", 1)[1]: line.split("  ", 1)[0]
        for line in (out / "SHA256SUMS").read_text().splitlines()
    }


def test_per_example_blocks_are_stripped(tmp_path, monkeypatch):
    results = tmp_path / "results"
    results.mkdir()
    (results / "missing_modality.json").write_text(
        json.dumps(
            {
                "test": {"full": {"mae": 10.4}},
                "predictions": {"full": [1, 2]},
                "predictions_val": {"full": [3]},
            }
        )
    )
    out = tmp_path / "bundle"
    _run(tmp_path, out, monkeypatch)

    exported = json.loads((out / "missing_modality.metrics.json").read_text())
    assert "predictions" not in exported and "predictions_val" not in exported
    assert exported["test"]["full"]["mae"] == 10.4


def test_a_file_whose_source_vanished_keeps_its_checksum(tmp_path, monkeypatch):
    """It stays committed either way, so dropping the line would let `shasum -c` pass
    over an artifact no run has verified."""
    results = tmp_path / "results"
    (results / "fairness").mkdir(parents=True)
    (results / "fairness" / "fairness_metrics.json").write_text(json.dumps({"overall": {}}))
    out = tmp_path / "bundle"
    _run(tmp_path, out, monkeypatch)
    assert "fairness_metrics.json" in {p.name for p in out.glob("*.json")}

    # second run, source gone: the exported copy is still on disk and still committed
    (results / "fairness" / "fairness_metrics.json").unlink()
    _run(tmp_path, out, monkeypatch)

    sums = _sums(out)
    assert any(name.endswith("fairness_metrics.json") for name in sums)


def test_checksums_match_the_files_on_disk(tmp_path, monkeypatch):
    from primed_ai.probes.common import sha256_file

    results = tmp_path / "results"
    results.mkdir()
    (results / "baseline_gap.json").write_text(json.dumps({"test_metrics": {}}))
    out = tmp_path / "bundle"
    _run(tmp_path, out, monkeypatch)

    sums = _sums(out)
    assert sums, "bundle produced no checksum lines"
    for name, digest in sums.items():
        assert sha256_file(Path(name)) == digest


def test_missing_results_dir_produces_an_empty_bundle(tmp_path, monkeypatch, capsys):
    out = tmp_path / "bundle"
    _run(tmp_path, out, monkeypatch, results="nope")
    assert "exported 0" in capsys.readouterr().out
