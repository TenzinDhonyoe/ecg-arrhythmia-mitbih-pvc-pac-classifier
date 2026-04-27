"""Tests for the JSON/text/CSV CLI output paths in ecg_arrhythmia.infer."""

from __future__ import annotations

import csv
import io
import json
from contextlib import redirect_stdout

import numpy as np

from ecg_arrhythmia.infer import main as infer_main
from ecg_arrhythmia.training import train_baseline_quick_smoke


def _synthetic_ecg(fs=360.0, duration=10.0, hr_bpm=72.0):
    t = np.arange(int(duration * fs)) / fs
    rr = 60.0 / hr_bpm
    sig = np.zeros_like(t)
    for tk in np.arange(rr, duration, rr):
        sig += np.exp(-((t - tk) / 0.02) ** 2)
    return sig


def _write_csv(path, sig):
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["signal"])
        for v in sig:
            w.writerow([f"{v:.6f}"])


def test_cli_text_output(tmp_path):
    artifacts = train_baseline_quick_smoke(tmp_path, seed=0)
    csv_path = tmp_path / "ecg.csv"
    _write_csv(csv_path, _synthetic_ecg())

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = infer_main(
            [
                "--csv",
                str(csv_path),
                "--input-fs",
                "360",
                "--lead",
                "MLII",
                "--model-path",
                str(artifacts.model_path),
                "--scaler-path",
                str(artifacts.scaler_path),
            ]
        )
    assert rc == 0
    out = buf.getvalue()
    assert "Detected and classified" in out


def test_cli_json_output_is_valid(tmp_path):
    artifacts = train_baseline_quick_smoke(tmp_path, seed=0)
    csv_path = tmp_path / "ecg.csv"
    _write_csv(csv_path, _synthetic_ecg())

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = infer_main(
            [
                "--csv",
                str(csv_path),
                "--input-fs",
                "360",
                "--lead",
                "MLII",
                "--model-path",
                str(artifacts.model_path),
                "--scaler-path",
                str(artifacts.scaler_path),
                "--output-format",
                "json",
                "--top-k",
                "2",
            ]
        )
    assert rc == 0
    payload = json.loads(buf.getvalue())
    assert "summary" in payload and "beats" in payload and "info" in payload
    assert payload["summary"]["n_beats"] == len(payload["beats"])
    if payload["beats"]:
        beat = payload["beats"][0]
        assert {"beat_index", "label", "confidence", "prob_N", "prob_V", "prob_a"}.issubset(beat.keys())
        assert "topk" in beat and len(beat["topk"]) == 2


def test_cli_csv_output(tmp_path):
    artifacts = train_baseline_quick_smoke(tmp_path, seed=0)
    csv_path = tmp_path / "ecg.csv"
    out_path = tmp_path / "preds.csv"
    _write_csv(csv_path, _synthetic_ecg())

    rc = infer_main(
        [
            "--csv",
            str(csv_path),
            "--input-fs",
            "360",
            "--lead",
            "MLII",
            "--model-path",
            str(artifacts.model_path),
            "--scaler-path",
            str(artifacts.scaler_path),
            "--output-format",
            "csv",
            "--out",
            str(out_path),
        ]
    )
    assert rc == 0
    rows = list(csv.DictReader(out_path.open()))
    assert len(rows) >= 1
    assert "label" in rows[0]
    assert "confidence" in rows[0]
