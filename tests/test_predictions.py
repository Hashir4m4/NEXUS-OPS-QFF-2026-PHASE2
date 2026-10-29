"""
NEXUS-OPS | QFF 2026 Phase 2
Automated Test Suite: Test Predictions and Official Validation
"""
from pathlib import Path
import sys
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT.parent / "QFF2026_QML_Track" / "QFF2026_QML_Challenge_Kit"
sys.path.insert(0, str(KIT / "tools"))

from validate_predictions import validate

PRED_FILE = ROOT / "outputs" / "predictions" / "predictions_q1.csv"


def test_prediction_file_exists():
    assert PRED_FILE.exists(), "predictions_q1.csv is missing"


def test_official_prediction_validator():
    errors = validate("q1", str(PRED_FILE))
    assert len(errors) == 0, f"Official validation failed: {errors}"


def test_prediction_rows_and_format():
    df = pd.read_csv(PRED_FILE)
    assert list(df.columns) == ["sample_id", "label"], f"Invalid columns: {df.columns}"
    assert len(df) == 1500, f"Expected 1500 rows, got {len(df)}"

    expected_ids = [f"s{i}" for i in range(1500)]
    assert df["sample_id"].tolist() == expected_ids, "Sample IDs mismatch s0..s1499 in order"

    labels = set(df["label"].unique())
    assert labels.issubset({0, 1}), f"Invalid labels: {labels}"

    counts = df["label"].value_counts()
    assert 700 <= counts[0] <= 1100, f"Normal count out of expected range: {counts[0]}"
    assert 400 <= counts[1] <= 800, f"Arrhythmia count out of expected range: {counts[1]}"

