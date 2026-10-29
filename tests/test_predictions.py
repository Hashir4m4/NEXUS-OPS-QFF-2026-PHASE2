"""
NEXUS-OPS | QFF 2026 Phase 2
Automated Test Suite: Test Predictions and Official Validation
Verifies:
- predictions_q1.csv exists and matches schema.
- Passes official tools/validate_predictions.py validator with 0 errors.
- Correct row count (1500), sample_id format, and binary labels.
- Balanced, reasonable class distribution.
"""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import get_tools_dir
sys.path.insert(0, str(get_tools_dir()))

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


if __name__ == "__main__":
    test_prediction_file_exists()
    test_official_prediction_validator()
    test_prediction_rows_and_format()
    print("All prediction validation tests passed successfully.")
