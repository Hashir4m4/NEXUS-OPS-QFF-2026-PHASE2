"""
NEXUS-OPS | QFF 2026 Phase 2
Automated Test Suite: Dataset Integrity and Schema Validation
Verifies:
- Official PS-Q1 dataset files exist (ecg_train, ecg_val, ecg_test_features).
- Row and column dimensions match official schema (6000 train, 1500 val, 1500 test).
- Feature columns match 30 statistical features from 5 windows.
- No missing / NaN values anywhere in the datasets.
- Binary labels strictly in {0, 1}.
- Test IDs are complete and unique across s0..s1499.
"""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import get_dataset_dir

DATA = get_dataset_dir("q1")


def test_dataset_files_exist():
    assert (DATA / "ecg_train.csv").exists(), "ecg_train.csv missing"
    assert (DATA / "ecg_val.csv").exists(), "ecg_val.csv missing"
    assert (DATA / "ecg_test_features.csv").exists(), "ecg_test_features.csv missing"


def test_dataset_dimensions():
    train = pd.read_csv(DATA / "ecg_train.csv")
    val = pd.read_csv(DATA / "ecg_val.csv")
    test = pd.read_csv(DATA / "ecg_test_features.csv")

    assert train.shape == (6000, 31), f"Unexpected train shape: {train.shape}"
    assert val.shape == (1500, 31), f"Unexpected val shape: {val.shape}"
    assert test.shape == (1500, 31), f"Unexpected test shape: {test.shape}"


def test_column_consistency():
    train = pd.read_csv(DATA / "ecg_train.csv")
    val = pd.read_csv(DATA / "ecg_val.csv")
    test = pd.read_csv(DATA / "ecg_test_features.csv")

    features = [c for c in train.columns if c != "label"]
    assert len(features) == 30, f"Expected 30 features, got {len(features)}"
    assert list(train.columns) == list(val.columns), "Train and val columns differ"

    test_features = [c for c in test.columns if c != "sample_id"]
    assert features == test_features, "Test feature columns do not match train"
    assert test.columns[0] == "sample_id", "First column of test file must be sample_id"


def test_no_missing_values():
    train = pd.read_csv(DATA / "ecg_train.csv")
    val = pd.read_csv(DATA / "ecg_val.csv")
    test = pd.read_csv(DATA / "ecg_test_features.csv")

    assert not train.isna().any().any(), "Train contains missing values"
    assert not val.isna().any().any(), "Validation contains missing values"
    assert not test.isna().any().any(), "Test contains missing values"


def test_label_validity():
    train = pd.read_csv(DATA / "ecg_train.csv")
    val = pd.read_csv(DATA / "ecg_val.csv")

    assert set(train["label"].unique()) == {0, 1}, "Train labels invalid"
    assert set(val["label"].unique()) == {0, 1}, "Val labels invalid"


def test_test_sample_ids():
    test = pd.read_csv(DATA / "ecg_test_features.csv")
    expected_ids = {f"s{i}" for i in range(1500)}
    actual_ids = set(test["sample_id"])

    assert len(test["sample_id"]) == 1500, "Test does not have 1500 IDs"
    assert not test["sample_id"].duplicated().any(), "Test contains duplicate sample IDs"
    assert actual_ids == expected_ids, "Test sample IDs mismatch expected s0..s1499"


if __name__ == "__main__":
    test_dataset_files_exist()
    test_dataset_dimensions()
    test_column_consistency()
    test_no_missing_values()
    test_label_validity()
    test_test_sample_ids()
    print("All dataset tests passed successfully.")
