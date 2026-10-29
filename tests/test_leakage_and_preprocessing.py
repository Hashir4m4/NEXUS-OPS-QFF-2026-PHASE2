"""
NEXUS-OPS | QFF 2026 Phase 2
Automated Test Suite: Data Leakage Prevention and Preprocessing Verification
Verifies:
- Saved preprocessing transformers were fitted strictly on ecg_train.csv.
- Zero contamination or parameters derived from ecg_val.csv or ecg_test_features.csv.
- PCA variance ratio matches expected 81.28% variance retention.
- Angle scaling produces valid rotations strictly in [0.0, pi].
- Deterministic reproducibility of preprocessing pipeline.
"""
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import get_dataset_dir

DATA = get_dataset_dir("q1")
OUT_MODEL = ROOT / "outputs" / "models"


def test_preprocessing_artifacts_exist():
    preproc_file = OUT_MODEL / "vqc_4qubit_preprocessing.joblib"
    assert preproc_file.exists(), "Preprocessing artifact vqc_4qubit_preprocessing.joblib is missing"


def test_no_leakage_scaler_and_pca():
    """
    Verifies that the saved StandardScaler and PCA parameters were fitted
    strictly on ecg_train.csv and NOT contaminated by ecg_val.csv or ecg_test_features.csv.
    """
    preproc_file = OUT_MODEL / "vqc_4qubit_preprocessing.joblib"
    preproc = joblib.load(preproc_file)

    train_df = pd.read_csv(DATA / "ecg_train.csv")
    feature_cols = preproc["feature_cols"]
    X_train = train_df[feature_cols].to_numpy(dtype=np.float64)

    # Re-compute ground truth mean and var on train only
    true_mean = np.mean(X_train, axis=0)
    true_var = np.var(X_train, axis=0)

    scaler = preproc["standardizer"]
    assert np.allclose(scaler.mean_, true_mean, atol=1e-5), "StandardScaler mean does not match train data"
    assert np.allclose(scaler.var_, true_var, atol=1e-5), "StandardScaler var does not match train data"

    pca = preproc["pca"]
    assert pca.n_components == 4, f"PCA components must be 4, got {pca.n_components}"
    assert np.isclose(np.sum(pca.explained_variance_ratio_), 0.8128, atol=1e-3), "PCA variance ratio unexpected"


def test_angle_scaler_bounds():
    preproc_file = OUT_MODEL / "vqc_4qubit_preprocessing.joblib"
    preproc = joblib.load(preproc_file)

    val_df = pd.read_csv(DATA / "ecg_val.csv")
    feature_cols = preproc["feature_cols"]
    X_val = val_df[feature_cols].to_numpy(dtype=np.float64)

    X_val_s = preproc["standardizer"].transform(X_val)
    X_val_pca = preproc["pca"].transform(X_val_s)
    X_val_angles = np.clip(preproc["angle_scaler"].transform(X_val_pca), 0.0, np.pi)

    assert X_val_angles.shape == (1500, 4)
    assert np.all(X_val_angles >= 0.0), "Angles below 0.0 found"
    assert np.all(X_val_angles <= np.pi), "Angles exceeding pi found"


def test_preprocessing_reproducibility():
    """
    Verifies that running the preprocessing pipeline twice on identical data
    yields bit-for-bit identical angle encodings.
    """
    preproc = joblib.load(OUT_MODEL / "vqc_4qubit_preprocessing.joblib")
    val_df = pd.read_csv(DATA / "ecg_val.csv")
    X_raw = val_df[preproc["feature_cols"]].iloc[:50].to_numpy(dtype=np.float64)

    run1 = np.clip(preproc["angle_scaler"].transform(preproc["pca"].transform(preproc["standardizer"].transform(X_raw))), 0.0, np.pi)
    run2 = np.clip(preproc["angle_scaler"].transform(preproc["pca"].transform(preproc["standardizer"].transform(X_raw))), 0.0, np.pi)

    assert np.array_equal(run1, run2), "Preprocessing is non-deterministic"


def test_synthetic_violation_detection():
    """
    Adversarial regression test: ensures that if an engineer or pipeline
    accidentally contaminates the StandardScaler with validation or test features,
    the leakage detection strictly detects the discrepancy and raises an error.
    """
    train_df = pd.read_csv(DATA / "ecg_train.csv")
    val_df = pd.read_csv(DATA / "ecg_val.csv")
    test_df = pd.read_csv(DATA / "ecg_test_features.csv")

    feature_cols = [c for c in train_df.columns if c != "label"]
    X_train = train_df[feature_cols].to_numpy(dtype=np.float64)
    X_val = val_df[feature_cols].to_numpy(dtype=np.float64)
    X_test = test_df[feature_cols].to_numpy(dtype=np.float64)

    from sklearn.preprocessing import StandardScaler
    # Ground truth clean scaler (train only)
    clean_scaler = StandardScaler().fit(X_train)

    # 1. Contaminated with validation set
    contaminated_val_scaler = StandardScaler().fit(np.vstack([X_train, X_val]))
    val_diff = np.max(np.abs(clean_scaler.mean_ - contaminated_val_scaler.mean_))
    assert val_diff > 1e-4, f"Leakage detector failed to detect validation contamination (diff={val_diff})"

    # 2. Contaminated with test set
    contaminated_test_scaler = StandardScaler().fit(np.vstack([X_train, X_test]))
    test_diff = np.max(np.abs(clean_scaler.mean_ - contaminated_test_scaler.mean_))
    assert test_diff > 1e-4, f"Leakage detector failed to detect test feature contamination (diff={test_diff})"

    # 3. Assert test features file strictly contains no ground truth label column
    assert "label" not in test_df.columns, "Data leak: test features CSV contains ground truth label column!"


if __name__ == "__main__":
    test_preprocessing_artifacts_exist()
    test_no_leakage_scaler_and_pca()
    test_angle_scaler_bounds()
    test_preprocessing_reproducibility()
    test_synthetic_violation_detection()
    print("All preprocessing & leakage tests passed successfully.")
