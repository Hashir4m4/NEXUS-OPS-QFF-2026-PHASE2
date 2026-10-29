"""
NEXUS-OPS | QFF 2026 Phase 2
Automated Test Suite: Data Leakage Prevention and Preprocessing Verification
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT.parent / "QFF2026_QML_Track" / "QFF2026_QML_Challenge_Kit"
DATA = KIT / "data" / "q1"
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

