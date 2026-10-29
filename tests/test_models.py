"""
NEXUS-OPS | QFF 2026 Phase 2
Automated Test Suite: Classical and Quantum Model Behavior & Probability Bounds
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
OUT_MODEL = ROOT / "outputs" / "models"
OUT_METRICS = ROOT / "outputs" / "metrics"


def test_model_files_exist():
    assert (OUT_MODEL / "logistic_regression_baseline.joblib").exists(), "Classical baseline model missing"
    assert (OUT_MODEL / "vqc_4qubit_weights.npy").exists(), "VQC weights file missing"
    assert (OUT_METRICS / "classical_baseline.json").exists(), "Classical metrics missing"
    assert (OUT_METRICS / "vqc_4qubit_ideal.json").exists(), "VQC ideal metrics missing"
    assert (OUT_METRICS / "vqc_4qubit_noisy.json").exists(), "VQC noisy metrics missing"


def test_vqc_weights_dimension():
    weights = np.load(OUT_MODEL / "vqc_4qubit_weights.npy")
    assert weights.shape == (2, 4, 3), f"Expected VQC weight shape (2, 4, 3), got {weights.shape}"
    assert np.all(np.isfinite(weights)), "Non-finite values found in VQC weights"


def test_vqc_validation_probabilities_bound():
    import pandas as pd
    df_ideal = pd.read_csv(OUT_METRICS / "vqc_4qubit_validation_predictions.csv")
    df_noisy = pd.read_csv(OUT_METRICS / "vqc_4qubit_noisy_validation_predictions.csv")

    for df, name in [(df_ideal, "ideal"), (df_noisy, "noisy")]:
        assert len(df) == 1500, f"{name} predictions should have 1500 rows"
        probs = df["probability_arrhythmia"]
        assert np.all(probs >= 0.0), f"Negative probability found in {name}"
        assert np.all(probs <= 1.0), f"Probability > 1.0 found in {name}"
        assert set(df["predicted_label"].unique()).issubset({0, 1}), f"Invalid labels in {name}"


def test_vqc_performance_thresholds():
    with open(OUT_METRICS / "vqc_4qubit_ideal.json") as f:
        ideal = json.load(f)
    with open(OUT_METRICS / "vqc_4qubit_noisy.json") as f:
        noisy = json.load(f)

    assert ideal["macro_f1"] >= 0.98, f"Ideal Macro-F1 too low: {ideal['macro_f1']}"
    assert ideal["accuracy"] >= 0.98, f"Ideal Accuracy too low: {ideal['accuracy']}"
    assert noisy["macro_f1"] >= 0.95, f"Noisy Macro-F1 too low: {noisy['macro_f1']}"
    assert noisy["accuracy"] >= 0.95, f"Noisy Accuracy too low: {noisy['accuracy']}"

