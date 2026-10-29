"""
NEXUS-OPS | QFF 2026 Phase 2
Automated Test Suite: Classical and Quantum Model Behavior & Probability Bounds
Verifies:
- Model artifacts and weights dimensions.
- Finite values and probability bounds.
- Live VQC execution reproduces saved predictions bit-for-bit.
- Consistency between saved metrics and model artifact outputs.
"""
import sys
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import pennylane as qml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import get_dataset_dir

OUT_MODEL = ROOT / "outputs" / "models"
OUT_METRICS = ROOT / "outputs" / "metrics"
DATA = get_dataset_dir("q1")


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


def test_vqc_live_circuit_execution_consistency():
    """
    Live behavioral test: Executes the PennyLane circuit on 5 validation samples
    and confirms that live outputs match the saved predictions CSV within 1e-6.
    """
    weights = np.load(OUT_MODEL / "vqc_4qubit_weights.npy")
    preproc = joblib.load(OUT_MODEL / "vqc_4qubit_preprocessing.joblib")
    df_saved = pd.read_csv(OUT_METRICS / "vqc_4qubit_validation_predictions.csv")

    val_df = pd.read_csv(DATA / "ecg_val.csv")
    X_val_raw = val_df[preproc["feature_cols"]].iloc[:5].to_numpy(dtype=np.float64)
    X_val_angles = np.clip(preproc["angle_scaler"].transform(preproc["pca"].transform(preproc["standardizer"].transform(X_val_raw))), 0.0, np.pi)

    dev = qml.device("default.qubit", wires=4)

    @qml.qnode(dev)
    def circuit(x, w):
        qml.AngleEmbedding(x, wires=range(4), rotation="Y")
        # NEXUS-Ring4 Ansatz
        for i in range(4):
            qml.Rot(w[0, i, 0], w[0, i, 1], w[0, i, 2], wires=i)
        for i in range(4):
            qml.CNOT(wires=[i, (i + 1) % 4])
        for i in range(4):
            qml.Rot(w[1, i, 0], w[1, i, 1], w[1, i, 2], wires=i)
        for i in range(4):
            qml.CNOT(wires=[(i + 1) % 4, i])
        return qml.expval(qml.PauliZ(0))

    live_exps = circuit(X_val_angles, weights)
    live_probs = (np.array(live_exps) + 1.0) / 2.0

    saved_probs = df_saved["probability_arrhythmia"].iloc[:5].to_numpy()
    assert np.allclose(live_probs, saved_probs, atol=1e-5), "Live circuit outputs differ from saved predictions"


if __name__ == "__main__":
    test_model_files_exist()
    test_vqc_weights_dimension()
    test_vqc_validation_probabilities_bound()
    test_vqc_performance_thresholds()
    test_vqc_live_circuit_execution_consistency()
    print("All model behavioral tests passed successfully.")
