"""
NEXUS-OPS | QFF 2026 Phase 2
Final Test Prediction Generation and Official Validation for PS-Q1
"""
from pathlib import Path
import json
import sys
import subprocess
import joblib
import numpy as np
import pandas as pd
import pennylane as qml

ROOT = Path(__file__).resolve().parents[2]
KIT = ROOT.parent / "QFF2026_QML_Track" / "QFF2026_QML_Challenge_Kit"
DATA = KIT / "data" / "q1"
TOOLS = KIT / "tools"

OUT_PRED = ROOT / "outputs" / "predictions"
OUT_MODEL = ROOT / "outputs" / "models"
OUT_PRED.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 65)
    print("NEXUS-OPS | GENERATE FINAL TEST PREDICTIONS | PS-Q1")
    print("=" * 65)

    test_file = DATA / "ecg_test_features.csv"
    if not test_file.exists():
        raise FileNotFoundError(f"Test features file not found: {test_file}")

    print(f"Reading test data from: {test_file.name}")
    test_df = pd.read_csv(test_file)
    print(f"Test rows: {len(test_df)}, columns: {len(test_df.columns)}")

    # Verify sample_id column
    if "sample_id" not in test_df.columns:
        raise ValueError("Missing 'sample_id' column in test features.")
    
    sample_ids = test_df["sample_id"].tolist()
    expected_ids = [f"s{i}" for i in range(1500)]
    if sample_ids != expected_ids:
        print(f"Warning: sample_ids order differs from sequential s0..s1499, verifying set equality...")
        if set(sample_ids) != set(expected_ids):
            raise ValueError("Test sample IDs do not match expected set s0..s1499.")

    # Load trained VQC model & preprocessors
    weights_path = OUT_MODEL / "vqc_4qubit_weights.npy"
    preproc_path = OUT_MODEL / "vqc_4qubit_preprocessing.joblib"

    if not weights_path.exists() or not preproc_path.exists():
        raise FileNotFoundError("Trained VQC weights or preprocessing artifacts not found.")

    weights = np.load(weights_path)
    preproc = joblib.load(preproc_path)
    standardizer = preproc["standardizer"]
    pca = preproc["pca"]
    angle_scaler = preproc["angle_scaler"]
    feature_cols = preproc["feature_cols"]

    print("\nPreparing test feature matrix (excluding sample_id)...")
    X_test_raw = test_df[feature_cols].to_numpy(dtype=np.float64)

    # Preprocessing strictly using train-fitted parameters
    X_test_std = standardizer.transform(X_test_raw)
    X_test_pca = pca.transform(X_test_std)
    X_test_angles = np.clip(angle_scaler.transform(X_test_pca), 0.0, np.pi)

    # Quantum Inference
    dev = qml.device("default.qubit", wires=4)

    @qml.qnode(dev)
    def circuit(x, w):
        qml.AngleEmbedding(x, wires=range(4), rotation="Y")
        qml.StronglyEntanglingLayers(w, wires=range(4))
        return qml.expval(qml.PauliZ(0))

    print("Running quantum inference across 1,500 test samples...")
    probs = []
    batch_size = 256
    for k in range(0, len(X_test_angles), batch_size):
        batch = X_test_angles[k:k + batch_size]
        exps = circuit(batch, weights)
        p = (exps + 1.0) / 2.0
        probs.extend(p)

    probs = np.array(probs)
    predictions = (probs >= 0.5).astype(int)

    class_counts = np.bincount(predictions)
    print("\nPredicted Class Distribution on Hidden Test Set:")
    print(f"  Class 0 (Normal):     {class_counts[0]} ({class_counts[0]/15:.1f}%)")
    print(f"  Class 1 (Arrhythmia): {class_counts[1]} ({class_counts[1]/15:.1f}%)")

    # Format predictions strictly as sample_id,label
    pred_df = pd.DataFrame({
        "sample_id": sample_ids,
        "label": predictions
    })

    pred_path = OUT_PRED / "predictions_q1.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"\nWritten {len(pred_df)} test predictions to: {pred_path}")

    # Run official validator
    validator_script = TOOLS / "validate_predictions.py"
    print(f"\nExecuting official validator: {validator_script.name}...")
    result = subprocess.run(
        [sys.executable, str(validator_script), "q1", str(pred_path)],
        capture_output=True,
        text=True
    )

    print("Validator STDOUT:")
    print(result.stdout)
    if result.stderr:
        print("Validator STDERR:")
        print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"Official prediction validator failed with return code {result.returncode}")

    print("[SUCCESS] Official predictions_q1.csv generated and passed official validation!")
    print("=" * 65)


if __name__ == "__main__":
    main()
