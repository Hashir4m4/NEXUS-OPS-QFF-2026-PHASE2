"""
NEXUS-OPS | QFF 2026 Phase 2
PS-Q1 4-Qubit VQC Noisy Evaluation under Organizer Noise Model QFF2026-v2
"""
from pathlib import Path
import json
import time
import sys
import joblib
import numpy as np
import pandas as pd

from qiskit import QuantumCircuit
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)

ROOT = Path(__file__).resolve().parents[2]
KIT = ROOT.parent / "QFF2026_QML_Track" / "QFF2026_QML_Challenge_Kit"
sys.path.insert(0, str(KIT / "noise_model"))

from benchmark_noise_model import (
    get_benchmark_noise_model,
    prepare_for_noisy_run,
    noisy_simulator,
    VERSION,
    SEED
)

DATA = KIT / "data" / "q1"
OUT_MODEL = ROOT / "outputs" / "models"
OUT_METRICS = ROOT / "outputs" / "metrics"
OUT_METRICS.mkdir(parents=True, exist_ok=True)


def build_vqc_circuit(x, weights):
    """
    Constructs the exact 4-qubit VQC circuit in Qiskit matching PennyLane's
    AngleEmbedding(Y) and StronglyEntanglingLayers(2).
    Wire 0 is measured on classical bit 0.
    """
    n_qubits = 4
    qc = QuantumCircuit(n_qubits, 1)

    # Angle embedding (Y-axis)
    for i in range(n_qubits):
        qc.ry(float(x[i]), i)

    # Layer 0 (range 1 entangling)
    for q in range(n_qubits):
        phi, theta, omega = weights[0, q]
        qc.rz(float(phi), q)
        qc.ry(float(theta), q)
        qc.rz(float(omega), q)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)
    qc.cx(3, 0)

    # Layer 1 (range 2 entangling)
    for q in range(n_qubits):
        phi, theta, omega = weights[1, q]
        qc.rz(float(phi), q)
        qc.ry(float(theta), q)
        qc.rz(float(omega), q)
    qc.cx(0, 2)
    qc.cx(1, 3)
    qc.cx(2, 0)
    qc.cx(3, 1)

    # Measure qubit 0 (corresponds to PennyLane PauliZ(0) expectation)
    qc.measure(0, 0)
    return qc


def main():
    start_total = time.time()
    print("=" * 65)
    print(f"NEXUS-OPS | 4-QUBIT VQC | NOISY EVALUATION ({VERSION})")
    print("=" * 65)

    # 1. Load trained model & preprocessors
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

    print(f"Loaded weights shape: {weights.shape}")
    print(f"Noise model: {VERSION} (single_q=0.01, two_q=0.05, readout=0.02)")

    # 2. Load and preprocess validation data
    val_path = DATA / "ecg_val.csv"
    val_df = pd.read_csv(val_path)
    X_val_raw = val_df[feature_cols].to_numpy(dtype=np.float64)
    y_val = val_df["label"].to_numpy(dtype=np.int64)

    X_val_std = standardizer.transform(X_val_raw)
    X_val_pca = pca.transform(X_val_std)
    X_val_angles = np.clip(angle_scaler.transform(X_val_pca), 0.0, np.pi)

    n_val = len(X_val_angles)
    print(f"Validation samples: {n_val}")

    # 3. Transpile and simulate under benchmark noise model
    shots = 8192
    sim = noisy_simulator(seed=SEED)

    print(f"\nTranspiling {n_val} circuits to fully-noised basis (rx, ry, rz, cx)...")
    t0 = time.time()
    transpiled_circuits = []
    for i in range(n_val):
        qc = build_vqc_circuit(X_val_angles[i], weights)
        tqc = prepare_for_noisy_run(qc)
        transpiled_circuits.append(tqc)
    transpile_time = time.time() - t0
    print(f"Transpilation completed in {transpile_time:.2f}s ({transpile_time/n_val*1000:.2f} ms/circuit)")
    print(f"Sample transpiled circuit depth: {transpiled_circuits[0].depth()}, ops: {dict(transpiled_circuits[0].count_ops())}")

    print(f"\nRunning noisy batch simulation on AerSimulator ({shots} shots/circuit)...")
    batch_size = 250
    counts_all = []
    sim_t0 = time.time()
    for b in range(0, n_val, batch_size):
        sub_circs = transpiled_circuits[b:b + batch_size]
        job = sim.run(sub_circs, shots=shots)
        result = job.result()
        for idx in range(len(sub_circs)):
            counts_all.append(result.get_counts(idx))
        print(f"  Simulated {min(b + batch_size, n_val)}/{n_val} circuits...")

    sim_time = time.time() - sim_t0
    print(f"Noisy simulation completed in {sim_time:.2f}s ({sim_time/n_val*1000:.2f} ms/circuit).")

    # 4. Compute expectation values and probabilities
    # <Z> = (N_0 - N_1) / N_total
    # P(Arrhythmia) = (<Z> + 1) / 2 = N_0 / N_total (since bit 0 = 0 corresponds to +1, bit 0 = 1 to -1)
    # Wait, in PennyLane: expval(PauliZ(0)) has outcome +1 for |0> and -1 for |1>.
    # In our training: y=1 (Arrhythmia) is mapped to P(Arrhythmia) = (expval + 1) / 2.
    # Therefore, expval = +1 gives P = 1.0 (state |0>), expval = -1 gives P = 0.0 (state |1>).
    # In Qiskit measurement counts: '0' count corresponds to eigenvalue +1, '1' to eigenvalue -1.
    val_probs = []
    for counts in counts_all:
        n0 = counts.get("0", 0)
        n1 = counts.get("1", 0)
        total = n0 + n1
        expval = (n0 - n1) / total
        p = (expval + 1.0) / 2.0
        val_probs.append(float(p))

    val_probs = np.array(val_probs)
    val_preds = (val_probs >= 0.5).astype(int)

    # 5. Compute Metrics
    acc = float(accuracy_score(y_val, val_preds))
    macro_f1 = float(f1_score(y_val, val_preds, average="macro"))
    macro_prec = float(precision_score(y_val, val_preds, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_val, val_preds, average="macro", zero_division=0))
    cm = confusion_matrix(y_val, val_preds, labels=[0, 1]).tolist()

    report = classification_report(
        y_val,
        val_preds,
        labels=[0, 1],
        target_names=["Normal", "Arrhythmia"],
        output_dict=True,
        zero_division=0
    )

    # Load ideal metrics for comparison
    ideal_metrics_path = OUT_METRICS / "vqc_4qubit_ideal.json"
    ideal_metrics = {}
    if ideal_metrics_path.exists():
        with open(ideal_metrics_path, "r", encoding="utf-8") as f:
            ideal_metrics = json.load(f)

    noisy_metrics = {
        "model": "4-qubit hybrid VQC",
        "simulator": "Qiskit AerSimulator (noise_model=QFF2026-v2)",
        "noise_condition": "noisy_benchmark",
        "noise_model_version": VERSION,
        "single_qubit_depolarizing": 0.01,
        "two_qubit_depolarizing": 0.05,
        "readout_error": 0.02,
        "shots_per_sample": shots,
        "dataset": "QFF2026 Phase 2 PS-Q1",
        "seed": SEED,
        "qubits": 4,
        "layers": 2,
        "trainable_parameters": int(weights.size),
        "circuit_depth_transpiled": transpiled_circuits[0].depth(),
        "two_qubit_gate_count": sum(1 for op in transpiled_circuits[0].data if op.operation.num_qubits == 2),
        "validation_samples": n_val,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "confusion_matrix": cm,
        "classification_report": report,
        "simulation_time_seconds": round(sim_time, 3),
        "transpile_time_seconds": round(transpile_time, 3),
        "total_time_seconds": round(time.time() - start_total, 3),
        "ideal_vs_noisy_comparison": {
            "ideal_macro_f1": ideal_metrics.get("macro_f1"),
            "noisy_macro_f1": macro_f1,
            "macro_f1_delta": round(macro_f1 - (ideal_metrics.get("macro_f1") or 0.0), 6),
            "ideal_accuracy": ideal_metrics.get("accuracy"),
            "noisy_accuracy": acc,
            "accuracy_delta": round(acc - (ideal_metrics.get("accuracy") or 0.0), 6),
            "ideal_confusion_matrix": ideal_metrics.get("confusion_matrix"),
            "noisy_confusion_matrix": cm
        }
    }

    # 6. Save Noisy Artifacts
    out_json = OUT_METRICS / "vqc_4qubit_noisy.json"
    out_csv = OUT_METRICS / "vqc_4qubit_noisy_validation_predictions.csv"

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(noisy_metrics, f, indent=4)

    pd.DataFrame({
        "actual_label": y_val,
        "predicted_label": val_preds,
        "probability_arrhythmia": val_probs
    }).to_csv(out_csv, index=False)

    print("\n" + "=" * 65)
    print("4-QUBIT VQC NOISY EVALUATION RESULTS")
    print("=" * 65)
    print(f"Accuracy:        {acc:.4f} (Ideal: {ideal_metrics.get('accuracy', 0):.4f})")
    print(f"Macro-F1:        {macro_f1:.4f} (Ideal: {ideal_metrics.get('macro_f1', 0):.4f})")
    print(f"Macro Precision: {macro_prec:.4f}")
    print(f"Macro Recall:    {macro_rec:.4f}")
    print("Confusion Matrix (TN, FP / FN, TP):")
    print(f"  Normal:     [{cm[0][0]:4d}, {cm[0][1]:4d}]")
    print(f"  Arrhythmia: [{cm[1][0]:4d}, {cm[1][1]:4d}]")
    print("\nSaved artifacts:")
    print(f"  Metrics JSON:   {out_json.name}")
    print(f"  Validation CSV: {out_csv.name}")
    print("=" * 65)


if __name__ == "__main__":
    main()

