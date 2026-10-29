"""
NEXUS-OPS | QFF 2026 Phase 2
Ablation Studies for PS-Q1 4-Qubit VQC
Studies:
1. Circuit Depth / Layer Count Ablation: 1 Layer vs 2 Layers vs 3 Layers
2. Qubit / Subspace Dimension Ablation: 2 Qubits vs 4 Qubits
"""
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score

SEED = 42
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parents[2]
KIT = ROOT.parent / "QFF2026_QML_Track" / "QFF2026_QML_Challenge_Kit"
DATA = KIT / "data" / "q1"
OUT_METRICS = ROOT / "outputs" / "metrics"
OUT_METRICS.mkdir(parents=True, exist_ok=True)


def train_eval_vqc(n_qubits, n_layers, epochs=5, batch_size=64, lr=0.03):
    train_df = pd.read_csv(DATA / "ecg_train.csv")
    val_df = pd.read_csv(DATA / "ecg_val.csv")

    feature_cols = [c for c in train_df.columns if c != "label"]
    X_train_raw = train_df[feature_cols].to_numpy(dtype=np.float64)
    y_train = train_df["label"].to_numpy(dtype=np.float64)
    X_val_raw = val_df[feature_cols].to_numpy(dtype=np.float64)
    y_val = val_df["label"].to_numpy(dtype=np.int64)

    # Preprocessing strictly on train
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train_raw)
    X_v_s = scaler.transform(X_val_raw)

    pca = PCA(n_components=n_qubits, random_state=SEED)
    X_tr_pca = pca.fit_transform(X_tr_s)
    X_v_pca = pca.transform(X_v_s)

    explained_var = float(np.sum(pca.explained_variance_ratio_))

    angle_scaler = MinMaxScaler(feature_range=(0.0, np.pi))
    X_tr_angles = np.clip(angle_scaler.fit_transform(X_tr_pca), 0.0, np.pi)
    X_v_angles = np.clip(angle_scaler.transform(X_v_pca), 0.0, np.pi)

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="autograd")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return qml.expval(qml.PauliZ(0))

    def predict_batch(inputs, weights):
        exp = circuit(inputs, weights)
        return (exp + 1.0) / 2.0

    def loss_fn(w, bx, by):
        preds = predict_batch(bx, w)
        return pnp.mean((preds - by) ** 2)

    weights = pnp.array(0.05 * np.random.randn(n_layers, n_qubits, 3), requires_grad=True)
    optimizer = qml.AdamOptimizer(stepsize=lr)

    n_samples = len(X_tr_angles)
    t0 = time.time()
    for _ in range(epochs):
        perm = np.random.permutation(n_samples)
        for s in range(0, n_samples, batch_size):
            idx = perm[s:s + batch_size]
            bx = pnp.array(X_tr_angles[idx], requires_grad=False)
            by = pnp.array(y_train[idx], requires_grad=False)
            weights, _ = optimizer.step_and_cost(lambda w: loss_fn(w, bx, by), weights)

    train_time = round(time.time() - t0, 3)

    # Evaluate
    val_probs = np.concatenate([
        np.array(predict_batch(X_v_angles[k:k+256], weights))
        for k in range(0, len(X_v_angles), 256)
    ])
    val_preds = (val_probs >= 0.5).astype(int)

    acc = float(accuracy_score(y_val, val_preds))
    macro_f1 = float(f1_score(y_val, val_preds, average="macro"))

    return {
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "trainable_parameters": int(weights.size),
        "pca_explained_variance": round(explained_var, 4),
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "training_time_seconds": train_time
    }


def main():
    print("=" * 65)
    print("NEXUS-OPS | VQC ABLATION EXPERIMENTS")
    print("=" * 65)

    experiments = [
        {"name": "1-Layer VQC (12 params)", "qubits": 4, "layers": 1},
        {"name": "2-Layer VQC (24 params - Baseline)", "qubits": 4, "layers": 2},
        {"name": "3-Layer VQC (36 params)", "qubits": 4, "layers": 3},
        {"name": "2-Qubit Subspace (PCA-2, 6 params)", "qubits": 2, "layers": 1},
        {"name": "2-Qubit Subspace (PCA-2, 12 params)", "qubits": 2, "layers": 2},
    ]

    results = {}
    for exp in experiments:
        print(f"\nRunning ablation: {exp['name']}...")
        res = train_eval_vqc(exp["qubits"], exp["layers"], epochs=5)
        results[exp["name"]] = res
        print(f"  Accuracy: {res['accuracy']:.4f} | Macro-F1: {res['macro_f1']:.4f} | Time: {res['training_time_seconds']}s")

    out_file = OUT_METRICS / "ablation_study.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"\nAblation study results written to: {out_file.name}")
    print("=" * 65)


if __name__ == "__main__":
    main()

