"""
NEXUS-OPS | QFF 2026 Phase 2
4-Qubit Variational Quantum Classifier (VQC) for PS-Q1 ECG Classification
"""
from pathlib import Path
import json
import time
import joblib
import numpy as np
import pandas as pd
import pennylane as qml
from pennylane import numpy as pnp

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)

SEED = 42
N_QUBITS = 4
N_LAYERS = 2
EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.03

np.random.seed(SEED)

ROOT = Path(__file__).resolve().parents[2]
KIT = ROOT.parent / "QFF2026_QML_Track" / "QFF2026_QML_Challenge_Kit"
DATA = KIT / "data" / "q1"

OUT_MODEL = ROOT / "outputs" / "models"
OUT_METRICS = ROOT / "outputs" / "metrics"

for folder in [OUT_MODEL, OUT_METRICS]:
    folder.mkdir(parents=True, exist_ok=True)


def create_circuit(n_qubits=N_QUBITS):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="autograd")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return qml.expval(qml.PauliZ(0))

    return circuit


def main():
    start_total = time.time()

    print("=" * 65)
    print("NEXUS-OPS | 4-QUBIT VQC CLASSIFIER | PS-Q1")
    print("=" * 65)

    # 1. Load Data
    train_path = DATA / "ecg_train.csv"
    val_path = DATA / "ecg_val.csv"

    print(f"Loading training data from: {train_path.name}")
    print(f"Loading validation data from: {val_path.name}")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    feature_cols = [c for c in train_df.columns if c != "label"]
    X_train_raw = train_df[feature_cols].to_numpy(dtype=np.float64)
    y_train_raw = train_df["label"].to_numpy(dtype=np.float64)

    X_val_raw = val_df[feature_cols].to_numpy(dtype=np.float64)
    y_val = val_df["label"].to_numpy(dtype=np.int64)

    print(f"Training samples: {len(X_train_raw)}, Validation samples: {len(X_val_raw)}")
    print(f"Original feature count: {len(feature_cols)}")

    # 2. Preprocessing (Fitted STRICTLY on training data only)
    print("\nFitting preprocessing on training data only...")
    standardizer = StandardScaler()
    X_train_std = standardizer.fit_transform(X_train_raw)
    X_val_std = standardizer.transform(X_val_raw)

    pca = PCA(n_components=N_QUBITS, random_state=SEED)
    X_train_pca = pca.fit_transform(X_train_std)
    X_val_pca = pca.transform(X_val_std)

    explained_var = float(np.sum(pca.explained_variance_ratio_))
    print(f"PCA explained variance ratio with {N_QUBITS} components: {explained_var:.4f}")
    for idx, ratio in enumerate(pca.explained_variance_ratio_):
        print(f"  PC{idx+1}: {ratio:.4f}")

    angle_scaler = MinMaxScaler(feature_range=(0.0, np.pi))
    X_train_angles = angle_scaler.fit_transform(X_train_pca)
    X_val_angles = angle_scaler.transform(X_val_pca)

    X_train_angles = np.clip(X_train_angles, 0.0, np.pi)
    X_val_angles = np.clip(X_val_angles, 0.0, np.pi)

    # 3. Model Definition
    circuit = create_circuit(N_QUBITS)

    def predict_batch(inputs, weights):
        exp = circuit(inputs, weights)
        return (exp + 1.0) / 2.0

    def loss_fn(w, bx, by):
        preds = predict_batch(bx, w)
        return pnp.mean((preds - by) ** 2)

    # Initialize weights
    weights = pnp.array(
        0.05 * np.random.randn(N_LAYERS, N_QUBITS, 3),
        requires_grad=True
    )
    optimizer = qml.AdamOptimizer(stepsize=LEARNING_RATE)

    print(f"\nVQC Architecture:")
    print(f"  Qubits: {N_QUBITS}")
    print(f"  Ansatz Layers: {N_LAYERS} (StronglyEntanglingLayers)")
    print(f"  Trainable Parameters: {weights.size}")
    print(f"  Batch size: {BATCH_SIZE}, Learning rate: {LEARNING_RATE}, Epochs: {EPOCHS}")

    # 4. Training Loop
    print("\nStarting optimization...")
    history = []
    best_val_f1 = -1.0
    best_weights = np.array(weights).copy()

    n_samples = len(X_train_angles)
    train_start_time = time.time()

    for epoch in range(EPOCHS):
        epoch_start = time.time()
        perm = np.random.permutation(n_samples)
        batch_losses = []

        for start_idx in range(0, n_samples, BATCH_SIZE):
            idx = perm[start_idx:start_idx + BATCH_SIZE]
            bx = pnp.array(X_train_angles[idx], requires_grad=False)
            by = pnp.array(y_train_raw[idx], requires_grad=False)

            weights, b_loss = optimizer.step_and_cost(
                lambda w: loss_fn(w, bx, by),
                weights
            )
            batch_losses.append(float(b_loss))

        avg_loss = float(np.mean(batch_losses))
        history.append(avg_loss)

        # Validation at epoch end
        val_probs = np.concatenate([
            np.array(predict_batch(X_val_angles[k:k+256], weights))
            for k in range(0, len(X_val_angles), 256)
        ])
        val_preds = (val_probs >= 0.5).astype(int)
        val_f1 = float(f1_score(y_val, val_preds, average="macro"))
        val_acc = float(accuracy_score(y_val, val_preds))

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_weights = np.array(weights).copy()

        epoch_duration = time.time() - epoch_start
        print(
            f"Epoch {epoch+1:02d}/{EPOCHS} | "
            f"Train MSE: {avg_loss:.5f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val Macro-F1: {val_f1:.4f} | "
            f"Time: {epoch_duration:.2f}s"
        )

    training_time = round(time.time() - train_start_time, 3)
    print(f"\nTraining completed in {training_time}s.")

    # 5. Final Evaluation on Validation Set
    weights = pnp.array(best_weights, requires_grad=False)
    final_val_probs = np.concatenate([
        np.array(predict_batch(X_val_angles[k:k+256], weights))
        for k in range(0, len(X_val_angles), 256)
    ])
    final_val_preds = (final_val_probs >= 0.5).astype(int)

    acc = float(accuracy_score(y_val, final_val_preds))
    macro_f1 = float(f1_score(y_val, final_val_preds, average="macro"))
    macro_prec = float(precision_score(y_val, final_val_preds, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_val, final_val_preds, average="macro", zero_division=0))
    cm = confusion_matrix(y_val, final_val_preds, labels=[0, 1]).tolist()

    report = classification_report(
        y_val,
        final_val_preds,
        labels=[0, 1],
        target_names=["Normal", "Arrhythmia"],
        output_dict=True,
        zero_division=0
    )

    metrics = {
        "model": "4-qubit hybrid VQC",
        "simulator": "PennyLane default.qubit",
        "noise_condition": "ideal",
        "dataset": "QFF2026 Phase 2 PS-Q1",
        "seed": SEED,
        "qubits": N_QUBITS,
        "layers": N_LAYERS,
        "trainable_parameters": int(weights.size),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "training_samples": len(X_train_raw),
        "validation_samples": len(X_val_raw),
        "pca_explained_variance": round(explained_var, 6),
        "pca_variance_per_component": [round(float(r), 6) for r in pca.explained_variance_ratio_],
        "accuracy": acc,
        "macro_f1": macro_f1,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "confusion_matrix": cm,
        "classification_report": report,
        "epoch_losses": history,
        "training_time_seconds": training_time,
        "total_time_seconds": round(time.time() - start_total, 3)
    }

    # 6. Save Artifacts
    weights_path = OUT_MODEL / "vqc_4qubit_weights.npy"
    preproc_path = OUT_MODEL / "vqc_4qubit_preprocessing.joblib"
    metrics_path = OUT_METRICS / "vqc_4qubit_ideal.json"
    val_preds_path = OUT_METRICS / "vqc_4qubit_validation_predictions.csv"

    np.save(weights_path, np.array(weights))

    joblib.dump({
        "standardizer": standardizer,
        "pca": pca,
        "angle_scaler": angle_scaler,
        "feature_cols": feature_cols,
        "seed": SEED
    }, preproc_path)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    pd.DataFrame({
        "actual_label": y_val,
        "predicted_label": final_val_preds,
        "probability_arrhythmia": final_val_probs
    }).to_csv(val_preds_path, index=False)

    print("\n" + "=" * 65)
    print("4-QUBIT VQC IDEAL EVALUATION RESULTS")
    print("=" * 65)
    print(f"Accuracy:        {acc:.4f}")
    print(f"Macro-F1:        {macro_f1:.4f}")
    print(f"Macro Precision: {macro_prec:.4f}")
    print(f"Macro Recall:    {macro_rec:.4f}")
    print(f"Confusion Matrix (TN, FP / FN, TP):")
    print(f"  Normal:     [{cm[0][0]:4d}, {cm[0][1]:4d}]")
    print(f"  Arrhythmia: [{cm[1][0]:4d}, {cm[1][1]:4d}]")
    print(f"\nArtifacts saved:")
    print(f"  Model weights:      {weights_path.name}")
    print(f"  Preprocessing:      {preproc_path.name}")
    print(f"  Metrics JSON:       {metrics_path.name}")
    print(f"  Validation CSV:     {val_preds_path.name}")
    print("=" * 65)


if __name__ == "__main__":
    main()
