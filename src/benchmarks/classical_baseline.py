from pathlib import Path
import json
import time
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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
DATA = KIT / "data" / "q1"

TRAIN_FILE = DATA / "ecg_train.csv"
VAL_FILE = DATA / "ecg_val.csv"

MODEL_DIR = ROOT / "outputs" / "models"
METRIC_DIR = ROOT / "outputs" / "metrics"
REPORT_DIR = ROOT / "reports"

for folder in [MODEL_DIR, METRIC_DIR, REPORT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

def main():
    print("=" * 65)
    print("NEXUS-OPS | PHASE 2 | CLASSICAL BASELINE")
    print("=" * 65)

    start = time.time()

    train = pd.read_csv(TRAIN_FILE)
    val = pd.read_csv(VAL_FILE)

    feature_columns = [c for c in train.columns if c != "label"]

    X_train = train[feature_columns]
    y_train = train["label"]

    X_val = val[feature_columns]
    y_val = val["label"]

    print("\nTraining samples:", len(X_train))
    print("Validation samples:", len(X_val))
    print("Features:", len(feature_columns))

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=2000,
            random_state=42
        ))
    ])

    print("\nTraining Logistic Regression...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_val)

    metrics = {
        "model": "Logistic Regression",
        "dataset": "QFF2026 Phase 2 PS-Q1",
        "random_state": 42,
        "training_samples": len(X_train),
        "validation_samples": len(X_val),
        "feature_count": len(feature_columns),
        "accuracy": float(accuracy_score(y_val, predictions)),
        "macro_f1": float(f1_score(y_val, predictions, average="macro")),
        "macro_precision": float(
            precision_score(y_val, predictions, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_val, predictions, average="macro", zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(
            y_val, predictions, labels=[0, 1]
        ).tolist(),
        "training_time_seconds": round(time.time() - start, 3)
    }

    metrics["classification_report"] = classification_report(
        y_val,
        predictions,
        labels=[0, 1],
        target_names=["Normal", "Arrhythmia"],
        output_dict=True,
        zero_division=0
    )

    model_path = MODEL_DIR / "logistic_regression_baseline.joblib"
    metrics_path = METRIC_DIR / "classical_baseline.json"
    predictions_path = METRIC_DIR / "classical_validation_predictions.csv"

    joblib.dump(model, model_path)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    pd.DataFrame({
        "actual_label": y_val,
        "predicted_label": predictions
    }).to_csv(predictions_path, index=False)

    print("\n" + "=" * 65)
    print("VALIDATION RESULTS")
    print("=" * 65)
    print(f"Accuracy:       {metrics['accuracy']:.4f}")
    print(f"Macro-F1:       {metrics['macro_f1']:.4f}")
    print(f"Macro-Precision:{metrics['macro_precision']:.4f}")
    print(f"Macro-Recall:   {metrics['macro_recall']:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_val, predictions, labels=[0, 1]))

    print("\nClassification Report:")
    print(classification_report(
        y_val,
        predictions,
        labels=[0, 1],
        target_names=["Normal", "Arrhythmia"],
        zero_division=0
    ))

    print("\nSaved model:", model_path)
    print("Saved metrics:", metrics_path)
    print("Saved validation predictions:", predictions_path)
    print("\nTraining time:", metrics["training_time_seconds"], "seconds")
    print("\nBASELINE COMPLETED SUCCESSFULLY")

if __name__ == "__main__":
    main()
