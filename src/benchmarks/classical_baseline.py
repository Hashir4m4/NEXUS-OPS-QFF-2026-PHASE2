"""
NEXUS-OPS | QFF 2026 Phase 2
Comprehensive Classical Baseline Models for PS-Q1 Benchmark
Fitted strictly on ecg_train.csv, evaluated on ecg_val.csv.
"""
from pathlib import Path
import json
import time
import sys
import joblib
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import get_dataset_dir, get_output_dir

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)

DATA = get_dataset_dir(problem="q1")
TRAIN_FILE = DATA / "ecg_train.csv"
VAL_FILE = DATA / "ecg_val.csv"

MODEL_DIR = ROOT / "outputs" / "models"
METRIC_DIR = ROOT / "outputs" / "metrics"
REPORT_DIR = ROOT / "reports"

for folder in [MODEL_DIR, METRIC_DIR, REPORT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


def evaluate_model(name, pipeline, X_train, y_train, X_val, y_val):
    start = time.time()
    pipeline.fit(X_train, y_train)
    train_time = round(time.time() - start, 4)

    preds = pipeline.predict(X_val)
    acc = float(accuracy_score(y_val, preds))
    macro_f1 = float(f1_score(y_val, preds, average="macro"))
    macro_prec = float(precision_score(y_val, preds, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_val, preds, average="macro", zero_division=0))
    cm = confusion_matrix(y_val, preds, labels=[0, 1]).tolist()

    return {
        "model": name,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "confusion_matrix": cm,
        "training_time_seconds": train_time,
        "pipeline": pipeline,
        "predictions": preds
    }


def main():
    print("=" * 65)
    print("NEXUS-OPS | PHASE 2 | REPRODUCIBLE CLASSICAL BASELINES")
    print("=" * 65)

    train = pd.read_csv(TRAIN_FILE)
    val = pd.read_csv(VAL_FILE)

    feature_columns = [c for c in train.columns if c != "label"]
    X_train = train[feature_columns]
    y_train = train["label"]
    X_val = val[feature_columns]
    y_val = val["label"]

    print(f"Loaded: {len(X_train)} train, {len(X_val)} val samples, {len(feature_columns)} features.")

    models_to_run = [
        ("Logistic Regression (Baseline)", Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000, random_state=42))
        ])),
        ("Linear SVM", Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(kernel="linear", C=1.0, random_state=42))
        ])),
        ("RBF SVM", Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(kernel="rbf", C=1.0, random_state=42))
        ])),
        ("Random Forest", Pipeline([
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
        ])),
        ("Logistic Regression (4 PCA components)", Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=4, random_state=42)),
            ("classifier", LogisticRegression(max_iter=2000, random_state=42))
        ]))
    ]

    all_results = {}
    print("\nTraining and evaluating models strictly on train/val splits:")
    for name, pipe in models_to_run:
        res = evaluate_model(name, pipe, X_train, y_train, X_val, y_val)
        all_results[name] = {
            "accuracy": res["accuracy"],
            "macro_f1": res["macro_f1"],
            "macro_precision": res["macro_precision"],
            "macro_recall": res["macro_recall"],
            "confusion_matrix": res["confusion_matrix"],
            "training_time_seconds": res["training_time_seconds"]
        }
        print(f"  {name:38s} | Acc: {res['accuracy']:.4f} | Macro-F1: {res['macro_f1']:.4f} | Time: {res['training_time_seconds']:.3f}s")

        if name == "Logistic Regression (Baseline)":
            joblib.dump(res["pipeline"], MODEL_DIR / "logistic_regression_baseline.joblib")
            val_df = pd.DataFrame({"true_label": y_val, "predicted_label": res["predictions"]})
            val_df.to_csv(METRIC_DIR / "classical_validation_predictions.csv", index=False)
            
            # Save single baseline json matching official schema
            single_metrics = {
                "model": "Logistic Regression",
                "dataset": "QFF2026 Phase 2 PS-Q1",
                "random_state": 42,
                "training_samples": len(X_train),
                "validation_samples": len(X_val),
                "feature_count": len(feature_columns),
                "accuracy": res["accuracy"],
                "macro_f1": res["macro_f1"],
                "macro_precision": res["macro_precision"],
                "macro_recall": res["macro_recall"],
                "confusion_matrix": res["confusion_matrix"],
                "training_time_seconds": res["training_time_seconds"]
            }
            with open(METRIC_DIR / "classical_baseline.json", "w") as f:
                json.dump(single_metrics, f, indent=4)

    # Save all baselines summary
    with open(METRIC_DIR / "classical_baselines_all.json", "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"\nSaved all baseline metrics to: {METRIC_DIR / 'classical_baselines_all.json'}")


if __name__ == "__main__":
    main()
