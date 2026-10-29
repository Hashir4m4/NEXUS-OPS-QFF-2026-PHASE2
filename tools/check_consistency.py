import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

def check_notebook(nb_path):
    print(f"\n--- Checking Notebook: {nb_path.name} ---")
    if not nb_path.exists():
        print("Does not exist!")
        return
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    print(f"Total cells: {len(nb.get('cells', []))}")
    for idx, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") == "code":
            outs = cell.get("outputs", [])
            for out in outs:
                lines = out.get("text", [])
                if isinstance(lines, str):
                    lines = [lines]
                for l in lines:
                    low = l.lower()
                    if any(k in low for k in ["macro-f1", "accuracy", "confusion", "noisy", "ideal"]):
                        print(f"  Cell {idx}: {l.strip()[:100]}")

def check_artifacts():
    print("\n--- Checking Saved Artifacts ---")
    metrics_dir = ROOT / "outputs" / "metrics"
    models_dir = ROOT / "outputs" / "models"
    preds_dir = ROOT / "outputs" / "predictions"
    
    for mf in sorted(metrics_dir.glob("*.json")):
        with open(mf, "r", encoding="utf-8") as f:
            data = json.load(f)
        acc = data.get("accuracy", "N/A")
        f1 = data.get("macro_f1", "N/A")
        cm = data.get("confusion_matrix", "N/A")
        print(f"  Metric {mf.name}: acc={acc}, macro_f1={f1}, cm={cm}")

    weights_file = models_dir / "vqc_4qubit_weights.npy"
    if weights_file.exists():
        w = np.load(weights_file)
        print(f"  Weights file: shape={w.shape}, min={w.min():.4f}, max={w.max():.4f}, mean={w.mean():.4f}")

    val_preds_file = metrics_dir / "vqc_4qubit_validation_predictions.csv"
    if val_preds_file.exists():
        df_val = pd.read_csv(val_preds_file)
        print(f"  Validation predictions: rows={len(df_val)}, cols={list(df_val.columns)}")
        if "pred_class" in df_val.columns and "true_label" in df_val.columns:
            from sklearn.metrics import confusion_matrix, f1_score
            cm = confusion_matrix(df_val["true_label"], df_val["pred_class"])
            f1 = f1_score(df_val["true_label"], df_val["pred_class"], average="macro")
            print(f"  Computed from CSV: Macro-F1={f1:.4f}, Confusion Matrix:\n{cm}")

if __name__ == "__main__":
    check_artifacts()
    check_notebook(ROOT / "solution.ipynb")
    check_notebook(ROOT / "notebooks" / "solution.ipynb")
