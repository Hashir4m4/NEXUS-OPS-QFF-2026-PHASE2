from pathlib import Path
import pandas as pd
import json

ROOT = Path(__file__).resolve().parents[2]
KIT = ROOT.parent / "QFF2026_QML_Track" / "QFF2026_QML_Challenge_Kit"
DATA = KIT / "data" / "q1"

FILES = {
    "train": DATA / "ecg_train.csv",
    "validation": DATA / "ecg_val.csv",
    "test": DATA / "ecg_test_features.csv",
}

def main():
    print("=" * 65)
    print("NEXUS-OPS | PHASE 2 | Q1 DATASET AUDIT")
    print("=" * 65)

    train = pd.read_csv(FILES["train"])
    val = pd.read_csv(FILES["validation"])
    test = pd.read_csv(FILES["test"])

    features = [c for c in train.columns if c != "label"]

    checks = {
        "Training shape": train.shape == (6000, 31),
        "Validation shape": val.shape == (1500, 31),
        "Test shape": test.shape == (1500, 31),
        "Training and validation columns match":
            list(train.columns) == list(val.columns),
        "Test feature columns match":
            features == [c for c in test.columns if c != "sample_id"],
        "Training labels valid": set(train["label"].unique()) == {0, 1},
        "Validation labels valid": set(val["label"].unique()) == {0, 1},
        "No missing values": not (
            train.isna().any().any()
            or val.isna().any().any()
            or test.isna().any().any()
        ),
        "No duplicate test IDs": not test["sample_id"].duplicated().any(),
        "Test IDs complete":
            set(test["sample_id"]) == {f"s{i}" for i in range(1500)},
    }

    for name, passed in checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")

    print("\nTraining class distribution:")
    print(train["label"].value_counts().sort_index())

    print("\nValidation class distribution:")
    print(val["label"].value_counts().sort_index())

    report = {
        "project": "NEXUS-OPS Phase 2 PS-Q1",
        "train_shape": list(train.shape),
        "validation_shape": list(val.shape),
        "test_shape": list(test.shape),
        "feature_count": len(features),
        "training_class_distribution":
            {str(k): int(v) for k, v in train["label"].value_counts().items()},
        "validation_class_distribution":
            {str(k): int(v) for k, v in val["label"].value_counts().items()},
        "checks": checks,
        "all_passed": all(checks.values()),
    }

    output = ROOT / "reports" / "dataset_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=4), encoding="utf-8")

    print("\nAudit report:", output)
    print("Overall status:", "PASS" if all(checks.values()) else "FAIL")

    if not all(checks.values()):
        raise SystemExit("Dataset audit failed. Do not proceed.")

if __name__ == "__main__":
    main()

