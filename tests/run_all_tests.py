"""
NEXUS-OPS | QFF 2026 Phase 2
Zero-dependency Test Runner (Runs in standard Python without extra packages)
"""
from pathlib import Path
import json
import sys
import pandas as pd
import numpy as np
import joblib

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT.parent / "QFF2026_QML_Track" / "QFF2026_QML_Challenge_Kit"
sys.path.insert(0, str(KIT / "tools"))
sys.path.insert(0, str(KIT / "benchmark"))

from validate_predictions import validate
from qff_qml_benchmark import validate_processor_d

DATA = KIT / "data" / "q1"
OUT_MODEL = ROOT / "outputs" / "models"
OUT_METRICS = ROOT / "outputs" / "metrics"
OUT_PRED = ROOT / "outputs" / "predictions"
CONFIG_FILE = ROOT / "configs" / "processor_d_config.json"


def run_tests():
    print("=" * 65)
    print("NEXUS-OPS | AUTOMATED TEST SUITE EXECUTION")
    print("=" * 65)

    tests = []

    # 1. Dataset Tests
    def test_datasets():
        train = pd.read_csv(DATA / "ecg_train.csv")
        val = pd.read_csv(DATA / "ecg_val.csv")
        test = pd.read_csv(DATA / "ecg_test_features.csv")

        assert train.shape == (6000, 31), f"train shape: {train.shape}"
        assert val.shape == (1500, 31), f"val shape: {val.shape}"
        assert test.shape == (1500, 31), f"test shape: {test.shape}"

        assert not train.isna().any().any(), "Train NaNs"
        assert not val.isna().any().any(), "Val NaNs"
        assert not test.isna().any().any(), "Test NaNs"

        assert set(train["label"].unique()) == {0, 1}, "Train labels"
        assert set(val["label"].unique()) == {0, 1}, "Val labels"
        assert test["sample_id"].tolist() == [f"s{i}" for i in range(1500)], "Test IDs"

    tests.append(("Dataset Integrity & Schema Checks", test_datasets))

    # 2. Preprocessing & Leakage Tests
    def test_preprocessing():
        preproc = joblib.load(OUT_MODEL / "vqc_4qubit_preprocessing.joblib")
        train_df = pd.read_csv(DATA / "ecg_train.csv")
        feature_cols = preproc["feature_cols"]
        X_train = train_df[feature_cols].to_numpy(dtype=np.float64)

        scaler = preproc["standardizer"]
        assert np.allclose(scaler.mean_, np.mean(X_train, axis=0), atol=1e-5), "Scaler mean leaked"
        assert np.allclose(scaler.var_, np.var(X_train, axis=0), atol=1e-5), "Scaler var leaked"

        pca = preproc["pca"]
        assert pca.n_components == 4, "PCA components != 4"
        assert np.isclose(np.sum(pca.explained_variance_ratio_), 0.8128, atol=1e-3), "PCA variance ratio"

    tests.append(("Data Leakage & Preprocessing Verification", test_preprocessing))

    # 3. Model Weights & Metrics
    def test_models():
        weights = np.load(OUT_MODEL / "vqc_4qubit_weights.npy")
        assert weights.shape == (2, 4, 3), f"Weight shape: {weights.shape}"
        assert np.all(np.isfinite(weights)), "Non-finite weights"

        with open(OUT_METRICS / "vqc_4qubit_ideal.json") as f:
            ideal = json.load(f)
        with open(OUT_METRICS / "vqc_4qubit_noisy.json") as f:
            noisy = json.load(f)

        assert ideal["macro_f1"] >= 0.98, f"Ideal Macro-F1 low: {ideal['macro_f1']}"
        assert noisy["macro_f1"] >= 0.95, f"Noisy Macro-F1 low: {noisy['macro_f1']}"

    tests.append(("Quantum Model Artifacts & Metric Thresholds", test_models))

    # 4. Task A Tests
    def test_task_a():
        with open(OUT_METRICS / "task_a_bell.json") as f:
            data = json.load(f)
        assert data["shots"] == 8192
        assert abs(data["ideal"]["fidelity"] - 1.0) < 1e-4
        assert 0.85 <= data["noisy"]["fidelity"] <= 0.91
        assert data["noisy"]["fidelity_se"] < 0.01

    tests.append(("Task A Bell-State Fidelity Benchmark", test_task_a))

    # 5. Task B Tests
    def test_task_b():
        with open(CONFIG_FILE) as f:
            cfg = json.load(f)
        errs = validate_processor_d(cfg, require_results=True)
        assert len(errs) == 0, f"Processor D errors: {errs}"
        b = cfg["benchmark_results"]
        assert b["qft3_fidelity_ideal"] == 1.0
        assert 0.5 <= b["qft3_fidelity_noisy"] <= 0.8

    tests.append(("Task B Processor D Schema & QFT-3 Benchmark", test_task_b))

    # 6. Official Prediction Validator
    def test_predictions():
        pred_file = OUT_PRED / "predictions_q1.csv"
        assert pred_file.exists(), "predictions_q1.csv missing"
        errs = validate("q1", str(pred_file))
        assert len(errs) == 0, f"Validator errors: {errs}"

        df = pd.read_csv(pred_file)
        assert len(df) == 1500, f"Expected 1500 rows, got {len(df)}"
        assert df["sample_id"].tolist() == [f"s{i}" for i in range(1500)]
        assert set(df["label"].unique()).issubset({0, 1})

    tests.append(("Official Prediction File & Validator Test", test_predictions))

    # Run all
    passed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"[PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"[FAIL] {name}: {e}")

    print("\n" + "=" * 65)
    print(f"TEST RESULTS: {passed}/{len(tests)} PASSED")
    print("=" * 65)

    if passed != len(tests):
        sys.exit(1)


if __name__ == "__main__":
    run_tests()
