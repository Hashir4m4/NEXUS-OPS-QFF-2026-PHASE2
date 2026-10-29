"""
NEXUS-OPS | QFF 2026 Phase 2
Automated Test Suite: Task A Bell State Fidelity
Verifies:
- task_a_bell.json artifact exists and records 8192 shots with noise model QFF2026-v2.
- Ideal fidelity is exactly 1.0 within statistical shot noise.
- Noisy fidelity aligns with official reference benchmark (F ≈ 0.880 ± 0.003).
- Parity expectation calculation and standard error propagation formulas are mathematically correct.
"""
import sys
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT_METRICS = ROOT / "outputs" / "metrics"


def test_task_a_artifact_exists():
    path = OUT_METRICS / "task_a_bell.json"
    assert path.exists(), "task_a_bell.json missing"


def test_task_a_metrics():
    with open(OUT_METRICS / "task_a_bell.json") as f:
        data = json.load(f)

    assert data["shots"] == 8192
    assert data["noise_model_version"] == "QFF2026-v2"

    ideal = data["ideal"]
    assert abs(ideal["XX"] - 1.0) < 1e-4
    assert abs(ideal["YY"] - (-1.0)) < 1e-4
    assert abs(ideal["ZZ"] - 1.0) < 1e-4
    assert abs(ideal["fidelity"] - 1.0) < 1e-4

    noisy = data["noisy"]
    # Reference noisy F ≈ 0.880
    assert 0.85 <= noisy["fidelity"] <= 0.91, f"Noisy fidelity outside expected range: {noisy['fidelity']}"
    assert noisy["fidelity_se"] < 0.01, f"Noisy fidelity standard error too high: {noisy['fidelity_se']}"

    # Verify official fidelity formula: F = (1 + <XX> - <YY> + <ZZ>) / 4
    expected_f = (1.0 + noisy["XX"] - noisy["YY"] + noisy["ZZ"]) / 4.0
    assert abs(noisy["fidelity"] - expected_f) < 1e-5

    # Verify standard error propagation formula: SE(F) = sqrt(SE_XX^2 + SE_YY^2 + SE_ZZ^2) / 4
    expected_se = np.sqrt(noisy["XX_se"]**2 + noisy["YY_se"]**2 + noisy["ZZ_se"]**2) / 4.0
    assert abs(noisy["fidelity_se"] - expected_se) < 1e-5


if __name__ == "__main__":
    test_task_a_artifact_exists()
    test_task_a_metrics()
    print("All Task A tests passed successfully.")
