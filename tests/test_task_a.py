"""
NEXUS-OPS | QFF 2026 Phase 2
Automated Test Suite: Task A Bell State Fidelity
"""
from pathlib import Path
import json
import pytest

ROOT = Path(__file__).resolve().parents[1]
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
    assert pytest.approx(ideal["XX"], abs=1e-4) == 1.0
    assert pytest.approx(ideal["YY"], abs=1e-4) == -1.0
    assert pytest.approx(ideal["ZZ"], abs=1e-4) == 1.0
    assert pytest.approx(ideal["fidelity"], abs=1e-4) == 1.0

    noisy = data["noisy"]
    # Reference noisy F ≈ 0.880
    assert 0.85 <= noisy["fidelity"] <= 0.91, f"Noisy fidelity outside expected range: {noisy['fidelity']}"
    assert noisy["fidelity_se"] < 0.01, f"Noisy fidelity standard error too high: {noisy['fidelity_se']}"

    # Verify parity calculation formula
    expected_f = (1.0 + noisy["XX"] - noisy["YY"] + noisy["ZZ"]) / 4.0
    assert pytest.approx(noisy["fidelity"], abs=1e-5) == expected_f

